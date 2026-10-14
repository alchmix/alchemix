#!/usr/bin/env python3
"""
Delete videos older than N days from `outputs/` and MongoDB.

Usage (from backend root, same folder as app.py / .env):
    python cleanup_old_videos.py              # dry-run, 7 days
    python cleanup_old_videos.py --apply        # actually delete
    python cleanup_old_videos.py --apply --yes  # skip confirmation
    python cleanup_old_videos.py --outputs-only --apply --yes  # disk only, fast

Uses mongodb.mongo_instance when available (same as app.py), else MONGODB_URI / MONGO_URI.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import env_loader  # noqa: F401
except ImportError:
    pass


def _log(msg: str) -> None:
    print(msg, flush=True)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_created_at(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, str):
        try:
            s = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except ValueError:
            return None
    return None


def _cleanup_mongo_client():
    """Dedicated client with longer socket timeout for bulk scans (not app 45s default)."""
    import certifi
    from pymongo import MongoClient

    uri = (os.getenv("MONGODB_URI") or os.getenv("MONGO_URI") or "").strip()
    if not uri:
        raise RuntimeError("MONGODB_URI or MONGO_URI is not set (check backend/.env)")
    db_name = (
        os.getenv("MONGODB_DB") or os.getenv("MONGO_DB_NAME") or "AlchemyMix"
    ).strip()
    connect_ms = max(5000, int(os.getenv("MONGODB_SERVER_SELECTION_MS", "15000")))
    socket_ms = max(
        connect_ms,
        int(os.getenv("MONGODB_CLEANUP_SOCKET_TIMEOUT_MS", "300000")),
    )
    client = MongoClient(
        uri,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=connect_ms,
        connectTimeoutMS=connect_ms,
        socketTimeoutMS=socket_ms,
        retryWrites=True,
    )
    client.admin.command("ping")
    return client[db_name]


def _get_db():
    # Prefer a maintenance client so large find/count/delete ops are not capped at 45s.
    try:
        return _cleanup_mongo_client()
    except Exception as exc:
        try:
            from mongodb import mongo_instance

            return mongo_instance.get_db()
        except Exception:
            raise exc


def _outputs_dir() -> Path:
    raw = os.getenv("OUTPUT_FOLDER") or os.getenv("OUTPUT_DIR") or "outputs"
    return Path(raw).resolve()


def _age_query(cutoff: datetime):
    """Match by created_at OR legacy docs with only ObjectId timestamp."""
    from bson import ObjectId

    cutoff_oid = ObjectId.from_datetime(
        cutoff.replace(tzinfo=timezone.utc) if cutoff.tzinfo is None else cutoff
    )
    return {
        "$or": [
            {"created_at": {"$lt": cutoff}},
            {"created_at": {"$exists": False}, "_id": {"$lt": cutoff_oid}},
            {"created_at": None, "_id": {"$lt": cutoff_oid}},
        ]
    }


def _collect_db_video_paths(doc: dict) -> set[str]:
    names: set[str] = set()
    for key in ("video_path", "intro_path", "body_path", "final_path"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            names.add(os.path.basename(val.strip()))
    return names


def _scan_outputs(outputs: Path, cutoff: datetime, *, apply: bool) -> tuple[int, int]:
    video_ext = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    deleted = 0
    scanned = 0

    if not outputs.is_dir():
        _log(f"\nWARNING: outputs folder does not exist: {outputs}")
        return 0, 0

    _log(f"\nScanning {outputs} ...")
    for path in outputs.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in video_ext:
            continue
        scanned += 1
        if scanned % 100 == 0:
            _log(f"  ... scanned {scanned} video files")

        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(
            tzinfo=None
        )
        if mtime >= cutoff:
            continue

        if apply:
            try:
                path.unlink()
                deleted += 1
            except OSError as exc:
                _log(f"  ! could not delete {path.name}: {exc}")
        else:
            deleted += 1  # count as "would delete" in dry-run

    return scanned, deleted


def _cleanup_collection(
    db,
    collection_name: str,
    query: dict,
    projection: dict,
    *,
    apply: bool,
    sample: int = 10,
    extra_label: str = "",
) -> tuple[int, set[str]]:
    col = db[collection_name]
    _log(f"\nQuerying {collection_name}{extra_label} ...")
    _log("  (large collections may take several minutes without a created_at index)")

    count: int | None = None
    try:
        count = col.count_documents(query, maxTimeMS=120_000)
        _log(f"  matched: {count}")
        if count == 0:
            return 0, set()
    except Exception as exc:
        _log(f"  ! count skipped ({exc}); streaming matches instead ...")

    video_names: set[str] = set()
    shown = 0
    streamed = 0
    cursor = col.find(query, projection, max_time_ms=300_000).batch_size(100)

    for doc in cursor:
        streamed += 1
        video_names.update(_collect_db_video_paths(doc))
        if shown < sample:
            created = _parse_created_at(doc.get("created_at"))
            title = (doc.get("title") or doc.get("status") or "")[:60]
            _log(f"  - {doc.get('_id')} | {created} | {title!r}")
            shown += 1
        if streamed % 500 == 0:
            _log(f"  ... processed {streamed} document(s)")

    if count is None:
        count = streamed
        _log(f"  matched (streamed): {count}")
    elif count > sample:
        _log(f"  ... and {max(0, count - sample)} more")

    if count == 0:
        return 0, set()

    if apply and count > 0:
        _log(f"  deleting {count} row(s) from {collection_name} ...")
        # delete_many has no max_time_ms; cleanup client uses a longer socketTimeoutMS.
        result = col.delete_many(query)
        _log(f"  deleted: {result.deleted_count}")

    return count, video_names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove videos older than N days from outputs/ and MongoDB."
    )
    parser.add_argument("--days", type=int, default=7, help="Age threshold (default: 7)")
    parser.add_argument("--apply", action="store_true", help="Actually delete")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation with --apply")
    parser.add_argument(
        "--outputs-only",
        action="store_true",
        help="Only delete old files in outputs/ (skip MongoDB)",
    )
    args = parser.parse_args()

    if args.days < 1:
        _log("ERROR: --days must be >= 1")
        return 1

    cutoff = _utcnow() - timedelta(days=args.days)
    outputs = _outputs_dir()
    mode = "APPLY" if args.apply else "DRY-RUN"

    _log(f"=== Video cleanup ({mode}) ===")
    _log(f"Cutoff: {cutoff.isoformat()} UTC (older than {args.days} day(s))")
    _log(f"Outputs folder: {outputs}")
    _log("")

    if args.apply and not args.yes:
        answer = input("Type yes to permanently delete old videos: ").strip().lower()
        if answer not in ("yes", "y"):
            _log("Aborted.")
            return 0

    files_from_db: set[str] = set()
    stats = {
        "completed_videos_db": 0,
        "video_data_db": 0,
        "dreamface_jobs_db": 0,
        "outputs_scanned": 0,
        "outputs_deleted": 0,
        "db_files_deleted": 0,
    }

    if not args.outputs_only:
        _log("Connecting to MongoDB ...")
        db = _get_db()
        _log("Connected.")

        age_q = _age_query(cutoff)
        cv_projection = {"title": 1, "created_at": 1, "video_path": 1, "intro_path": 1, "body_path": 1, "final_path": 1}

        n, names = _cleanup_collection(
            db, "completed_videos", age_q, cv_projection, apply=args.apply
        )
        stats["completed_videos_db"] = n
        files_from_db.update(names)

        vd_query = {
            **age_q,
            "status": {"$in": ["completed", "failed", "processing", "awaiting_dreamface"]},
        }
        n, _ = _cleanup_collection(
            db,
            "video_data",
            vd_query,
            {"title": 1, "created_at": 1, "status": 1},
            apply=args.apply,
            extra_label=" (non-draft)",
        )
        stats["video_data_db"] = n

        n, _ = _cleanup_collection(
            db,
            "dreamface_jobs",
            age_q,
            {"created_at": 1, "status": 1, "user_id": 1},
            apply=args.apply,
        )
        stats["dreamface_jobs_db"] = n

        if files_from_db:
            _log(f"\nDeleting {len(files_from_db)} file(s) referenced in DB ...")
            for name in sorted(files_from_db):
                path = outputs / name
                if not path.is_file():
                    continue
                if args.apply:
                    try:
                        path.unlink()
                        stats["db_files_deleted"] += 1
                    except OSError as exc:
                        _log(f"  ! {name}: {exc}")
                else:
                    stats["db_files_deleted"] += 1
                    _log(f"  would delete: {path}")

    scanned, deleted = _scan_outputs(outputs, cutoff, apply=args.apply)
    stats["outputs_scanned"] = scanned
    stats["outputs_deleted"] = deleted

    _log("")
    _log("=== Summary ===")
    if not args.outputs_only:
        _log(f"  completed_videos removed: {stats['completed_videos_db']}")
        _log(f"  video_data removed:       {stats['video_data_db']}")
        _log(f"  dreamface_jobs removed:   {stats['dreamface_jobs_db']}")
        _log(f"  DB-linked files removed:  {stats['db_files_deleted']}")
    _log(f"  outputs scanned:          {stats['outputs_scanned']}")
    if args.apply:
        _log(f"  outputs files deleted:    {stats['outputs_deleted']}")
    else:
        _log(f"  outputs files to delete:  {stats['outputs_deleted']}")
        _log("  (dry-run — use --apply to delete for real)")

    _log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
