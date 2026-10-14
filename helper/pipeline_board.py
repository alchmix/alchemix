"""
Unified pipeline board for admin dashboard — merges queues, DreamFace, drafts, events.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

from bson.objectid import ObjectId

BOARD_MAX_AGE_DAYS = max(1, int(os.getenv("PIPELINE_BOARD_MAX_AGE_DAYS", "3")))


def board_age_cutoff(now: datetime | None = None) -> datetime:
    now = now or datetime.utcnow()
    return now - timedelta(days=BOARD_MAX_AGE_DAYS)


def document_timestamp(doc: dict) -> datetime | None:
    created = doc.get("created_at")
    if isinstance(created, datetime):
        return created
    oid = doc.get("_id")
    if not oid:
        return None
    try:
        if isinstance(oid, str):
            oid = ObjectId(oid)
        return oid.generation_time.replace(tzinfo=None)
    except Exception:
        return None


def is_recent_board_job(doc: dict, cutoff: datetime) -> bool:
    ts = document_timestamp(doc)
    if ts is None:
        return True
    return ts >= cutoff


def filter_recent_jobs(docs: list[dict], cutoff: datetime) -> list[dict]:
    return [d for d in docs if is_recent_board_job(d, cutoff)]


def board_mongo_cutoff_filter(cutoff: datetime) -> dict:
    """Mongo filter: created_at within window, or legacy docs via ObjectId time."""
    min_oid = ObjectId.from_datetime(cutoff)
    return {
        "$or": [
            {"created_at": {"$gte": cutoff}},
            {"created_at": {"$exists": False}, "_id": {"$gte": min_oid}},
        ]
    }


def _datetime_to_iso(value: datetime | str | None) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    return None


def _unix_to_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.utcfromtimestamp(ts).isoformat() + "Z"


def _doc_created_iso(doc: dict | None) -> str | None:
    if not doc:
        return None
    return _datetime_to_iso(document_timestamp(doc))


def _resolve_job_link(
    *,
    job_type: str,
    dreamface_job_id: str | None = None,
    vid_data_id: str | None = None,
    ref_id: str | None = None,
) -> dict[str, str | None]:
    dreamface_link = None
    compose_link = None

    df_id = dreamface_job_id or (ref_id if job_type == "dreamface" else None)
    if df_id:
        dreamface_link = f"/dreamface/{df_id}"

    vid = vid_data_id or (ref_id if job_type in ("standard", "regenerate") else None)
    if vid:
        compose_link = f"/compose/{vid}"

    if job_type == "dreamface":
        primary = dreamface_link or compose_link
    else:
        primary = compose_link or dreamface_link

    return {
        "link": primary,
        "dreamface_link": dreamface_link,
        "compose_link": compose_link,
    }


def _phase_label(item: dict) -> str:
    status = item.get("status", "")
    job_type = item.get("job_type", "")
    column = item.get("column", "")

    if column == "queued":
        return "Waiting in video queue"
    if status == "awaiting_dreamface":
        return "Waiting for avatar links (Talking Heads)"
    if status == "failed":
        return "Failed — needs retry or reset"
    if job_type == "dreamface":
        return "Talking Heads — encode & composite"
    if job_type == "regenerate":
        return "Regenerating video"
    if column == "running":
        return "Audio → segments → merge"
    return "Processing"


def _board_item(
    *,
    item_id: str,
    title: str,
    user_id: str,
    user_name: str,
    job_type: str,
    status: str,
    column: str,
    runtime_seconds: float | None = None,
    ref_id: str | None = None,
    dreamface_job_id: str | None = None,
    vid_data_id: str | None = None,
    queue_id: str | None = None,
    stuck: bool = False,
    error: str | None = None,
    created_at: str | None = None,
    sort_at: str | None = None,
    extra: dict | None = None,
) -> dict:
    links = _resolve_job_link(
        job_type=job_type,
        dreamface_job_id=dreamface_job_id,
        vid_data_id=vid_data_id,
        ref_id=ref_id,
    )

    item = {
        "id": item_id,
        "title": title or "Untitled video",
        "user_id": user_id,
        "user_name": user_name or user_id,
        "job_type": job_type,
        "status": status,
        "column": column,
        "phase_label": _phase_label({"status": status, "job_type": job_type, "column": column}),
        "runtime_seconds": runtime_seconds,
        "ref_id": ref_id,
        "dreamface_job_id": dreamface_job_id,
        "vid_data_id": vid_data_id,
        "queue_id": queue_id,
        "link": links["link"],
        "dreamface_link": links["dreamface_link"],
        "compose_link": links["compose_link"],
        "stuck": stuck,
        "error": error,
        "created_at": created_at,
        "sort_at": sort_at or created_at,
    }
    if extra:
        item.update(extra)
    return item


def compute_health(base: dict) -> dict:
    stall_sec = float(base.get("stall_seconds") or 0)
    stall_limit = float(base.get("stall_unstick_sec") or 300)
    slots_used = int(base.get("global_in_flight") or 0)
    slots_cap = int(base.get("global_in_flight_cap") or 0)
    running = len(base.get("running_video_jobs") or [])
    queued = len(base.get("queued_video_jobs") or [])

    stalled = stall_sec >= stall_limit and slots_used > 0
    slots_full = slots_cap > 0 and slots_used >= slots_cap

    if stalled:
        return {
            "level": "critical",
            "label": "Stalled",
            "message": (
                f"No segment progress for {int(stall_sec)}s with {slots_used}/{slots_cap} "
                "slots held — use Force recovery if CPU is idle."
            ),
        }
    if slots_full and queued > 0:
        return {
            "level": "warning",
            "label": "Contended",
            "message": f"All {slots_cap} segment slots in use — {queued} video(s) waiting.",
        }
    if running >= int(base.get("concurrent_video_jobs_limit") or 1):
        return {
            "level": "busy",
            "label": "At capacity",
            "message": f"{running} video job(s) running at parallel limit.",
        }
    return {
        "level": "healthy",
        "label": "Healthy",
        "message": "Pipeline accepting work normally.",
    }


def build_pipeline_board(
    *,
    base: dict,
    dreamface_jobs: list[dict],
    video_drafts: list[dict],
    user_names: dict[str, str],
    job_events: list[dict],
    recent_completions: list[dict],
) -> dict:
    """Organize all work into board columns + activity feed."""
    cutoff = board_age_cutoff()
    dreamface_jobs = filter_recent_jobs(dreamface_jobs, cutoff)
    video_drafts = filter_recent_jobs(video_drafts, cutoff)
    recent_completions = filter_recent_jobs(recent_completions, cutoff)

    columns: dict[str, list[dict]] = {
        "running": [],
        "queued": [],
        "awaiting": [],
        "failed": [],
    }
    seen_ids: set[str] = set()

    def uname(email: str | None) -> str:
        if not email:
            return "—"
        return user_names.get(email, email)

    df_vid_map = {
        str(j.get("_id")): str(j["vid_data_id"])
        for j in dreamface_jobs
        if j.get("vid_data_id")
    }

    for job in base.get("running_video_jobs") or []:
        qid = job.get("queue_id") or job.get("ref_id") or ""
        seen_ids.add(f"run-{qid}")
        jtype = job.get("job_type") or "standard"
        df_id = job.get("ref_id") if jtype == "dreamface" else None
        vid_id = (
            df_vid_map.get(str(job.get("ref_id") or ""))
            if jtype == "dreamface"
            else (job.get("ref_id") if jtype in ("standard", "regenerate") else None)
        )
        started_iso = _unix_to_iso(job.get("started_at"))
        columns["running"].append(
            _board_item(
                item_id=f"run-{qid}",
                title=job.get("label") or job.get("func_name"),
                user_id=job.get("user_id"),
                user_name=uname(job.get("user_id")),
                job_type=jtype,
                status="running",
                column="running",
                runtime_seconds=job.get("running_seconds"),
                ref_id=job.get("ref_id"),
                dreamface_job_id=df_id,
                vid_data_id=vid_id,
                queue_id=job.get("queue_id"),
                created_at=started_iso,
                sort_at=started_iso,
            )
        )

    for job in base.get("queued_video_jobs") or []:
        qid = job.get("queue_id") or ""
        seen_ids.add(f"q-{qid}")
        jtype = job.get("job_type") or "standard"
        df_id = job.get("ref_id") if jtype == "dreamface" else None
        vid_id = (
            df_vid_map.get(str(job.get("ref_id") or ""))
            if jtype == "dreamface"
            else (job.get("ref_id") if jtype in ("standard", "regenerate") else None)
        )
        enqueued_iso = _unix_to_iso(job.get("enqueued_at"))
        columns["queued"].append(
            _board_item(
                item_id=f"q-{qid}",
                title=job.get("label") or job.get("func_name"),
                user_id=job.get("user_id"),
                user_name=uname(job.get("user_id")),
                job_type=jtype,
                status="queued",
                column="queued",
                runtime_seconds=job.get("wait_seconds"),
                ref_id=job.get("ref_id"),
                dreamface_job_id=df_id,
                vid_data_id=vid_id,
                queue_id=job.get("queue_id"),
                created_at=enqueued_iso,
                sort_at=enqueued_iso,
                extra={"position": job.get("position")},
            )
        )

    running_df_refs = {
        j.get("ref_id") for j in (base.get("running_video_jobs") or [])
        if j.get("job_type") == "dreamface" and j.get("ref_id")
    }

    for job in dreamface_jobs:
        jid = str(job.get("_id", ""))
        status = job.get("status") or "unknown"
        recovery = job.get("recovery") or {}
        stuck = bool(recovery.get("is_likely_stuck"))
        title = job.get("title") or f"Talking Heads · {jid[-8:]}"
        user_id = job.get("user_id")
        vid_data_id = str(job.get("vid_data_id")) if job.get("vid_data_id") else None
        created_iso = _datetime_to_iso(job.get("created_at")) or _doc_created_iso(job)
        processing_iso = _datetime_to_iso(job.get("processing_started_at"))

        if status == "awaiting_dreamface":
            if f"df-{jid}" not in seen_ids:
                columns["awaiting"].append(
                    _board_item(
                        item_id=f"df-{jid}",
                        title=title,
                        user_id=user_id,
                        user_name=uname(user_id),
                        job_type="dreamface",
                        status=status,
                        column="awaiting",
                        dreamface_job_id=jid,
                        vid_data_id=vid_data_id,
                        created_at=created_iso,
                        sort_at=created_iso,
                        extra={"batches": len(job.get("batches") or [])},
                    )
                )
        elif status == "failed":
            columns["failed"].append(
                _board_item(
                    item_id=f"df-{jid}",
                    title=title,
                    user_id=user_id,
                    user_name=uname(user_id),
                    job_type="dreamface",
                    status=status,
                    column="failed",
                    dreamface_job_id=jid,
                    vid_data_id=vid_data_id,
                    error=(job.get("error") or "")[:200],
                    created_at=created_iso,
                    sort_at=created_iso,
                    extra={"can_retry": recovery.get("can_retry")},
                )
            )
        elif status == "processing" and jid not in running_df_refs:
            columns["running"].append(
                _board_item(
                    item_id=f"df-{jid}",
                    title=title,
                    user_id=user_id,
                    user_name=uname(user_id),
                    job_type="dreamface",
                    status=status,
                    column="running",
                    runtime_seconds=recovery.get("processing_seconds"),
                    dreamface_job_id=jid,
                    vid_data_id=vid_data_id,
                    stuck=stuck,
                    created_at=created_iso,
                    sort_at=processing_iso or created_iso,
                    extra={"can_retry": recovery.get("can_retry"), "can_reset": recovery.get("can_reset")},
                )
            )

    for draft in video_drafts:
        did = str(draft.get("_id", ""))
        status = draft.get("status") or "unknown"
        if status == "awaiting_dreamface":
            continue
        df_id = str(draft.get("dreamface_job_id")) if draft.get("dreamface_job_id") else None
        title = draft.get("title") or f"Video · {did[-8:]}"
        user_id = draft.get("user_id")
        created_iso = _datetime_to_iso(draft.get("created_at")) or _doc_created_iso(draft)

        if status == "failed":
            columns["failed"].append(
                _board_item(
                    item_id=f"vd-{did}",
                    title=title,
                    user_id=user_id,
                    user_name=uname(user_id),
                    job_type="standard",
                    status=status,
                    column="failed",
                    vid_data_id=did,
                    dreamface_job_id=df_id,
                    created_at=created_iso,
                    sort_at=created_iso,
                )
            )
        elif status == "processing":
            already = any(
                c.get("vid_data_id") == did or c.get("ref_id") == did
                for col in columns.values()
                for c in col
            )
            if not already:
                columns["running"].append(
                    _board_item(
                        item_id=f"vd-{did}",
                        title=title,
                        user_id=user_id,
                        user_name=uname(user_id),
                        job_type="standard",
                        status=status,
                        column="running",
                        vid_data_id=did,
                        dreamface_job_id=df_id,
                        created_at=created_iso,
                        sort_at=created_iso,
                    )
                )

    for col_name in columns:
        if col_name == "queued":
            columns[col_name].sort(
                key=lambda item: item.get("position") if item.get("position") is not None else 9999,
            )
        else:
            columns[col_name].sort(
                key=lambda item: item.get("sort_at") or item.get("created_at") or "",
                reverse=True,
            )

    activity: list[dict] = []
    for ev in job_events[:30]:
        activity.append(
            {
                "kind": "event",
                "at": ev.get("at"),
                "type": ev.get("type"),
                "title": ev.get("label"),
                "user_name": uname(ev.get("user_id")),
                "job_type": ev.get("job_type"),
                "message": ev.get("message"),
                "duration_seconds": ev.get("duration_seconds"),
            }
        )

    for vid in recent_completions[:15]:
        created = vid.get("created_at")
        if isinstance(created, datetime):
            created = created.isoformat() + "Z"
        activity.append(
            {
                "kind": "completion",
                "at": created,
                "type": "completed",
                "title": vid.get("title") or vid.get("video_path"),
                "user_name": uname(vid.get("user_id")),
                "job_type": "video",
                "message": "Video saved to completed library",
                "video_path": vid.get("video_path"),
            }
        )

    activity.sort(key=lambda x: x.get("at") or "", reverse=True)
    activity = activity[:40]

    health = compute_health(base)
    summary = {
        "running": len(columns["running"]),
        "queued": len(columns["queued"]),
        "awaiting": len(columns["awaiting"]),
        "failed": len(columns["failed"]),
        "active_video_slots": base.get("active_video_jobs"),
        "video_slot_limit": base.get("concurrent_video_jobs_limit"),
        "segment_slots_used": base.get("global_in_flight"),
        "segment_slots_cap": base.get("global_in_flight_cap"),
        "script_queue": base.get("script_queue_depth"),
        "stall_seconds": base.get("stall_seconds"),
    }

    return {
        "health": health,
        "summary": summary,
        "columns": columns,
        "activity": activity,
        "max_age_days": BOARD_MAX_AGE_DAYS,
    }
