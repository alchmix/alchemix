"""
Visual Engine — the AlchemyMix side. Logged-in users ask for documentary videos here; this backend checks who they
are, forwards the request to the engine (signed), and turns the engine's callbacks into completed_videos records
and Telegram notices.

    The flow (manual — Claude writes the plan):
      1. POST videos        script + channel  → the engine makes the voice and timings (Telegram: "timings ready")
      2. GET  package       copy the timings, rulebook, instructions … into Claude
      3. POST plan          paste Claude's JSON → checked at once (blocking problems / warnings)
      4. POST render        render the plan → video (re-render with an improved plan any time, within limits)

    GET    /api/visual-engine/options                 voices, looks, genres, modes, limits (for the form)
    GET    /api/visual-engine/channels                my channels with their brand kits
    PUT    /api/visual-engine/channels/<id>/brand-kit {voice, look, genre, accent, notes, visual_mode}
    POST   /api/visual-engine/videos                  start a project
           {"title", "script", "channel_id", "voice"?, "look"?, "genre"?, "accent"?, "brand_notes"?,
            "visual_mode"?: "footage"|"mograph", "save_brand_kit"?: true}
           missing fields come from the channel's brand kit; header Idempotency-Key recommended
    GET    /api/visual-engine/videos/<id>/package     the files for Claude + "copy_all"
    GET    /api/visual-engine/videos/<id>/package.zip
    POST   /api/visual-engine/videos/<id>/plan        {"plan": "<Claude's JSON>"} → video.plan {ok, blocking, warnings}
    POST   /api/visual-engine/videos/<id>/render
    GET    /api/visual-engine/videos                  my videos (newest first; ?status=queued,running&before=<id>)
    GET    /api/visual-engine/videos/<id>             one video: status, stage, progress, queue position, links
    POST   /api/visual-engine/videos/<id>/cancel
    POST   /api/visual-engine/videos/<id>/retry
    GET    /api/visual-engine/videos/<id>/download    the finished MP4

    admin only (users.status == "admin"):
    GET    /api/visual-engine/admin/status            engine readiness + queue
    GET    /api/visual-engine/admin/videos            everyone's videos (?user=email)
    POST   /api/visual-engine/admin/videos/<id>/priority   {"priority": 10}
    POST   /api/visual-engine/admin/videos/<id>/fail       {"reason": "..."}
    POST   /api/visual-engine/admin/callbacks/resend
    GET    /api/visual-engine/admin/audit

    engine → this backend (no user login; HMAC-signed, see _verify_callback):
    POST   /api/visual-engine/callback

Reads (lists, one video) come straight from the shared `ve_jobs` collection, so users still see their videos while
the engine restarts. Every change goes through the engine, which owns the job state.
Errors: {"error": {"code": "...", "message": "..."}}.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any

from bson.objectid import ObjectId
from flask import Blueprint, Response, jsonify, request, send_from_directory, stream_with_context
from flask_jwt_extended import get_jwt_identity, jwt_required

from visual_engine_client import EngineError, get_client

log = logging.getLogger("visual_engine_routes")

visual_engine_bp = Blueprint("visual_engine", __name__, url_prefix="/api/visual-engine")

JOBS = "ve_jobs"
CALLBACK_LOG = "ve_callback_log"
OUTPUT_FOLDER = os.getenv("VE_OUTPUT_FOLDER", "outputs")  # where the engine publishes (its VE_PUBLISH_DIR)
CALLBACK_MAX_SKEW_SEC = 300
_PUBLIC = ("user_id", "title", "status", "stage", "progress", "priority", "attempts", "created_at", "queued_at",
           "started_at", "finished_at", "updated_at", "error", "result", "cancel_requested", "request",
           "manual_retries", "phase", "package", "plan", "renders")
BRAND_FIELDS = ("voice", "look", "genre", "accent", "notes", "visual_mode", "voice_name")


def _db():
    from mongodb import mongo_instance

    return mongo_instance.get_db()


def _err(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


def _me() -> tuple[str, str]:
    """(email, role) of the logged-in user; role from the database, never from the request."""
    email = get_jwt_identity()
    user = _db().users.find_one({"email": email}, {"status": 1}) if email else None
    if not user:
        return "", ""
    return email, ("admin" if user.get("status") == "admin" else "user")


def logged_in(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*a, **kw):
        email, role = _me()
        if not email:
            return _err("unknown_user", "Your account was not found.", 403)
        return fn(email, role, *a, **kw)

    return wrapper


def admin_only(fn):
    @wraps(fn)
    @logged_in
    def wrapper(email, role, *a, **kw):
        if role != "admin":
            return _err("forbidden", "Admin access required.", 403)
        return fn(email, role, *a, **kw)

    return wrapper


def _engine(fn):
    """Engine errors → the same JSON error shape, with the engine's status code (429, 409 …) passed through."""

    @wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except EngineError as e:
            body, status = e.to_response()
            resp = jsonify(body)
            resp.status_code = status
            retry = (body.get("error") or {}).get("retry_after_sec")
            if retry:
                resp.headers["Retry-After"] = str(retry)
            return resp

    return wrapper


def _utcnow() -> datetime:
    """Naive UTC, like the rest of this backend's records."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(v):
    return v.isoformat() + "Z" if isinstance(v, datetime) else v


def view(doc: dict, *, admin: bool = False) -> dict:
    """A ve_jobs document as the frontend sees it (script and internal paths left out)."""
    out = {k: doc.get(k) for k in _PUBLIC}
    out["id"] = doc["_id"]
    req = dict(out.get("request") or {})
    req.pop("script", None)
    out["request"] = req
    res = dict(out.get("result") or {})
    for k in ("internal_job_id", "video_path", "credits"):
        res.pop(k, None)
    if doc.get("status") == "completed" and res.get("video_file"):
        res["download_url"] = f"/api/visual-engine/videos/{doc['_id']}/download"
        if res.get("publish") == "shared":
            res["video_url"] = f"/videos/{res['video_file']}"
            if res.get("thumbnail_file"):
                res["thumbnail_url"] = f"/videos/{res['thumbnail_file']}"
    out["result"] = res or None
    for k in ("created_at", "queued_at", "started_at", "finished_at", "updated_at"):
        out[k] = _iso(out.get(k))
    if isinstance(out.get("plan"), dict):
        out["plan"] = {**out["plan"], "saved_at": _iso(out["plan"].get("saved_at"))}
    out["step"] = step_of(doc)
    if doc.get("status") == "queued":
        col = _db()[JOBS]
        pr = doc.get("priority", 0)
        out["queue_position"] = 1 + col.count_documents({"status": "queued", "$or": [
            {"priority": {"$gt": pr}}, {"priority": pr, "queued_at": {"$lt": doc.get("queued_at")}}]})
    if admin:
        out["logs"] = doc.get("logs") or []
        out["worker_id"] = doc.get("worker_id")
    else:  # users get a plain reason by code; raw messages and log lines (server paths) are for admins
        out["error"] = _user_error(doc.get("error"))
    return out


def step_of(doc: dict) -> str:
    """voice → plan → rendering → done (or voice_failed / render_failed / canceled)."""
    st, ph = doc.get("status"), doc.get("phase") or "voice"
    if ph == "voice":
        return {"queued": "voice", "running": "voice", "failed": "voice_failed",
                "canceled": "canceled"}.get(st, "plan")
    if st in ("queued", "running"):
        return "rendering"
    return {"completed": "done", "failed": "render_failed"}.get(st, "plan")


FRIENDLY = {
    "voice_failed": "The voice could not be made. Check the script and the voice, then retry.",
    "render_failed": "The video could not be rendered. You can retry; if it fails again, contact support.",
    "publish_failed": "The video rendered but could not be saved. Retry, or contact support.",
    "timeout": "Rendering took too long and was stopped. Retry, or try a shorter script.",
    "engine_error": "Something went wrong on the render machine. Retry, or contact support.",
    "admin_failed": "This video was stopped by an admin.",
    "plan_missing": "No checked plan was found. Paste Claude's plan again, then render.",
    "canceled": "Canceled.",
}


def _user_error(err):
    if not err:
        return None
    code = err.get("code") or "engine_error"
    return {"code": code, "message": FRIENDLY.get(code, FRIENDLY["engine_error"])}


# ── user routes ───────────────────────────────────────────────────────────────────────────────────────────────────

_catalog_cache: dict[str, Any] = {"at": 0.0, "data": None}


@visual_engine_bp.route("/options", methods=["GET"])
@logged_in
@_engine
def options(email, role):
    if not _catalog_cache["data"] or time.time() - _catalog_cache["at"] > 300:
        _catalog_cache["data"] = get_client().catalog(email, role)
        _catalog_cache["at"] = time.time()
    return jsonify(_catalog_cache["data"])


def _my_channels(email: str, role: str, channel_id: str | None = None):
    user = _db().users.find_one({"email": email}, {"_id": 1})
    q: dict[str, Any] = {} if (role == "admin" and channel_id) else {"userId": str(user["_id"])}
    if channel_id:
        try:
            q["_id"] = ObjectId(str(channel_id))
        except Exception:
            return []
    return list(_db().channels.find(q).sort("createdAt", -1))


_CARTESIA: dict[str, str] | None = None


def _cartesia_voices() -> dict[str, str]:
    """The Cartesia voice names → ids the main pipeline knows (helper/cartesia.py). Read once; if importing that
    module fails (it needs audio libraries), the dict is read straight from the file."""
    global _CARTESIA
    if _CARTESIA is None:
        try:
            from helper.cartesia import CARTESIA_VOICES

            _CARTESIA = dict(CARTESIA_VOICES)
        except Exception:
            import re
            from pathlib import Path

            _CARTESIA = {}
            try:
                src = (Path(__file__).parent / "helper" / "cartesia.py").read_text(encoding="utf-8")
                m = re.search(r"CARTESIA_VOICES\s*=\s*\{(.*?)\}", src, re.S)
                if m:
                    _CARTESIA = dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', m.group(1)))
            except Exception:
                pass
    return _CARTESIA


def _channel_voice(ch: dict) -> dict | None:
    """The voice the channel already uses everywhere (set on the Voices page). Documentaries need Speechify
    (it gives the word timings), so a Cartesia voice is reported as such and the user picks a Speechify one."""
    vid = (ch.get("voice") or ch.get("speechify_voice_id") or "").strip()
    if not vid:
        return None
    cart = _cartesia_voices()
    provider = "cartesia" if (vid in cart or vid in cart.values()) else "speechify"
    return {"id": vid, "name": ch.get("speechify_voice_name") or None, "provider": provider}


def _channel_view(ch: dict) -> dict:
    kit = ch.get("brandKit") or {}
    logo = ch.get("logo")
    return {"id": str(ch["_id"]), "name": ch.get("channelName"), "niche": ch.get("niche"), "logo": logo,
            "logo_url": f"/channel_media/{logo}" if logo else None, "voice": _channel_voice(ch),
            "brand_kit": {k: kit.get(k) for k in BRAND_FIELDS if kit.get(k)}}


@visual_engine_bp.route("/channels", methods=["GET"])
@logged_in
def my_channels(email, role):
    return jsonify({"channels": [_channel_view(c) for c in _my_channels(email, "user")]})


def _clean_kit(data: dict) -> dict:
    kit = {}
    for k in BRAND_FIELDS:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            kit[k] = v.strip()[:1000 if k == "notes" else 40]
    return kit


@visual_engine_bp.route("/channels/<channel_id>/brand-kit", methods=["PUT"])
@logged_in
def save_brand_kit(email, role, channel_id):
    chans = _my_channels(email, role, channel_id)
    if not chans:
        return _err("unknown_channel", "That channel was not found in your account.", 404)
    kit = _clean_kit(request.get_json(silent=True) or {})
    _db().channels.update_one({"_id": chans[0]["_id"]}, {"$set": {"brandKit": kit}})
    return jsonify({"channel": _channel_view({**chans[0], "brandKit": kit})})


@visual_engine_bp.route("/videos", methods=["POST"])
@logged_in
@_engine
def create_video(email, role):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err("invalid_json", "Send the request as JSON.", 400)
    if not data.get("channel_id"):
        return _err("channel_required", "Choose the channel this video is for.", 400)
    chans = _my_channels(email, role, data["channel_id"])
    if not chans:
        return _err("unknown_channel", "That channel was not found in your account.", 404)
    ch = chans[0]
    kit = {**(ch.get("brandKit") or {}), **_clean_kit({k: data.get(k) for k in BRAND_FIELDS}),
           **({"notes": data["brand_notes"].strip()} if isinstance(data.get("brand_notes"), str)
              and data["brand_notes"].strip() else {})}
    if data.get("save_brand_kit"):
        _db().channels.update_one({"_id": ch["_id"]}, {"$set": {"brandKit": _clean_kit(kit)}})
    body = {"title": data.get("title"), "script": data.get("script"), "channel_id": str(ch["_id"]),
            "channel_name": ch.get("channelName"), "niche": (str(ch.get("niche"))[:60] if ch.get("niche") else None)}
    if not kit.get("voice"):  # the channel's own voice, unless this video (or its brand kit) names another
        cv = _channel_voice(ch)
        if cv and cv["provider"] == "speechify":
            kit["voice"] = cv["id"]
            kit.setdefault("voice_name", cv.get("name") or "")
        elif cv:
            return _err("voice_not_supported", "This channel's voice is a Cartesia voice. Documentaries need a "
                        "Speechify voice for the word timings: choose one for this channel.", 400)
    for src, dst in (("voice", "voice"), ("voice_name", "voice_name"), ("look", "look"), ("genre", "genre"), ("accent", "accent"),
                     ("notes", "brand_notes"), ("visual_mode", "visual_mode")):
        if kit.get(src):
            body[dst] = kit[src]
    body = {k: v for k, v in body.items() if v not in (None, "")}
    if role == "admin" and data.get("dry_run"):
        body["dry_run"] = True
    key = request.headers.get("Idempotency-Key") or data.get("client_request_id")
    res = get_client().create_video(email, role, body, idempotency_key=str(key) if key else None)
    status = 202 if res.get("created", True) else 200
    doc = _db()[JOBS].find_one({"_id": res["video"]["id"]}) or {**res["video"], "_id": res["video"]["id"]}
    return jsonify({"video": view(doc, admin=role == "admin"), "created": res.get("created", True)}), status


@visual_engine_bp.route("/videos", methods=["GET"])
@logged_in
def list_videos(email, role):
    q: dict[str, Any] = {"user_id": email}
    st = request.args.get("status")
    if st:
        q["status"] = {"$in": [s for s in st.split(",") if s]}
    before = request.args.get("before")
    col = _db()[JOBS]
    if before:
        ref = col.find_one({"_id": before, "user_id": email}, {"created_at": 1})
        if ref:
            q["created_at"] = {"$lt": ref["created_at"]}
    limit = max(1, min(100, int(request.args.get("limit", 30) or 30)))
    docs = list(col.find(q).sort("created_at", -1).limit(limit))
    return jsonify({"videos": [view(d) for d in docs],
                    "next_before": docs[-1]["_id"] if len(docs) == limit else None})


@visual_engine_bp.route("/videos/<job_id>", methods=["GET"])
@logged_in
def get_video(email, role, job_id):
    q = {"_id": job_id} if role == "admin" else {"_id": job_id, "user_id": email}
    doc = _db()[JOBS].find_one(q)
    if not doc:
        return _err("not_found", "No such video.", 404)
    return jsonify({"video": view(doc, admin=role == "admin")})


@visual_engine_bp.route("/videos/<job_id>/cancel", methods=["POST"])
@logged_in
@_engine
def cancel_video(email, role, job_id):
    res = get_client().call("POST", f"/v1/videos/{job_id}/cancel", user=email, role=role)
    return jsonify({"video": view(_db()[JOBS].find_one({"_id": job_id}), admin=role == "admin")
                    if res.get("video") else None})


@visual_engine_bp.route("/videos/<job_id>/retry", methods=["POST"])
@logged_in
@_engine
def retry_video(email, role, job_id):
    get_client().call("POST", f"/v1/videos/{job_id}/retry", user=email, role=role)
    return jsonify({"video": view(_db()[JOBS].find_one({"_id": job_id}), admin=role == "admin")})


@visual_engine_bp.route("/videos/<job_id>/package", methods=["GET"])
@logged_in
@_engine
def get_package(email, role, job_id):
    return jsonify(get_client().call("GET", f"/v1/videos/{job_id}/package", user=email, role=role))


@visual_engine_bp.route("/videos/<job_id>/package.zip", methods=["GET"])
@logged_in
@_engine
def get_package_zip(email, role, job_id):
    up = get_client().call("GET", f"/v1/videos/{job_id}/package.zip", user=email, role=role, stream=True)
    return Response(b"".join(up.iter_content(65536)), mimetype="application/zip",
                    headers={"Content-Disposition": up.headers.get("Content-Disposition",
                                                                   'attachment; filename="for Claude.zip"')})


@visual_engine_bp.route("/videos/<job_id>/plan", methods=["POST"])
@logged_in
@_engine
def post_plan(email, role, job_id):
    data = request.get_json(silent=True) or {}
    get_client().call("POST", f"/v1/videos/{job_id}/plan", user=email, role=role, json={"plan": data.get("plan")},
                      timeout=(5, 120))
    return jsonify({"video": view(_db()[JOBS].find_one({"_id": job_id}), admin=role == "admin")})


@visual_engine_bp.route("/videos/<job_id>/render", methods=["POST"])
@logged_in
@_engine
def post_render(email, role, job_id):
    get_client().call("POST", f"/v1/videos/{job_id}/render", user=email, role=role)
    return jsonify({"video": view(_db()[JOBS].find_one({"_id": job_id}), admin=role == "admin")})


@visual_engine_bp.route("/videos/<job_id>/download", methods=["GET"])
@logged_in
@_engine
def download_video(email, role, job_id):
    q = {"_id": job_id} if role == "admin" else {"_id": job_id, "user_id": email}
    doc = _db()[JOBS].find_one(q)
    if not doc:
        return _err("not_found", "No such video.", 404)
    res = doc.get("result") or {}
    if doc.get("status") != "completed" or not res.get("video_file"):
        return _err("not_ready", "The video is not finished yet.", 409)
    safe = "".join(c for c in (doc.get("title") or "video") if c.isalnum() or c in " -_")[:60].strip() or "video"
    if res.get("publish") == "shared":
        name = os.path.basename(res["video_file"])
        if not os.path.isfile(os.path.join(OUTPUT_FOLDER, name)):
            return _err("file_missing", "The video file is no longer on the server.", 410)
        return send_from_directory(OUTPUT_FOLDER, name, as_attachment=True, download_name=f"{safe}.mp4",
                                   conditional=True)
    up = get_client().call("GET", f"/v1/videos/{job_id}/file", user=email, role=role, stream=True,
                           timeout=(5, 300))
    return Response(stream_with_context(up.iter_content(1024 * 256)), mimetype="video/mp4",
                    headers={"Content-Disposition": f'attachment; filename="{safe}.mp4"',
                             **({"Content-Length": up.headers["Content-Length"]}
                                if up.headers.get("Content-Length") else {})})


# ── admin routes ──────────────────────────────────────────────────────────────────────────────────────────────────


@visual_engine_bp.route("/admin/status", methods=["GET"])
@admin_only
@_engine
def admin_status(email, role):
    c = get_client()
    return jsonify({"ready": c.call("GET", "/v1/ready", user=email, role=role),
                    "queue": c.call("GET", "/v1/admin/queue", user=email, role=role)})


@visual_engine_bp.route("/admin/videos", methods=["GET"])
@admin_only
def admin_videos(email, role):
    q: dict[str, Any] = {}
    if request.args.get("user"):
        q["user_id"] = request.args["user"]
    if request.args.get("status"):
        q["status"] = {"$in": request.args["status"].split(",")}
    limit = max(1, min(200, int(request.args.get("limit", 50) or 50)))
    docs = list(_db()[JOBS].find(q).sort("created_at", -1).limit(limit))
    return jsonify({"videos": [view(d, admin=True) for d in docs]})


@visual_engine_bp.route("/admin/videos/<job_id>/priority", methods=["POST"])
@admin_only
@_engine
def admin_priority(email, role, job_id):
    body = request.get_json(silent=True) or {}
    return jsonify(get_client().call("POST", f"/v1/admin/videos/{job_id}/priority", user=email, role=role,
                                     json={"priority": int(body.get("priority", 0))}))


@visual_engine_bp.route("/admin/videos/<job_id>/fail", methods=["POST"])
@admin_only
@_engine
def admin_fail(email, role, job_id):
    body = request.get_json(silent=True) or {}
    return jsonify(get_client().call("POST", f"/v1/admin/videos/{job_id}/fail", user=email, role=role,
                                     json={"reason": str(body.get("reason") or "force-failed by an admin")[:300]}))


@visual_engine_bp.route("/admin/callbacks/resend", methods=["POST"])
@admin_only
@_engine
def admin_resend(email, role):
    body = request.get_json(silent=True) or {}
    return jsonify(get_client().call("POST", "/v1/admin/callbacks/resend", user=email, role=role,
                                     json={"job_id": body.get("job_id")}))


@visual_engine_bp.route("/admin/audit", methods=["GET"])
@admin_only
@_engine
def admin_audit(email, role):
    params = {k: request.args[k] for k in ("job_id", "limit") if request.args.get(k)}
    return jsonify(get_client().call("GET", "/v1/admin/audit", user=email, role=role, params=params))


# ── callbacks from the engine ─────────────────────────────────────────────────────────────────────────────────────


def _verify_callback(raw: bytes) -> bool:
    secret = os.getenv("VE_CALLBACK_SECRET") or os.getenv("VE_SHARED_SECRET") or ""
    ts = request.headers.get("X-VE-Timestamp", "")
    sig = request.headers.get("X-VE-Signature", "")
    if len(secret) < 32 or not ts.isdigit() or not sig.startswith("v1="):
        return False
    if abs(time.time() - int(ts)) > CALLBACK_MAX_SKEW_SEC:
        return False
    want = hmac.new(secret.encode(), f"{ts}.".encode() + raw, "sha256").hexdigest()
    return hmac.compare_digest(want, sig[3:])


def _ensure_indexes(db) -> None:
    if getattr(_ensure_indexes, "done", False):
        return
    try:
        db[CALLBACK_LOG].create_index("received_at", expireAfterSeconds=30 * 86400)
        db.completed_videos.create_index("ve_job_id", unique=True,
                                         partialFilterExpression={"ve_job_id": {"$type": "string"}})
        _ensure_indexes.done = True  # type: ignore[attr-defined]
    except Exception as e:
        log.warning("visual engine indexes: %s", e)


def _notify(kind: str, job: dict) -> None:
    try:
        from helper.telegram_notification_engine import send_telegram_error, send_telegram_notification

        user = job.get("user_id") or "?"
        front = os.getenv("FRONTEND_URL", "").rstrip("/")
        link = f"{front}/documentaries/{job['id']}" if front else "https://alchemymix.com"
        if kind == "video.timings_ready":
            send_telegram_notification(f"⏱ Timings ready: {job.get('title')} — copy them to Claude", "Open", link,
                                       user)
        elif kind == "video.completed":
            send_telegram_notification(f"🎬 Documentary ready: {job.get('title')}", "View", link, user)
        elif kind == "video.failed":
            err = job.get("error") or {}
            send_telegram_error(f"❌ Documentary failed: {job.get('title')} — {err.get('code')}: "
                                f"{(err.get('message') or '')[:200]}", user)
    except Exception as e:  # a notice must never fail the callback
        log.warning("telegram notice failed: %s", e)


@visual_engine_bp.route("/callback", methods=["POST"])
def engine_callback():
    raw = request.get_data(cache=True)
    if not _verify_callback(raw):
        return _err("bad_signature", "invalid or expired signature", 401)
    ev = request.get_json(silent=True) or {}
    eid, kind, job = ev.get("event_id"), ev.get("type"), ev.get("job") or {}
    if not eid or not kind or not job.get("id"):
        return _err("bad_event", "missing event_id, type or job", 400)
    db = _db()
    _ensure_indexes(db)
    try:
        db[CALLBACK_LOG].insert_one({"_id": eid, "type": kind, "job_id": job["id"], "received_at": _utcnow()})
    except Exception:
        return jsonify({"ok": True, "duplicate": True})  # already handled — the engine retried a delivered event
    try:
        if kind == "video.completed":
            res = job.get("result") or {}
            req = job.get("request") or {}
            db.completed_videos.update_one(
                {"ve_job_id": job["id"]},
                {"$setOnInsert": {"ve_job_id": job["id"], "created_at": _utcnow()},
                 "$set": {"title": job.get("title", ""), "user_id": job.get("user_id"), "source": "visual_engine",
                          "video_path": res.get("video_file"), "thumbnail": res.get("thumbnail_file"),
                          "duration_sec": res.get("duration_sec"), "niche": req.get("niche") or req.get("genre"),
                          "channel_id": req.get("channel_id"), "publish": res.get("publish"), "sections": []}},
                upsert=True)
        if kind in ("video.completed", "video.failed", "video.timings_ready"):
            _notify(kind, job)
    except Exception as e:
        db[CALLBACK_LOG].delete_one({"_id": eid})  # let the engine's retry try again
        log.exception("visual engine callback %s failed: %s", kind, e)
        return _err("callback_failed", "could not record the event", 500)
    return jsonify({"ok": True})
