"""
App-level job queues with admin visibility and queue reordering.

Concurrency model:
  - Script jobs (OpenAI, compose): SCRIPT_WORKERS parallel threads, never touch GPU.
  - Video jobs (create_video): up to CONCURRENT_VIDEO_JOBS in parallel threads.
  - Segment workers (global_render_scheduler): CPU/API/ImageFX/ffmpeg-cpu in parallel.
  - GPU only (Ken Burns + NVENC): global gpu_semaphore — everything else keeps running.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import traceback
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from global_render_scheduler import get_scheduler, init_global_render_scheduler
from worker_config import get_worker_limits, log_worker_limits

_limits = get_worker_limits()
CONCURRENT_VIDEO_JOBS = _limits.concurrent_video_jobs
SCRIPT_WORKERS = _limits.script_workers
MAX_JOBS_PER_USER = int(os.getenv("MAX_JOBS_PER_USER", "10"))
JOB_QUEUE_TIMEOUT = 5
VIDEO_JOB_TIMEOUT = int(os.getenv("VIDEO_JOB_TIMEOUT", "0")) or None
SCRIPT_JOB_TIMEOUT = int(os.getenv("SCRIPT_JOB_TIMEOUT", "300"))

_VIDEO_QUEUE_MAX = max(10, int(os.getenv("VIDEO_QUEUE_MAX", "100")))
_SCRIPT_QUEUE_MAX = max(10, int(os.getenv("SCRIPT_QUEUE_MAX", "200")))

script_job_queue = queue.Queue(maxsize=_SCRIPT_QUEUE_MAX)
video_job_semaphore = threading.Semaphore(CONCURRENT_VIDEO_JOBS)
active_users: defaultdict[str, int] = defaultdict(int)
job_lock = threading.Lock()
_active_video_jobs = 0
_active_video_lock = threading.Lock()


@dataclass
class VideoJobEntry:
    queue_id: str
    user_id: str
    func: Callable
    args: tuple
    kwargs: dict
    func_name: str
    job_type: str
    ref_id: str | None
    label: str
    enqueued_at: float = field(default_factory=time.time)


class VideoJobQueue:
    """Ordered deque queue — supports list + move up/down for admin."""

    def __init__(self, maxsize: int) -> None:
        self._deque: deque[VideoJobEntry] = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._maxsize = maxsize

    def put(self, entry: VideoJobEntry, timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while len(self._deque) >= self._maxsize:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Full
                self._cond.wait(timeout=remaining)
            self._deque.append(entry)
            self._cond.notify()

    def get(self, timeout: float = JOB_QUEUE_TIMEOUT) -> VideoJobEntry:
        with self._cond:
            deadline = time.monotonic() + timeout
            while not self._deque:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self._cond.wait(timeout=remaining)
            entry = self._deque.popleft()
            return entry

    def qsize(self) -> int:
        with self._lock:
            return len(self._deque)

    def list_entries(self) -> list[VideoJobEntry]:
        with self._lock:
            return list(self._deque)

    def move(self, queue_id: str, direction: str) -> bool:
        with self._cond:
            ids = [e.queue_id for e in self._deque]
            if queue_id not in ids:
                return False
            idx = ids.index(queue_id)
            if direction == "up" and idx > 0:
                self._deque[idx], self._deque[idx - 1] = (
                    self._deque[idx - 1],
                    self._deque[idx],
                )
                return True
            if direction == "down" and idx < len(self._deque) - 1:
                self._deque[idx], self._deque[idx + 1] = (
                    self._deque[idx + 1],
                    self._deque[idx],
                )
                return True
            return False


video_job_queue = VideoJobQueue(maxsize=_VIDEO_QUEUE_MAX)

_running_jobs: dict[str, dict[str, Any]] = {}
_running_lock = threading.Lock()

_JOB_EVENTS_MAX = max(20, int(os.getenv("PIPELINE_JOB_EVENTS_MAX", "80")))
_job_events: deque[dict[str, Any]] = deque(maxlen=_JOB_EVENTS_MAX)
_events_lock = threading.Lock()


def _record_job_event(
    event_type: str,
    entry: VideoJobEntry,
    *,
    message: str = "",
    duration_seconds: float | None = None,
) -> None:
    with _events_lock:
        _job_events.appendleft(
            {
                "type": event_type,
                "at": datetime.utcnow().isoformat() + "Z",
                "queue_id": entry.queue_id,
                "user_id": entry.user_id,
                "label": entry.label,
                "job_type": entry.job_type,
                "ref_id": entry.ref_id,
                "func_name": entry.func_name,
                "message": message,
                "duration_seconds": duration_seconds,
            }
        )


def get_job_events(limit: int = 50) -> list[dict]:
    with _events_lock:
        return list(_job_events)[:limit]


def _infer_job_meta(
    func: Callable,
    args: tuple,
    *,
    job_label: str | None = None,
    job_type: str | None = None,
) -> tuple[str, str | None, str]:
    fn = getattr(func, "__name__", "video_job")
    jtype = job_type or {
        "resume_dreamface_pipeline": "dreamface",
        "process_regenerated_video": "regenerate",
        "process_full_video": "standard",
    }.get(fn, "standard")

    ref_id: str | None = None
    label = job_label or fn

    if fn == "resume_dreamface_pipeline" and args:
        ref_id = str(args[0])
        label = job_label or f"Talking Heads · {ref_id[-8:]}"
    elif fn == "process_full_video" and len(args) >= 3:
        ref_id = str(args[2])
        payload = args[1] if len(args) > 1 else {}
        if isinstance(payload, dict):
            title = payload.get("title")
            if not title:
                nested = payload.get("data") or {}
                if isinstance(nested, dict):
                    title = nested.get("title")
            if title:
                label = str(title)[:120]
    elif fn == "process_regenerated_video" and len(args) >= 2:
        payload = args[1]
        if isinstance(payload, dict) and payload.get("title"):
            label = str(payload["title"])[:120]

    return jtype, ref_id, label


def _entry_to_dict(entry: VideoJobEntry, position: int) -> dict:
    return {
        "queue_id": entry.queue_id,
        "position": position,
        "user_id": entry.user_id,
        "func_name": entry.func_name,
        "job_type": entry.job_type,
        "ref_id": entry.ref_id,
        "label": entry.label,
        "enqueued_at": entry.enqueued_at,
        "wait_seconds": round(time.time() - entry.enqueued_at, 1),
    }


def _register_running(entry: VideoJobEntry) -> None:
    with _running_lock:
        _running_jobs[entry.queue_id] = {
            "queue_id": entry.queue_id,
            "user_id": entry.user_id,
            "func_name": entry.func_name,
            "job_type": entry.job_type,
            "ref_id": entry.ref_id,
            "label": entry.label,
            "started_at": time.time(),
        }


def _unregister_running(queue_id: str) -> None:
    with _running_lock:
        _running_jobs.pop(queue_id, None)


def get_admin_pipeline_snapshot() -> dict:
    queued = video_job_queue.list_entries()
    with _running_lock:
        running = []
        now = time.time()
        for job in _running_jobs.values():
            started = job.get("started_at", now)
            running.append(
                {
                    **job,
                    "running_seconds": round(now - started, 1),
                }
            )
    return {
        "queued_video_jobs": [_entry_to_dict(e, i + 1) for i, e in enumerate(queued)],
        "running_video_jobs": running,
        "video_queue_max": _VIDEO_QUEUE_MAX,
        "job_events": get_job_events(40),
    }


def move_queued_video_job(queue_id: str, direction: str) -> bool:
    if direction not in ("up", "down"):
        raise ValueError("direction must be up or down")
    return video_job_queue.move(queue_id, direction)


def _decrement_user(user_id: str) -> None:
    with job_lock:
        active_users[user_id] -= 1
        if active_users[user_id] <= 0:
            del active_users[user_id]


def _run_video_job(entry: VideoJobEntry) -> None:
    global _active_video_jobs
    user_id = entry.user_id
    func = entry.func
    args = entry.args
    kwargs = entry.kwargs
    label = entry.func_name
    queue_id = entry.queue_id

    _register_running(entry)
    started = time.time()
    _record_job_event("started", entry, message="Video job started")
    try:
        logging.info(
            f"🚀 Video job running | user={user_id} fn={label} "
            f"type={entry.job_type} ref={entry.ref_id or '—'}"
        )

        if VIDEO_JOB_TIMEOUT:
            done = threading.Event()
            error_holder: list = [None]

            def _inner() -> None:
                try:
                    func(*args, **kwargs)
                except Exception as exc:
                    error_holder[0] = exc
                finally:
                    done.set()

            inner = threading.Thread(
                target=_inner, daemon=True, name=f"video-{label}"
            )
            inner.start()
            if not done.wait(timeout=VIDEO_JOB_TIMEOUT):
                logging.error(
                    f"⏰ Video job TIMED OUT ({VIDEO_JOB_TIMEOUT}s) | "
                    f"user={user_id} fn={label} — thread may still be running"
                )
                return
            if error_holder[0]:
                raise error_holder[0]
        else:
            func(*args, **kwargs)

        logging.info(f"✅ Video job finished | user={user_id} fn={label}")
        _record_job_event(
            "completed",
            entry,
            message="Video job finished successfully",
            duration_seconds=round(time.time() - started, 1),
        )
    except Exception as exc:
        logging.error(
            f"❌ Video job error | user={user_id} fn={label}\n{traceback.format_exc()}"
        )
        _record_job_event(
            "failed",
            entry,
            message=str(exc)[:200],
            duration_seconds=round(time.time() - started, 1),
        )
    finally:
        _unregister_running(queue_id)
        _decrement_user(user_id)
        video_job_semaphore.release()
        with _active_video_lock:
            _active_video_jobs -= 1
        logging.info(
            f"📤 Video slot freed | user={user_id} fn={label} | "
            f"active={_active_video_jobs}/{CONCURRENT_VIDEO_JOBS} | "
            f"queue={video_job_queue.qsize()}"
        )


def _video_drainer() -> None:
    global _active_video_jobs
    while True:
        try:
            entry = video_job_queue.get(timeout=JOB_QUEUE_TIMEOUT)
        except queue.Empty:
            continue

        with job_lock:
            if active_users[entry.user_id] >= MAX_JOBS_PER_USER:
                logging.warning(
                    f"User {entry.user_id} at job limit — re-queuing video job"
                )
                video_job_queue.put(entry, timeout=30)
                time.sleep(0.5)
                continue
            active_users[entry.user_id] += 1

        video_job_semaphore.acquire()

        with _active_video_lock:
            _active_video_jobs += 1

        logging.info(
            f"📥 Video job dispatched | user={entry.user_id} fn={entry.func_name} | "
            f"type={entry.job_type} | active={_active_video_jobs}/{CONCURRENT_VIDEO_JOBS} | "
            f"queue={video_job_queue.qsize()}"
        )

        threading.Thread(
            target=_run_video_job,
            args=(entry,),
            daemon=True,
            name=f"video-job-{entry.func_name}",
        ).start()


def _script_drainer(worker_id: int) -> None:
    logging.info(f"Script worker {worker_id} started")
    while True:
        try:
            user_id, func, args, kwargs = script_job_queue.get(timeout=JOB_QUEUE_TIMEOUT)
        except queue.Empty:
            continue

        logging.info(
            f"✍️  Script job starting | worker={worker_id} user={user_id} fn={func.__name__}"
        )

        def _run() -> None:
            try:
                func(*args, **kwargs)
                logging.info(f"✅ Script job finished | user={user_id} fn={func.__name__}")
            except Exception:
                logging.error(
                    f"❌ Script job error | user={user_id} fn={func.__name__}\n"
                    f"{traceback.format_exc()}"
                )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=SCRIPT_JOB_TIMEOUT)

        if t.is_alive():
            logging.error(
                f"⏰ Script job TIMED OUT | worker={worker_id} user={user_id} "
                f"fn={func.__name__}"
            )


def enqueue_video_job(
    user_id: str,
    func: Callable,
    *args,
    job_label: str | None = None,
    job_type: str | None = None,
    **kwargs,
) -> str:
    """Queue a full video render. Returns queue_id for admin tracking."""
    job_type, ref_id, label = _infer_job_meta(
        func, args, job_label=job_label, job_type=job_type
    )
    entry = VideoJobEntry(
        queue_id=uuid.uuid4().hex[:12],
        user_id=user_id,
        func=func,
        args=args,
        kwargs=kwargs,
        func_name=func.__name__,
        job_type=job_type,
        ref_id=ref_id,
        label=label,
    )
    try:
        video_job_queue.put(entry, timeout=30)
    except queue.Full:
        logging.error(
            f"❌ Video queue full ({_VIDEO_QUEUE_MAX}) — rejecting job for user={user_id}"
        )
        raise RuntimeError("Video queue is full; try again later") from None
    logging.info(
        f"📥 Video job enqueued | id={entry.queue_id} user={user_id} "
        f"fn={entry.func_name} type={job_type} depth={video_job_queue.qsize()}"
    )
    return entry.queue_id


def enqueue_script_job(user_id: str, func: Callable, *args, **kwargs) -> None:
    script_job_queue.put((user_id, func, args, kwargs))
    logging.info(
        f"📥 Script job enqueued | user={user_id} fn={func.__name__} "
        f"depth={script_job_queue.qsize()}"
    )


def get_active_video_job_count() -> int:
    with _active_video_lock:
        return _active_video_jobs


def get_queue_status() -> dict:
    sched = get_scheduler().status()
    with _active_video_lock:
        running = _active_video_jobs
    admin = get_admin_pipeline_snapshot()
    return {
        "video_queue_depth": video_job_queue.qsize(),
        "script_queue_depth": script_job_queue.qsize(),
        "active_video_jobs": running,
        "active_users": dict(active_users),
        "concurrent_video_jobs_limit": CONCURRENT_VIDEO_JOBS,
        "script_worker_count": SCRIPT_WORKERS,
        **sched,
        **get_worker_limits().as_dict(),
        **admin,
    }


def start_video_worker() -> None:
    log_worker_limits()
    init_global_render_scheduler()

    threading.Thread(target=_video_drainer, daemon=True, name="video-drainer").start()
    logging.info(
        f"🔄 Video drainer started | up to {CONCURRENT_VIDEO_JOBS} parallel video jobs | "
        f"GPU capped at {get_worker_limits().gpu_concurrent_jobs} via global scheduler"
    )

    for i in range(SCRIPT_WORKERS):
        threading.Thread(
            target=_script_drainer,
            args=(i,),
            daemon=True,
            name=f"script-drainer-{i}",
        ).start()
    logging.info(f"🔄 {SCRIPT_WORKERS} script worker(s) started")
