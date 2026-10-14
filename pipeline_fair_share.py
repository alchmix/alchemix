"""
Fair-share parallelism for concurrent video pipeline jobs.

One video job = audio TTS + segment encode (+ storyboard inside segments).
When CONCURRENT_VIDEO_JOBS videos run, split shared caps equally so no job
monopolizes segment slots, storyboard Gradio calls, or audio workers.

Import from test2.py, process_full_video.py, global_render_scheduler.py.
"""

from __future__ import annotations

import logging
import os


def active_video_jobs() -> int:
    """Videos currently running (audio + encode + merge in one thread each)."""
    try:
        from video_job_controller import get_active_video_job_count

        n = get_active_video_job_count()
        if n > 0:
            return n
    except Exception:
        pass
    # Never call scheduler.status() here — status() may already hold _stats_lock (deadlock).
    return 1


def fair_share(total_cap: int, active: int | None = None) -> int:
    """Split total_cap equally across active video jobs (minimum 1 each)."""
    n = max(1, active if active is not None else active_video_jobs())
    return max(1, total_cap // n)


def audio_max_workers(active: int | None = None) -> int:
    cap = max(1, int(os.getenv("AUDIO_MAX_WORKERS", "20")))
    n = active if active is not None else active_video_jobs()
    share = fair_share(cap, n)
    if n > 1:
        logging.info(
            f"🎙️ Audio fair-share | {share} worker(s)/video "
            f"({cap} total ÷ {n} active video(s))"
        )
    return share


def segment_max_in_flight(base: int, global_cap: int, active: int | None = None) -> int:
    n = active if active is not None else active_video_jobs()
    share = fair_share(global_cap, n)
    capped = min(base, share)
    if n > 1 and capped < base:
        logging.info(
            f"🌐 Segment fair-share | max_in_flight {base} → {capped} "
            f"({global_cap} global cap ÷ {n} video(s))"
        )
    return capped


def storyboard_max_in_flight(base: int, active: int | None = None) -> int:
    sb = max(1, int(os.getenv("STORYBOARD_MAX_CONCURRENT", "5")))
    slack = max(0, int(os.getenv("STORYBOARD_IN_FLIGHT_SLACK", "0")))
    n = active if active is not None else active_video_jobs()
    share = fair_share(sb + slack, n)
    capped = min(base, share)
    if n > 1 and capped < base:
        logging.info(
            f"🎨 Storyboard fair-share | max_in_flight {base} → {capped} "
            f"({sb + slack} sb cap ÷ {n} video(s))"
        )
    return capped
