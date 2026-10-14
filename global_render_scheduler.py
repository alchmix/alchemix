"""
Global render scheduler — one GPU queue + one segment worker pool for the whole app.

Multiple create_video() jobs share:
  - gpu_semaphore / nvenc_semaphore (VRAM-safe cap across all videos)
  - ProcessPoolExecutor (CPU prep: ImageFX, prompts, Ken Burns workers)

Call init_global_render_scheduler() once at app startup (before any video jobs).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, BrokenExecutor, ProcessPoolExecutor, wait

from multiprocessing import Manager

from worker_config import get_worker_limits
from segment_pool_worker import resolve_encode_module_name, run_segment_encode
from pipeline_fair_share import active_video_jobs, segment_max_in_flight as _fair_segment_in_flight
from pipeline_fair_share import storyboard_max_in_flight as _fair_storyboard_in_flight

GPU_SEM_INDEX = 17
NVENC_SEM_INDEX = 18
STORYBOARD_SEM_INDEX = 26


def _storyboard_max_concurrent() -> int:
    return max(1, int(os.getenv("STORYBOARD_MAX_CONCURRENT", "5")))

_scheduler: "GlobalRenderScheduler | None" = None
_scheduler_lock = threading.Lock()


def _segment_wall_timeout_sec() -> int:
    return max(60, int(os.getenv("SEGMENT_WALL_TIMEOUT_SEC", "420")))


def _global_pool_in_flight_cap(pool_workers: int) -> int:
    """Total segment slots app-wide.

    Default: pool_workers (10) — safe for one heavy video with OS headroom.
    GLOBAL_POOL_DEDICATED=1: concurrent_video_jobs × per-video slots so each
    video can pipeline its own SEGMENT_MAX_IN_FLIGHT without fighting others.
    """
    dedicated = os.getenv("GLOBAL_POOL_DEDICATED", "").strip().lower() in ("1", "true", "yes")
    if dedicated:
        from worker_config import get_worker_limits

        limits = get_worker_limits()
        per_video_env = os.getenv("GLOBAL_POOL_PER_VIDEO_SLOTS") or os.getenv("SEGMENT_MAX_IN_FLIGHT")
        per_video = max(1, int(per_video_env)) if per_video_env else max(1, pool_workers)
        cap = limits.concurrent_video_jobs * per_video
        logging.info(
            f"🌐 Global pool dedicated mode | {limits.concurrent_video_jobs} video(s) × "
            f"{per_video} slots = {cap} total"
        )
        return cap

    env_val = os.getenv("GLOBAL_POOL_IN_FLIGHT_CAP")
    if env_val is not None:
        return max(1, int(env_val))
    return pool_workers


def _segment_max_in_flight(
    pool_workers: int,
    segment_info_list: list | None = None,
    global_cap: int | None = None,
    active_jobs: int | None = None,
) -> int:
    env_val = os.getenv("SEGMENT_MAX_IN_FLIGHT")
    base = max(1, int(env_val)) if env_val is not None else max(1, pool_workers)
    active = active_jobs if active_jobs is not None else active_video_jobs()

    if segment_info_list:
        try:
            uses_storyboard = any(bool(info[14]) for info in segment_info_list if len(info) > 14)
            if uses_storyboard:
                base = _fair_storyboard_in_flight(base, active)
        except (IndexError, TypeError):
            pass

    if active > 1 and global_cap is not None:
        base = _fair_segment_in_flight(base, global_cap, active)
    return base


def _max_wall_retries() -> int:
    return max(1, int(os.getenv("SEGMENT_WALL_MAX_RETRIES", "2")))


def _pool_recreate_cooldown_sec() -> int:
    return max(0, int(os.getenv("POOL_RECREATE_COOLDOWN_SEC", "90")))


def _pool_recreate_delay_sec() -> int:
    return max(0, int(os.getenv("POOL_RECREATE_DELAY_SEC", "8")))


def _global_slot_timeout_sec() -> int:
    return max(120, int(os.getenv("GLOBAL_SLOT_TIMEOUT_SEC", "3600")))


def _slot_wait_log_interval_sec() -> int:
    return max(5, int(os.getenv("GLOBAL_SLOT_WAIT_LOG_SEC", "15")))


def _pool_stall_unstick_sec() -> int:
    """No segment finished anywhere for this long while slots are held → force recovery."""
    return max(60, int(os.getenv("POOL_STALL_UNSTICK_SEC", "300")))


def _watchdog_interval_sec() -> int:
    return max(15, int(os.getenv("POOL_WATCHDOG_INTERVAL_SEC", "60")))


class GlobalRenderScheduler:
    """App-wide segment pool + GPU semaphore."""

    def __init__(self):
        self._init_lock = threading.Lock()
        self._pool_lock = threading.Lock()
        self._manager: Manager | None = None
        self._gpu_semaphore = None
        self._storyboard_semaphore = None
        self._storyboard_slots = 5
        self._pool: ProcessPoolExecutor | None = None
        self._gpu_slots = 1
        self._pool_workers = 4
        self._initialized = False
        self._active_segment_jobs = 0
        self._active_video_jobs = 0
        self._stats_lock = threading.Lock()
        self._global_in_flight = 0
        self._global_in_flight_lock = threading.Lock()
        self._global_in_flight_cap = 4
        self._pool_generation = 0
        self._last_pool_recreate = 0.0
        self._pool_recreate_skips = 0
        self._pool_recreate_times: list[float] = []
        self._encode_validated = False
        self._last_slot_wait_log: dict[str, float] = {}
        self._last_progress_at = time.time()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._recovery_lock = threading.Lock()

    def _touch_progress(self) -> None:
        self._last_progress_at = time.time()

    def _stall_seconds(self) -> float:
        return time.time() - self._last_progress_at

    def force_recovery(self, reason: str) -> bool:
        """
        Nuclear recovery when workers hung or slot counters leaked.
        Clears global in-flight count and recreates the worker pool + IPC semaphores.
        """
        with self._recovery_lock:
            with self._global_in_flight_lock:
                leaked = self._global_in_flight
                self._global_in_flight = 0
            logging.error(
                f"🚨 FORCE RECOVERY | cleared {leaked} global slot(s) | "
                f"stall={self._stall_seconds():.0f}s | {reason}"
            )
            self._recreate_pool(f"force recovery | {reason}", force=True)
            self._touch_progress()
            return True

    def _maybe_force_recovery(self, reason: str) -> bool:
        """Recover if slots are held but nothing has completed recently."""
        if not self._initialized:
            return False
        with self._global_in_flight_lock:
            in_use = self._global_in_flight
        if in_use <= 0:
            return False
        stall_limit = _pool_stall_unstick_sec()
        if self._stall_seconds() < stall_limit:
            return False
        return self.force_recovery(reason)

    def _start_watchdog(self) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return

        def _loop():
            logging.info(
                f"🐕 Pool watchdog started | check every {_watchdog_interval_sec()}s | "
                f"unstick after {_pool_stall_unstick_sec()}s stall"
            )
            while not self._watchdog_stop.wait(_watchdog_interval_sec()):
                try:
                    self._maybe_force_recovery("background watchdog")
                except Exception as exc:
                    logging.error(f"Watchdog error: {exc}")

        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=_loop, daemon=True, name="pool-watchdog"
        )
        self._watchdog_thread.start()

    def _acquire_global_slot(self, label: str, seg_idx: int) -> None:
        deadline = time.monotonic() + _global_slot_timeout_sec()
        log_interval = _slot_wait_log_interval_sec()
        wait_started = time.monotonic()
        while True:
            # Hung workers with leaked slots: recover before waiting an hour.
            if time.monotonic() - wait_started >= _pool_stall_unstick_sec():
                self._maybe_force_recovery(
                    f"slot wait {int(time.monotonic() - wait_started)}s | {label} seg {seg_idx}"
                )
                wait_started = time.monotonic()

            acquired = False
            with self._global_in_flight_lock:
                cap = self._global_in_flight_cap
                cur = self._global_in_flight
                if self._global_in_flight < self._global_in_flight_cap:
                    self._global_in_flight += 1
                    cur = self._global_in_flight
                    acquired = True
            if acquired:
                self._touch_progress()
                logging.info(
                    f"🔒 Slot acquired | {label} seg {seg_idx} | "
                    f"{cur}/{cap} in use | stall={self._stall_seconds():.0f}s | "
                    f"pool_gen={self._pool_generation}"
                )
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Global pool slot unavailable after {_global_slot_timeout_sec()}s | "
                    f"{label} seg {seg_idx} ({cur}/{cap} in use)"
                )
            now = time.monotonic()
            last = self._last_slot_wait_log.get(label, 0.0)
            if now - last >= log_interval:
                self._last_slot_wait_log[label] = now
                active_videos = active_video_jobs()
                logging.info(
                    f"⏳ Global pool slot wait | {label} seg {seg_idx} | "
                    f"{cur}/{cap} slots in use | {active_videos} video(s) | "
                    f"fair-share≈{max(1, cap // max(1, active_videos))}/video"
                )
            time.sleep(2)

    def _release_global_slot(self, label: str = "", seg_idx: int | None = None) -> None:
        with self._global_in_flight_lock:
            self._global_in_flight = max(0, self._global_in_flight - 1)
            cur = self._global_in_flight
            cap = self._global_in_flight_cap
        if label or seg_idx is not None:
            logging.info(
                f"🔓 Slot released | {label} seg {seg_idx} | "
                f"{cur}/{cap} in use | pool_gen={self._pool_generation}"
            )

    def _reset_ipc_semaphores(self, reason: str) -> None:
        """Fresh GPU/storyboard semaphores after pool recycle (avoids leaked locks)."""
        if self._manager is None:
            return
        self._gpu_semaphore = self._manager.Semaphore(self._gpu_slots)
        self._storyboard_semaphore = self._manager.Semaphore(self._storyboard_slots)
        self._pool_generation += 1
        logging.warning(
            f"🔁 Reset GPU/storyboard semaphores (gen={self._pool_generation}) — {reason}"
        )

    def init(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            limits = get_worker_limits()
            self._gpu_slots = limits.gpu_concurrent_jobs
            self._pool_workers = limits.encode_pool_workers
            from segment_pool_worker import validate_encode_module_path

            logging.info("🌐 Initializing global render scheduler...")
            validate_encode_module_path()
            self._encode_validated = True
            logging.info("🌐 Starting multiprocessing Manager + segment pool...")
            self._manager = Manager()
            self._gpu_semaphore = self._manager.Semaphore(self._gpu_slots)
            self._storyboard_slots = _storyboard_max_concurrent()
            self._storyboard_semaphore = self._manager.Semaphore(self._storyboard_slots)
            self._global_in_flight_cap = _global_pool_in_flight_cap(self._pool_workers)
            self._create_pool()
            max_in_flight = _segment_max_in_flight(self._pool_workers)
            self._initialized = True
            self._touch_progress()
            self._start_watchdog()
            logging.info(
                f"🌐 Global render scheduler ready | "
                f"GPU slots={self._gpu_slots} (Ken Burns+NVENC only) | "
                f"storyboard slots={self._storyboard_slots} (shared Gradio cap) | "
                f"segment workers={self._pool_workers} | "
                f"max_in_flight={max_in_flight} | "
                f"global_cap={self._global_in_flight_cap} | "
                f"segment_wall_timeout={_segment_wall_timeout_sec()}s"
            )

    def _storm_worker_cap(self) -> int | None:
        """After several recreates in a short window, halve workers to avoid OOM."""
        window = max(60, int(os.getenv("POOL_STORM_WINDOW_SEC", "300")))
        threshold = max(2, int(os.getenv("POOL_STORM_THRESHOLD", "3")))
        now = time.time()
        self._pool_recreate_times = [t for t in self._pool_recreate_times if now - t < window]
        if len(self._pool_recreate_times) >= threshold:
            reduced = max(2, self._pool_workers // 2)
            logging.warning(
                f"🌩️  Pool recreate storm ({len(self._pool_recreate_times)} in {window}s) — "
                f"temporarily using {reduced} workers (normal={self._pool_workers})"
            )
            return reduced
        return None

    def _create_pool(self, workers_override: int | None = None) -> None:
        max_tasks = int(os.getenv("SEGMENT_WORKER_MAX_TASKS", "1"))
        workers = workers_override or self._pool_workers
        self._pool = ProcessPoolExecutor(
            max_workers=workers,
            max_tasks_per_child=max_tasks,
        )

    def _recreate_pool(self, reason: str, *, force: bool = False) -> bool:
        """
        Replace the process pool. Returns True if recreated, False if skipped (cooldown).
        Cooldown prevents OOM storms when many segments hit wall timeout at once.
        """
        cooldown = _pool_recreate_cooldown_sec()
        now = time.time()
        if not force and cooldown > 0 and self._last_pool_recreate > 0:
            elapsed = now - self._last_pool_recreate
            if elapsed < cooldown:
                self._pool_recreate_skips += 1
                logging.warning(
                    f"⏸️  Pool recreate skipped ({int(cooldown - elapsed)}s cooldown left, "
                    f"skips={self._pool_recreate_skips}) — {reason}"
                )
                # Still reset IPC semaphores — workers may hold leaked locks.
                self._reset_ipc_semaphores(f"cooldown skip | {reason}")
                return False

        storm_cap = self._storm_worker_cap()
        self._pool_recreate_times.append(now)

        with self._pool_lock:
            logging.warning(f"♻️  Recreating segment worker pool — {reason}")
            old = self._pool
            self._pool = None
            if old is not None:
                try:
                    old.shutdown(wait=False, cancel_futures=True)
                except Exception as exc:
                    logging.debug(f"Pool shutdown: {exc}")

            delay = _pool_recreate_delay_sec()
            if delay > 0:
                logging.info(
                    f"⏳ Waiting {delay}s before spawning new workers (let memory settle)"
                )
                time.sleep(delay)

            self._create_pool(workers_override=storm_cap)
            self._last_pool_recreate = time.time()

        self._reset_ipc_semaphores(reason)
        self._touch_progress()
        return True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def gpu_semaphore(self):
        if not self._initialized:
            self.init()
        return self._gpu_semaphore

    def storyboard_semaphore(self):
        if not self._initialized:
            self.init()
        return self._storyboard_semaphore

    def _patch_segment_info(self, segment_info: tuple, job_label: str) -> tuple:
        gpu = self.gpu_semaphore()
        sb = self.storyboard_semaphore()
        info = list(segment_info)
        if len(info) > GPU_SEM_INDEX:
            info[GPU_SEM_INDEX] = gpu
        if len(info) > NVENC_SEM_INDEX:
            info[NVENC_SEM_INDEX] = gpu
        if len(info) > STORYBOARD_SEM_INDEX:
            info[STORYBOARD_SEM_INDEX] = sb
        elif len(info) >= STORYBOARD_SEM_INDEX:
            info.append(sb)
        return tuple(info)

    def _submit_segment(self, segment_info: tuple, job_label: str):
        self._acquire_global_slot(job_label, segment_info[5])
        try:
            with self._pool_lock:
                if self._pool is None:
                    self._create_pool()
                patched = self._patch_segment_info(segment_info, job_label)
                return self._pool.submit(run_segment_encode, patched)
        except Exception:
            self._release_global_slot()
            raise

    def _handle_stalled_segments(
        self,
        in_flight: dict,
        pending: deque,
        label: str,
        wall_timeout: int,
        wall_retries: dict,
        chunk_paths: list,
        completed_holder: list,
    ) -> bool:
        """Recycle workers that exceeded wall clock. Returns True if any stalled."""
        now = time.time()
        stalled = [
            (fut, seg_idx, info, started)
            for fut, (seg_idx, info, started) in list(in_flight.items())
            if now - started > wall_timeout
        ]
        if not stalled:
            return False

        max_retries = _max_wall_retries()
        ages = ", ".join(
            f"seg{seg_idx}({int(now - started)}s)"
            for _, seg_idx, _, started in stalled
        )
        logging.error(
            f"⏱️ Global pool | {label} | {len(stalled)} segment(s) exceeded "
            f"{wall_timeout}s — recycling workers: {ages}"
        )
        for fut, seg_idx, info, _started in stalled:
            in_flight.pop(fut, None)
            try:
                fut.cancel()
            except Exception:
                pass
            self._release_global_slot(label, seg_idx)

            wall_retries[seg_idx] = wall_retries.get(seg_idx, 0) + 1
            if wall_retries[seg_idx] > max_retries:
                logging.error(
                    f"❌ Global pool | {label} | segment {seg_idx} hit "
                    f"{wall_retries[seg_idx]} wall timeouts — giving up (no more retries)"
                )
                chunk_paths[seg_idx] = None
                completed_holder[0] += 1
            else:
                logging.warning(
                    f"🔁 Global pool | {label} | re-queue segment {seg_idx} "
                    f"(wall retry {wall_retries[seg_idx]}/{max_retries})"
                )
                pending.appendleft(info)

        recreated = self._recreate_pool(f"segment wall timeout | {label}", force=True)
        if not recreated:
            self.force_recovery(f"wall timeout + recreate blocked | {label}")
        self._touch_progress()
        return True
    def encode_segments(self, segment_info_list, job_label: str = "") -> list:
        """Encode all segments for one video using the global worker pool."""
        if not self._initialized:
            self.init()
        if not segment_info_list:
            return []

        label = job_label or "video"
        total = len(segment_info_list)
        chunk_paths: list = [None] * total
        pending: deque = deque(segment_info_list)

        with self._stats_lock:
            self._active_video_jobs += 1
            self._active_segment_jobs += total

        active_jobs = active_video_jobs()
        max_in_flight = _segment_max_in_flight(
            self._pool_workers,
            segment_info_list,
            global_cap=self._global_in_flight_cap,
            active_jobs=active_jobs,
        )
        heartbeat_sec = int(os.getenv("SEGMENT_POOL_HEARTBEAT_SEC", "45"))
        wall_timeout = _segment_wall_timeout_sec()
        wall_retries: dict = {}
        completed_holder = [0]

        in_flight: dict = {}

        logging.info(
            f"🌐 Global pool | {label} | {total} segment(s) | "
            f"max_in_flight={max_in_flight} | wall_timeout={wall_timeout}s"
        )
        self._touch_progress()

        try:
            while completed_holder[0] < total:
                while len(in_flight) < max_in_flight and pending:
                    info = pending.popleft()
                    seg_idx = info[5]
                    try:
                        fut = self._submit_segment(info, label)
                    except BrokenExecutor as exc:
                        logging.error(f"Broken pool on submit seg {seg_idx}: {exc}")
                        self._recreate_pool("broken on submit", force=True)
                        pending.appendleft(info)
                        break
                    in_flight[fut] = (seg_idx, info, time.time())

                if not in_flight:
                    if pending:
                        logging.error(
                            f"❌ Global pool | {label} | no in-flight workers but "
                            f"{len(pending)} segment(s) still queued — aborting batch"
                        )
                    break

                done, _ = wait(
                    set(in_flight.keys()),
                    timeout=heartbeat_sec,
                    return_when=FIRST_COMPLETED,
                )

                if not done:
                    if self._maybe_force_recovery(
                        f"no progress {int(self._stall_seconds())}s | {label}"
                    ):
                        for fut, (seg_idx, info, _started) in list(in_flight.items()):
                            in_flight.pop(fut, None)
                            try:
                                fut.cancel()
                            except Exception:
                                pass
                            pending.appendleft(info)
                        continue

                    self._handle_stalled_segments(
                        in_flight,
                        pending,
                        label,
                        wall_timeout,
                        wall_retries,
                        chunk_paths,
                        completed_holder,
                    )
                    with self._global_in_flight_lock:
                        global_used = self._global_in_flight
                        global_cap = self._global_in_flight_cap
                    hb_now = time.time()
                    in_flight_detail = ", ".join(
                        f"seg{seg_idx}({int(hb_now - started)}s)"
                        for _f, (seg_idx, _info, started) in in_flight.items()
                    )
                    logging.info(
                        f"⏳ Global pool | {label} | {completed_holder[0]}/{total} done, "
                        f"{len(in_flight)} in flight [{in_flight_detail}], "
                        f"{len(pending)} queued | global {global_used}/{global_cap} | "
                        f"stall={self._stall_seconds():.0f}s | pool_gen={self._pool_generation}"
                    )
                    continue

                now = time.time()
                for fut in done:
                    seg_idx, info, started = in_flight.pop(fut)
                    self._release_global_slot(label, seg_idx)
                    try:
                        _, chunk_path = fut.result(timeout=5)
                        chunk_paths[seg_idx] = chunk_path
                    except BrokenExecutor as exc:
                        logging.error(
                            f"❌ Pool broken on segment {seg_idx} | {label}: {exc}"
                        )
                        self._recreate_pool(f"broken result seg {seg_idx}", force=True)
                        pending.appendleft(info)
                        continue
                    except Exception as exc:
                        logging.error(
                            f"❌ Global pool segment {seg_idx} failed | {label}: {exc}",
                            exc_info=True,
                        )
                        chunk_paths[seg_idx] = None
                    else:
                        elapsed = now - started
                        logging.info(
                            f"📊 Global pool | {label} | {completed_holder[0] + 1}/{total} "
                            f"(segment {seg_idx}, {elapsed:.0f}s)"
                        )
                    self._touch_progress()
                    completed_holder[0] += 1

        finally:
            with self._stats_lock:
                self._active_video_jobs = max(0, self._active_video_jobs - 1)
                self._active_segment_jobs = max(0, self._active_segment_jobs - total)

        return chunk_paths

    def admin_recreate_pool(self, reason: str) -> bool:
        """Admin: replace worker pool without clearing global slot counter."""
        if not self._initialized:
            self.init()
        return self._recreate_pool(reason, force=True)

    def admin_reset_semaphores(self, reason: str) -> None:
        """Admin: reset GPU/storyboard IPC semaphores (leaked lock recovery)."""
        if not self._initialized:
            self.init()
        self._reset_ipc_semaphores(reason)

    def status(self) -> dict:
        try:
            from video_job_controller import get_active_video_job_count

            pipeline_jobs = get_active_video_job_count()
        except Exception:
            pipeline_jobs = 0
        with self._stats_lock:
            encode_jobs = self._active_video_jobs
            if pipeline_jobs <= 0 and encode_jobs > 0:
                pipeline_jobs = encode_jobs
            return {
                "initialized": self._initialized,
                "gpu_slots": self._gpu_slots,
                "storyboard_slots": self._storyboard_slots,
                "segment_workers": self._pool_workers,
                "global_in_flight": self._global_in_flight,
                "global_in_flight_cap": self._global_in_flight_cap,
                "pool_generation": self._pool_generation,
                "segment_wall_timeout_sec": _segment_wall_timeout_sec(),
                "active_video_jobs": encode_jobs,
                "active_encode_jobs": encode_jobs,
                "active_pipeline_jobs": pipeline_jobs,
                "active_segment_tasks": self._active_segment_jobs,
                "stall_seconds": round(self._stall_seconds(), 1),
                "stall_unstick_sec": _pool_stall_unstick_sec(),
            }

    def shutdown(self) -> None:
        with self._init_lock:
            self._watchdog_stop.set()
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None
            if self._manager is not None:
                self._manager.shutdown()
                self._manager = None
            self._gpu_semaphore = None
            self._storyboard_semaphore = None
            self._initialized = False
            logging.info("🌐 Global render scheduler shut down")


def get_scheduler() -> GlobalRenderScheduler:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = GlobalRenderScheduler()
        return _scheduler


def init_global_render_scheduler() -> GlobalRenderScheduler:
    """Call once at app startup (e.g. from start_video_worker)."""
    scheduler = get_scheduler()
    scheduler.init()
    return scheduler
