"""
Hardware-aware worker limits for the video pipeline.

Defaults target high-RAM / many-core boxes (e.g. dual E5-2697v2, 128GB, A4000).
Override any value via environment variables.

Env vars:
  CONCURRENT_VIDEO_JOBS   — parallel create_video() threads (default: auto)
  SCRIPT_WORKERS          — parallel script/compose threads (default: auto)
  ENCODE_POOL_WORKERS     — global segment CPU pool (default: auto)
  GLOBAL_SEGMENT_WORKERS  — alias for ENCODE_POOL_WORKERS
  GPU_CONCURRENT_JOBS     — Ken Burns + NVENC slots (default: auto from VRAM)
  KENBURNS_FFMPEG_UPSCALE  — smooth zoom upscale width (default 8000; try 4000 for speed)
  KENBURNS_FFMPEG_THREADS  — ffmpeg -threads per job (default 0 = auto; try 6 on many-core)
  KENBURNS_FFMPEG_SPLIT    — two-pass: parallel CPU zoompan then NVENC (default auto when USE_NVENC=1)
  KENBURNS_NVENC_PRESET      — NVENC preset (default p1)
  STORYBOARD_MAX_CONCURRENT  — max parallel Gradio storyboard API calls app-wide (default 5)
  STORYBOARD_TIMEOUT_SEC     — per-segment storyboard timeout; fallback to Ken Burns (default 600)
  STORYBOARD_MAX_RENDER_JOBS — img2Sketch server-side encode slots (default 5, match MAX_CONCURRENT)
  STORYBOARD_GRADIO_CONCURRENCY — Gradio queue limit on img2Sketch (default 5)
  STORYBOARD_ENCODER         — libx264 | nvenc | auto (img2Sketch ffmpeg writer)
  STORYBOARD_GRADIO_PORT     — img2Sketch listen port (default 7860)
  STORYBOARD_SPLIT_LEN       — grid cell size (default 20; larger = faster sketch)
  STORYBOARD_FPS             — storyboard encode fps (default: segment fps; try 24)
  STORYBOARD_OBJ_SKIP        — object pass skip rate fallback (default 12)
  STORYBOARD_BG_SKIP         — background pass skip rate fallback (default 20)
  STORYBOARD_HOLD_FRACTION   — end color hold fraction 0–0.9 (default 0.25)
  STORYBOARD_IN_FLIGHT_SLACK — extra segment slots beyond sb cap when storyboard on (default 2)
  BANNER_SEGMENT_TAIL_SEC  — outgoing segment tail in visual transition (default 0.95)
  BANNER_SEGMENT_HEAD_SEC  — incoming segment head in visual transition (default 0.95)
  BANNER_XFADE_SEC         — each xfade leg duration (default 0.85)
  BANNER_HOLD_SEC          — full banner hold between xfades (default 0.8)
  BANNER_FPS                 — banner encode fps (default 30)
  STATIC_ROOT                — local static files root for /static/ image URLs
  (BANNER_OVERLAP_SEC / BANNER_DURATION_SEC are legacy aliases for xfade / 2×xfade)
"""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class HardwareProfile:
    logical_cpus: int
    ram_gb: float
    vram_gb: float | None
    gpu_name: str | None


@dataclass(frozen=True)
class WorkerLimits:
    concurrent_video_jobs: int
    script_workers: int
    encode_pool_workers: int
    gpu_concurrent_jobs: int
    hardware: HardwareProfile

    def as_dict(self) -> dict:
        return {
            "concurrent_video_jobs": self.concurrent_video_jobs,
            "script_workers": self.script_workers,
            "encode_pool_workers": self.encode_pool_workers,
            "gpu_concurrent_jobs": self.gpu_concurrent_jobs,
            "logical_cpus": self.hardware.logical_cpus,
            "ram_gb": round(self.hardware.ram_gb, 1),
            "vram_gb": self.hardware.vram_gb,
            "gpu_name": self.hardware.gpu_name,
        }


def _ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024**3)
    except Exception:
        pass
    return 16.0


def _gpu_info() -> tuple[float | None, str | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3), props.name
    except Exception:
        return None, None


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    vram_gb, gpu_name = _gpu_info()
    return HardwareProfile(
        logical_cpus=os.cpu_count() or 8,
        ram_gb=_ram_gb(),
        vram_gb=vram_gb,
        gpu_name=gpu_name,
    )


def _auto_concurrent_video_jobs(hw: HardwareProfile) -> int:
    by_ram = max(2, min(6, int(hw.ram_gb // 32)))
    by_cpu = max(2, min(6, hw.logical_cpus // 12))
    return max(2, min(by_ram, by_cpu))


def _auto_script_workers(hw: HardwareProfile) -> int:
    return max(8, min(24, hw.logical_cpus // 3))


def _auto_gpu_slots(hw: HardwareProfile) -> int:
    """Conservative defaults — override with GPU_CONCURRENT_JOBS env."""
    if hw.vram_gb is None:
        return 1
    if hw.vram_gb >= 16:
        return 2
    if hw.vram_gb >= 10:
        return 1
    return 1


def _auto_encode_pool_workers(hw: HardwareProfile, video_jobs: int) -> int:
    env_val = os.getenv("ENCODE_POOL_WORKERS") or os.getenv("GLOBAL_SEGMENT_WORKERS")
    if env_val is not None:
        return max(1, min(16, int(env_val)))
    # RAM-aware: ~1 worker per 8 GB, capped for stability
    by_ram = max(4, min(12, int(hw.ram_gb // 8)))
    by_jobs = max(4, min(12, video_jobs * 2))
    return min(by_ram, by_jobs)


@lru_cache(maxsize=1)
def get_worker_limits() -> WorkerLimits:
    hw = detect_hardware()

    video_jobs = int(os.getenv("CONCURRENT_VIDEO_JOBS", str(_auto_concurrent_video_jobs(hw))))
    video_jobs = max(1, min(8, video_jobs))

    script_workers = int(os.getenv("SCRIPT_WORKERS", str(_auto_script_workers(hw))))
    script_workers = max(1, min(48, script_workers))

    pool_env = os.getenv("ENCODE_POOL_WORKERS") or os.getenv("GLOBAL_SEGMENT_WORKERS")
    if pool_env is not None:
        encode_pool = max(1, min(48, int(pool_env)))
    else:
        encode_pool = _auto_encode_pool_workers(hw, video_jobs)

    gpu_env = os.getenv("GPU_CONCURRENT_JOBS")
    if gpu_env is not None:
        gpu_jobs = max(1, min(4, int(gpu_env)))
    else:
        gpu_jobs = _auto_gpu_slots(hw)

    return WorkerLimits(
        concurrent_video_jobs=video_jobs,
        script_workers=script_workers,
        encode_pool_workers=encode_pool,
        gpu_concurrent_jobs=gpu_jobs,
        hardware=hw,
    )


def log_worker_limits() -> WorkerLimits:
    limits = get_worker_limits()
    hw = limits.hardware
    gpu_label = f"{hw.gpu_name} ({hw.vram_gb:.1f}GB)" if hw.gpu_name else "none"
    logging.info(
        f"⚙️  Worker limits | CPUs={hw.logical_cpus} RAM={hw.ram_gb:.0f}GB GPU={gpu_label} | "
        f"videos={limits.concurrent_video_jobs} script={limits.script_workers} "
        f"segments={limits.encode_pool_workers} gpu={limits.gpu_concurrent_jobs}"
    )
    return limits
