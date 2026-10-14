"""
Ken Burns — semantic effect selection + FFmpeg zoompan (default) or GPU torch path.

Set KENBURNS_BACKEND=gpu to use the legacy per-frame CUDA renderer.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import random
import re
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageFilter

# Lazy torch — ffmpeg Ken Burns (default) must not init CUDA in every pool worker.
_torch_mod = None
_F_mod = None
_transforms_mod = None


def _ensure_torch():
    global _torch_mod, _F_mod, _transforms_mod
    if _torch_mod is None:
        import torch as _t
        import torch.nn.functional as _f
        from torchvision import transforms as _tr

        _torch_mod, _F_mod, _transforms_mod = _t, _f, _tr
    return _torch_mod, _F_mod, _transforms_mod


class _LazyTorch:
    """Defer CUDA init until the GPU Ken Burns path actually runs."""

    def __getattr__(self, name):
        return getattr(_ensure_torch()[0], name)


class _LazyF:
    def __getattr__(self, name):
        return getattr(_ensure_torch()[1], name)


class _LazyTransforms:
    def __getattr__(self, name):
        return getattr(_ensure_torch()[2], name)


torch = _LazyTorch()
F = _LazyF()
transforms = _LazyTransforms()

logger = logging.getLogger(__name__)

DEFAULT_NVENC_PRESET = os.getenv("KENBURNS_NVENC_PRESET", "p1")
KENBURNS_FFMPEG_TIMEOUT_SEC = int(os.getenv("KENBURNS_FFMPEG_TIMEOUT_SEC", "600"))


def _gpu_lock_timeout_sec() -> int:
    return max(60, int(os.getenv("GPU_LOCK_TIMEOUT_SEC", "900")))
# 8000 upscale made zoompan crawl for 10+ min/segment (portrait → ~8000x14200 frames);
# 4000 keeps the motion smooth at a fraction of the cost.
SMOOTH_UPSCALE = int(os.getenv("KENBURNS_FFMPEG_UPSCALE", "4000"))
# 0 (ffmpeg auto) spawns threads for every core in EACH worker → thrash with 12 workers.
FFMPEG_THREADS = int(os.getenv("KENBURNS_FFMPEG_THREADS", "4"))
INTERMEDIATE_PRESET = os.getenv("KENBURNS_FFMPEG_INTERMEDIATE_PRESET", "ultrafast")


def _ffmpeg_split_encode() -> bool:
    """
    Two-pass Ken Burns: CPU zoompan in parallel (libx264), then short NVENC pass.
    auto = on when USE_NVENC=1 (default). Set KENBURNS_FFMPEG_SPLIT=0 for single-pass.
    """
    env = os.getenv("KENBURNS_FFMPEG_SPLIT", "auto").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    return _use_nvenc()


def _kenburns_backend() -> str:
    return os.getenv("KENBURNS_BACKEND", "ffmpeg").strip().lower()


def kenburns_uses_gpu_lock() -> bool:
    """Torch path needs the global GPU lock for the whole render; FFmpeg only for NVENC."""
    return _kenburns_backend() == "gpu"


def _use_nvenc() -> bool:
    env = os.getenv("USE_NVENC", "1").strip().lower()
    return env not in ("0", "false", "no", "off")

# ============================================================================
# VALID EFFECTS
# ============================================================================

VALID_EFFECTS = ["zoom_in", "zoom_out", "pan_left", "pan_right"]

# ============================================================================
# SEMANTIC SIGNALS
# ============================================================================

_SIGNALS = {
    "forward": re.compile(
        r"\b(grow|rise|increase|build|expand|achieve|unlock|future|will|next|discover|reveal|hope|success|launch|start)\b",
        re.I,
    ),
    "receding": re.compile(
        r"\b(fall|drop|decline|lose|collapse|fail|past|was|were|history|once|ago|before|ancient|remember|end)\b",
        re.I,
    ),
    "lateral": re.compile(
        r"\b(world|global|across|throughout|every|millions|billions|spread|wide|all|entire|journey|move|travel|walk|run)\b",
        re.I,
    ),
    "intimate": re.compile(
        r"\b(you|your|feel|heart|soul|alone|single|one|moment|personal|inside|within|deep|close|face)\b",
        re.I,
    ),
    "high_energy": re.compile(
        r"\b(war|fight|crash|explode|attack|force|power|destroy|crisis|urgent|never|always|impossible|unstoppable|greatest)\b",
        re.I,
    ),
    "low_energy": re.compile(
        r"\b(peace|quiet|still|gentle|slow|rest|breathe|wonder|reflect|imagine|calm|soft|silence|dream)\b",
        re.I,
    ),
}


def _gpu_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_signals(text: str) -> dict:
    return {k: bool(p.search(text)) for k, p in _SIGNALS.items()}


# ============================================================================
# EDITOR PERSONALITY — stable per video, different across videos
# ============================================================================

def _editor_personality(video_seed: str) -> dict:
    h = int(hashlib.md5(video_seed.encode()).hexdigest(), 16)
    rng = random.Random(h)
    return {
        "depth_bias": rng.uniform(0.3, 0.7),
        "approach_bias": rng.uniform(0.35, 0.65),
        "pan_direction_bias": rng.uniform(0.35, 0.65),
        "text_sensitivity": rng.uniform(0.45, 0.85),
    }


def _pick_pan(candidates: list, personality: dict, rng: random.Random) -> str:
    if "pan_left" in candidates and "pan_right" in candidates:
        return "pan_left" if rng.random() < personality["pan_direction_bias"] else "pan_right"
    return candidates[0] if candidates else "pan_left"


def _pick_effect(
    signals: dict,
    personality: dict,
    history: list,
    rng: random.Random,
) -> str:
    last = history[-1] if history else None
    recent = history[-4:] if len(history) >= 4 else history[:]

    candidates = [e for e in VALID_EFFECTS if e != last and recent.count(e) < 2]
    if not candidates:
        candidates = [e for e in VALID_EFFECTS if e != last] or VALID_EFFECTS[:]

    if rng.random() < personality["text_sensitivity"]:
        if signals.get("intimate") and "zoom_in" in candidates:
            return "zoom_in"

        if signals.get("forward") and not signals.get("receding"):
            fwd = [e for e in candidates if e in ("zoom_in", "pan_left", "pan_right")]
            if fwd:
                if rng.random() < personality["depth_bias"] and "zoom_in" in fwd:
                    return "zoom_in"
                pans = [e for e in fwd if "pan" in e]
                if pans:
                    return _pick_pan(pans, personality, rng)

        if signals.get("receding") and not signals.get("forward"):
            rec = [e for e in candidates if e in ("zoom_out", "pan_left", "pan_right")]
            if rec:
                if rng.random() < personality["depth_bias"] and "zoom_out" in rec:
                    return "zoom_out"
                pans = [e for e in rec if "pan" in e]
                if pans:
                    return _pick_pan(pans, personality, rng)

        if signals.get("lateral") and not signals.get("intimate"):
            pans = [e for e in candidates if "pan" in e]
            if pans:
                return _pick_pan(pans, personality, rng)

    zooms = [e for e in candidates if "zoom" in e]
    pans = [e for e in candidates if "pan" in e]

    if rng.random() < personality["depth_bias"] and zooms:
        if "zoom_in" in zooms and "zoom_out" in zooms:
            return "zoom_in" if rng.random() < personality["approach_bias"] else "zoom_out"
        return zooms[0]
    if pans:
        return _pick_pan(pans, personality, rng)
    return rng.choice(candidates)


# ============================================================================
# PRECOMPUTE — called once in main process before workers start
# ============================================================================

def precompute_segment_effects(vid_segments: list) -> tuple:
    all_text = " ".join(s.get("text", "") for s in vid_segments)
    video_seed = hashlib.md5(all_text.encode()).hexdigest()[:16]

    personality = _editor_personality(video_seed)
    effect_history = []
    assignments = []

    for global_clip_idx, segment in enumerate(vid_segments):
        text = segment.get("text", "")
        signals = _read_signals(text)

        clip_seed = f"{video_seed}_clip_{global_clip_idx}"
        rng = random.Random(int(hashlib.md5(clip_seed.encode()).hexdigest(), 16))

        effect = _pick_effect(signals, personality, effect_history, rng)
        effect_history.append(effect)

        assignments.append(
            {
                "primary_effect": effect,
                "video_seed": video_seed,
                "global_clip_idx": global_clip_idx,
            }
        )

    dist = {e: sum(1 for a in assignments if a["primary_effect"] == e) for e in VALID_EFFECTS}
    logger.info(
        f"[EffectPrecompute] {len(assignments)} segments | seed={video_seed} | dist={dist}"
    )
    return assignments, video_seed


def get_secondary_effect(
    primary_effect: str,
    item_idx: int,
    global_clip_idx: int,
    video_seed: str,
) -> str:
    others = [e for e in VALID_EFFECTS if e != primary_effect]
    rng = random.Random(
        int(hashlib.md5(f"{video_seed}_{global_clip_idx}_{item_idx}".encode()).hexdigest(), 16)
    )
    return rng.choice(others)


def get_motion_speed(
    duration: float,
    segment_text: str = "",
    effect_type: str = "zoom_in",
    segment_index: int = 0,
    video_seed: str = "default",
) -> float:
    signals = _read_signals(segment_text)

    if duration <= 2.5:
        base = 0.92
    elif duration <= 4.0:
        base = 0.78
    elif duration <= 6.0:
        base = 0.62
    elif duration <= 9.0:
        base = 0.50
    elif duration <= 13.0:
        base = 0.42
    else:
        base = 0.38

    if signals.get("high_energy"):
        energy_push = +0.12
    elif signals.get("low_energy") or signals.get("intimate"):
        energy_push = -0.08
    elif signals.get("forward"):
        energy_push = +0.06
    elif signals.get("receding"):
        energy_push = -0.06
    else:
        energy_push = 0.0

    drift_seed = f"{video_seed}_speed_{segment_index}_{effect_type}"
    drift_rng = random.Random(int(hashlib.md5(drift_seed.encode()).hexdigest(), 16))
    drift = drift_rng.uniform(-0.08, +0.08) * base

    if "pan" in effect_type:
        effect_nudge = +0.04
    elif effect_type == "zoom_out":
        effect_nudge = -0.03
    else:
        effect_nudge = 0.0

    speed = max(0.35, min(base + energy_push + drift + effect_nudge, 1.0))

    logger.debug(
        f"[MotionSpeed] {effect_type} | dur={duration:.1f}s "
        f"base={base:.2f} energy={energy_push:+.2f} drift={drift:+.3f} → {speed:.3f}"
    )
    return speed


# ============================================================================
# GPU HELPERS
# ============================================================================

@torch.no_grad()
def gaussian_blur_gpu(img_tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    channels = img_tensor.shape[1]
    kernel = _get_gaussian_kernel2d(kernel_size, sigma).to(img_tensor.device)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return F.conv2d(img_tensor, kernel, padding=kernel_size // 2, groups=channels)


def _get_gaussian_kernel2d(kernel_size: int, sigma: float) -> torch.Tensor:
    k = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    k = torch.exp(-k.pow(2) / (2 * sigma**2))
    k = k / k.sum()
    return (k[:, None] * k[None, :]).reshape(1, 1, kernel_size, kernel_size)


@torch.no_grad()
def create_composite_gpu(image_path: str, target_w: int, target_h: int) -> torch.Tensor:
    """[1,3,H,W] float32 tensor at exactly target_w×target_h."""
    device = _gpu_device()
    img = Image.open(image_path).convert("RGB")
    iw, ih = img.size
    t = transforms.ToTensor()(img).unsqueeze(0).to(device)

    if abs((iw / ih) - (target_w / target_h)) <= 0.01:
        return F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)

    blurred = gaussian_blur_gpu(
        F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False),
        kernel_size=41,
        sigma=13.0,
    )
    scale = target_h / ih
    new_w = int(iw * scale)

    if new_w <= target_w:
        fore = F.interpolate(t, size=(target_h, new_w), mode="bilinear", align_corners=False)
        x = (target_w - new_w) // 2
        composite = blurred.clone()
        composite[:, :, :, x : x + new_w] = fore
    else:
        new_h = int(ih * (target_w / iw))
        fore = F.interpolate(t, size=(new_h, target_w), mode="bilinear", align_corners=False)
        y = (target_h - new_h) // 2
        composite = blurred.clone()
        composite[:, :, y : y + new_h, :] = fore

    return composite


# ============================================================================
# ASYNC GPU → FFMPEG PIPELINE
# ============================================================================

_SENTINEL = None
_active_pipeline = threading.local()


class GPUToFFmpegPipeline:
    """GPU generates frames → queue → consumer thread writes to FFmpeg stdin."""

    def __init__(
        self,
        width,
        height,
        fps=30,
        output_path="out.mp4",
        preset=DEFAULT_NVENC_PRESET,
        crf=23,
        queue_depth=60,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path
        self.frame_count = 0

        if torch.cuda.is_available():
            self.ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "h264_nvenc",
                "-preset",
                preset,
                "-rc",
                "vbr",
                "-cq",
                str(crf),
                "-b:v",
                "0",
                "-rc-lookahead",
                "0",
                "-surfaces",
                "32",
                "-async_depth",
                "4",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_path,
            ]
        else:
            self.ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
        "-s",
        f"{width}x{height}",
                "-pix_fmt",
                "rgb24",
        "-r",
        str(fps),
        "-i",
                "-",
        "-c:v",
                "libx264",
        "-preset",
                "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

        self._q: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._process = None
        self._consumer = None
        self._stderr_q: queue.Queue = queue.Queue()
        self._error = None

    def start(self):
        self._process = subprocess.Popen(
            self.ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
            bufsize=10**8,
        )
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._consumer = threading.Thread(target=self._write_loop, daemon=False)
        self._consumer.start()
        _active_pipeline.current = self
        logger.debug(f"Pipeline started → {self.output_path}")

    def kill(self):
        """Force-stop FFmpeg and writer thread (used when parent times out)."""
        try:
            self._q.put(_SENTINEL, timeout=1)
        except queue.Full:
            pass
        proc = self._process
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
        consumer = self._consumer
        if consumer is not None and consumer.is_alive():
            consumer.join(timeout=2)
        if getattr(_active_pipeline, "current", None) is self:
            _active_pipeline.current = None

    def enqueue_frame(self, frame_tensor: torch.Tensor):
        if self._error:
            raise self._error

        _, _, h, w = frame_tensor.shape
        if h != self.height or w != self.width:
            frame_tensor = F.interpolate(
                frame_tensor,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )

        frame_u8 = (
            frame_tensor.squeeze(0)
            .permute(1, 2, 0)
            .mul(255.0)
            .clamp(0, 255)
            .to(dtype=torch.uint8, non_blocking=True)
            .cpu()
            .numpy()
        )

        if not frame_u8.flags["C_CONTIGUOUS"]:
            frame_u8 = np.ascontiguousarray(frame_u8)

        payload = frame_u8.tobytes()
        try:
            self._q.put(payload, timeout=60)
        except queue.Full:
            logger.warning("⚠️ FFmpeg queue full — NVENC backpressure, waiting...")
            self._q.put(payload, timeout=300)
        self.frame_count += 1

    def _write_loop(self):
        try:
            while True:
                item = self._q.get()
                if item is _SENTINEL:
                    break
                self._process.stdin.write(item)
        except Exception as e:
            self._error = e
            logger.error(f"❌ Write error: {e}")

    def _read_stderr(self):
        for line in iter(self._process.stderr.readline, b""):
            self._stderr_q.put(line.decode("utf-8", errors="ignore"))

    def _drain_stderr(self):
        while not self._stderr_q.empty():
            try:
                line = self._stderr_q.get_nowait()
                if "error" in line.lower() or "invalid" in line.lower():
                    logger.error(line.strip())
            except queue.Empty:
                break

    def finish(self) -> str:
        self._q.put(_SENTINEL)
        self._consumer.join()
        if self._error:
            raise self._error
        try:
            self._process.stdin.close()
            rc = self._process.wait(timeout=120)
            self._drain_stderr()
            if rc != 0:
                logger.error(f"❌ FFmpeg exit {rc}")
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        if getattr(_active_pipeline, "current", None) is self:
            _active_pipeline.current = None
        return self.output_path


# ============================================================================
# GPU EFFECT PROCESSOR
# ============================================================================

class GPUEffectProcessor:
    _ZOOM_IN_BASE = 1.20
    _ZOOM_OUT_BASE = 1.30
    _PAN_BASE = 1.20

    _ZOOM_IN_MIN = 1.04
    _ZOOM_OUT_MIN = 1.05
    _PAN_MIN = 1.05

    def __init__(self, base_tensor: torch.Tensor, duration: float, fps: int, speed_multiplier: float = 1.0):
        device = _gpu_device()
        self.base = base_tensor.to(device) if base_tensor.device != device else base_tensor
        self.N = int(duration * fps)
        self.H = base_tensor.shape[2]
        self.W = base_tensor.shape[3]

        spd = max(0.35, min(speed_multiplier, 1.0))

        self.ZI_START = 1.0
        self.ZI_END = self._ZOOM_IN_MIN + (self._ZOOM_IN_BASE - self._ZOOM_IN_MIN) * spd
        self.ZO_START = self._ZOOM_OUT_MIN + (self._ZOOM_OUT_BASE - self._ZOOM_OUT_MIN) * spd
        self.ZO_END = 1.0

        pan_W = int(self.W * (self._PAN_MIN + (self._PAN_BASE - self._PAN_MIN) * spd))
        self._pan_canvas = F.interpolate(
            self.base, size=(self.H, pan_W), mode="bilinear", align_corners=False
        )
        self._pan_max_x = pan_W - self.W

        # Pre-scale once for zoom — avoids per-frame full-image F.interpolate
        max_z = max(self.ZI_END, self.ZO_START, 1.001)
        self._zoom_max = max_z
        zh = int(round(self.H * max_z))
        zw = int(round(self.W * max_z))
        self._zoom_canvas = F.interpolate(
            self.base, size=(zh, zw), mode="bilinear", align_corners=False
        )

        self.total_frames = self.N
        logger.debug(
            f"ZoomIn {self.ZI_START:.2f}→{self.ZI_END:.2f} | "
            f"ZoomOut {self.ZO_START:.2f}→{self.ZO_END:.2f} | "
            f"Pan {pan_W}px | spd={spd:.2f} | dur={duration:.1f}s"
        )

    def _zoom_frame(self, progress: float, zoom_start: float, zoom_end: float) -> torch.Tensor:
        z = max(1.001, zoom_start + (zoom_end - zoom_start) * progress)
        sh, sw = self._zoom_canvas.shape[2], self._zoom_canvas.shape[3]
        crop_h = max(self.H, min(sh, int(round(self.H * self._zoom_max / z))))
        crop_w = max(self.W, min(sw, int(round(self.W * self._zoom_max / z))))
        y = max(0, (sh - crop_h) // 2)
        x = max(0, (sw - crop_w) // 2)
        cropped = self._zoom_canvas[:, :, y : y + crop_h, x : x + crop_w]
        if cropped.shape[2] != self.H or cropped.shape[3] != self.W:
            cropped = F.interpolate(
                cropped, size=(self.H, self.W), mode="bilinear", align_corners=False
            )
        return cropped

    @torch.no_grad()
    def zoom_in(self, i: int) -> torch.Tensor:
        progress = i / max(self.N - 1, 1)
        return self._zoom_frame(progress, self.ZI_START, self.ZI_END)

    @torch.no_grad()
    def zoom_out(self, i: int) -> torch.Tensor:
        progress = i / max(self.N - 1, 1)
        return self._zoom_frame(progress, self.ZO_START, self.ZO_END)

    @torch.no_grad()
    def pan_left_to_right(self, i: int) -> torch.Tensor:
        progress = i / max(self.N - 1, 1)
        x = max(0, min(self._pan_max_x, int(self._pan_max_x * progress)))
        return self._pan_canvas[:, :, :, x : x + self.W]

    @torch.no_grad()
    def pan_right_to_left(self, i: int) -> torch.Tensor:
        progress = i / max(self.N - 1, 1)
        x = max(0, min(self._pan_max_x, int(self._pan_max_x * (1.0 - progress))))
        return self._pan_canvas[:, :, :, x : x + self.W]

    def cleanup(self):
        for attr in ("_pan_canvas", "_zoom_canvas", "base"):
            setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# MAIN ENTRY
# ============================================================================

def _composite_image_pil(image_path: str, width: int, height: int) -> str:
    """Letterbox/pillarbox with blurred fill — matches create_composite_gpu intent."""
    with Image.open(image_path).convert("RGB") as im0:
        img = im0.copy()

    iw, ih = img.size
    out_dir = os.path.dirname(os.path.abspath(image_path)) or "."
    out_path = os.path.join(out_dir, f"kb_comp_{uuid.uuid4().hex}.jpg")

    if abs((iw / ih) - (width / height)) <= 0.01:
        img.resize((width, height), Image.Resampling.LANCZOS).save(out_path, quality=95)
        return out_path

    blurred = img.resize((width, height), Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(40)
    )
    scale = height / ih
    new_w = int(iw * scale)

    if new_w <= width:
        fore = img.resize((new_w, height), Image.Resampling.LANCZOS)
        x = (width - new_w) // 2
        comp = blurred.copy()
        comp.paste(fore, (x, 0))
    else:
        new_h = int(ih * (width / iw))
        fore = img.resize((width, new_h), Image.Resampling.LANCZOS)
        y = (height - new_h) // 2
        comp = blurred.copy()
        comp.paste(fore, (0, y))

    comp.save(out_path, quality=95)
    return out_path


def _zoom_increment(speed_multiplier: float, fps: int) -> float:
    speed = max(0.35, min(float(speed_multiplier), 1.0))
    base = 0.001 * (25.0 / max(fps, 1))
    return round(base * (0.55 + speed * 0.9), 6)


def _pan_zoom_level(speed_multiplier: float) -> float:
    speed = max(0.35, min(float(speed_multiplier), 1.0))
    return round(1.05 + 0.12 * speed, 4)


def _max_zoom(speed_multiplier: float) -> float:
    speed = max(0.35, min(float(speed_multiplier), 1.0))
    return round(1.08 + 0.14 * speed, 4)


def _zoompan_filter_chain(
    effect_type: str,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
    speed_multiplier: float,
) -> str:
    """Bannerbear-style zoompan: upscale for smoothness, then pan/zoom."""
    w, h = width, height
    d = max(num_frames, 1)
    inc = _zoom_increment(speed_multiplier, fps)
    pan_z = _pan_zoom_level(speed_multiplier)
    max_z = _max_zoom(speed_multiplier)
    denom = max(d - 1, 1)

    if effect_type == "zoom_in":
        z_expr = f"min({max_z},zoom+{inc})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif effect_type == "zoom_out":
        z_expr = f"max(1.0,{max_z}-{inc}*on)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif effect_type == "pan_left":
        z_expr = str(pan_z)
        x_expr = f"(iw-iw/zoom)*on/{denom}"
        y_expr = "ih/2-(ih/zoom/2)"
    elif effect_type == "pan_right":
        z_expr = str(pan_z)
        x_expr = f"(iw-iw/zoom)*(1-on/{denom})"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        z_expr = f"min({max_z},zoom+{inc})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    return (
        f"scale={w}:-2,setsar=1:1,crop={w}:{h},"
        f"scale={SMOOTH_UPSCALE}:-1,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={d}:s={w}x{h}:fps={fps}"
    )


def _run_ffmpeg_kb(
    cmd: list[str],
    label: str,
    total_frames: int = 0,
    nvenc_semaphore=None,
    *,
    nvenc_lock: bool | None = None,
) -> None:
    """Run ffmpeg; log frame progress; optionally limit concurrent NVENC."""
    if FFMPEG_THREADS > 0:
        cmd = cmd[:1] + ["-threads", str(FFMPEG_THREADS)] + cmd[1:]

    cmd_uses_nvenc = any("nvenc" in str(a) for a in cmd)
    if nvenc_lock is None:
        nvenc_lock = _use_nvenc() and cmd_uses_nvenc
    acquired = False
    if nvenc_lock and nvenc_semaphore is not None:
        nvenc_timeout = _gpu_lock_timeout_sec()
        logger.info(f"⏳ Ken Burns NVENC slot wait ({label})...")
        if not nvenc_semaphore.acquire(timeout=nvenc_timeout):
            raise TimeoutError(
                f"Ken Burns NVENC slot unavailable after {nvenc_timeout}s ({label}) — "
                f"a prior worker likely died while holding the GPU lock; restart wsgi.py"
            )
        acquired = True
        logger.info(f"🔒 Ken Burns NVENC slot acquired ({label})")

    import re

    frame_re = re.compile(r"frame=\s*(\d+)")
    log_every = max(min((total_frames or 300) // 15, 90), 1)
    last_logged = -log_every
    progress_prefix = "Ken Burns" if not cmd_uses_nvenc else "Ken Burns NVENC"
    ffmpeg_timeout = KENBURNS_FFMPEG_TIMEOUT_SEC

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        err_tail: list[str] = []
        assert proc.stderr is not None
        deadline = time.monotonic() + ffmpeg_timeout
        for line in proc.stderr:
            if time.monotonic() > deadline:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                raise TimeoutError(
                    f"FFmpeg Ken Burns timed out after {ffmpeg_timeout}s ({label})"
                )
            err_tail.append(line)
            if len(err_tail) > 80:
                err_tail.pop(0)
            if total_frames > 0:
                m = frame_re.search(line)
                if m:
                    frame = int(m.group(1))
                    if frame - last_logged >= log_every or frame >= total_frames:
                        pct = min(100.0, frame / total_frames * 100.0)
                        logger.info(
                            f"⏳ {progress_prefix} {frame}/{total_frames} ({pct:.1f}%)"
                        )
                        last_logged = frame
        rc = proc.wait(timeout=max(1, int(deadline - time.monotonic())))
        if rc != 0:
            tail = "".join(err_tail)[-1200:]
            raise RuntimeError(f"FFmpeg Ken Burns failed ({label}): {tail}")
    except subprocess.TimeoutExpired:
        proc.kill()
        raise TimeoutError(
            f"FFmpeg Ken Burns timed out after {ffmpeg_timeout}s ({label})"
        ) from None
    finally:
        if acquired:
            nvenc_semaphore.release()
            logger.info(f"🔓 Ken Burns NVENC slot released ({label})")


def _zoompan_cpu_cmd(
    composite_path: str,
    vf: str,
    kb_duration: float,
    fps: int,
    output_path: str,
) -> list[str]:
    """Pass 1 — CPU zoompan only; no GPU lock (libx264 intermediate)."""
    return [
        "ffmpeg", "-y", "-loop", "1", "-i", composite_path,
        "-vf", vf, "-t", str(kb_duration), "-r", str(fps),
        "-c:v", "libx264", "-preset", INTERMEDIATE_PRESET, "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        output_path,
    ]


def _nvenc_from_file_cmd(input_path: str, output_path: str) -> list[str]:
    """Pass 2 — decode intermediate → NVENC (short; holds GPU lock only here)."""
    return [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "h264_nvenc", "-preset", DEFAULT_NVENC_PRESET,
        "-rc", "vbr", "-cq", "23", "-b:v", "0",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        output_path,
    ]


def _create_video_ffmpeg_direct(
    image_path: str,
    duration: float,
    effect_type: str,
    width: int,
    height: int,
    fps: int,
    output_path: str,
    speed_multiplier: float,
    head_freeze_sec: float,
    nvenc_semaphore=None,
) -> str:
    freeze_sec = max(0.0, float(head_freeze_sec))
    kb_duration = max(0.1, duration - freeze_sec)
    kb_frames = max(1, int(round(kb_duration * fps)))

    composite_path = _composite_image_pil(image_path, width, height)
    temp_paths = [composite_path]

    try:
        use_nvenc = _use_nvenc()
        split = use_nvenc and _ffmpeg_split_encode()
        vf = _zoompan_filter_chain(effect_type, width, height, fps, kb_frames, speed_multiplier)

        motion_path = output_path
        if freeze_sec > 0.01 or split:
            motion_path = os.path.join(tempfile.gettempdir(), f"kb_motion_{uuid.uuid4().hex}.mp4")
            temp_paths.append(motion_path)

        if split:
            logger.info("🎬 Ken Burns pass 1/2: CPU zoompan (parallel, no GPU lock)...")
            _run_ffmpeg_kb(
                _zoompan_cpu_cmd(composite_path, vf, kb_duration, fps, motion_path),
                "zoompan",
                total_frames=kb_frames,
                nvenc_semaphore=nvenc_semaphore,
                nvenc_lock=False,
            )
        else:
            codec = "h264_nvenc" if use_nvenc else "libx264"
            preset = DEFAULT_NVENC_PRESET if use_nvenc else "ultrafast"
            cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", composite_path,
                "-vf", vf, "-t", str(kb_duration), "-r", str(fps),
                "-c:v", codec, "-preset", preset,
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                motion_path,
            ]
            if use_nvenc:
                cmd.extend(["-rc", "vbr", "-cq", "23", "-b:v", "0"])
            _run_ffmpeg_kb(cmd, "motion", total_frames=kb_frames, nvenc_semaphore=nvenc_semaphore)

        if freeze_sec > 0.01:
            freeze_frames = max(1, int(round(freeze_sec * fps)))
            total_out_frames = freeze_frames + kb_frames
            fc = (
                f"[0:v]scale={width}:{height},setsar=1,fps={fps},trim=duration={freeze_sec}[hold];"
                f"[1:v]scale={width}:{height},setsar=1,fps={fps}[motion];"
                f"[hold][motion]concat=n=2:v=1:a=0[out]"
            )
            concat_out = output_path
            if split:
                concat_out = os.path.join(tempfile.gettempdir(), f"kb_concat_{uuid.uuid4().hex}.mp4")
                temp_paths.append(concat_out)
            concat_codec = "libx264" if split else ("h264_nvenc" if use_nvenc else "libx264")
            concat_preset = INTERMEDIATE_PRESET if split else (DEFAULT_NVENC_PRESET if use_nvenc else "ultrafast")
            concat_cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", composite_path,
                "-i", motion_path, "-filter_complex", fc,
                "-map", "[out]", "-c:v", concat_codec, "-preset", concat_preset,
                "-pix_fmt", "yuv420p", "-an",
                concat_out if split else output_path,
            ]
            if not split:
                concat_cmd.insert(-1, "-movflags")
                concat_cmd.insert(-1, "+faststart")
            if use_nvenc and not split:
                concat_cmd.extend(["-rc", "vbr", "-cq", "23", "-b:v", "0"])
            elif split:
                concat_cmd.extend(["-crf", "18"])
            _run_ffmpeg_kb(
                concat_cmd, "freeze+concat",
                total_frames=total_out_frames,
                nvenc_semaphore=nvenc_semaphore,
                nvenc_lock=not split,
            )
            nvenc_src = concat_out
        elif split:
            nvenc_src = motion_path
        else:
            nvenc_src = None

        if split:
            src = nvenc_src or motion_path
            logger.info("🎬 Ken Burns pass 2/2: NVENC (GPU queue)...")
            _run_ffmpeg_kb(
                _nvenc_from_file_cmd(src, output_path),
                "nvenc",
                total_frames=kb_frames + (max(1, int(round(freeze_sec * fps))) if freeze_sec > 0.01 else 0),
                nvenc_semaphore=nvenc_semaphore,
                nvenc_lock=True,
            )

        return output_path
    finally:
        for p in temp_paths:
            if p != output_path and os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _create_static_ffmpeg(
    image_path: str,
    duration: float,
    width: int,
    height: int,
    fps: int,
    output_path: str,
    nvenc_semaphore=None,
) -> str:
    composite_path = _composite_image_pil(image_path, width, height)
    try:
        use_nvenc = _use_nvenc()
        codec = "h264_nvenc" if use_nvenc else "libx264"
        preset = DEFAULT_NVENC_PRESET if use_nvenc else "ultrafast"
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", composite_path,
            "-t", str(duration), "-r", str(fps),
            "-c:v", codec, "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            output_path,
        ]
        if use_nvenc:
            cmd.extend(["-rc", "vbr", "-cq", "23", "-b:v", "0"])
        total_frames = max(1, int(round(duration * fps)))
        _run_ffmpeg_kb(cmd, "static", total_frames=total_frames, nvenc_semaphore=nvenc_semaphore)
        return output_path
    finally:
        if os.path.isfile(composite_path):
            try:
                os.remove(composite_path)
            except OSError:
                pass


def _create_video_torch_direct(
    image_path: str,
    duration: float,
    effect_type: str,
    width: int,
    height: int,
    fps: int,
    output_path: str,
    preset: str,
    crf: int,
    speed_multiplier: float,
    head_freeze_sec: float,
) -> str:
    freeze_frames = int(head_freeze_sec * fps)
    kb_duration = max(0.1, duration - head_freeze_sec)

    logger.info(
        f"🚀 torch {effect_type} | {duration:.2f}s @ {width}x{height} {fps}fps "
        f"speed={speed_multiplier:.2f} freeze={head_freeze_sec:.3f}s "
        f"({freeze_frames}fr) kb_dur={kb_duration:.3f}s → {output_path}"
    )

    base_tensor = create_composite_gpu(image_path, width, height)
    processor = GPUEffectProcessor(base_tensor, kb_duration, fps, speed_multiplier)

    effect_map = {
        "zoom_in": processor.zoom_in,
        "zoom_out": processor.zoom_out,
        "pan_left": processor.pan_left_to_right,
        "pan_right": processor.pan_right_to_left,
    }

    pipeline = GPUToFFmpegPipeline(width, height, fps, output_path, preset, crf)
    try:
        pipeline.start()

        if freeze_frames > 0:
            first_frame = effect_map[effect_type](0)
            for _ in range(freeze_frames):
                pipeline.enqueue_frame(first_frame)
            logger.debug(f"🧊 Froze {freeze_frames} frames ({head_freeze_sec:.3f}s)")

        log_every = max(min(fps * 2, 60), 1)
        render_fn = effect_map[effect_type]
        total = processor.total_frames
        for i in range(total):
            pipeline.enqueue_frame(render_fn(i))
            if i % log_every == 0 or i == total - 1:
                logger.info(
                    f"⏳ Ken Burns {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)"
                )

        output = pipeline.finish()
    except Exception:
        pipeline.kill()
        raise
    finally:
        processor.cleanup()
        del base_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info(f"✅ Ken Burns done (torch): {output}")
    return output


def create_video_gpu_direct(
    image_path: str,
    duration: float,
    effect_type: Literal["zoom_in", "zoom_out", "pan_left", "pan_right"] = "zoom_in",
    image_index: Optional[int] = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    output_path: str = "out.mp4",
    preset: str = DEFAULT_NVENC_PRESET,
    crf: int = 23,
    segment_text: str = "",
    speed_multiplier: Optional[float] = None,
    head_freeze_sec: float = 0.0,
    video_seed: str = "direct",
    nvenc_semaphore=None,
) -> str:
    """Pipeline entry — FFmpeg zoompan by default; KENBURNS_BACKEND=gpu for torch path."""
    if effect_type not in VALID_EFFECTS:
        logger.warning(f"Unknown effect {effect_type!r}, defaulting to zoom_in")
        effect_type = "zoom_in"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    if speed_multiplier is None:
        speed_multiplier = get_motion_speed(
        duration=duration,
        segment_text=segment_text,
        effect_type=effect_type,
            segment_index=image_index or 0,
            video_seed=video_seed,
        )

    backend = _kenburns_backend()
    logger.info(
        f"🚀 {effect_type} ({backend}) | {duration:.2f}s @ {width}x{height} {fps}fps "
        f"speed={speed_multiplier:.2f} freeze={head_freeze_sec:.3f}s → {output_path}"
    )

    if backend == "gpu":
        return _create_video_torch_direct(
            image_path=image_path,
            duration=duration,
            effect_type=effect_type,
            width=width,
            height=height,
            fps=fps,
            output_path=output_path,
            preset=preset,
            crf=crf,
            speed_multiplier=speed_multiplier,
            head_freeze_sec=head_freeze_sec,
        )

    try:
        out = _create_video_ffmpeg_direct(
            image_path=image_path,
            duration=duration,
            effect_type=effect_type,
            width=width,
            height=height,
            fps=fps,
            output_path=output_path,
            speed_multiplier=speed_multiplier,
            head_freeze_sec=head_freeze_sec,
            nvenc_semaphore=nvenc_semaphore,
        )
        logger.info(f"✅ Ken Burns done (ffmpeg): {out}")
        return out
    except Exception as exc:
        logger.error(f"❌ FFmpeg Ken Burns failed: {exc} — falling back to torch")
        return _create_video_torch_direct(
            image_path=image_path,
            duration=duration,
            effect_type=effect_type,
            width=width,
            height=height,
            fps=fps,
            output_path=output_path,
            preset=preset,
            crf=crf,
            speed_multiplier=speed_multiplier,
            head_freeze_sec=head_freeze_sec,
        )


def create_static_video_gpu(
    image_path: str,
    duration: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    output_path: str = "out.mp4",
    preset: str = DEFAULT_NVENC_PRESET,
    nvenc_semaphore=None,
) -> str:
    """No motion — FFmpeg by default; KENBURNS_BACKEND=gpu for torch path."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    if _kenburns_backend() == "gpu":
        base_tensor = create_composite_gpu(image_path, width, height)
        pipeline = GPUToFFmpegPipeline(width, height, fps, output_path, preset)
        pipeline.start()
        frame_bytes = (
            base_tensor.squeeze(0)
            .permute(1, 2, 0)
            .mul_(255.0)
            .clamp_(0, 255)
            .to(dtype=torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
        total = int(duration * fps)
        for _ in range(total):
            pipeline._q.put(frame_bytes)
            pipeline.frame_count += 1
        output = pipeline.finish()
        del base_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output

    return _create_static_ffmpeg(
        image_path, duration, width, height, fps, output_path, nvenc_semaphore
    )
