import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import subprocess
import logging
import threading
import queue
import re
import hashlib
import random
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# VALID EFFECTS
# ============================================================================

VALID_EFFECTS = ['zoom_in', 'zoom_out', 'pan_left', 'pan_right']

# ============================================================================
# SEMANTIC SIGNALS
# ============================================================================

_SIGNALS = {
    "forward":     re.compile(r'\b(grow|rise|increase|build|expand|achieve|unlock|future|will|next|discover|reveal|hope|success|launch|start)\b', re.I),
    "receding":    re.compile(r'\b(fall|drop|decline|lose|collapse|fail|past|was|were|history|once|ago|before|ancient|remember|end)\b', re.I),
    "lateral":     re.compile(r'\b(world|global|across|throughout|every|millions|billions|spread|wide|all|entire|journey|move|travel|walk|run)\b', re.I),
    "intimate":    re.compile(r'\b(you|your|feel|heart|soul|alone|single|one|moment|personal|inside|within|deep|close|face)\b', re.I),
    "high_energy": re.compile(r'\b(war|fight|crash|explode|attack|force|power|destroy|crisis|urgent|never|always|impossible|unstoppable|greatest)\b', re.I),
    "low_energy":  re.compile(r'\b(peace|quiet|still|gentle|slow|rest|breathe|wonder|reflect|imagine|calm|soft|silence|dream)\b', re.I),
}

def _read_signals(text: str) -> dict:
    return {k: bool(p.search(text)) for k, p in _SIGNALS.items()}


# ============================================================================
# EDITOR PERSONALITY — stable per video, different across videos
# ============================================================================

def _editor_personality(video_seed: str) -> dict:
    h = int(hashlib.md5(video_seed.encode()).hexdigest(), 16)
    rng = random.Random(h)
    return {
        "depth_bias":         rng.uniform(0.3,  0.7),
        "approach_bias":      rng.uniform(0.35, 0.65),
        "pan_direction_bias": rng.uniform(0.35, 0.65),
        "text_sensitivity":   rng.uniform(0.45, 0.85),
    }


# ============================================================================
# EFFECT PICKER — history-aware, text-motivated, never robotic
# ============================================================================

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
    last   = history[-1] if history else None
    recent = history[-4:] if len(history) >= 4 else history[:]

    # Hard rules — never repeat immediately, cap at 2 in any 4-clip window
    candidates = [e for e in VALID_EFFECTS if e != last and recent.count(e) < 2]
    if not candidates:
        candidates = [e for e in VALID_EFFECTS if e != last] or VALID_EFFECTS[:]

    # Text-motivated selection
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

    # Personality-driven fallback
    zooms = [e for e in candidates if "zoom" in e]
    pans  = [e for e in candidates if "pan"  in e]

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

def precompute_segment_effects(
    vid_segments: list,
) -> tuple:
    """
    Compute effect for every segment sequentially in the main process.
    Returns (effect_assignments, video_seed).

    Video seed derived entirely from segment texts — no script_context,
    no timestamps, no external state. Same script = same seed = same
    editor personality across re-renders.

    Workers receive their assignment in segment_info — zero shared state,
    zero locks, zero race conditions at render time.
    """
    all_text   = " ".join(s.get("text", "") for s in vid_segments)
    video_seed = hashlib.md5(all_text.encode()).hexdigest()[:16]

    personality    = _editor_personality(video_seed)
    effect_history = []
    assignments    = []

    for global_clip_idx, segment in enumerate(vid_segments):
        text    = segment.get("text", "")
        signals = _read_signals(text)

        clip_seed = f"{video_seed}_clip_{global_clip_idx}"
        rng       = random.Random(int(hashlib.md5(clip_seed.encode()).hexdigest(), 16))

        effect = _pick_effect(signals, personality, effect_history, rng)
        effect_history.append(effect)

        assignments.append({
            "primary_effect":  effect,
            "video_seed":      video_seed,
            "global_clip_idx": global_clip_idx,
        })

    dist = {e: sum(1 for a in assignments if a["primary_effect"] == e)
            for e in VALID_EFFECTS}
    logger.info(
        f"[EffectPrecompute] {len(assignments)} segments | "
        f"seed={video_seed} | dist={dist}"
    )

    return assignments, video_seed


def get_secondary_effect(
    primary_effect: str,
    item_idx: int,
    global_clip_idx: int,
    video_seed: str,
) -> str:
    """
    Secondary clips within a segment (item_idx > 0) get a different effect
    from the primary so multiple images in one segment don't all do the same move.
    Deterministic — same inputs always return same output.
    """
    others = [e for e in VALID_EFFECTS if e != primary_effect]
    rng    = random.Random(
        int(hashlib.md5(
            f"{video_seed}_{global_clip_idx}_{item_idx}".encode()
        ).hexdigest(), 16)
    )
    return rng.choice(others)


# ============================================================================
# SPEED — variable, never mechanical
# ============================================================================

def get_motion_speed(
    duration: float,
    segment_text: str = "",
    effect_type: str = "zoom_in",
    segment_index: int = 0,
    video_seed: str = "default",
) -> float:
    """
    Returns speed multiplier (0.35–1.0) that scales zoom/pan canvas size.

    Floor is 0.35 — below this the per-frame pixel step gets so small
    that even pre-rounded integer sequences produce visible stutter.

    Three forces act on speed:
      1. Duration pressure  — short clips need faster movement to read
      2. Content energy     — text signals push speed up or down
      3. Controlled drift   — per-clip micro-variation breaks mechanical feel
    """
    signals = _read_signals(segment_text)

    # Duration base — floor raised to 0.38 so all clips stay smooth
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
        base = 0.38   # long clips drift slowly but visibly and smoothly

    # Energy push
    if signals.get("high_energy"):
        energy_push = +0.12
    elif signals.get("low_energy") or signals.get("intimate"):
        energy_push = -0.08   # reduced from -0.14 to protect floor
    elif signals.get("forward"):
        energy_push = +0.06
    elif signals.get("receding"):
        energy_push = -0.06
    else:
        energy_push = 0.0

    # Per-clip drift — breaks mechanical regularity
    # Seeded so same clip always gets same drift across re-renders
    drift_seed = f"{video_seed}_speed_{segment_index}_{effect_type}"
    drift_rng  = random.Random(int(hashlib.md5(drift_seed.encode()).hexdigest(), 16))
    drift      = drift_rng.uniform(-0.08, +0.08) * base   # tighter than before

    # Effect nudge
    if "pan" in effect_type:
        effect_nudge = +0.04
    elif effect_type == "zoom_out":
        effect_nudge = -0.03
    else:
        effect_nudge = 0.0

    # Floor at 0.35 — hard minimum for smooth motion
    speed = max(0.35, min(base + energy_push + drift + effect_nudge, 1.0))

    logger.debug(
        f"[MotionSpeed] {effect_type} | dur={duration:.1f}s "
        f"base={base:.2f} energy={energy_push:+.2f} "
        f"drift={drift:+.3f} → {speed:.3f}"
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
    k = torch.exp(-k.pow(2) / (2 * sigma ** 2))
    k = k / k.sum()
    return (k[:, None] * k[None, :]).reshape(1, 1, kernel_size, kernel_size)


@torch.no_grad()
def create_composite_gpu(image_path: str, target_w: int, target_h: int) -> torch.Tensor:
    """[1,3,H,W] float32 CUDA tensor at exactly target_w×target_h."""
    img = Image.open(image_path).convert("RGB")
    iw, ih = img.size
    t = transforms.ToTensor()(img).unsqueeze(0).cuda()

    if abs((iw / ih) - (target_w / target_h)) <= 0.01:
        return F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)

    blurred = gaussian_blur_gpu(
        F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False),
        kernel_size=41, sigma=13.0
    )
    scale = target_h / ih
    new_w = int(iw * scale)

    if new_w <= target_w:
        fore = F.interpolate(t, size=(target_h, new_w), mode='bilinear', align_corners=False)
        x = (target_w - new_w) // 2
        composite = blurred.clone()
        composite[:, :, :, x:x + new_w] = fore
    else:
        new_h = int(ih * (target_w / iw))
        fore = F.interpolate(t, size=(new_h, target_w), mode='bilinear', align_corners=False)
        y = (target_h - new_h) // 2
        composite = blurred.clone()
        composite[:, :, y:y + new_h, :] = fore

    return composite


# ============================================================================
# ASYNC GPU → FFMPEG PIPELINE
# ============================================================================

_SENTINEL = None


class GPUToFFmpegPipeline:
    """GPU generates frames → queue → consumer thread writes to FFmpeg stdin."""

    def __init__(self, width, height, fps=30, output_path="out.mp4",
                 preset="p3", crf=23, queue_depth=60):
        self.width       = width
        self.height      = height
        self.fps         = fps
        self.output_path = output_path
        self.frame_count = 0

        self.ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', '-',
            '-c:v', 'h264_nvenc', '-preset', preset,
            '-rc', 'vbr', '-cq', str(crf), '-b:v', '0',
            '-rc-lookahead', '20', '-surfaces', '64', '-async_depth', '4',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            output_path,
        ]

        self._q:       queue.Queue = queue.Queue(maxsize=queue_depth)
        self._process  = None
        self._consumer = None
        self._stderr_q: queue.Queue = queue.Queue()
        self._error    = None

    def start(self):
        self._process = subprocess.Popen(
            self.ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=10 ** 8,
        )
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._consumer = threading.Thread(target=self._write_loop, daemon=False)
        self._consumer.start()
        logger.info(f"✅ Pipeline started → {self.output_path}")

    def enqueue_frame(self, frame_tensor: torch.Tensor):
        if self._error:
            raise self._error

        _, _, h, w = frame_tensor.shape
        if h != self.height or w != self.width:
            frame_tensor = F.interpolate(frame_tensor, size=(self.height, self.width),
                                         mode='bilinear', align_corners=False)

        frame_u8 = (frame_tensor
                    .clone()
                    .squeeze_(0)
                    .permute(1, 2, 0)
                    .mul_(255.0)
                    .clamp_(0, 255)
                    .to(dtype=torch.uint8, non_blocking=True)
                    .cpu()
                    .numpy())

        if not frame_u8.flags['C_CONTIGUOUS']:
            frame_u8 = np.ascontiguousarray(frame_u8)

        self._q.put(frame_u8.tobytes())
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
        for line in iter(self._process.stderr.readline, b''):
            self._stderr_q.put(line.decode('utf-8', errors='ignore'))

    def _drain_stderr(self):
        while not self._stderr_q.empty():
            try:
                line = self._stderr_q.get_nowait()
                if 'error' in line.lower() or 'invalid' in line.lower():
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
        return self.output_path


# ============================================================================
# GPU EFFECT PROCESSOR
# — identical frame methods to original (smooth)
# — speed_multiplier scales canvas sizes (variable motion)
# — pre-rounded long tensors at construction (no per-frame jitter)
# ============================================================================

class GPUEffectProcessor:
    """
    Zoom: per-frame F.interpolate from original base tensor.
    No easing — pure linear progress.
    Speed is always dynamic based on clip duration via speed_multiplier.
    Pan: pre-built canvas.
    """

    _ZOOM_IN_BASE  = 1.20
    _ZOOM_OUT_BASE = 1.30
    _PAN_BASE      = 1.20

    _ZOOM_IN_MIN  = 1.04
    _ZOOM_OUT_MIN = 1.05
    _PAN_MIN      = 1.05

    def __init__(self, base_tensor: torch.Tensor, duration: float, fps: int,
                 speed_multiplier: float = 1.0):
        self.base = base_tensor.cuda() if not base_tensor.is_cuda else base_tensor
        self.N    = int(duration * fps)
        self.H    = base_tensor.shape[2]
        self.W    = base_tensor.shape[3]

        # speed_multiplier is always driven by get_motion_speed(duration=...)
        # short clip → high spd → big zoom range → fast feeling
        # long clip  → low spd → small zoom range → slow feeling
        spd = max(0.35, min(speed_multiplier, 1.0))

        self.ZI_START = 1.0
        self.ZI_END   = self._ZOOM_IN_MIN  + (self._ZOOM_IN_BASE  - self._ZOOM_IN_MIN)  * spd
        self.ZO_START = self._ZOOM_OUT_MIN + (self._ZOOM_OUT_BASE - self._ZOOM_OUT_MIN) * spd
        self.ZO_END   = 1.0

        pan_W            = int(self.W * (self._PAN_MIN + (self._PAN_BASE - self._PAN_MIN) * spd))
        self._pan_canvas = F.interpolate(self.base, size=(self.H, pan_W),
                                         mode='bilinear', align_corners=False)
        self._pan_max_x  = pan_W - self.W

        self.total_frames = self.N
        logger.info(
            f"📐 ZoomIn {self.ZI_START:.2f}→{self.ZI_END:.2f} | "
            f"ZoomOut {self.ZO_START:.2f}→{self.ZO_END:.2f} | "
            f"Pan {pan_W}px | spd={spd:.2f} | dur={duration:.1f}s"
        )

    def _zoom_frame(self, progress: float, zoom_start: float, zoom_end: float) -> torch.Tensor:
        """
        Fresh resize from original base at exact zoom level — no canvas, no easing.
        progress is raw linear 0→1 driven by frame index / total frames.
        """
        zoom  = zoom_start + (zoom_end - zoom_start) * progress
        zoom  = max(1.001, zoom)

        new_H = int(self.H * zoom)
        new_W = int(self.W * zoom)
        new_H += new_H % 2
        new_W += new_W % 2

        enlarged = F.interpolate(
            self.base, size=(new_H, new_W),
            mode='bilinear', align_corners=False
        )

        y = (new_H - self.H) // 2
        x = (new_W - self.W) // 2
        return enlarged[:, :, y:y + self.H, x:x + self.W]

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
        return self._pan_canvas[:, :, :, x:x + self.W]

    @torch.no_grad()
    def pan_right_to_left(self, i: int) -> torch.Tensor:
        progress = i / max(self.N - 1, 1)
        x = max(0, min(self._pan_max_x, int(self._pan_max_x * (1.0 - progress))))
        return self._pan_canvas[:, :, :, x:x + self.W]

    def cleanup(self):
        for attr in ('_pan_canvas', 'base'):
            setattr(self, attr, None)
        torch.cuda.empty_cache()
# ============================================================================
# MAIN ENTRY
# ============================================================================

def create_video_gpu_direct(
    image_path: str,
    duration: float,
    effect_type: Literal['zoom_in', 'zoom_out', 'pan_left', 'pan_right'] = 'zoom_in',
    image_index: Optional[int] = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    output_path: str = "out.mp4",
    preset: str = "p3",
    crf: int = 23,
    segment_text: str = "",
    speed_multiplier: Optional[float] = None,
    head_freeze_sec: float = 0.0,  # ✅ NEW: hold first frame before KB starts
) -> str:
    if effect_type not in VALID_EFFECTS:
        logger.warning(f"Unknown effect {effect_type!r}, defaulting to zoom_in")
        effect_type = 'zoom_in'

    if speed_multiplier is None:
        speed_multiplier = get_motion_speed(
            duration=duration,
            segment_text=segment_text,
            effect_type=effect_type,
            segment_index=image_index or 0,
            video_seed="direct",
        )

    # ✅ KB only runs over the non-frozen portion of the clip
    freeze_frames = int(head_freeze_sec * fps)
    kb_duration   = max(0.1, duration - head_freeze_sec)

    logger.info(
        f"🚀 {effect_type} | {duration:.2f}s @ {width}x{height} {fps}fps "
        f"speed={speed_multiplier:.2f} freeze={head_freeze_sec:.3f}s "
        f"({freeze_frames}fr) kb_dur={kb_duration:.3f}s → {output_path}"
    )

    base_tensor = create_composite_gpu(image_path, width, height)

    # ✅ GPUEffectProcessor only sees kb_duration — its total_frames
    # and zoom/pan range are computed over the motion-only portion.
    # The frozen frames are prepended before the KB loop below.
    processor = GPUEffectProcessor(base_tensor, kb_duration, fps, speed_multiplier)

    effect_map = {
        'zoom_in':   processor.zoom_in,
        'zoom_out':  processor.zoom_out,
        'pan_left':  processor.pan_left_to_right,
        'pan_right': processor.pan_right_to_left,
    }

    pipeline = GPUToFFmpegPipeline(width, height, fps, output_path, preset, crf)
    pipeline.start()

    # ✅ PHASE 1: Freeze — hold exact frame 0 of KB motion still.
    # The transition dissolves over this still frame.
    # KB starts moving only once the transition is complete.
    if freeze_frames > 0:
        first_frame = effect_map[effect_type](0)
        for _ in range(freeze_frames):
            pipeline.enqueue_frame(first_frame)
        logger.info(f"🧊 Froze {freeze_frames} frames ({head_freeze_sec:.3f}s)")

    # ✅ PHASE 2: KB motion runs normally over remaining duration
    log_every = max(fps * 2, 1)
    for i in range(processor.total_frames):
        pipeline.enqueue_frame(effect_map[effect_type](i))
        if i % log_every == 0 or i == processor.total_frames - 1:
            logger.info(
                f"⏳ {i+1}/{processor.total_frames} "
                f"({(i+1)/processor.total_frames*100:.1f}%)"
            )

    output = pipeline.finish()
    processor.cleanup()
    del base_tensor
    torch.cuda.empty_cache()
    logger.info(f"✅ Done: {output}")
    return output

# ============================================================================
# STATIC CONVENIENCE
# ============================================================================

def create_static_video_gpu(
    image_path: str,
    duration: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    output_path: str = "out.mp4",
    preset: str = "p1",
) -> str:
    """No motion — converts frame once, repeats bytes. Fastest path."""
    base_tensor = create_composite_gpu(image_path, width, height)
    pipeline    = GPUToFFmpegPipeline(width, height, fps, output_path, preset)
    pipeline.start()

    frame_bytes = (
        base_tensor.squeeze(0).permute(1, 2, 0)
        .mul_(255.0).clamp_(0, 255)
        .to(dtype=torch.uint8).cpu().numpy().tobytes()
    )
    total = int(duration * fps)
    for _ in range(total):
        pipeline._q.put(frame_bytes)
        pipeline.frame_count += 1

    output = pipeline.finish()
    del base_tensor
    torch.cuda.empty_cache()
    return output