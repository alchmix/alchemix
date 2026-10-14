"""
Remove DreamFace mandatory top-left "Ai" disclosure tag from avatar MP4s.

Controlled entirely via environment variables — no changes needed elsewhere
to tune or disable. Wired from avatar_extractor / process_full_video only.

Enable / disable:
  DREAMFACE_STRIP_AI_TAG=1          (default on; set 0/false/no to skip)

Method:
  DREAMFACE_STRIP_METHOD=crop       (default — tight corner crop+scale)
  DREAMFACE_STRIP_METHOD=delogo     (ffmpeg delogo inpaint)

Crop tuning (matched to preview script):
  DREAMFACE_CROP_TAG_X_PCT=0.008
  DREAMFACE_CROP_TAG_Y_PCT=0.008
  DREAMFACE_CROP_TAG_W_PCT=0.145
  DREAMFACE_CROP_TAG_H_PCT=0.075
  DREAMFACE_CROP_SCALE=1.055
  DREAMFACE_CROP_MARGIN_PCT=0.018
  DREAMFACE_CROP_NUDGE_X=12
  DREAMFACE_CROP_NUDGE_Y=8

Delogo tuning:
  DREAMFACE_DELOGO_X_PCT=0.02
  DREAMFACE_DELOGO_Y_PCT=0.015
  DREAMFACE_DELOGO_W_PCT=0.12
  DREAMFACE_DELOGO_H_PCT=0.06
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path


def _even(n: int) -> int:
    return max(2, n - (n % 2))


def strip_enabled() -> bool:
    return os.environ.get("DREAMFACE_STRIP_AI_TAG", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def strip_method() -> str:
    return os.environ.get("DREAMFACE_STRIP_METHOD", "crop").strip().lower()


def _probe_video_size(path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{r.stderr[-600:]}")
    data = json.loads(r.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video size: {path}")
    return width, height


def crop_tag_region(width: int, height: int) -> tuple[int, int, int, int]:
    x_pct = float(os.environ.get("DREAMFACE_CROP_TAG_X_PCT", "0.008"))
    y_pct = float(os.environ.get("DREAMFACE_CROP_TAG_Y_PCT", "0.008"))
    w_pct = float(os.environ.get("DREAMFACE_CROP_TAG_W_PCT", "0.145"))
    h_pct = float(os.environ.get("DREAMFACE_CROP_TAG_H_PCT", "0.075"))

    x = _even(int(width * x_pct))
    y = _even(int(height * y_pct))
    w = _even(int(width * w_pct))
    h = _even(int(height * h_pct))
    w = min(w, width - x - 2)
    h = min(h, height - y - 2)
    return x, y, w, h


def delogo_region(width: int, height: int) -> tuple[int, int, int, int]:
    x_pct = float(os.environ.get("DREAMFACE_DELOGO_X_PCT", "0.02"))
    y_pct = float(os.environ.get("DREAMFACE_DELOGO_Y_PCT", "0.015"))
    w_pct = float(os.environ.get("DREAMFACE_DELOGO_W_PCT", "0.12"))
    h_pct = float(os.environ.get("DREAMFACE_DELOGO_H_PCT", "0.06"))

    x = _even(int(width * x_pct))
    y = _even(int(height * y_pct))
    w = _even(int(width * w_pct))
    h = _even(int(height * h_pct))
    w = min(w, width - x - 2)
    h = min(h, height - y - 2)
    return x, y, w, h


def crop_scale_params(width: int, height: int) -> dict:
    tx, ty, tw, th = crop_tag_region(width, height)
    margin_pct = float(os.environ.get("DREAMFACE_CROP_MARGIN_PCT", "0.018"))
    margin_x = max(2, int(width * margin_pct))
    margin_y = max(2, int(height * margin_pct))

    hide_w = tx + tw + margin_x
    hide_h = ty + th + margin_y

    scale = float(os.environ.get("DREAMFACE_CROP_SCALE", "1.055"))
    scale = max(1.02, min(scale, 1.10))

    scaled_w = _even(int(width * scale))
    scaled_h = _even(int(height * scale))
    crop_w = _even(width)
    crop_h = _even(height)

    crop_x = _even(int(hide_w * scale))
    crop_y = _even(int(hide_h * scale))

    nudge_x = int(os.environ.get("DREAMFACE_CROP_NUDGE_X", "12"))
    nudge_y = int(os.environ.get("DREAMFACE_CROP_NUDGE_Y", "8"))
    crop_x += _even(max(0, nudge_x))
    crop_y += _even(max(0, nudge_y))

    max_x = max(0, scaled_w - crop_w - 2)
    max_y = max(0, scaled_h - crop_h - 2)
    crop_x = min(crop_x, max_x)
    crop_y = min(crop_y, max_y)

    return {
        "scale": scale,
        "scaled_w": scaled_w,
        "scaled_h": scaled_h,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "tag_box": (tx, ty, tw, th),
    }


def _run_ffmpeg(src: str, dst: str, vf: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        "-movflags", "+faststart",
        dst,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"ffmpeg tag strip failed:\n{r.stderr[-1000:]}")


def stripped_output_path(input_path: str) -> str:
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}_notag{p.suffix}"))


def strip_dreamface_ai_tag(input_path: str, output_path: str | None = None) -> str:
    """
    Remove top-left Ai tag. Returns output path (creates *_notag.mp4 by default).
    """
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Avatar not found: {input_path}")

    out = output_path or stripped_output_path(input_path)
    width, height = _probe_video_size(input_path)
    method = strip_method()

    if method == "delogo":
        x, y, w, h = delogo_region(width, height)
        vf = f"delogo=x={x}:y={y}:w={w}:h={h}:show=0"
        logging.info(
            f"🏷️ DreamFace delogo | {width}x{height} | box=({x},{y},{w},{h})"
        )
    else:
        params = crop_scale_params(width, height)
        tx, ty, tw, th = params["tag_box"]
        vf = (
            f"scale={params['scaled_w']}:{params['scaled_h']}:flags=lanczos,"
            f"crop={params['crop_w']}:{params['crop_h']}:"
            f"{params['crop_x']}:{params['crop_y']}"
        )
        logging.info(
            f"🏷️ DreamFace crop+scale | {width}x{height} | "
            f"tag=({tx},{ty},{tw},{th}) scale={params['scale']:.3f} "
            f"crop@({params['crop_x']},{params['crop_y']})"
        )

    _run_ffmpeg(input_path, out, vf)
    mb = os.path.getsize(out) / (1024 * 1024)
    logging.info(f"✅ DreamFace tag stripped → {out} ({mb:.1f}MB)")
    return out


def maybe_strip_dreamface_avatar(input_path: str | None) -> str | None:
    """
    Pipeline hook: strip tag when enabled, with *_notag.mp4 cache.
    On failure logs warning and returns original path.
    """
    if not input_path or not os.path.exists(input_path):
        return input_path
    if not strip_enabled():
        return input_path

    out = stripped_output_path(input_path)
    try:
        raw_mtime = os.path.getmtime(input_path)
        if os.path.exists(out) and os.path.getsize(out) >= 100_000:
            if os.path.getmtime(out) >= raw_mtime:
                logging.info(f"♻️ DreamFace stripped cache hit | {out}")
                return out
    except OSError:
        pass

    try:
        return strip_dreamface_ai_tag(input_path, out)
    except Exception as exc:
        logging.warning(
            f"⚠️ DreamFace tag strip failed — using original avatar: {exc}"
        )
        return input_path
