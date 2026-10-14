"""
Timeline video editor — FFmpeg pipeline synced with frontend Timeline.jsx.

Speed strategy (same idea as gpu_kenburns_simple split encode):
  1. Parallel CPU workers pre-render image overlays that need per-frame scale
     animation (Ken Burns and/or zoom transitions) via zoompan
  2. Single filter_complex pass on all CPU threads (libx264 ultrafast)
  3. Optional NVENC finalize pass when USE_NVENC=1 and TIMELINE_FFMPEG_SPLIT=auto

v2 pipeline (synced with timelineFx.js / timeline_fx.py):
  - Overlays are always best-fit to the base frame (aspect preserved, centered)
  - Overlay stream PTS shifted onto the main timeline so clips play their
    content inside [start, end] instead of being consumed at t=0
  - Per-overlay transitions: fade / slide-* (main graph), zoom (pre-render)
  - Global FX chain after compositing: color fx → vignette → noise → fades
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from timeline_fx import (
    bg_blur_sigma,
    color_fx_filters,
    cover_blur_filters,
    fade_filters,
    fg_frame_filters,
    gradient_source,
    noise_ffmpeg_level,
    overlay_alpha_fade_filters,
    overlay_prerender_graph,
    overlay_xy_exprs,
    parse_fit,
    parse_focus,
    parse_global_fx,
    parse_ken_burns,
    parse_overlay_bg,
    parse_transition,
    parse_vignette,
    PRERENDER_KEY_COLOR,
    transition_needs_zoom_prerender,
)

logger = logging.getLogger(__name__)

OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", "outputs"))
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(os.getenv("OVERLAY_CACHE", "overlay_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = Path(os.getenv("TIMELINE_WORK_DIR", "timeline_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FPS = os.getenv("OUTPUT_FPS", "30")
OUTPUT_CRF = int(os.getenv("OUTPUT_CRF", "20"))
NVENC_CQ = int(os.getenv("NVENC_CQ", "23"))
NVENC_PRESET = os.getenv("KENBURNS_NVENC_PRESET", os.getenv("TIMELINE_NVENC_PRESET", "p1"))
INTERMEDIATE_PRESET = os.getenv(
    "TIMELINE_FFMPEG_INTERMEDIATE_PRESET",
    os.getenv("KENBURNS_FFMPEG_INTERMEDIATE_PRESET", "ultrafast"),
)
FFMPEG_THREADS = int(os.getenv("TIMELINE_FFMPEG_THREADS", os.getenv("KENBURNS_FFMPEG_THREADS", "0")))


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _guess_ext_from_headers(resp: requests.Response) -> str:
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
    if not ctype:
        return ""
    ext = mimetypes.guess_extension(ctype) or ""
    return {".jpe": ".jpg"}.get(ext, ext)


def _hash_name(url: str, ext_hint: str = "") -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    ext = ext_hint or Path(urlparse(url).path).suffix or ".bin"
    return f"{h}{ext}"


def download_with_cache(url: str, timeout: int = 120) -> Path:
    local_name = _hash_name(url)
    local_path = CACHE_DIR / local_name
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    headers = {"User-Agent": "TimelineEditor/1.0", "Accept": "*/*"}
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        ext2 = _guess_ext_from_headers(r)
        if ext2 and Path(local_name).suffix.lower() != ext2.lower():
            local_name = _hash_name(url, ext2)
            local_path = CACHE_DIR / local_name
            if local_path.exists() and local_path.stat().st_size > 0:
                return local_path
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return local_path


def _resolve_src(src: str) -> Path:
    if _is_url(src):
        return download_with_cache(src)
    p = Path(src)
    if p.exists():
        return p
    raise FileNotFoundError(f"Overlay file not found: {src}")


def _detect_media_type(path: Path, fallback: str) -> str:
    ext = path.suffix.lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}:
        return "video"
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}:
        return "image"
    return fallback


def _ffmpeg_threads_args() -> list[str]:
    if FFMPEG_THREADS > 0:
        return ["-threads", str(FFMPEG_THREADS)]
    return ["-threads", "0"]


def _use_nvenc() -> bool:
    env = os.getenv("USE_NVENC", "1").strip().lower()
    return env not in ("0", "false", "no", "off")


def _split_encode() -> bool:
    env = os.getenv("TIMELINE_FFMPEG_SPLIT", os.getenv("KENBURNS_FFMPEG_SPLIT", "auto")).strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    return _use_nvenc()


def _get_best_encoder() -> tuple[str, str]:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if "h264_nvenc" in out:
            return "h264_nvenc", "nvenc"
        if "h264_amf" in out:
            return "h264_amf", "amf"
        if "h264_qsv" in out:
            return "h264_qsv", "qsv"
        if "h264_videotoolbox" in out:
            return "h264_videotoolbox", "videotoolbox"
    except Exception:
        pass
    return "libx264", "cpu"


def _parse_fps(fps_raw: Any) -> float:
    """Safe replacement for eval() on ffprobe r_frame_rate."""
    try:
        return float(Fraction(str(fps_raw)))
    except (ValueError, ZeroDivisionError):
        return 30.0


def _get_video_info(video_path: Path) -> tuple[float, float, int, int]:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not video_stream:
        raise ValueError("No video stream found")
    duration = float(data["format"]["duration"])
    fps = _parse_fps(video_stream.get("r_frame_rate", "30/1"))
    width = int(video_stream.get("width") or 1280)
    height = int(video_stream.get("height") or 720)
    return duration, fps, width, height


def _run_ffmpeg(cmd: list[str], label: str, timeout: int = 7200) -> None:
    cmd = cmd[:1] + _ffmpeg_threads_args() + cmd[1:]
    logger.info("FFmpeg [%s]: %s", label, " ".join(cmd[:12]) + ("..." if len(cmd) > 12 else ""))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        tail = (result.stderr or "")[-2000:]
        raise RuntimeError(f"FFmpeg failed ({label}): {tail}")


def _editor_pool_workers() -> int:
    env = os.getenv("EDITOR_POOL_WORKERS") or os.getenv("ENCODE_POOL_WORKERS")
    if env:
        return max(1, int(env))
    try:
        from worker_config import get_worker_limits

        return max(1, get_worker_limits().encode_pool_workers)
    except Exception:
        return max(1, min(8, (os.cpu_count() or 4)))


def _render_animated_overlay(args: tuple) -> tuple[int, str | None, bool, str | None]:
    """Worker: image → short mp4 clip with Ken Burns + zoom transitions +
    framing (fit/focus) + background (blur/color/gradient) baked in."""
    (idx, image_path, out_path, width, height, fps, duration,
     ken, tin, tout, fit, focus, bg, aspect_mismatch) = args
    ken = parse_ken_burns(ken)
    needs_zoom = "zoom" in (tin.get("type"), tout.get("type"))
    if not ken["enabled"] and not needs_zoom:
        return idx, None, False, None

    frames = max(1, int(round(duration * fps)))
    graph, needs_key = overlay_prerender_graph(
        width, height, fps, frames, ken, tin, tout,
        fit, focus, bg, duration, aspect_mismatch,
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-i", image_path,
        "-filter_complex", graph, "-map", "[vout]",
        "-t", f"{duration:.6f}",
        "-r", str(int(round(fps))),
        "-c:v", "libx264", "-preset", INTERMEDIATE_PRESET, "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        out_path,
    ]
    try:
        _run_ffmpeg(cmd, f"animated_overlay_{idx}")
        return idx, out_path, needs_key, None
    except Exception as exc:
        return idx, None, False, str(exc)


def _probe_image_size(path: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        s = next((x for x in json.loads(out)["streams"] if x.get("width")), None)
        if s:
            return int(s["width"]), int(s["height"])
    except Exception:
        pass
    return None


def _prep_overlay(
    ov: dict[str, Any],
    idx: int,
    video_duration: float,
    base_w: int,
    base_h: int,
    target_fps: float,
    work_dir: Path,
) -> dict[str, Any]:
    src = ov.get("url") or ov.get("path")
    src_path = _resolve_src(str(src))
    start = max(0.0, float(ov.get("start", 0)))
    end = min(video_duration, float(ov.get("end", start + 1.0)))
    if end <= start:
        raise ValueError(f"Overlay {idx}: invalid timing")

    typ = _detect_media_type(src_path, (ov.get("type") or "image").lower())
    ken = parse_ken_burns(ov.get("kenBurns") or ov.get("ken"))
    tin = parse_transition(ov.get("transIn") or ov.get("transitionIn"), typ)
    tout = parse_transition(ov.get("transOut") or ov.get("transitionOut"), typ)
    fit = parse_fit(ov.get("fit"))
    focus = parse_focus(ov.get("focus"))
    bg = parse_overlay_bg(ov.get("bg"))
    if typ == "video" or fit == "cover":
        # backgrounds only apply to letterboxed images (matches the preview)
        bg = {**bg, "type": "none"}
    dur = end - start

    item = {
        "path": src_path,
        "start": start,
        "end": end,
        "type": typ,
        "opacity": float(ov.get("opacity", 1.0)),
        "scale": float(ov["scale"]) if ov.get("scale") not in (None, "", False) else None,
        "ken": ken,
        "trans_in": tin,
        "trans_out": tout,
        "fit": fit,
        "focus": focus,
        "bg": bg,
        "framed": False,
        "pre_rendered": None,
        "idx": idx,
    }

    needs_prerender = typ == "image" and (
        ken["enabled"] or transition_needs_zoom_prerender(tin, tout, typ)
    )
    if needs_prerender:
        out_path = str(work_dir / f"ov_anim_{idx}_{uuid.uuid4().hex[:8]}.mp4")
        item["pre_render_path"] = out_path
        item["needs_anim_render"] = True
        # zoompan flattens alpha: when the letterbox must stay transparent
        # (contain fit, no background), pad with a key color and chroma-key
        # it back out in the main graph. Cover fit and bg fills are opaque
        # full-frame composites, so no keying is needed there.
        dims = _probe_image_size(src_path)
        if dims:
            src_ar, frame_ar = dims[0] / max(1, dims[1]), base_w / max(1, base_h)
            item["aspect_mismatch"] = abs(src_ar - frame_ar) > 0.01
        else:
            item["aspect_mismatch"] = True
        item["colorkey"] = False  # set from worker result
    else:
        item["needs_anim_render"] = False
        item["aspect_mismatch"] = False
        item["colorkey"] = False

    item["duration"] = dur
    item["target_fps"] = target_fps
    item["base_w"] = base_w
    item["base_h"] = base_h
    return item


def _parallel_prerender(items: list[dict[str, Any]]) -> None:
    jobs = []
    for item in items:
        if not item.get("needs_anim_render"):
            continue
        jobs.append((
            item["idx"],
            str(item["path"]),
            item["pre_render_path"],
            item["base_w"],
            item["base_h"],
            item["target_fps"],
            item["duration"],
            item["ken"],
            item["trans_in"],
            item["trans_out"],
            item["fit"],
            item["focus"],
            item["bg"],
            bool(item.get("aspect_mismatch")),
        ))

    if not jobs:
        return

    workers = min(_editor_pool_workers(), len(jobs))
    logger.info("Timeline: parallel overlay prep — %s overlays, %s workers", len(jobs), workers)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_render_animated_overlay, job): job[0] for job in jobs}
        errors = []
        for fut in as_completed(futures):
            idx, out_path, needs_key, err = fut.result()
            if err:
                errors.append(f"overlay {idx}: {err}")
                continue
            if out_path is None:
                continue
            for item in items:
                if item["idx"] == idx:
                    item["pre_rendered"] = Path(out_path)
                    item["type"] = "video"
                    item["path"] = Path(out_path)
                    item["framed"] = True   # frame-size clip; skip framing in main graph
                    item["colorkey"] = needs_key
                    break
        if errors:
            raise RuntimeError("Overlay pre-render failed: " + "; ".join(errors))


def _build_filter_graph(
    norm: list[dict[str, Any]],
    target_fps: float,
    gfx: dict[str, Any],
    video_duration: float,
) -> list[str]:
    graph: list[str] = []
    vprev = "v0"
    graph.append(f"[0:v]setsar=1,fps={target_fps}[{vprev}]")

    for i, ov in enumerate(norm):
        idx = i + 1
        start, end = ov["start"], ov["end"]
        dur = end - start
        ovlbl = f"ov{i}"
        outlbl = f"v{i + 1}"

        w, h = ov["base_w"], ov["base_h"]
        fit, focus, bg = ov["fit"], ov["focus"], ov["bg"]

        if ov.get("framed"):
            # Pre-rendered clip: already framed + composed at frame size.
            chain = [f"[{idx}:v]format=rgba"]
            if ov.get("colorkey"):
                chain.append(f"colorkey={PRERENDER_KEY_COLOR}:0.08:0.05")
        elif fit == "cover":
            # Fill the frame, cropping at the focus point
            chain = [f"[{idx}:v]format=rgba"] + fg_frame_filters(w, h, "cover", focus)
        elif bg["type"] == "none":
            # Letterbox with real transparency (no zoompan → alpha survives)
            chain = [f"[{idx}:v]format=rgba"] + fg_frame_filters(w, h, "contain", focus)
        elif bg["type"] == "color":
            chain = (
                [f"[{idx}:v]format=rgba"]
                + fg_frame_filters(w, h, "contain", focus)
                + [f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={bg['color']}", "format=rgba"]
            )
        elif bg["type"] == "blur":
            sigma = bg_blur_sigma(bg["blur"], w)
            graph += [
                f"[{idx}:v]format=rgba,split[bga{i}][fga{i}]",
                f"[bga{i}]" + ",".join(cover_blur_filters(w, h, focus, sigma)) + f"[bgb{i}]",
                f"[fga{i}]" + ",".join(fg_frame_filters(w, h, "contain", focus)) + f"[fgc{i}]",
                f"[bgb{i}][fgc{i}]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1[frm{i}]",
            ]
            chain = [f"[frm{i}]format=rgba"]
        else:  # gradient
            graph += [
                gradient_source(w, h, bg["color"], bg["color2"], bg["angle"], 0.2) + f"[bgg{i}]",
                f"[{idx}:v]format=rgba," + ",".join(fg_frame_filters(w, h, "contain", focus)) + f"[fgc{i}]",
                f"[bgg{i}][fgc{i}]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1[frm{i}]",
            ]
            chain = [f"[frm{i}]format=rgba"]
        if ov.get("scale") and ov["scale"] > 0 and ov["scale"] != 1:
            chain.append(f"scale=iw*{ov['scale']}:ih*{ov['scale']}:flags=fast_bilinear")

        opacity = max(0.0, min(1.0, float(ov.get("opacity", 1.0))))
        if opacity < 1.0:
            chain.append(f"colorchannelmixer=aa={opacity:.3f}")

        # ── Shift stream PTS onto the main timeline: clip content must live
        #    inside [start, end], not be consumed at t=0 and freeze. ──
        if ov["type"] == "video":
            chain += [
                f"trim=duration={dur:.6f}",
                "setpts=PTS-STARTPTS",
                f"fps={target_fps}",
                f"setpts=PTS+{start:.6f}/TB",
            ]
        else:
            chain += [
                "loop=loop=-1:size=1:start=0",
                f"trim=duration={dur:.6f}",
                f"setpts=PTS-STARTPTS+{start:.6f}/TB",
            ]

        # ── Transition opacity ramps (fade / zoom) on the alpha channel ──
        chain += overlay_alpha_fade_filters(start, end, ov["trans_in"], ov["trans_out"])

        graph.append(",".join(chain) + f"[{ovlbl}]")

        # ── Composite: centered + slide offsets, gated to [start, end] ──
        x_expr, y_expr = overlay_xy_exprs(start, end, ov["trans_in"], ov["trans_out"])
        enable = f"between(t\\,{start:.3f}\\,{end:.3f})"
        graph.append(
            f"[{vprev}][{ovlbl}]overlay=x='{x_expr}':y='{y_expr}'"
            f":enable='{enable}':eval=frame[{outlbl}]"
        )
        vprev = outlbl

    # ── Global FX: color chain → vignette → noise → fades (preview order) ──
    cf = color_fx_filters(gfx)
    if cf:
        graph.append(f"[{vprev}]" + ",".join(cf) + "[v_col]")
        vprev = "v_col"

    vignette_cfg = gfx["vignette"]
    if vignette_cfg.get("enabled"):
        opacity = vignette_cfg.get("opacity", 0.0)
        graph += [
            f"[{vprev}]split[clean][vigsrc]",
            "[vigsrc]vignette=angle=PI/4[vv]",
            f"[clean][vv]blend=all_mode=multiply:all_opacity={opacity:.4f}[v_vig]",
        ]
        vprev = "v_vig"

    n = noise_ffmpeg_level(gfx["noise"])
    if n > 0:
        graph.append(f"[{vprev}]noise=alls={n}:allf=t+u[v_noi]")
        vprev = "v_noi"

    ff = fade_filters(video_duration, gfx["fade_in"], gfx["fade_out"])
    if ff:
        graph.append(f"[{vprev}]" + ",".join(ff) + "[v_fad]")
        vprev = "v_fad"

    graph.append(f"[{vprev}]format=yuv420p[vout]")
    return graph


def _encode_pass(
    input_args: list[str],
    filter_graph: str,
    out_path: Path,
    *,
    use_hw: bool,
    audio_map: str = "0:a?",
) -> None:
    encoder, hw_type = _get_best_encoder()
    map_args = ["-map", "[vout]", "-map", audio_map]
    audio_args = ["-c:a", "aac", "-b:a", "192k"]

    if use_hw and hw_type == "nvenc":
        vcodec = [
            "-c:v", "h264_nvenc",
            "-preset", NVENC_PRESET,
            "-rc", "vbr", "-cq", str(NVENC_CQ),
            "-b:v", "0",
        ]
    elif use_hw and hw_type == "amf":
        vcodec = ["-c:v", "h264_amf", "-quality", "speed", "-rc", "vbr_peak"]
    elif use_hw and hw_type == "qsv":
        vcodec = ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", str(NVENC_CQ + 5)]
    else:
        vcodec = [
            "-c:v", "libx264",
            "-preset", INTERMEDIATE_PRESET if not use_hw else "veryfast",
            "-crf", str(OUTPUT_CRF),
            "-tune", "fastdecode",
        ]

    cmd = input_args + [
        "-filter_complex", filter_graph,
        *map_args,
        *audio_args,
        *vcodec,
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    _run_ffmpeg(cmd, "timeline_encode")


def apply_edits(
    video_path: str,
    overlays: list[dict[str, Any]],
    *,
    fx: dict[str, Any] | None = None,
    vignette: dict[str, Any] | float | None = None,  # legacy
    noise: float = 0.0,  # legacy
    ken_burns_default: dict[str, Any] | None = None,
    output_name_hint: str = "output.mp4",
    target_fps: float | None = None,
    public_base_url: str | None = None,
) -> tuple[Path, str | None]:
    base = Path(video_path)
    if not base.exists():
        raise FileNotFoundError(f"Base video not found: {base}")

    video_duration, original_fps, base_w, base_h = _get_video_info(base)
    fps = min(60.0, target_fps or original_fps or float(OUTPUT_FPS))

    gfx = parse_global_fx(fx, legacy_vignette=vignette, legacy_noise=noise)
    work_dir = WORK_DIR / f"job_{uuid.uuid4().hex[:10]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    norm: list[dict[str, Any]] = []
    for i, ov in enumerate(overlays or []):
        if not ov:
            continue
        merged = dict(ov)
        if ken_burns_default and not merged.get("kenBurns") and not merged.get("ken"):
            merged["kenBurns"] = ken_burns_default
        try:
            norm.append(_prep_overlay(merged, i, video_duration, base_w, base_h, fps, work_dir))
        except Exception as exc:
            logger.warning("Skipping overlay %s: %s", i, exc)

    _parallel_prerender(norm)

    args = [
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-fflags", "+genpts+discardcorrupt",
        "-i", str(base),
    ]
    for ov in norm:
        args += ["-i", str(ov["path"])]

    graph = _build_filter_graph(norm, fps, gfx, video_duration)

    out_name = f"edited_{Path(output_name_hint).stem}_{_ts()}.mp4"
    out_path = OUTPUT_FOLDER / out_name
    intermediate = work_dir / "intermediate.mp4"

    _encode_pass(args, ";".join(graph), intermediate, use_hw=False)

    if _split_encode() and _use_nvenc():
        logger.info("Timeline pass 2/2: NVENC finalize")
        nvenc_out = work_dir / "nvenc.mp4"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(intermediate),
            "-c:v", "h264_nvenc", "-preset", NVENC_PRESET,
            "-rc", "vbr", "-cq", str(NVENC_CQ), "-b:v", "0",
            "-c:a", "copy", "-movflags", "+faststart",
            str(nvenc_out),
        ]
        _run_ffmpeg(cmd, "timeline_nvenc")
        current = nvenc_out
    else:
        current = intermediate

    if current != out_path:
        shutil.copyfile(current, out_path)

    public_url = None
    if public_base_url:
        public_url = public_base_url.rstrip("/") + "/" + out_name

    logger.info("Timeline edit complete: %s", out_path)
    return out_path, public_url