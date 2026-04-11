# editor.py - Hybrid approach: GPU vignette + FFmpeg overlays + Captions
import os
import hashlib
import mimetypes
import requests
import subprocess
import json
import tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import torch
import threading
from captions import generate_captions, save_captions_srt, write_ass_file


# ---------------- Dirs / Exports ----------------
OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", "outputs"))
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.getenv("OVERLAY_CACHE", "overlay_cache"))
CACHE.mkdir(parents=True, exist_ok=True)

# ---------------- Tunables ----------------
RENDER_MODE = os.getenv("RENDER_MODE", "fast").lower()  # fast | hq
NVENC_CQ = int(os.getenv("NVENC_CQ", "25"))
OUT_FPS_CAP = float(os.getenv("OUT_FPS_CAP", "60"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Using device: {DEVICE}")

def escape_ffmpeg(s: str) -> str:
    return (
        s.replace("'", "\\'")
         .replace(":", "\\:")
         .replace(",", "\\,")
    )


def build_karaoke_filters(words, fontsize, highlight_color, highlight_opacity, x_expr, y_expr):
    """
    Build the FULL karaoke filter script:
    - Base full text (white outline)
    - Per-word highlights (red background)
    - Per-word white outlined text on top
    """

    filters = []

    # ---------------------------
    # BASE FULL SENTENCE (white)
    # ---------------------------

    full_sentence = " ".join([w[2] for w in words])
    full_escaped = escape_ffmpeg(full_sentence)

    filters.append(
        f"drawtext=text='{full_escaped}':"
        f"fontsize={fontsize}:"
        f"fontcolor=white:"
        f"borderw=4:bordercolor=black:"
        f"x={x_expr}:y={y_expr}"
    )

    # ---------------------------
    # WORD-BY-WORD HIGHLIGHT LAYER
    # ---------------------------

    x_offset = 0  # rough horizontal offset accumulator

    for (start, end, word) in words:
        escaped_word = escape_ffmpeg(word)

        approx_width = len(word) * (fontsize * 0.55)
        height = int(fontsize * 1.4)

        filters.append(
            f"drawbox=enable='between(t,{start},{end})':"
            f"x={x_expr}+{int(x_offset)}:"
            f"y={y_expr}-10:"
            f"w={int(approx_width)+20}:h={height}:"
            f"color={highlight_color}@{highlight_opacity}:t=fill"
        )

        filters.append(
            f"drawtext=enable='between(t,{start},{end})':"
            f"text='{escaped_word}':"
            f"fontsize={fontsize}:"
            f"fontcolor=white:borderw=4:bordercolor=black:"
            f"x={x_expr}+{int(x_offset)}:"
            f"y={y_expr}"
        )

        x_offset += approx_width + 18

    return filters


def burn_ass_subtitles(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
):
    """
    Hard-burn ASS subtitles into the video using FFmpeg.
    This replaces the giant drawtext filter logic completely.
    """
    print(f"🎬 Burning ASS subtitles into video...")
    ass_arg = ass_path.as_posix()  # safer for ffmpeg

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"ass='{ass_arg}'",
        "-c:v", "h264_nvenc",
        "-preset", ("p1" if RENDER_MODE == "fast" else "p4"),
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", str(NVENC_CQ),
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FFmpeg (ASS) error:\n{result.stderr}")
        raise RuntimeError("FFmpeg ASS subtitle burn failed")
    print(f"✅ Subtitles burned: {output_path}")

def burn_filter_script(video_path, filter_file, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex_script", str(filter_file),
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "vbr", "-cq", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path)
    ]

    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        raise RuntimeError("❌ FFmpeg failed while burning karaoke subtitles.")

    print(f"✅ Karaoke subtitles burned → {output_path}")


# ---------------- Caption Application Function ----------------
from typing import List, Tuple
from pathlib import Path
import subprocess
import json
import os


def escape_text_for_ffmpeg(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter"""
    return (
        text.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
            .replace("%", "\\%")
    )


# ---------------------------------------------------------------------
# WORD WRAPPING LOGIC — WIDE, LANDSCAPE-FRIENDLY, NO TEXT LOSS
# ---------------------------------------------------------------------
def wrap_caption_text(text: str, max_chars_per_line: int, max_lines: int) -> str:
    """
    Safe, flexible word wrapper for captions.
    - Never cuts words.
    - Never removes content.
    - If text is too long for max_lines, all remaining text stays on last line.
    """
    words = text.split()
    if not words:
        return ""

    lines = [""]
    line_index = 0

    for w in words:
        current = lines[line_index]
        new_length = len(current) + (1 if current else 0) + len(w)

        if new_length <= max_chars_per_line:
            # put inside current line
            lines[line_index] = w if not current else current + " " + w
        else:
            # need new line
            if line_index + 1 < max_lines:
                lines.append(w)
                line_index += 1
            else:
                # overflow → append into last line (TikTok style)
                lines[line_index] = current + " " + w

    return "\n".join(lines)



# ---------------------------------------------------------------------
# MAIN FUNCTION — APPLY CAPTIONS USING FFmpeg filter_complex_script
# ---------------------------------------------------------------------
def apply_captions_to_video(
    video_path: Path,
    captions: List[Tuple[float, float, str]],
    output_path: Path,
    caption_style: dict
):
    """
    Apply burned-in subtitles using FFmpeg + filter script.
    Works on Windows, Linux, Mac. No long command errors.
    """

    print(f"🎬 Applying {len(captions)} captions to video...")

    # -------------------------
    # Extract video width (for auto line length)
    # -------------------------
    try:
        info = _get_video_info(video_path)
        width = int(info.get("width", 1920) or 1920)
    except:
        width = 1920

    # Auto-wide line width (landscape)
    auto_chars = max(28, int(width / 46))  # 1920 → ~42 chars
    max_chars_per_line = int(caption_style.get("maxCharsPerLine", auto_chars))
    max_lines = int(caption_style.get("maxLines", 2))

    # -------------------------
    # Style settings from React
    # -------------------------
    fontsize = int(caption_style.get("fontsize", 48))
    fontcolor = caption_style.get("fontcolor", "white")

    box_enabled = int(caption_style.get("box", 1))
    boxcolor = caption_style.get("boxcolor", "black@0.6")
    boxborderw = int(caption_style.get("boxborderw", 5))

    borderw = int(caption_style.get("borderWidth", 3))
    bordercolor = caption_style.get("borderColor", "black")

    x_pos = caption_style.get("x", "(w-text_w)/2")
    y_pos = caption_style.get("y", "h-th-50")

    # -------------------------
    # Build FFmpeg drawtext filters
    # -------------------------
    draw_filters = []

    for start, end, text in captions:
        text = (text or "").strip()
        if not text:
            continue

        wrapped = wrap_caption_text(text, max_chars_per_line, max_lines)
        escaped = escape_text_for_ffmpeg(wrapped)

        fx = (
            f"drawtext=text='{escaped}':"
            f"fontsize={fontsize}:"
            f"fontcolor={fontcolor}:"
            f"borderw={borderw}:"
            f"bordercolor={bordercolor}:"
            f"box={box_enabled}:"
            f"boxcolor={boxcolor}:"
            f"boxborderw={boxborderw}:"
            f"x={x_pos}:"
            f"y={y_pos}:"
            f"line_spacing=10:"
            f"enable='between(t,{start:.3f},{end:.3f})'"
        )

        draw_filters.append(fx)

    if not draw_filters:
        print("⚠️ No captions found → copying video unchanged")
        import shutil
        shutil.copy2(video_path, output_path)
        return

    # -------------------------
    # WRITE FILTER SCRIPT FILE
    # -------------------------
    filter_file = output_path.with_suffix(".filters.txt")

    with open(filter_file, "w", encoding="utf-8") as f:
        # Each drawtext on its own line
        for fx in draw_filters:
            f.write(f"{fx}\n")

    print(f"📝 FFmpeg filter file written: {filter_file}")

    # -------------------------
    # RUN FFmpeg using filter_complex_script
    # -------------------------
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex_script", str(filter_file),  # <-- FIX
        "-c:v", "h264_nvenc",
        "-preset", ("p1" if RENDER_MODE == "fast" else "p4"),
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", str(NVENC_CQ),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    print("🎬 Running FFmpeg for captions...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ FFmpeg Error:", result.stderr)
        raise RuntimeError("FFmpeg caption encoding failed.")

    print(f"✅ Captions applied successfully → {output_path}")

# ---------------- GPU Vignette ----------------
def make_vignette_mask_gpu(w: int, h: int, vignette_config: dict) -> Optional[torch.Tensor]:
    """
    Create vignette mask on GPU using config from frontend.
    
    Args:
        w, h: Video dimensions
        vignette_config: {
            'enabled': bool,
            'strength': float,
            'centerX': float (0-1),
            'centerY': float (0-1),
            'innerRadius': float (0-1),
            'outerRadius': float (0-1),
            'color': [r, g, b],
            'maxAlpha': float
        }
    """
    if not vignette_config.get('enabled', False):
        return None
    
    strength = float(vignette_config.get('strength', 0))
    if strength <= 0:
        return None
    
    center_x = float(vignette_config.get('centerX', 0.5))
    center_y = float(vignette_config.get('centerY', 0.5))
    inner = float(vignette_config.get('innerRadius', 0.4))
    max_alpha = float(vignette_config.get('maxAlpha', 0.99 * strength))
    
    # Create coordinate grids on GPU
    y = torch.linspace(0, h - 1, h, device=DEVICE)
    x = torch.linspace(0, w - 1, w, device=DEVICE)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Center point
    cx, cy = w * center_x, h * center_y
    
    # Radial distance
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_radius = torch.sqrt(torch.tensor(cx ** 2 + cy ** 2, device=DEVICE))
    rN = r / max_radius
    
    # Linear ramp from inner to outer
    ramp = (rN - inner) / (1.0 - inner)
    ramp = torch.clamp(ramp, 0, 1)
    
    # Alpha channel
    alpha = max_alpha * ramp
    multiply_factor = 1.0 - alpha
    
    # Shape: (h, w, 3) - RGB multiply mask
    mask = torch.stack([multiply_factor, multiply_factor, multiply_factor], dim=2)
    return mask


def apply_vignette_gpu_batch(video_path: Path, vignette_config: dict, output_path: Path, batch_size: int = 120):
    """Apply GPU vignette to video in batches"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Applying GPU vignette: {w}x{h} @ {fps}fps, {total_frames} frames")
    
    # Create vignette mask on GPU
    mask_gpu = make_vignette_mask_gpu(w, h, vignette_config)
    if mask_gpu is None:
        print("⚠️  Vignette disabled, copying video")
        cap.release()
        # Just copy the file
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path
    
    print(f"✓ Vignette mask created on GPU")
    
    # Setup FFmpeg pipe for writing
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'h264_nvenc',
        '-preset', ('p1' if RENDER_MODE == 'fast' else 'p4'),
        '-tune', 'hq',
        '-rc', 'vbr',
        '-cq', str(NVENC_CQ),
        '-b:v', '0',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    
    proc = subprocess.Popen(
        ffmpeg_cmd, 
        stdin=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    
    # Consume stderr in background thread to prevent blocking
    def consume_stderr():
        """Read and discard FFmpeg's stderr output to prevent pipe blocking"""
        try:
            for line in proc.stderr:
                pass  # Discard or log if needed for debugging
        except Exception:
            pass
    
    stderr_thread = threading.Thread(target=consume_stderr, daemon=True)
    stderr_thread.start()
    
    processed_frames = 0
    batch_frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process remaining batch
                if batch_frames:
                    process_batch_gpu(batch_frames, mask_gpu, proc)
                    processed_frames += len(batch_frames)
                break
            
            batch_frames.append(frame)
            
            if len(batch_frames) >= batch_size:
                process_batch_gpu(batch_frames, mask_gpu, proc)
                processed_frames += len(batch_frames)
                batch_frames = []
                
                if processed_frames % 300 == 0:
                    progress = (processed_frames / total_frames) * 100
                    print(f"⏳ Vignette progress: {processed_frames}/{total_frames} ({progress:.1f}%)")
    
    except Exception as e:
        print(f"❌ Error during vignette processing: {e}")
        raise
    
    finally:
        cap.release()
        
        # Critical: Close stdin to signal FFmpeg we're done
        try:
            proc.stdin.flush()
            proc.stdin.close()
        except Exception as e:
            print(f"⚠️  Error closing stdin: {e}")
        
        # Wait for FFmpeg to finish encoding
        print("⏳ Waiting for FFmpeg to finish encoding...")
        return_code = proc.wait(timeout=60)  # 60 second timeout
        
        # Join stderr thread
        stderr_thread.join(timeout=5)
        
        if return_code != 0:
            raise RuntimeError(f"FFmpeg encoding failed with code {return_code}")
    
    print(f"✅ GPU vignette applied: {output_path}")
    return output_path

def process_batch_gpu(batch_frames: list, mask_gpu: torch.Tensor, proc):
    """Process a batch of frames with GPU vignette"""
    # Stack to numpy array
    batch_np = np.stack(batch_frames, axis=0).astype(np.float32) / 255.0
    
    # Move to GPU
    batch_gpu = torch.from_numpy(batch_np).to(DEVICE)
    
    # Apply vignette (element-wise multiply)
    batch_gpu = batch_gpu * mask_gpu.unsqueeze(0)
    
    # Download from GPU
    batch_cpu = (batch_gpu.cpu().numpy() * 255).astype(np.uint8)
    
    # Write frames to FFmpeg
    for frame in batch_cpu:
        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            raise RuntimeError("FFmpeg pipe broken")


# ---------------- Helpers ----------------
def _normalize_url(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith("//"):
        s = "https:" + s
    try:
        u = urlparse(s)
        if u.scheme in ("http", "https") and u.netloc:
            if u.scheme == "http" and "pexels.com" in u.netloc.lower():
                u = u._replace(scheme="https")
            return urlunparse(u)
    except:
        pass
    return s


def _is_url(s: str) -> bool:
    try:
        u = urlparse(_normalize_url(s))
        return u.scheme in ("http", "https") and bool(u.netloc)
    except:
        return False


def _ext_from_resp(resp: requests.Response) -> str:
    ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
    ext = mimetypes.guess_extension(ct) or ""
    return {".jpe": ".jpg"}.get(ext, ext) or ".bin"


def _hash_name(url: str, ext_hint: str = "") -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    ext = ext_hint or Path(urlparse(url).path).suffix or ".bin"
    return f"{h}{ext}"


EDIT_FOLDER = Path(os.getenv("EDIT_FOLDER", "edit_uploads"))
EDIT_FOLDER.mkdir(parents=True, exist_ok=True)

def _download(src: str) -> Path:
    """Download media to cache if URL, otherwise return local path"""
    if not _is_url(src):
        # Check if it's a filename in local folders
        p = Path(src)
        
        # If it's not an absolute path, check common locations
        if not p.is_absolute():
            # Try EDIT_FOLDER first (where frontend uploads sit)
            edit_path = EDIT_FOLDER / p.name
            if edit_path.exists():
                return edit_path
            
            # Also try OUTPUT_FOLDER (where base videos are stored)
            output_path = OUTPUT_FOLDER / p.name
            if output_path.exists():
                return output_path
            
            # Try cache folder
            cache_path = CACHE / p.name
            if cache_path.exists():
                return cache_path
        
        # If absolute path or not found in common locations, check as-is
        if not p.exists():
            raise FileNotFoundError(f"Overlay not found: {src}")
        return p
    
    # It's a URL - download to cache
    url = _normalize_url(src)
    local_name = _hash_name(url)
    p = CACHE / local_name
    
    if p.exists() and p.stat().st_size > 0:
        return p
    
    headers = {
        "User-Agent": "VideoEditor/1.0",
        "Accept": "*/*"
    }
    
    if "pexels.com" in (urlparse(url).netloc or "").lower():
        headers["Referer"] = "https://www.pexels.com/"
    
    print(f"📥 Downloading: {url}")
    with requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        ext2 = _ext_from_resp(r)
        if ext2 and Path(local_name).suffix.lower() != ext2.lower():
            local_name = _hash_name(url, ext2)
            p = CACHE / local_name
            if p.exists() and p.stat().st_size > 0:
                return p
        
        with open(p, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    
    print(f"✓ Downloaded: {p.name}")
    return p

def _safe_div(num: str) -> float:
    """Parse rational like '30000/1001' into float"""
    if not num or num == '0/0':
        return 0.0
    if '/' in num:
        n, d = num.split('/', 1)
        try:
            n = float(n)
            d = float(d) if float(d) != 0 else 1.0
            return n / d
        except:
            return 0.0
    try:
        return float(num)
    except:
        return 0.0


def _get_video_info(video_path: Path) -> Dict:
    """Get video metadata using FFprobe"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout or "{}")
    
    video_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break
    
    if not video_stream:
        raise ValueError("No video stream found")
    
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    duration = float(info.get("format", {}).get("duration", 0.0))
    
    fps = _safe_div(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1")
    if fps <= 0:
        fps = 30.0
    
    if OUT_FPS_CAP and fps > OUT_FPS_CAP:
        fps = OUT_FPS_CAP
    
    return {"width": width, "height": height, "duration": duration, "fps": fps}


def _ken_zoom_expr(mode: str, strength: float, speed: float, seg_len: float, base_fps: float, base_scale: float) -> str:
    """Build FFmpeg zoompan expression for Ken Burns effect"""
    seg_len = max(0.001, float(seg_len))
    base_fps = max(1.0, float(base_fps))
    speed = max(0.001, float(speed))
    base_scale = max(0.0001, float(base_scale))
    
    t = f"(on/{base_fps})"
    
    if mode == "in":
        z_core = f"(1+({strength})*(({t})/{seg_len}))"
    elif mode == "out":
        z_core = f"(1+({strength})*(1-(({t})/{seg_len})))"
    else:  # auto
        z_core = f"(1+({strength})*0.5*(1-cos(2*PI*({t})/{speed})))"
    
    return f"({base_scale})*({z_core})"


def add_audio_track(original_video: Path, processed_video: Path, output_path: Path):
    """Copy audio from original to processed video"""
    cmd = [
        'ffmpeg', '-y',
        '-i', str(processed_video),
        '-i', str(original_video),
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest',
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"⚠️  Warning: Could not add audio (video may not have audio track)")


# ---------------- Main Function ----------------
def apply_edits(
    video_path: str,
    overlays: List[Dict],
    replace_audio: bool = False,
    vignette: dict = None,
    noise: float = 0.0,
    ken_burns_default: Optional[Dict] = None,
    output_name_hint: str = "output.mp4",
    target_fps: Optional[float] = None,
    captions: Optional[dict] = None,
    progress_callback=None
) -> Tuple[Path, Optional[str]]:
    """
    Apply edits to video:
    - Overlays (if any)
    - Captions (burned-in before vignette)
    - GPU Vignette (preserves captions)
    - Audio restore (after vignette)
    """
    base_path = Path(video_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base not found: {base_path}")

    base_info = _get_video_info(base_path)
    base_width = base_info["width"]
    base_height = base_info["height"]
    base_fps = base_info["fps"]

    print(f"📹 Processing: {base_width}x{base_height} @ {base_fps}fps")

    vignette_config = vignette if isinstance(vignette, dict) else {'enabled': False}
    captions_config = captions if isinstance(captions, dict) else {'enabled': False}

    has_overlays = overlays and len(overlays) > 0
    has_vignette = vignette_config.get('enabled', False)
    has_captions = captions_config.get('enabled', False)

    if not has_overlays and not has_vignette and not has_captions:
        raise ValueError("No effects to apply (no overlays, vignette disabled, and captions disabled)")

    out_name = f"processed_{Path(output_name_hint).stem}.mp4"
    out_path = OUTPUT_FOLDER / out_name

    # ---------------- Step 1: Overlays ----------------
    if has_overlays:
        print(f"✨ Step 1: Applying {len(overlays)} overlay(s)...")
        temp_overlays = OUTPUT_FOLDER / f"temp_overlays_{Path(output_name_hint).stem}.mp4"
        apply_overlays_ffmpeg(base_path, overlays, base_info, temp_overlays, ken_burns_default, replace_audio)
        video_after_overlays = temp_overlays
    else:
        print("⏭️  Step 1: No overlays to apply, skipping...")
        video_after_overlays = base_path

    # ---------------- Step 2: Captions ----------------
    if has_captions:
        print("✨ Step 2: Generating and applying ASS subtitles...")
        model_size = captions_config.get("model", "base")
        max_chars = int(captions_config.get("style", {}).get("maxCharsPerLine", 42))
        max_lines = int(captions_config.get("style", {}).get("maxLines", 2))

        # 2.1 Transcribe + segment (TS-style chunking)
        caption_list = generate_captions(
            video_after_overlays,
            model_size=model_size,
            max_chars_per_line=max_chars,
            max_lines=max_lines,
        )

        if not caption_list:
            print("⚠️ No captions generated (video may have no audio).")
            video_after_captions = video_after_overlays
        else:
            # 2.2 Save optional SRT
            if captions_config.get("saveSrt", False):
                srt_path = OUTPUT_FOLDER / f"{Path(output_name_hint).stem}.srt"
                save_captions_srt(caption_list, srt_path)
                print(f"💾 SRT file saved: {srt_path}")

            # 2.3 Write ASS with nice styling (bottom center, outline, safe margins)
            ass_path = OUTPUT_FOLDER / f"{Path(output_name_hint).stem}.ass"
            write_ass_file(
                caption_list,
                ass_path,
                playres_x=base_width,
                playres_y=base_height,
                fontsize=int(captions_config.get("style", {}).get("fontsize", 56)),
                bottom_margin=80,
            )

            temp_captions = OUTPUT_FOLDER / f"temp_captions_{Path(output_name_hint).stem}.mp4"
            burn_ass_subtitles(video_after_overlays, ass_path, temp_captions)
            video_after_captions = temp_captions
    else:
        print("⏭️  Step 2: Captions disabled, skipping...")
        video_after_captions = video_after_overlays

    # ---------------- Step 3: Vignette ----------------
    if has_vignette:
        print(f"✨ Step 3: Applying GPU vignette (preserving captions)...")
        temp_vignette = OUTPUT_FOLDER / f"temp_vignette_{Path(output_name_hint).stem}.mp4"

        # ✅ FIX: ensure vignette uses captioned file
        apply_vignette_gpu_batch(video_after_captions, vignette_config, temp_vignette)

        video_after_vignette = temp_vignette
    else:
        print("⏭️  Step 3: Vignette disabled, skipping...")
        video_after_vignette = video_after_captions

    # ---------------- Step 4: Audio Restore ----------------
    if not replace_audio and has_vignette:
        print("🔊 Step 4: Adding audio back after vignette...")
        print(f"   📹 Video source (with vignette, no audio): {video_after_vignette}")
        print(f"   🎵 Audio source: {base_path}")
        print(f"   💾 Final output: {out_path}")

        add_audio_track(base_path, video_after_vignette, out_path)

        if out_path.exists():
            print(f"   ✅ Final video created: {out_path.stat().st_size:,} bytes")

        # Cleanup
        for temp in [video_after_overlays, video_after_captions, video_after_vignette]:
            if temp != base_path and temp.exists():
                temp.unlink()
    else:
        print("⏭️  Step 4: No vignette audio restore needed...")
        import shutil
        shutil.move(str(video_after_vignette), str(out_path))

    # ---------------- Step 5: Public URL ----------------
    public_url = None
    if PUBLIC_BASE_URL:
        public_url = PUBLIC_BASE_URL.rstrip("/") + "/" + out_name

    print(f"✅ Processing complete: {out_path}")
    print(f"   📁 Location: {out_path.absolute()}")
    if out_path.exists():
        print(f"   📊 Size: {out_path.stat().st_size:,} bytes")

    return out_path, public_url

def apply_overlays_ffmpeg(
    base_path: Path,
    overlays: List[Dict],
    base_info: Dict,
    output_path: Path,
    ken_burns_default: Optional[Dict],
    replace_audio: bool
):
    """Apply overlays with Ken Burns using FFmpeg"""
    base_width = base_info["width"]
    base_height = base_info["height"]
    base_fps = base_info["fps"]
    
    # Download all overlays
    local_overlays = []
    for ov in overlays:
        src = ov.get("url") or ov.get("path")
        if not src:
            continue
        
        p = _download(src)
        
        start = float(ov.get("start", 0.0))
        end = float(ov.get("end", start))
        if end <= start:
            continue
        
        otype = str(ov.get("type", "image")).lower()
        x = int(ov.get("x", 0))
        y = int(ov.get("y", 0))
        opacity = float(ov.get("opacity", 1.0))
        pre_scale = ov.get("scale")
        pre_scale = float(pre_scale) if pre_scale not in (None, "", False) else 1.0
        
        # Ken Burns config
        ken = ov.get("kenBurns")
        if ken is None and ken_burns_default:
            ken = ken_burns_default
        
        if isinstance(ken, dict) and ken.get("enabled") is not False:
            kmode = ken.get("mode", "auto")
            kstrength = float(ken.get("strength", 0.2))
            kspeed = float(ken.get("speed", 8))
            ken_clean = {"mode": kmode, "strength": kstrength, "speed": kspeed}
        else:
            ken_clean = None
        
        local_overlays.append({
            "src": p,
            "start": start,
            "end": end,
            "type": ("video" if p.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm'] or otype == "video" else "image"),
            "x": x,
            "y": y,
            "opacity": max(0.0, min(1.0, opacity)),
            "pre_scale": max(0.0001, float(pre_scale)),
            "ken": ken_clean
        })
    
    if not local_overlays:
        print("⚠️  No valid overlays to apply")
        return
    
    print(f"✓ Downloaded {len(local_overlays)} overlay(s)")
    
    # Build FFmpeg inputs
    ff_inputs = ["-i", str(base_path)]
    for ov in local_overlays:
        ff_inputs.extend(["-i", str(ov["src"])])
    
    # Build filter chain
    filter_parts = []
    last_label = "0:v"
    
    for i, ov in enumerate(local_overlays, start=1):
        duration = ov["end"] - ov["start"]
        
        # Scale to cover canvas
        base_fit = (
            f"scale={base_width}:{base_height}:force_original_aspect_ratio=increase,"
            f"crop={base_width}:{base_height}:(iw-ow)/2:(ih-oh)/2,"
            f"setsar=1"
        )
        
        # Opacity
        alpha_pipe = ""
        if ov["opacity"] < 1.0:
            alpha_pipe = f",format=rgba,colorchannelmixer=aa={ov['opacity']:.6f}"
        
        # Ken Burns zoom
        ken = ov["ken"]
        if ken:
            zexpr = _ken_zoom_expr(
                ken.get("mode", "auto"),
                float(ken.get("strength", 0.2)),
                float(ken.get("speed", 8)),
                duration,
                base_fps,
                ov["pre_scale"]
            )
            kp = f",zoompan=zoom='{zexpr}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d=1:s={base_width}x{base_height}:fps={int(round(base_fps))}"
        else:
            zexpr = f"{ov['pre_scale']:.6f}"
            kp = f",zoompan=zoom='{zexpr}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d=1:s={base_width}x{base_height}:fps={int(round(base_fps))}"
        
        # Build filter for this overlay
        if ov["type"] == "image":
            filter_parts.append(
                f"[{i}:v]{base_fit},loop=loop=-1:size=1:start=0{alpha_pipe}{kp},"
                f"setpts=PTS-STARTPTS+{ov['start']}/TB,trim=duration={duration}[ov{i}];"
            )
        else:
            filter_parts.append(
                f"[{i}:v]{base_fit}{alpha_pipe}{kp},setpts=PTS-STARTPTS+{ov['start']}/TB[ov{i}];")
        
        # Overlay on base
        enable_expr = f"between(t,{ov['start']},{ov['end']})"
        out_label = f"tmp{i}"
        pos_x = f"(W-w)/2+{ov['x']}"
        pos_y = f"(H-h)/2+{ov['y']}"
        
        filter_parts.append(
            f"[{last_label}][ov{i}] overlay={pos_x}:{pos_y}:enable='{enable_expr}' [{out_label}];"
        )
        last_label = out_label
    
    # Final output
    filter_parts.append(f"[{last_label}]format=yuv420p[final]")
    filtergraph = " ".join(filter_parts)
    
    # Encode with NVENC
    nvenc_params = [
        "-map", "[final]",
        "-c:v", "h264_nvenc",
        "-preset", ("p1" if RENDER_MODE == "fast" else "p4"),
        "-rc", "vbr",
        "-cq", str(NVENC_CQ),
        "-b:v", "0",
        "-spatial_aq", "1",
        "-temporal_aq", "1",
        "-movflags", "+faststart"
    ]
    
    if not replace_audio:
        nvenc_params.extend(["-map", "0:a?", "-c:a", "aac", "-b:a", "192k"])
    
    cmd = (
        ["ffmpeg", "-y", "-hwaccel", "cuda"]
        + ff_inputs
        + ["-filter_complex", filtergraph]
        + nvenc_params
        + [str(output_path)]
    )
    
    print("🎬 Applying overlays with FFmpeg...")
    subprocess.run(cmd, check=True)
    print(f"✓ Overlays applied: {output_path}")