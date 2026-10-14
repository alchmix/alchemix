"""
zoom2.py — news_broadcast layout composite test
─────────────────────────────────────────────────
Tests _composite_avatar_over_full_video logic with
the exact news_broadcast layout config.
"""

import os, json, logging, subprocess, time, re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

BACKEND_ROOT  = r"D:\Alchemy Mix\backend"
CONTENT_VIDEO = os.path.join(BACKEND_ROOT, "outputs",           "final_20260610_152907.mp4")
AVATAR_0      = os.path.join(BACKEND_ROOT, "static", "avatars", "avatar_900374860e64.mp4")
AVATAR_1      = os.path.join(BACKEND_ROOT, "static", "avatars", "avatar_430cc42f411e.mp4")
OUT_DIR       = os.path.join(BACKEND_ROOT, "test_output")

CANVAS_W  = 1920
CANVAS_H  = 1080
PREVIEW_W = 560

LAYOUT_CONFIG = {
    "layout_id": "news_broadcast",
    "background": "blurred",
    "avatar_layer": {
        "x": 26, "y": 437, "width": 499, "height": 603,
        "border": {"width": 4, "color": "#f5f5f5", "radius": 0},
    },
    "content_layer": {
        "x": 617, "y": 101, "width": 1251, "height": 617,
        "border": {"width": 4, "color": "#ffffff", "radius": 0},
    },
}


def probe_duration(path):
    if not path or not os.path.exists(path):
        return 0.0
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True,
        )
        for s in json.loads(r.stdout).get("streams", []):
            d = float(s.get("duration", 0))
            if d > 0:
                return d
    except Exception as e:
        logging.warning(f"probe_duration failed: {e}")
    return 0.0


def drawbox(x, y, w, h, border_w, color_hex, radius):
    t   = max(1, min(border_w, min(w, h) // 2))
    col = f"{color_hex}@1.0"
    if radius <= 0:
        return f"drawbox=x={x}:y={y}:w={w}:h={h}:color={col}:t={t}"
    r = max(1, min(radius, w // 2, h // 2))
    return ",".join([
        f"drawbox=x={x+r}:y={y}:w={w-2*r}:h={t}:color={col}:t=fill",
        f"drawbox=x={x+r}:y={y+h-t}:w={w-2*r}:h={t}:color={col}:t=fill",
        f"drawbox=x={x}:y={y+r}:w={t}:h={h-2*r}:color={col}:t=fill",
        f"drawbox=x={x+w-t}:y={y+r}:w={t}:h={h-2*r}:color={col}:t=fill",
    ])


def concat_avatars(avatar_paths, tmp_dir):
    """Stream-copy concat — no decode, ~1s."""
    out = os.path.join(tmp_dir, f"avatar_track_{int(time.time()*1000)}.mp4")
    txt = os.path.join(tmp_dir, "avatar_concat.txt")
    with open(txt, "w") as f:
        for p in avatar_paths:
            f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")
    t0 = time.time()
    r  = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", txt,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        out,
    ], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        logging.error(f"❌ avatar concat failed:\n{r.stderr[-400:]}")
        return None
    dur = probe_duration(out)
    mb  = os.path.getsize(out) / (1024*1024)
    logging.info(f"✅ avatar track: {dur:.3f}s  {mb:.1f}MB  ({time.time()-t0:.1f}s)")
    return out, txt


def composite(content_video, avatar_track, layout_config, canvas_w, canvas_h, output_path):
    """
    Single-pass NVENC composite.
    input 0: content video  (anchor + audio)
    input 1: avatar track   (visual only, from t=0)
    Background 'blurred' uses content video downscale as BG — no extra input needed.
    """
    layout_id  = layout_config.get("layout_id",  "news_broadcast")
    background = layout_config.get("background", "#0a0a0a")

    # ── geometry ──────────────────────────────────────────────────────────────
    av_layer        = layout_config.get("avatar_layer")  or {}
    av_x            = int(av_layer.get("x",      0))
    av_y            = int(av_layer.get("y",      0))
    av_w            = int(av_layer.get("width",  320))
    av_h            = int(av_layer.get("height", 450))
    av_border       = av_layer.get("border") or {}
    av_border_w     = max(0, int(av_border.get("width",  0)))
    av_border_color = av_border.get("color", "#ffffff").lstrip("#")
    av_border_r     = max(0, int(av_border.get("radius", 0)))
    if not re.match(r'^[0-9a-fA-F]{6}$', av_border_color):
        av_border_color = "ffffff"

    ct_layer        = layout_config.get("content_layer") or {}
    ct_x            = int(ct_layer.get("x",      0))
    ct_y            = int(ct_layer.get("y",      0))
    ct_w            = int(ct_layer.get("width",  canvas_w))
    ct_h            = int(ct_layer.get("height", canvas_h))
    ct_border       = ct_layer.get("border") or {}
    ct_border_w     = max(0, int(ct_border.get("width",  0)))
    ct_border_color = ct_border.get("color", "#ffffff").lstrip("#")
    ct_border_r     = max(0, int(ct_border.get("radius", 0)))
    if not re.match(r'^[0-9a-fA-F]{6}$', ct_border_color):
        ct_border_color = "ffffff"

    # ── scale borders preview → canvas ────────────────────────────────────────
    scale       = canvas_w / PREVIEW_W
    av_border_w = max(0, round(av_border_w * scale))
    av_border_r = max(0, round(av_border_r * scale))
    ct_border_w = max(0, round(ct_border_w * scale))
    ct_border_r = max(0, round(ct_border_r * scale))

    logging.info(
        f"🖼️  layout={layout_id}  bg={background}  canvas={canvas_w}x{canvas_h}\n"
        f"   content : ({ct_x},{ct_y}) {ct_w}x{ct_h}  border={ct_border_w}px\n"
        f"   avatar  : ({av_x},{av_y}) {av_w}x{av_h}  border={av_border_w}px"
    )

    # ── filter graph ──────────────────────────────────────────────────────────
    # input 0 : content video
    # input 1 : avatar track
    # (no input 2 — blurred BG is derived from input 0, no extra file)
    parts   = []
    current = "base"

    if background == "blurred":
        # Derive blurred background from the content video itself.
        # Downscale to 1/8 then upscale back — fast CPU blur approximation.
        # Same technique as original production code.
        parts.append(
            f"[0:v]"
            f"scale={canvas_w//8}:{canvas_h//8}:flags=bilinear,"
            f"scale={canvas_w}:{canvas_h}:flags=bilinear,"
            f"setsar=1,format=yuv420p[base]"
        )
    elif re.match(r'^#[0-9a-fA-F]{6}$', str(background)):
        bg_color = background.lstrip("#")
        parts.append(
            f"color=c=#{bg_color}:s={canvas_w}x{canvas_h}:r=30,"
            f"format=yuv420p[base]"
        )
    else:
        parts.append(
            f"color=c=#0a0a0a:s={canvas_w}x{canvas_h}:r=30,"
            f"format=yuv420p[base]"
        )

    # Content layer
    if ct_layer and layout_id != "full_avatar":
        parts.append(
            f"[0:v]scale={ct_w}:{ct_h}:flags=lanczos,"
            f"setsar=1,format=yuv420p[ct]"
        )
        parts.append(
            f"[{current}][ct]overlay=x={ct_x}:y={ct_y}:format=yuv420[after_ct]"
        )
        current = "after_ct"
        if ct_border_w > 0:
            parts.append(
                f"[{current}]"
                f"{drawbox(ct_x,ct_y,ct_w,ct_h,ct_border_w,ct_border_color,ct_border_r)}"
                f"[after_ct_border]"
            )
            current = "after_ct_border"
    elif layout_id == "full_avatar":
        parts.append(
            f"[0:v]scale={canvas_w}:{canvas_h}:flags=lanczos,"
            f"setsar=1,format=yuv420p[base_full]"
        )
        current = "base_full"

    # Avatar overlay
    sw = canvas_w if layout_id == "full_avatar" else av_w
    sh = canvas_h if layout_id == "full_avatar" else av_h
    ox = 0        if layout_id == "full_avatar" else av_x
    oy = 0        if layout_id == "full_avatar" else av_y

    parts.append(
        f"[1:v]scale={sw}:{sh}:flags=lanczos,"
        f"setsar=1,format=yuv420p[av]"
    )
    parts.append(
        f"[{current}][av]overlay=x={ox}:y={oy}:format=yuv420[after_av]"
    )
    current = "after_av"

    # Avatar border
    if av_border_w > 0 and layout_id not in ("full_avatar", "full_content"):
        parts.append(
            f"[{current}]"
            f"{drawbox(av_x,av_y,av_w,av_h,av_border_w,av_border_color,av_border_r)}"
            f"[after_av_border]"
        )
        current = "after_av_border"

    filter_complex = ";".join(parts)
    logging.info(f"🔧 filter_complex ({len(filter_complex)} chars):\n{filter_complex}\n")

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda", "-i", content_video,
        "-hwaccel", "cuda", "-i", avatar_track,
        "-filter_complex", filter_complex,
        "-map",       f"[{current}]",
        "-map",       "0:a",
        "-shortest",
        "-c:v",       "h264_nvenc",
        "-preset",    "p4",
        "-rc",        "constqp",
        "-qp",        "18",
        "-profile:v", "high",
        "-pix_fmt",   "yuv420p",
        "-c:a",       "copy",
        "-movflags",  "+faststart",
        output_path,
    ]

    logging.info("🚀 single-pass NVENC composite …")
    t0     = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    el     = time.time() - t0

    if result.returncode != 0 or not os.path.exists(output_path):
        logging.error(f"❌ failed ({el:.1f}s):\n{result.stderr[-1500:]}")
        return None

    content_dur = probe_duration(content_video)
    out_dur     = probe_duration(output_path)
    mb          = os.path.getsize(output_path) / (1024*1024)

    logging.info("")
    logging.info("=" * 55)
    logging.info("🎉 DONE")
    logging.info(f"   output  : {output_path}")
    logging.info(f"   size    : {mb:.1f} MB")
    logging.info(f"   time    : {el:.1f}s  ({el/60:.1f} min)")
    logging.info(f"   content : {content_dur:.3f}s")
    logging.info(f"   output  : {out_dur:.3f}s")
    logging.info(f"   delta   : {out_dur - content_dur:+.3f}s  ← target 0.000s")
    logging.info("=" * 55)

    return output_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = os.path.join(OUT_DIR, f"_tmp_{int(time.time())}")
    os.makedirs(tmp_dir, exist_ok=True)

    for label, path in [
        ("content", CONTENT_VIDEO),
        ("avatar0", AVATAR_0),
        ("avatar1", AVATAR_1),
    ]:
        if not os.path.exists(path):
            logging.error(f"❌ missing {label}: {path}")
            return

    content_dur = probe_duration(CONTENT_VIDEO)
    av0_dur     = probe_duration(AVATAR_0)
    av1_dur     = probe_duration(AVATAR_1)
    av_total    = av0_dur + av1_dur

    logging.info(f"📹 content    : {content_dur:.3f}s")
    logging.info(f"🎭 avatar0    : {av0_dur:.3f}s")
    logging.info(f"🎭 avatar1    : {av1_dur:.3f}s")
    logging.info(f"🎭 av total   : {av_total:.3f}s")
    logging.info(f"   delta      : {av_total - content_dur:+.3f}s")
    logging.info(f"\n📐 Layout     : {LAYOUT_CONFIG['layout_id']}")
    logging.info(f"   background : {LAYOUT_CONFIG['background']}")
    logging.info(
        f"   content    : ({LAYOUT_CONFIG['content_layer']['x']},"
        f"{LAYOUT_CONFIG['content_layer']['y']}) "
        f"{LAYOUT_CONFIG['content_layer']['width']}x"
        f"{LAYOUT_CONFIG['content_layer']['height']}"
    )
    logging.info(
        f"   avatar     : ({LAYOUT_CONFIG['avatar_layer']['x']},"
        f"{LAYOUT_CONFIG['avatar_layer']['y']}) "
        f"{LAYOUT_CONFIG['avatar_layer']['width']}x"
        f"{LAYOUT_CONFIG['avatar_layer']['height']}"
    )

    # step 1: concat avatar batches → one continuous track
    logging.info("\n▶ step 1 — concat avatar batches …")
    result = concat_avatars([AVATAR_0, AVATAR_1], tmp_dir)
    if not result:
        return
    avatar_track, concat_txt = result

    # step 2: composite
    logging.info("\n▶ step 2 — composite …")
    output_path = os.path.join(
        OUT_DIR,
        f"composite_news_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    )

    out = composite(
        content_video = CONTENT_VIDEO,
        avatar_track  = avatar_track,
        layout_config = LAYOUT_CONFIG,
        canvas_w      = CANVAS_W,
        canvas_h      = CANVAS_H,
        output_path   = output_path,
    )

    # cleanup
    try: os.remove(avatar_track)
    except: pass
    try: os.remove(concat_txt)
    except: pass
    try: os.rmdir(tmp_dir)
    except: pass

    if out:
        logging.info(f"\n🎉 SUCCESS → {out}")
    else:
        logging.error("\n💥 FAILED")


if __name__ == "__main__":
    main()