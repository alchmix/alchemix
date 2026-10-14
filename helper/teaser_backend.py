"""
Intro teaser + narration image helpers for segment encode.

Never use `from tests import …` — use segment_pool_worker.get_encode_attr.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

TEASER_SCHEMA_VERSION = 2

DEFAULT_TEASER_CONFIG: dict[str, Any] = {
    "schemaVersion": TEASER_SCHEMA_VERSION,
    "chyronText": "COMING UP...",
    "layoutMode": "free",
    "xPct": 5.0,
    "yPct": 78.0,
    "widthPct": 90.0,
    "colorMode": "color",
    "accentColor": "#FF1A1A",
    "stripColor": "#000000",
    "stripOpacity": 0.93,
    "textColor": "#FFFFFF",
    "contentEffect": "none",
    "contentDesaturate": 0.65,
    "contrast": 1.15,
    "vignette": False,
    "showAccentBar": True,
    "showFlagTab": False,
    "showBottomLine": False,
    "borderEnabled": False,
    "borderColor": "#FFFFFF",
    "borderWidth": 2,
    "borderRadiusPct": 0.0,
    "chyronHeightPct": 0.12,
    "textSizePct": 0.075,
    "accentBarHeightPct": 0.008,
    "flagWidthPct": 0.018,
    "textPadXPct": 0.02,
    "bottomLineHeightPct": 0.004,
    "padYPct": 0.01,
    "fontFamily": "Arial Black",
    "fontWeight": "bold",
    "letterSpacing": 0.06,
    "textShadow": True,
    # v2.1 (optional — defaults keep the original look)
    "textAlign": "left",          # left | center | right
    "textTransform": "none",      # none | uppercase | lowercase
    "startSec": 0.0,              # appear N seconds into the intro segment
    "durationSec": 0.0,           # 0 = until the segment ends
}


def unwrap_intro_teaser_payload(raw: dict | None) -> dict | None:
    """Frontend sends { presetId, config } — extract inner config."""
    if not isinstance(raw, dict):
        return raw
    inner = raw.get("config")
    if isinstance(inner, dict):
        return inner
    return raw


def normalize_teaser_config(raw: dict | None) -> dict:
    cfg = dict(DEFAULT_TEASER_CONFIG)
    unwrapped = unwrap_intro_teaser_payload(raw)
    if isinstance(unwrapped, dict):
        cfg.update({k: v for k, v in unwrapped.items() if v is not None})
    cfg["schemaVersion"] = TEASER_SCHEMA_VERSION
    if not cfg.get("layoutMode"):
        cfg["layoutMode"] = (
            "free" if cfg.get("xPct") is not None or cfg.get("widthPct") is not None else "classic"
        )
    return cfg


def compute_intro_chyron_geometry(
    teaser_config: dict | None,
    canvas_w: int,
    canvas_h: int,
    *,
    pos_x: int = 0,
    pos_y: int = 0,
    framed_w: int | None = None,
    framed_h: int | None = None,
) -> dict[str, Any]:
    """Pixel layout for intro chyron — matches frontend teaserLayoutMetrics."""
    tc = normalize_teaser_config(teaser_config)
    layout_mode = tc.get("layoutMode", "free")

    show_accent = bool(tc.get("showAccentBar", True))
    show_flag = bool(tc.get("showFlagTab", False))
    show_bottom = bool(tc.get("showBottomLine", False))

    chyron_h = max(32, int(canvas_h * float(tc.get("chyronHeightPct", 0.12))))
    accent_bar_h = max(3, int(canvas_h * float(tc.get("accentBarHeightPct", 0.008)))) if show_accent else 0
    text_size = max(14, int(canvas_h * float(tc.get("textSizePct", 0.075))))
    flag_w = max(6, int(canvas_w * float(tc.get("flagWidthPct", 0.018)))) if show_flag else 0
    text_pad_x = max(8, int(canvas_w * float(tc.get("textPadXPct", 0.02))))
    bottom_h = max(2, int(canvas_h * float(tc.get("bottomLineHeightPct", 0.004)))) if show_bottom else 0
    strip_h = max(18, chyron_h)
    widget_h = accent_bar_h + strip_h + bottom_h

    if layout_mode == "classic":
        region_w = framed_w if framed_w is not None else canvas_w
        region_h = framed_h if framed_h is not None else canvas_h
        box_w = region_w
        box_x = pos_x
        widget_top = pos_y + region_h - widget_h
    else:
        box_w = max(60, int(canvas_w * float(tc.get("widthPct", 90)) / 100.0))
        box_x = int(canvas_w * float(tc.get("xPct", 5)) / 100.0)
        widget_top = int(canvas_h * float(tc.get("yPct", 78)) / 100.0)
        box_x = max(0, min(box_x, canvas_w - box_w))
        widget_top = max(0, min(widget_top, canvas_h - widget_h))

    accent_y = widget_top
    strip_y = widget_top + accent_bar_h
    text_y = strip_y + max(2, (strip_h - text_size) // 2)
    text_x = box_x + flag_w + text_pad_x

    return {
        "tc": tc,
        "box_x": box_x,
        "box_w": box_w,
        "widget_top": widget_top,
        "widget_h": widget_h,
        "accent_y": accent_y,
        "accent_bar_h": accent_bar_h,
        "strip_y": strip_y,
        "strip_h": strip_h,
        "text_x": text_x,
        "text_y": text_y,
        "text_size": text_size,
        "flag_w": flag_w,
        "bottom_h": bottom_h,
        "show_accent": show_accent,
        "show_flag": show_flag,
        "show_bottom": show_bottom,
    }


def is_intro_tag(tag: str | None) -> bool:
    return isinstance(tag, str) and tag.strip().lower() == "intro"


def hex_to_ffmpeg_color(hex_color: str, alpha: float = 1.0) -> str:
    s = (hex_color or "#000000").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        s = "000000"
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    a = max(0, min(255, int(alpha * 255)))
    return f"0x{r:02X}{g:02X}{b:02X}{a:02X}"


# UI fontFamily → Windows Fonts filenames (bold preferred for chyron)
_FONT_CANDIDATES: dict[str, list[str]] = {
    "arial black": ["ariblk.ttf", "Arial Black.ttf", "arialbd.ttf"],
    "impact": ["impact.ttf", "Impact.ttf"],
    "arial": ["arialbd.ttf", "arial.ttf"],
    "arial bold": ["arialbd.ttf", "arial.ttf"],
    "georgia": ["georgiab.ttf", "georgia.ttf"],
    "georgia bold": ["georgiab.ttf", "georgia.ttf"],
    "times new roman": ["timesbd.ttf", "times.ttf"],
    "times new roman bold": ["timesbd.ttf", "times.ttf"],
    "verdana": ["verdanab.ttf", "verdana.ttf"],
    "trebuchet ms": ["trebucbd.ttf", "trebuc.ttf"],
    "comic sans ms": ["comicbd.ttf", "comic.ttf"],
    "segoe ui": ["segoeuib.ttf", "segoeui.ttf"],
    "tahoma": ["tahomabd.ttf", "tahoma.ttf"],
    "calibri": ["calibrib.ttf", "calibri.ttf"],
    "courier new": ["courbd.ttf", "cour.ttf"],
}


def _fonts_dir() -> str:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    return os.path.join(windir, "Fonts")


def _ffmpeg_font_path(path: str | None) -> str | None:
    """Normalize font path for ffmpeg-python (forward slashes, no colon escape)."""
    if not path:
        return None
    return os.path.normpath(path).replace("\\", "/")


def _escape_drawtext(text: str) -> str:
    """Escape user text for FFmpeg drawtext."""
    t = (text or "").replace("\\", "\\\\")
    t = t.replace(":", "\\:")
    t = t.replace("'", "\\'")
    t = t.replace("%", "\\%")
    return t


def resolve_font_file(teaser_config: dict | None) -> str | None:
    """Resolve UI fontFamily to a real .ttf for FFmpeg drawtext."""
    cfg = teaser_config or {}
    family = (cfg.get("fontFamily") or "Arial Black").strip()
    weight = str(cfg.get("fontWeight") or "bold").strip().lower()
    key = family.lower()

    fonts_dir = _fonts_dir()
    candidates: list[str] = []

    if key in _FONT_CANDIDATES:
        candidates = list(_FONT_CANDIDATES[key])
    elif f"{key} bold" in _FONT_CANDIDATES and weight in ("bold", "700", "bolder"):
        candidates = list(_FONT_CANDIDATES[f"{key} bold"])
    else:
        compact = family.replace(" ", "")
        candidates = [f"{compact}.ttf", f"{compact.lower()}.ttf", f"{compact}bd.ttf"]

    # Prefer regular weights when not bold
    if weight not in ("bold", "700", "bolder") and key in _FONT_CANDIDATES:
        regular = [c for c in _FONT_CANDIDATES[key] if "bd" not in c.lower() and "blk" not in c.lower()]
        if regular:
            candidates = regular + candidates

    for name in candidates:
        path = os.path.join(fonts_dir, name)
        if os.path.isfile(path):
            return path

    # Case-insensitive filename match in Fonts folder
    needle = family.replace(" ", "").lower()
    try:
        for fn in os.listdir(fonts_dir):
            low = fn.lower()
            if not low.endswith((".ttf", ".otf")):
                continue
            if needle in low.replace(" ", ""):
                if weight in ("bold", "700", "bolder") and "bd" not in low and "blk" not in low:
                    continue
                return os.path.join(fonts_dir, fn)
    except OSError:
        pass

    logging.warning(f"⚠️ Chyron font not found for {family!r} — FFmpeg will use default")
    return None


def narration_override_mode(narration_images_config: dict | None) -> str | None:
    if not isinstance(narration_images_config, dict):
        return None
    mode = (narration_images_config.get("mode") or "").strip().lower()
    if mode in ("global", "per_segment"):
        return mode
    return None


def resolve_narration_image(
    narration_images_config: dict | None,
    section_type: str,
    segment_index: int,
) -> str | None:
    if not isinstance(narration_images_config, dict):
        return None
    mode = narration_override_mode(narration_images_config)
    if mode == "global":
        url = narration_images_config.get("globalImage")
        return (url or "").strip() or None
    if mode == "per_segment":
        per = narration_images_config.get("perSegment") or {}
        section = per.get(section_type) or per.get(str(section_type)) or {}
        if isinstance(section, dict):
            url = section.get(str(segment_index), section.get(segment_index))
            return (url or "").strip() or None
    return None


def download_image_to_local(url: str, dest_path: str, static_root: str | None = None) -> str | None:
    from helper.banner_backend import resolve_media_url

    resolved = resolve_media_url(url, static_root)
    if not resolved:
        return None
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    if os.path.isfile(resolved):
        return resolved
    if resolved.startswith(("http://", "https://")):
        try:
            r = requests.get(resolved, timeout=45)
            r.raise_for_status()
            with open(dest_path, "wb") as fh:
                fh.write(r.content)
            return dest_path if os.path.isfile(dest_path) else None
        except Exception as exc:
            logging.warning(f"⚠️ Image download failed {resolved}: {exc}")
            return None
    if os.path.isfile(resolved):
        return resolved
    return None


def _ffmpeg_font_color(color: str) -> str:
    """drawtext fontcolor — accept #RRGGBB or named colors."""
    c = (color or "#FFFFFF").strip()
    if c.startswith("#"):
        return f"0x{c[1:]}"
    return c


def _teaser_enable_expr(tc: dict) -> str:
    """ffmpeg timeline expression for when the chyron is visible."""
    try:
        start = max(0.0, float(tc.get("startSec") or 0))
    except (TypeError, ValueError):
        start = 0.0
    try:
        dur = max(0.0, float(tc.get("durationSec") or 0))
    except (TypeError, ValueError):
        dur = 0.0
    if dur > 0:
        return f"between(t,{start:.3f},{start + dur:.3f})"
    return f"gte(t,{start:.3f})"


def apply_chyron_to_stream(
    video_stream,
    teaser_config: dict | None,
    canvas_w: int,
    canvas_h: int,
    *,
    pos_x: int = 0,
    pos_y: int = 0,
    framed_w: int | None = None,
    framed_h: int | None = None,
):
    """Burn intro chyron onto an ffmpeg video stream (panel or fullscreen)."""
    tc = normalize_teaser_config(teaser_config)
    geo = compute_intro_chyron_geometry(
        tc,
        canvas_w,
        canvas_h,
        pos_x=pos_x,
        pos_y=pos_y,
        framed_w=framed_w,
        framed_h=framed_h,
    )

    raw_text = str(tc.get("chyronText", "COMING UP...") or "")
    transform = str(tc.get("textTransform") or "none").lower()
    if transform == "uppercase":
        raw_text = raw_text.upper()
    elif transform == "lowercase":
        raw_text = raw_text.lower()
    chyron_text = _escape_drawtext(raw_text)
    enable = _teaser_enable_expr(tc)
    accent = hex_to_ffmpeg_color(tc.get("accentColor", "#FF1A1A"), 1.0)
    strip = hex_to_ffmpeg_color(tc.get("stripColor", "#000000"), tc.get("stripOpacity", 0.93))
    text_color = _ffmpeg_font_color(tc.get("textColor", "#FFFFFF"))
    fontfile = _ffmpeg_font_path(resolve_font_file(tc))
    use_shadow = tc.get("textShadow", True)

    box_x = geo["box_x"]
    box_w = geo["box_w"]
    accent_y = geo["accent_y"]
    accent_bar_h = geo["accent_bar_h"]
    strip_y = geo["strip_y"]
    strip_h = geo["strip_h"]
    text_x = geo["text_x"]
    text_y = geo["text_y"]
    text_size = geo["text_size"]
    flag_w = geo["flag_w"]
    bottom_h = geo["bottom_h"]

    logging.info(
        f"📺 Chyron burn | text={chyron_text!r} | font={tc.get('fontFamily')} | "
        f"fontfile={fontfile or 'default'} | layout={tc.get('layoutMode')} | "
        f"pos=({box_x},{geo['widget_top']}) | size={box_w}x{geo['widget_h']} | "
        f"canvas={canvas_w}x{canvas_h}"
    )

    if geo["show_accent"] and accent_bar_h > 0:
        video_stream = video_stream.filter(
            "drawbox", x=box_x, y=accent_y, w=box_w, h=accent_bar_h, color=accent, t="fill", enable=enable
        )

    video_stream = video_stream.filter(
        "drawbox", x=box_x, y=strip_y, w=box_w, h=strip_h, color=strip, t="fill", enable=enable
    )

    if geo["show_flag"] and flag_w > 0:
        video_stream = video_stream.filter(
            "drawbox", x=box_x, y=strip_y, w=flag_w, h=strip_h, color=accent, t="fill", enable=enable
        )
        video_stream = video_stream.filter(
            "drawbox", x=box_x + flag_w, y=strip_y, w=3, h=strip_h, color="0xFFFFFFCC", t="fill", enable=enable
        )

    align = str(tc.get("textAlign") or "left").lower()
    text_pad_x = max(0, text_x - box_x - flag_w)
    if align == "center":
        x_expr = f"{box_x + flag_w}+({box_w - flag_w}-text_w)/2"
    elif align == "right":
        x_expr = f"{box_x + box_w - text_pad_x}-text_w"
    else:
        x_expr = text_x

    drawtext_kw = dict(
        text=chyron_text,
        fontsize=text_size,
        fontcolor=text_color,
        x=x_expr,
        y=text_y,
        enable=enable,
    )
    if fontfile:
        drawtext_kw["fontfile"] = fontfile
    if use_shadow:
        drawtext_kw.update(shadowcolor="0x000000CC", shadowx=4, shadowy=4)

    video_stream = video_stream.filter("drawtext", **drawtext_kw)

    if geo["show_bottom"] and bottom_h > 0:
        video_stream = video_stream.filter(
            "drawbox",
            x=box_x,
            y=strip_y + strip_h - bottom_h,
            w=box_w,
            h=bottom_h,
            color="0xFFFFFFAA",
            t="fill",
            enable=enable,
        )

    if tc.get("borderEnabled"):
        border_color = hex_to_ffmpeg_color(tc.get("borderColor", "#FFFFFF"), 1.0)
        border_w = max(1, int(tc.get("borderWidth", 2)))
        video_stream = video_stream.filter(
            "drawbox",
            x=box_x,
            y=geo["widget_top"],
            w=box_w,
            h=geo["widget_h"],
            color=border_color,
            t=border_w,
            enable=enable,
        )

    return video_stream


def apply_intro_teaser_overlay(
    *,
    input_path: str,
    output_path: str,
    background_path: str | None = None,
    teaser_config: dict | None = None,
) -> str:
    """
    Burn intro chyron onto an already-encoded segment (keeps panel layout).
    Does NOT re-run fullscreen overlay — only adds the banner layer.
    Uses libx264 (CPU) — fast second pass; avoids NVENC slot contention.
    """
    import ffmpeg

    cfg = normalize_teaser_config(teaser_config)
    font_path = resolve_font_file(cfg)
    logging.info(
        f"📺 Intro teaser overlay | {os.path.basename(input_path)} → {os.path.basename(output_path)} | "
        f"text={cfg.get('chyronText')!r} | font={cfg.get('fontFamily')} | "
        f"fontfile={font_path or 'default'} | layout={cfg.get('layoutMode')}"
    )

    probe = ffmpeg.probe(input_path)
    vstream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    canvas_w = int(vstream["width"])
    canvas_h = int(vstream["height"])

    inp = ffmpeg.input(input_path)
    video = apply_chyron_to_stream(inp.video, cfg, canvas_w, canvas_h)
    video = video.filter("format", "yuv420p")

    out_kwargs: dict[str, Any] = dict(
        vcodec="libx264",
        preset="ultrafast",
        crf=18,
        r=30,
        pix_fmt="yuv420p",
        movflags="+faststart",
    )
    outputs = [video]
    if has_audio:
        audio = (
            inp.audio
            .filter("aresample", 48000)
            .filter("aformat", channel_layouts="stereo")
        )
        outputs.append(audio)
        out_kwargs.update(acodec="aac", audio_bitrate="192k", ar=48000, ac=2)

    try:
        (
            ffmpeg.output(*outputs, output_path, **out_kwargs)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = ""
        if exc.stderr:
            stderr = exc.stderr.decode("utf-8", errors="replace")
        logging.error(
            f"❌ Intro teaser ffmpeg failed | {os.path.basename(input_path)} | "
            f"text={cfg.get('chyronText')!r} | fontfile={font_path}\n{stderr[-2500:]}"
        )
        raise
    return output_path