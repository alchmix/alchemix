"""
Transition Banner Studio — backend renderer + narration↔CLIP boundary transitions.

Frontend payload (schema v2):
  transitionBanner: { enabled, presetId, config: { schemaVersion, transitionType, layers[] } }

Boundary behaviour (legacy splice + hold-aligned xfade head):
  • Video: left body + xfade block (tail ↔ banner hold ↔ head) + right body.
  • Banner hold extends video by BANNER_HOLD_SEC; incoming voice plays under banner.
  • Audio: full left + full right concat, apad=pad_dur at end to match longer video.
  • Xfade sync rule: incoming head is trimmed from [hold, hold+head] on the right
    chunk (not [0, head]), matching what continuous audio plays at each fade frame.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import uuid

from PIL import Image, ImageDraw, ImageFont

BANNER_SCHEMA_VERSION = 2

BANNER_SEGMENT_TAIL_SEC = float(os.getenv("BANNER_SEGMENT_TAIL_SEC", "0.95"))
BANNER_SEGMENT_HEAD_SEC = float(os.getenv("BANNER_SEGMENT_HEAD_SEC", "0.95"))
BANNER_XFADE_SEC = float(os.getenv("BANNER_XFADE_SEC", "0.85"))
BANNER_HOLD_SEC = float(os.getenv("BANNER_HOLD_SEC", "0.8"))
BANNER_FPS = int(os.getenv("BANNER_FPS", "30"))

FFMPEG_TRANSITIONS = {
    "swipe_left": "slideleft",
    "swipe_right": "slideright",
    "swipe_up": "slideup",
    "swipe_down": "slidedown",
    "wipe_left": "wipeleft",
    "wipe_right": "wiperight",
    "fade": "fade",
    "fade_black": "fadeblack",
    "fade_white": "fadewhite",
    "circle_open": "circleopen",
    "circle_close": "circleclose",
    "pixelize": "pixelize",
    "radial": "radial",
    "smooth_left": "smoothleft",
    "zoom_blur": "fade",  # zoom_fade composite — fade fallback
}


def _is_clip_segment(seg: dict) -> bool:
    text = (seg.get("text") or "").strip()
    return text.startswith("[CLIP:") and text.endswith("]")


def resolve_media_url(url: str, static_root: str | None = None) -> str:
    """Normalize image URLs from the composition UI."""
    if not url or not isinstance(url, str):
        return ""
    u = url.strip()
    if not u:
        return ""
    if u.startswith(("http://", "https://", "file://")):
        return u
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/static/") or u.startswith("static/"):
        rel = u.lstrip("/")
        if static_root:
            return os.path.join(static_root, rel.replace("static/", "", 1))
        return u
    if "api.alchemymix.com" in u or "alchemymix.com" in u:
        return "https://" + u.lstrip("/")
    if static_root and not os.path.isabs(u):
        return os.path.join(static_root, u)
    return u


def _download_image(url: str, dest: str, static_root: str | None = None) -> str | None:
    resolved = resolve_media_url(url, static_root)
    if not resolved:
        return None
    try:
        from helper.teaser_backend import download_image_to_local

        return download_image_to_local(resolved, dest)
    except Exception:
        pass

    if resolved.startswith(("http://", "https://")):
        try:
            import requests

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            r = requests.get(resolved, timeout=45)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            return dest if os.path.exists(dest) else None
        except Exception as exc:
            logging.warning(f"Banner image download failed {resolved}: {exc}")
            return None

    if os.path.exists(resolved):
        return resolved
    return None


def _hex_to_rgba(color: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    c = (color or "#000000").strip()
    if c.startswith("#"):
        c = c[1:]
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return (0, 0, 0, int(max(0, min(1, opacity)) * 255))
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (r, g, b, int(max(0, min(1, opacity)) * 255))


def _migrate_legacy_config(legacy: dict) -> dict:
    layers: list[dict] = []
    z = 0

    if legacy.get("backgroundType") == "image" and legacy.get("backgroundImageUrl"):
        layers.append(
            {
                "type": "image",
                "zIndex": z,
                "imageUrl": legacy.get("backgroundImageUrl", ""),
                "objectFit": legacy.get("backgroundImageFit", "cover"),
                "xPct": 0,
                "yPct": 0,
                "widthPct": 100,
                "heightPct": 100,
                "anchor": "top-left",
                "borderRadiusPct": 0,
                "imageFocusXPct": 50,
                "imageFocusYPct": 50,
                "borderWidth": 0,
                "borderColor": "#ffffff",
                "opacity": 1,
                "visible": True,
            }
        )
        z += 1
        if float(legacy.get("backgroundOverlayOpacity") or 0) > 0:
            layers.append(
                {
                    "type": "fill",
                    "zIndex": z,
                    "color": legacy.get("backgroundOverlayColor", "#000000"),
                    "opacity": legacy.get("backgroundOverlayOpacity"),
                    "xPct": 0,
                    "yPct": 0,
                    "widthPct": 100,
                    "heightPct": 100,
                    "anchor": "top-left",
                    "visible": True,
                }
            )
            z += 1
    elif legacy.get("backgroundType") == "gradient":
        layers.append(
            {
                "type": "gradient",
                "zIndex": z,
                "gradientFrom": legacy.get("gradientFrom", "#7c3aed"),
                "gradientTo": legacy.get("gradientTo", "#2563eb"),
                "gradientAngle": legacy.get("gradientAngle", 135),
                "opacity": 1,
                "visible": True,
            }
        )
        z += 1
    else:
        layers.append(
            {
                "type": "fill",
                "zIndex": z,
                "color": legacy.get("backgroundColor", "#0f172a"),
                "opacity": 1,
                "visible": True,
            }
        )
        z += 1

    if legacy.get("showLogo") and legacy.get("logoUrl"):
        layers.append(
            {
                "type": "image",
                "zIndex": z,
                "imageUrl": legacy.get("logoUrl"),
                "xPct": 50,
                "yPct": 45,
                "widthPct": float(legacy.get("logoWidthPct", 0.2)) * 100,
                "heightPct": float(legacy.get("logoMaxHeightPct", 0.35)) * 100,
                "borderRadiusPct": legacy.get("logoBorderRadiusPct", 0),
                "objectFit": "contain",
                "anchor": "center",
                "opacity": 1,
                "visible": True,
            }
        )
        z += 1

    if legacy.get("headline"):
        layers.append(
            {
                "type": "text",
                "zIndex": z,
                "text": legacy["headline"],
                "color": legacy.get("headlineColor", "#ffffff"),
                "fontSizePct": legacy.get("headlineSizePct", 0.065),
                "yPct": 72,
                "xPct": 50,
                "widthPct": 90,
                "heightPct": 12,
                "anchor": "center",
                "textAlign": "center",
                "opacity": 1,
                "visible": True,
            }
        )
        z += 1

    if legacy.get("subtext"):
        layers.append(
            {
                "type": "text",
                "zIndex": z,
                "text": legacy["subtext"],
                "color": legacy.get("subtextColor", "#ffffff"),
                "fontSizePct": legacy.get("subtextSizePct", 0.05),
                "yPct": 82,
                "xPct": 50,
                "widthPct": 90,
                "heightPct": 12,
                "anchor": "center",
                "textAlign": "center",
                "fontWeight": "500",
                "opacity": 1,
                "visible": True,
            }
        )
        z += 1

    swipe_map = {"left": "swipe_left", "right": "swipe_right", "up": "swipe_up", "down": "swipe_down"}
    return {
        "schemaVersion": BANNER_SCHEMA_VERSION,
        "transitionType": swipe_map.get(legacy.get("swipeDirection"), "swipe_left"),
        "layers": layers or _default_layers(),
    }


def _default_layers() -> list[dict]:
    return [
        {
            "type": "gradient",
            "zIndex": 0,
            "gradientFrom": "#7c3aed",
            "gradientTo": "#2563eb",
            "gradientAngle": 135,
            "opacity": 1,
            "visible": True,
        },
        {
            "type": "text",
            "zIndex": 10,
            "text": "UP NEXT",
            "color": "#ffffff",
            "fontSizePct": 0.07,
            "yPct": 72,
            "xPct": 50,
            "widthPct": 90,
            "heightPct": 12,
            "anchor": "center",
            "textAlign": "center",
            "opacity": 1,
            "visible": True,
        },
    ]


def normalize_banner_config(raw: dict | None) -> dict:
    if not raw or not isinstance(raw, dict):
        return {
            "schemaVersion": BANNER_SCHEMA_VERSION,
            "transitionType": "swipe_left",
            "layers": _default_layers(),
        }
    if raw.get("schemaVersion") == BANNER_SCHEMA_VERSION and isinstance(raw.get("layers"), list):
        layers = []
        for layer in raw["layers"]:
            if isinstance(layer, dict):
                layers.append(dict(layer))
        out = {
            "schemaVersion": BANNER_SCHEMA_VERSION,
            "transitionType": raw.get("transitionType") or "swipe_left",
            "layers": layers or _default_layers(),
        }
        # optional per-banner timing + camera motion (editor v2.1)
        for key, lo, hi in (("holdSec", 0.0, 4.0), ("transitionSec", 0.2, 1.6), ("motionAmount", 0.0, 0.3)):
            if raw.get(key) is not None:
                try:
                    out[key] = max(lo, min(hi, float(raw[key])))
                except (TypeError, ValueError):
                    pass
        if raw.get("motion") in ("none", "zoom_in", "zoom_out", "pan_left", "pan_right"):
            out["motion"] = raw["motion"]
        return out
    return _migrate_legacy_config(raw)


def normalize_transition_banner(payload: dict | None) -> dict:
    if not payload or not isinstance(payload, dict):
        return {"enabled": False, "preset_id": None, "config": normalize_banner_config(None)}

    cfg = payload.get("config")
    if not cfg and payload.get("schemaVersion"):
        cfg = payload

    return {
        "enabled": bool(payload.get("enabled", True)),
        "preset_id": payload.get("presetId") or payload.get("preset_id"),
        "config": normalize_banner_config(cfg or {}),
    }


def ffmpeg_transition_name(transition_type: str) -> str:
    return FFMPEG_TRANSITIONS.get(transition_type or "swipe_left", "slideleft")


def _num(value, default: float) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _layer_geometry(layer: dict, canvas_w: int, canvas_h: int) -> tuple[float, float, float, float, float]:
    """(cx, cy, w, h, rotation°) in canvas px — same math as the editor (bannerPresets.layerBox).

    Boxes may extend past the canvas; whatever is outside is simply cropped,
    exactly like the browser preview.
    """
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    if layer.get("type") == "image" and layer.get("shape") == "circle":
        size_pct = _num(layer.get("sizePct") or layer.get("widthPct"), 28.0)
        w = h = max(1.0, min(canvas_w, canvas_h) * size_pct / 100.0)
    else:
        w = max(1.0, canvas_w * _num(layer.get("widthPct", 100), 100.0) / 100.0)
        h = max(1.0, canvas_h * _num(layer.get("heightPct", 100), 100.0) / 100.0)
    px = canvas_w * _num(layer.get("xPct", 0), 0.0) / 100.0
    py = canvas_h * _num(layer.get("yPct", 0), 0.0) / 100.0
    if (layer.get("anchor") or "top-left").lower() == "center":
        cx, cy = px, py
    else:
        cx, cy = px + w / 2.0, py + h / 2.0
    return cx, cy, w, h, _num(layer.get("rotation", 0), 0.0)


def _layer_box(layer: dict, canvas_w: int, canvas_h: int) -> tuple[int, int, int, int]:
    """Integer (x, y, w, h) of the unrotated layer box (may be partly off-canvas)."""
    cx, cy, w, h, _rot = _layer_geometry(layer, canvas_w, canvas_h)
    return int(round(cx - w / 2)), int(round(cy - h / 2)), max(1, int(round(w))), max(1, int(round(h)))


def _radius_px(fraction, w: int, h: int) -> int:
    r = _num(fraction, 0.0)
    if r <= 0:
        return 0
    if r > 1:  # also accept 0–100 style values
        r = r / 100.0
    return max(0, min(int(min(w, h) * min(r, 0.5)), min(w, h) // 2))


def _border_radius_px(layer: dict, w: int, h: int) -> int:
    if layer.get("type") == "image" and layer.get("shape") == "circle":
        return max(0, min(w, h) // 2)
    return _radius_px(layer.get("borderRadiusPct", 0), w, h)


def _safe_rounded_outline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    color: tuple[int, int, int, int],
    border_w: int,
) -> None:
    x, y, w, h = box
    if w < 2 or h < 2:
        return
    border_w = max(1, int(border_w))
    border_w = min(border_w, w // 2, h // 2)
    if border_w < 1:
        return
    r = max(0, min(radius, w // 2, h // 2))
    try:
        draw.rounded_rectangle((x, y, x + w - 1, y + h - 1), radius=r, outline=color, width=border_w)
    except Exception as exc:
        logging.warning(f"Banner border skipped ({w}x{h}): {exc}")


# ------------------------------------------------------------------- fonts --
# family → (regular files, bold files, heavy files). Windows font file names;
# Linux/mac fallbacks are searched too so local tests still render.
_BANNER_FONTS: dict[str, tuple[list[str], list[str], list[str]]] = {
    "impact": (["impact.ttf"], ["impact.ttf"], ["impact.ttf"]),
    "arial black": (["ariblk.ttf"], ["ariblk.ttf"], ["ariblk.ttf"]),
    "arial": (["arial.ttf"], ["arialbd.ttf"], ["ariblk.ttf", "arialbd.ttf"]),
    "segoe ui": (["segoeui.ttf"], ["segoeuib.ttf"], ["seguibl.ttf", "segoeuib.ttf"]),
    "verdana": (["verdana.ttf"], ["verdanab.ttf"], ["verdanab.ttf"]),
    "tahoma": (["tahoma.ttf"], ["tahomabd.ttf"], ["tahomabd.ttf"]),
    "trebuchet ms": (["trebuc.ttf"], ["trebucbd.ttf"], ["trebucbd.ttf"]),
    "calibri": (["calibri.ttf"], ["calibrib.ttf"], ["calibrib.ttf"]),
    "georgia": (["georgia.ttf"], ["georgiab.ttf"], ["georgiab.ttf"]),
    "times new roman": (["times.ttf"], ["timesbd.ttf"], ["timesbd.ttf"]),
    "courier new": (["cour.ttf"], ["courbd.ttf"], ["courbd.ttf"]),
    "comic sans ms": (["comic.ttf"], ["comicbd.ttf"], ["comicbd.ttf"]),
    "dm sans": (["arial.ttf"], ["arialbd.ttf"], ["arialbd.ttf"]),
}
_FALLBACK_FONTS = (
    ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"],
    ["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
)
_font_index: dict[str, str] | None = None
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font_dirs() -> list[str]:
    dirs = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")]
    dirs += [
        os.path.expanduser(r"~\AppData\Local\Microsoft\Windows\Fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        "/Library/Fonts",
        "/System/Library/Fonts",
    ]
    return [d for d in dirs if os.path.isdir(d)]


def _find_font_file(name: str) -> str | None:
    global _font_index
    if _font_index is None:
        _font_index = {}
        for d in _font_dirs():
            for root, _dirs, files in os.walk(d):
                for fn in files:
                    if fn.lower().endswith((".ttf", ".otf", ".ttc")):
                        _font_index.setdefault(fn.lower(), os.path.join(root, fn))
    return _font_index.get(name.lower())


def _primary_family(font_family) -> str:
    first = str(font_family or "Impact").split(",")[0].strip().strip("'\"")
    return first.lower() or "impact"


def _load_font(size: int, weight="800", family="Impact"):
    """Truetype font for a UI font family + weight (cached)."""
    size = max(6, int(size))
    w = str(weight or "700").lower()
    heavy = w in ("800", "900", "black", "heavy")
    bold = heavy or w in ("bold", "600", "700", "bolder")
    regular, bold_files, heavy_files = _BANNER_FONTS.get(_primary_family(family), _BANNER_FONTS["arial"])
    candidates = (heavy_files if heavy else bold_files if bold else regular) + (
        _FALLBACK_FONTS[1] if bold else _FALLBACK_FONTS[0]
    )
    for name in candidates:
        path = _find_font_file(name)
        if not path:
            continue
        key = (path, size)
        if key not in _font_cache:
            try:
                _font_cache[key] = ImageFont.truetype(path, size)
            except Exception:
                continue
        return _font_cache[key]
    return ImageFont.load_default()


# ------------------------------------------------------------- layer art --
def _apply_opacity(img: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 0.999:
        return img
    alpha = img.getchannel("A").point(lambda p: int(p * max(0.0, opacity)))
    img.putalpha(alpha)
    return img


def _gradient_patch(layer: dict, w: int, h: int) -> Image.Image:
    """CSS-accurate linear-gradient(angle) / radial-gradient(ellipse farthest-corner)."""
    c1 = _hex_to_rgba(layer.get("gradientFrom", "#7c3aed"), 1.0)
    c2 = _hex_to_rgba(layer.get("gradientTo", "#2563eb"), 1.0)
    radial = (layer.get("gradientType") or "linear").lower() == "radial"
    angle = math.radians(_num(layer.get("gradientAngle", 135), 135.0))
    try:
        import numpy as np

        xs = (np.arange(w, dtype=np.float32) + 0.5) - w / 2.0
        ys = (np.arange(h, dtype=np.float32) + 0.5) - h / 2.0
        X, Y = np.meshgrid(xs, ys)
        if radial:
            rx = max(w / 2.0 * math.sqrt(2), 1e-6)
            ry = max(h / 2.0 * math.sqrt(2), 1e-6)
            t = np.sqrt((X / rx) ** 2 + (Y / ry) ** 2)
        else:
            sa, ca = math.sin(angle), math.cos(angle)
            length = max(abs(w * sa) + abs(h * ca), 1e-6)
            t = (X * sa - Y * ca) / length + 0.5
        t = np.clip(t, 0.0, 1.0)[..., None]
        a = np.array(c1, dtype=np.float32)
        b = np.array(c2, dtype=np.float32)
        arr = (a + (b - a) * t).round().astype(np.uint8)
        return Image.fromarray(arr, "RGBA")
    except ImportError:
        img = Image.new("RGBA", (w, h))
        px = img.load()
        sa, ca = math.sin(angle), math.cos(angle)
        length = max(abs(w * sa) + abs(h * ca), 1e-6)
        rx, ry = max(w / 2.0 * math.sqrt(2), 1e-6), max(h / 2.0 * math.sqrt(2), 1e-6)
        for j in range(h):
            y = j + 0.5 - h / 2.0
            for i in range(w):
                x = i + 0.5 - w / 2.0
                t = math.hypot(x / rx, y / ry) if radial else (x * sa - y * ca) / length + 0.5
                t = max(0.0, min(1.0, t))
                px[i, j] = tuple(int(round(c1[k] + (c2[k] - c1[k]) * t)) for k in range(4))
        return img


def _draw_gradient(img: Image.Image, box: tuple[int, int, int, int], layer: dict, opacity: float) -> None:
    """Back-compat helper: draw a gradient into `img` at box."""
    x, y, w, h = box
    if w < 1 or h < 1:
        return
    _paste_clipped(img, _apply_opacity(_gradient_patch(layer, w, h), opacity), x, y)


def _fit_image(
    src: Image.Image,
    tw: int,
    th: int,
    object_fit: str,
    focus_x_pct: float,
    focus_y_pct: float,
) -> Image.Image:
    if tw <= 0 or th <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    sw, sh = src.size
    if sw <= 0 or sh <= 0:
        return Image.new("RGBA", (tw, th), (0, 0, 0, 0))

    fit = (object_fit or "cover").lower()
    if fit == "contain":
        scale = min(tw / sw, th / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        resized = src.resize((nw, nh), Image.LANCZOS)
        out = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        out.paste(resized, ((tw - nw) // 2, (th - nh) // 2), resized if resized.mode == "RGBA" else None)
        return out
    if fit == "fill":
        return src.resize((tw, th), Image.LANCZOS)

    scale = max(tw / sw, th / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    resized = src.resize((nw, nh), Image.LANCZOS)
    fx = max(0.0, min(100.0, _num(focus_x_pct, 50.0))) / 100.0
    fy = max(0.0, min(100.0, _num(focus_y_pct, 50.0))) / 100.0
    cx = int(max(0, nw - tw) * fx)
    cy = int(max(0, nh - th) * fy)
    return resized.crop((cx, cy, cx + tw, cy + th))


def _rounded_mask(w: int, h: int, radius: int, ellipse: bool = False) -> Image.Image:
    w, h = max(1, w), max(1, h)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if ellipse:
        draw.ellipse((0, 0, w - 1, h - 1), fill=255)
        return mask
    r = max(0, min(radius, w // 2, h // 2))
    try:
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=255)
    except Exception:
        draw.rectangle((0, 0, w - 1, h - 1), fill=255)
    return mask


def _image_patch(layer: dict, w: int, h: int, work_dir: str, static_root: str | None,
                 border_scale: float) -> Image.Image | None:
    local = os.path.join(work_dir, f"banner_img_{uuid.uuid4().hex[:8]}.img")
    path = _download_image(layer.get("imageUrl", "") or layer.get("image_url", ""), local, static_root)
    if not path:
        return None
    try:
        src = Image.open(path).convert("RGBA")
    except Exception as exc:
        logging.warning(f"Banner layer image load failed: {exc}")
        return None
    fitted = _fit_image(
        src, w, h,
        layer.get("objectFit", "cover"),
        layer.get("imageFocusXPct", 50),
        layer.get("imageFocusYPct", 50),
    )
    circle = layer.get("shape") == "circle"
    radius = _border_radius_px(layer, w, h)
    if radius > 0 or circle:
        mask = _rounded_mask(w, h, radius, ellipse=circle)
        fitted.putalpha(Image.composite(fitted.getchannel("A"), Image.new("L", (w, h), 0), mask))
    border_w = int(round(_num(layer.get("borderWidth"), 0) * border_scale))
    if border_w > 0:
        draw = ImageDraw.Draw(fitted)
        col = _hex_to_rgba(layer.get("borderColor", "#ffffff"), 1.0)
        if circle:
            draw.ellipse((0, 0, w - 1, h - 1), outline=col, width=min(border_w, w // 2))
        else:
            _safe_rounded_outline(draw, (0, 0, w, h), radius, col, border_w)
    return fitted


def _shape_patch(layer: dict, w: int, h: int, canvas_h: int) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ellipse = (layer.get("shapeKind") or "rect").lower() == "ellipse"
    radius = 0 if ellipse else _radius_px(layer.get("borderRadiusPct", 0), w, h)
    fill = _hex_to_rgba(layer.get("color", "#7c3aed"), 1.0) if layer.get("fillEnabled", True) else None
    stroke_w = 0
    if layer.get("strokeEnabled"):
        stroke_w = max(1, int(round(canvas_h * _num(layer.get("strokeWidthPct"), 0.004))))
        stroke_w = min(stroke_w, max(1, min(w, h) // 2))
    outline = _hex_to_rgba(layer.get("strokeColor", "#ffffff"), 1.0) if stroke_w else None
    box = (0, 0, w - 1, h - 1)
    if ellipse:
        draw.ellipse(box, fill=fill, outline=outline, width=stroke_w or 1)
    else:
        try:
            draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=stroke_w or 1)
        except Exception:
            draw.rectangle(box, fill=fill, outline=outline, width=stroke_w or 1)
    return img


def _text_width(text: str, font, spacing_px: float) -> float:
    if not text:
        return 0.0
    try:
        base = font.getlength(text)
    except Exception:
        bbox = font.getbbox(text)
        base = bbox[2] - bbox[0]
    return base + spacing_px * len(text)


def _wrap_lines(text: str, font, max_w: float, spacing_px: float) -> list[str]:
    """Greedy word wrap per paragraph — the same rule the browser uses."""
    lines: list[str] = []
    for para in text.split("\n"):
        words = para.split(" ")
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if not line or _text_width(candidate, font, spacing_px) <= max_w + 0.5:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def _draw_text_line(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font, fill,
                    spacing_px: float, stroke_w: int = 0, stroke_fill=None) -> None:
    kw = {"font": font, "fill": fill}
    if stroke_w > 0:
        kw.update(stroke_width=stroke_w, stroke_fill=stroke_fill)
    if abs(spacing_px) < 0.05:
        draw.text((x, y), text, **kw)
        return
    cx = x
    for ch in text:
        draw.text((cx, y), ch, **kw)
        try:
            adv = font.getlength(ch)
        except Exception:
            adv = font.getbbox(ch)[2]
        cx += adv + spacing_px


def _text_patch(layer: dict, w: int, h: int, canvas_h: int) -> tuple[Image.Image, int, int] | None:
    """Render a text layer. Returns (patch, ox, oy): the patch's top-left
    relative to the layer box (text may overflow the box, like in CSS)."""
    text = str(layer.get("text") or "")
    transform = (layer.get("textTransform") or "none").lower()
    if transform == "uppercase":
        text = text.upper()
    elif transform == "lowercase":
        text = text.lower()

    size = max(6, int(round(canvas_h * _num(layer.get("fontSizePct", 0.065), 0.065))))
    font = _load_font(size, str(layer.get("fontWeight", "800")), layer.get("fontFamily", "Impact"))
    spacing = _num(layer.get("letterSpacing", 0), 0.0) * 0.1 * size
    line_h = size * _num(layer.get("lineHeight", 1.15), 1.15)
    pad = canvas_h * _num(layer.get("paddingPct", 0.004), 0.004)
    align = (layer.get("textAlign") or "center").lower()

    stroke_w = int(round(canvas_h * _num(layer.get("strokeWidthPct", 0), 0.0)))
    shadow_on = layer.get("shadowEnabled", True) is not False
    blur = canvas_h * _num(layer.get("shadowBlurPct", 0.011), 0.011)
    sx = canvas_h * _num(layer.get("shadowOffsetXPct", 0), 0.0)
    sy = canvas_h * _num(layer.get("shadowOffsetYPct", 0.004), 0.004)

    lines = _wrap_lines(text, font, max(1.0, w - 2 * pad), spacing) if text.strip() else []
    block_h = line_h * len(lines)
    try:
        ascent, descent = font.getmetrics()
    except Exception:
        ascent, descent = size, int(size * 0.25)

    margin = int(size * 1.5 + stroke_w * 2 + (blur * 2 + abs(sx) + abs(sy) if shadow_on else 0)) + 4
    widest = max([_text_width(ln, font, spacing) for ln in lines] + [0.0])
    pw = int(max(w, widest + 2 * pad) + 2 * margin)
    ph = int(max(h, block_h) + 2 * margin)
    ox = -margin - int(max(0.0, (widest + 2 * pad - w) / 2))
    oy = -margin - int(max(0.0, (block_h - h) / 2))
    # box origin inside the patch
    bx, by = -ox, -oy

    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))

    if layer.get("bgEnabled"):
        bg = _hex_to_rgba(layer.get("bgColor", "#000000"), _num(layer.get("bgOpacity", 0.6), 0.6))
        radius = _radius_px(layer.get("bgRadiusPct", 0.2), w, h)
        box_img = Image.new("RGBA", (w, h), bg)
        box_img.putalpha(Image.composite(box_img.getchannel("A"), Image.new("L", (w, h), 0), _rounded_mask(w, h, radius)))
        patch.alpha_composite(box_img, (bx, by))

    if not lines:
        return patch, ox, oy

    top = by + (h - block_h) / 2.0
    positions = []
    for i, ln in enumerate(lines):
        lw = _text_width(ln, font, spacing)
        if align == "left":
            lx = bx + pad
        elif align == "right":
            lx = bx + w - pad - lw
        else:
            lx = bx + (w - lw) / 2.0
        ly = top + i * line_h + (line_h - (ascent + descent)) / 2.0
        positions.append((lx, ly, ln))

    stroke_fill = _hex_to_rgba(layer.get("strokeColor", "#000000"), 1.0)

    if shadow_on:
        a = _num(layer.get("shadowOpacity", 0.45), 0.45)
        mask = Image.new("L", (pw, ph), 0)
        md = ImageDraw.Draw(mask)
        for lx, ly, ln in positions:
            _draw_text_line(md, lx + sx, ly + sy, ln, font, 255, spacing, stroke_w, 255)
        if blur > 0.5:
            from PIL import ImageFilter

            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur / 2.0))
        sc = _hex_to_rgba(layer.get("shadowColor", "#000000"), 1.0)
        shadow = Image.new("RGBA", (pw, ph), sc[:3] + (0,))
        shadow.putalpha(mask.point(lambda p: int(p * max(0.0, min(1.0, a)))))
        patch.alpha_composite(shadow)

    draw = ImageDraw.Draw(patch)
    color = _hex_to_rgba(layer.get("color", "#ffffff"), 1.0)
    for lx, ly, ln in positions:
        _draw_text_line(draw, lx, ly, ln, font, color, spacing, stroke_w, stroke_fill)
    return patch, ox, oy


def _paste_clipped(canvas: Image.Image, patch: Image.Image, x: int, y: int) -> None:
    """alpha_composite that tolerates negative / off-canvas positions."""
    cw, ch = canvas.size
    pw, ph = patch.size
    left, top = max(0, x), max(0, y)
    right, bottom = min(cw, x + pw), min(ch, y + ph)
    if right <= left or bottom <= top:
        return
    crop = patch.crop((left - x, top - y, right - x, bottom - y))
    canvas.alpha_composite(crop, (left, top))


def _place_patch(canvas: Image.Image, patch: Image.Image, geom, ox: int = 0, oy: int = 0) -> None:
    """Rotate `patch` about the layer centre and composite it.

    (ox, oy) = patch top-left relative to the unrotated layer box.
    """
    cx, cy, w, h, rot = geom
    box_x = cx - w / 2.0
    box_y = cy - h / 2.0
    if abs(rot) < 0.01:
        _paste_clipped(canvas, patch, int(round(box_x + ox)), int(round(box_y + oy)))
        return
    pw, ph = patch.size
    # centre of the layer inside the patch
    pcx, pcy = w / 2.0 - ox, h / 2.0 - oy
    r = int(math.ceil(max(
        math.hypot(pcx, pcy), math.hypot(pw - pcx, pcy),
        math.hypot(pcx, ph - pcy), math.hypot(pw - pcx, ph - pcy),
    ))) + 2
    square = Image.new("RGBA", (2 * r, 2 * r), (0, 0, 0, 0))
    square.alpha_composite(patch, (int(round(r - pcx)), int(round(r - pcy))))
    # CSS rotate() is clockwise; PIL rotate() is counter-clockwise
    rotated = square.rotate(-rot, resample=Image.BICUBIC, center=(r, r))
    _paste_clipped(canvas, rotated, int(round(cx - r)), int(round(cy - r)))


def _render_image_layer(canvas, layer, box, work_dir, static_root, opacity) -> None:
    """Back-compat entry point (unrotated)."""
    x, y, w, h = box
    patch = _image_patch(layer, w, h, work_dir, static_root, 1.0)
    if patch is not None:
        _paste_clipped(canvas, _apply_opacity(patch, opacity), x, y)


def _render_text_layer(canvas, layer, box, canvas_h, opacity) -> None:
    """Back-compat entry point (unrotated)."""
    x, y, w, h = box
    res = _text_patch(layer, w, h, canvas_h)
    if res:
        patch, ox, oy = res
        _paste_clipped(canvas, _apply_opacity(patch, opacity), x + ox, y + oy)


def render_banner_frame(
    config: dict,
    canvas_w: int,
    canvas_h: int,
    work_dir: str,
    static_root: str | None = None,
) -> Image.Image:
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    layers = sorted(
        [l for l in config.get("layers", []) if isinstance(l, dict) and l.get("visible", True) is not False],
        key=lambda l: _num(l.get("zIndex", 0), 0.0),
    )
    os.makedirs(work_dir, exist_ok=True)
    # image borders are stored in px of a 1080p-class canvas (editor preview scales them)
    border_scale = min(canvas_w, canvas_h) / 1080.0

    for layer in layers:
        try:
            opacity = max(0.0, min(1.0, _num(layer.get("opacity", 1), 1.0)))
            if opacity <= 0:
                continue
            ltype = layer.get("type")
            geom = _layer_geometry(layer, canvas_w, canvas_h)
            w = max(1, int(round(geom[2])))
            h = max(1, int(round(geom[3])))
            ox = oy = 0

            if ltype == "fill":
                patch = Image.new("RGBA", (w, h), _hex_to_rgba(layer.get("color", "#0f172a"), 1.0))
            elif ltype == "gradient":
                patch = _gradient_patch(layer, w, h)
            elif ltype == "shape":
                patch = _shape_patch(layer, w, h, canvas_h)
            elif ltype == "image":
                patch = _image_patch(layer, w, h, work_dir, static_root, border_scale)
                if patch is None:
                    continue
            elif ltype == "text":
                res = _text_patch(layer, w, h, canvas_h)
                if not res:
                    continue
                patch, ox, oy = res
            else:
                continue

            _place_patch(canvas, _apply_opacity(patch, opacity), geom, ox, oy)
        except Exception as exc:
            logging.warning(
                f"Banner layer skipped ({layer.get('type', '?')} id={layer.get('id', '?')}): {exc}"
            )

    return canvas.convert("RGB")


def _motion_filter(config: dict, canvas_w: int, canvas_h: int, fps: int, duration: float) -> str | None:
    """Slow camera move over the still banner (zoompan on a 2× supersample)."""
    motion = (config.get("motion") or "none").lower()
    if motion not in ("zoom_in", "zoom_out", "pan_left", "pan_right"):
        return None
    amount = max(0.02, min(0.3, _num(config.get("motionAmount", 0.08), 0.08)))
    n = max(2, int(round(duration * fps)))
    p = f"(on/{n - 1})"
    if motion == "zoom_in":
        z = f"1+{amount:.4f}*{p}"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif motion == "zoom_out":
        z = f"{1 + amount:.4f}-{amount:.4f}*{p}"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif motion == "pan_left":
        z = f"{1 + amount:.4f}"
        x, y = f"(iw-iw/zoom)*{p}", "ih/2-(ih/zoom/2)"
    else:  # pan_right
        z = f"{1 + amount:.4f}"
        x, y = f"(iw-iw/zoom)*(1-{p})", "ih/2-(ih/zoom/2)"
    W, H = int(canvas_w), int(canvas_h)
    return (
        f"scale={2 * W}:{2 * H}:flags=lanczos,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={W}x{H}:fps={fps},"
        f"format=yuv420p"
    )


def render_banner_video(
    config: dict,
    canvas_w: int,
    canvas_h: int,
    output_path: str,
    work_dir: str,
    duration_sec: float | None = None,
    fps: int | None = None,
    static_root: str | None = None,
) -> str | None:
    hold = _num(config.get("holdSec"), BANNER_HOLD_SEC) if isinstance(config, dict) else BANNER_HOLD_SEC
    xf = _num(config.get("transitionSec"), BANNER_XFADE_SEC) if isinstance(config, dict) else BANNER_XFADE_SEC
    duration = float(duration_sec or (hold + 2 * xf))
    fps = int(fps or BANNER_FPS)
    frame_path = os.path.join(work_dir, f"banner_frame_{uuid.uuid4().hex[:8]}.png")
    try:
        os.makedirs(work_dir, exist_ok=True)
        frame = render_banner_frame(config, canvas_w, canvas_h, work_dir, static_root)
        frame.save(frame_path, "PNG")
    except Exception as exc:
        logging.error(f"Banner frame render failed: {exc}")
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    motion_vf = _motion_filter(config or {}, canvas_w, canvas_h, fps, duration)

    def _encode(vf: str | None) -> None:
        # Video-only: banner is an overlay source; audio comes from the join.
        cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", str(fps), "-i", frame_path, "-t", f"{duration:.3f}"]
        if vf:
            cmd += ["-vf", vf]
        cmd += [
            "-r", str(fps),
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    try:
        try:
            _encode(motion_vf)
        except subprocess.CalledProcessError as exc:
            if not motion_vf:
                raise
            logging.warning(f"Banner motion failed, rendering still instead:\n{(exc.stderr or '')[-400:]}")
            _encode(None)
    except subprocess.CalledProcessError as exc:
        logging.error(f"Banner video encode failed:\n{(exc.stderr or '')[-600:]}")
        return None
    finally:
        try:
            os.remove(frame_path)
        except Exception:
            pass

    return output_path if os.path.exists(output_path) else None


# ------------------------------------------------------------------ probing --
def _ffprobe_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(r.stdout)
    except Exception:
        return {}


def _parse_fps(rate: str | None) -> float:
    if not rate or rate in ("0/0", "N/A"):
        return float(BANNER_FPS)
    try:
        num, den = rate.split("/", 1)
        den_f = float(den)
        if den_f <= 0:
            return float(BANNER_FPS)
        return float(num) / den_f
    except Exception:
        return float(BANNER_FPS)


def _probe_duration(path: str) -> float:
    """Conservative video duration for trim/xfade (avoid reading past EOF).

    Original Ken Burns / intro-teaser chunks often report a longer container
    duration than the video stream actually has — trimming to the max breaks
    xfade (0 frames → encoder EOF). Use the minimum trustworthy estimate.
    """
    data = _ffprobe_json(path)
    if not data:
        return 0.0

    estimates: list[float] = []
    for s in data.get("streams", []):
        if s.get("codec_type") != "video":
            continue
        d = float(s.get("duration") or 0)
        if d > 0:
            estimates.append(d)
        nbf = s.get("nb_frames")
        if nbf not in (None, "N/A", ""):
            try:
                fps = _parse_fps(s.get("avg_frame_rate") or s.get("r_frame_rate"))
                frame_est = int(nbf) / max(fps, 1.0)
                if frame_est > 0:
                    estimates.append(frame_est)
            except (TypeError, ValueError):
                pass

    fmt_d = float(data.get("format", {}).get("duration") or 0)
    if fmt_d > 0:
        estimates.append(fmt_d)

    if not estimates:
        for s in data.get("streams", []):
            d = float(s.get("duration") or 0)
            if d > 0:
                estimates.append(d)
        if fmt_d > 0:
            estimates.append(fmt_d)

    return min(estimates) if estimates else 0.0


def _has_audio_stream(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True,
            text=True,
            check=True,
        )
        return any(
            s.get("codec_type") == "audio"
            for s in json.loads(r.stdout).get("streams", [])
        )
    except Exception:
        return False


# ----------------------------------------------------------------- timings --
def _compute_transition_timings(
    left_dur: float,
    right_dur: float,
    hold_sec: float | None = None,
    xfade_sec: float | None = None,
) -> dict | None:
    """
    Outgoing tail → xfade → banner hold → xfade → incoming head.

    Video timeline extends by hold (left + hold + right). Audio is left + right
    with apad at the end — incoming voice starts when hold begins visually.

    hold_sec / xfade_sec come from the banner config (editor "Timing"); when
    absent the BANNER_* environment defaults apply, exactly as before.
    """
    frame = 1.0 / max(BANNER_FPS, 1)
    min_seg = max(0.25, 3.0 * frame)
    if left_dur < min_seg or right_dur < min_seg:
        return None

    xf_target = BANNER_XFADE_SEC if xfade_sec is None else max(0.1, min(float(xfade_sec), 1.6))
    cap = max(1.05, xf_target + 0.1)
    tail_target = max(BANNER_SEGMENT_TAIL_SEC, xf_target + 0.1) if xfade_sec is not None else BANNER_SEGMENT_TAIL_SEC
    head_target = max(BANNER_SEGMENT_HEAD_SEC, xf_target + 0.1) if xfade_sec is not None else BANNER_SEGMENT_HEAD_SEC

    tail = min(tail_target, left_dur * 0.4, cap, max(frame, left_dur - frame))
    head = min(head_target, right_dur * 0.4, cap, max(frame, right_dur - frame))
    tail = max(frame, tail)
    head = max(frame, head)
    hold = max(0.0, min(BANNER_HOLD_SEC if hold_sec is None else float(hold_sec), 4.0))
    xf = min(xf_target, tail * 0.98, head * 0.98, cap)
    xf = max(0.05, min(xf, tail, head))
    left_main = max(frame, left_dur - tail)
    tail = min(tail, left_dur - left_main)
    transition_window = tail + hold + head
    return {
        "tail": tail,
        "head": head,
        "hold": hold,
        "xfade": xf,
        "banner_dur": hold + 2.0 * xf,
        "left_main": left_main,
        "transition_window": transition_window,
        "video_total": left_dur + hold + right_dur,
    }


def _legacy_audio_filter(
    left_path: str,
    right_path: str,
    left_dur: float,
    right_dur: float,
    hold: float,
) -> tuple[str, bool]:
    left_a = _has_audio_stream(left_path)
    right_a = _has_audio_stream(right_path)

    if left_a and right_a:
        filt = (
            f"[0:a]atrim=0:{left_dur:.3f},asetpts=PTS-STARTPTS,aresample=48000[a0];"
            f"[2:a]atrim=0:{right_dur:.3f},asetpts=PTS-STARTPTS,aresample=48000[a1];"
            f"[a0][a1]concat=n=2:v=0:a=1[ac];"
        )
        if hold > 0.01:
            filt += f"[ac]apad=pad_dur={hold:.3f}[aout]"
        else:
            filt += "[ac]asetpts=PTS-STARTPTS[aout]"
        return filt, True
    if left_a:
        filt = (
            f"[0:a]atrim=0:{left_dur:.3f},asetpts=PTS-STARTPTS,aresample=48000[a0];"
        )
        if hold > 0.01:
            filt += f"[a0]apad=pad_dur={hold:.3f}[aout]"
        else:
            filt += "[a0]asetpts=PTS-STARTPTS[aout]"
        return filt, True
    if right_a:
        filt = (
            f"[2:a]atrim=0:{right_dur:.3f},asetpts=PTS-STARTPTS,aresample=48000[a0];"
        )
        if hold > 0.01:
            filt += f"[a0]apad=pad_dur={hold:.3f}[aout]"
        else:
            filt += "[a0]asetpts=PTS-STARTPTS[aout]"
        return filt, True
    return "", False


def _banner_video_norm(canvas_w: int, canvas_h: int, fps: int | None = None) -> str:
    """Single canvas + SAR + CFR — required for xfade and concat filter chains."""
    W, H = int(canvas_w), int(canvas_h)
    r = int(fps or BANNER_FPS)
    return (
        f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={r}:round=up,"
        f"setpts=PTS-STARTPTS,format=yuv420p"
    )


def _head_window(hold: float, head: float, right_dur: float, xf: float) -> tuple[float, float, float]:
    """Pick a head slice on the right chunk with enough frames for xfade."""
    head_start = min(hold, max(0.0, right_dur - 0.05))
    head_end = min(right_dur, head_start + head)
    if head_end - head_start < xf * 0.85:
        head_start = 0.0
        head_end = min(right_dur, head)
    if head_end - head_start < 1.0 / BANNER_FPS:
        head_start = 0.0
        head_end = min(right_dur, max(head, 1.0 / BANNER_FPS))
    right_body_start = head_end
    return head_start, head_end, right_body_start


def _build_banner_video_filter(
    *,
    left_dur: float,
    right_dur: float,
    need_banner: float,
    left_main: float,
    tail: float,
    head_start: float,
    head_end: float,
    right_body_start: float,
    trans_dur: float,
    xf: float,
    offset1: float,
    offset2: float,
    ffmpeg_transition: str,
    canvas_w: int,
    canvas_h: int,
) -> str:
    fps = BANNER_FPS
    norm = _banner_video_norm(canvas_w, canvas_h, fps)
    return (
        f"[0:v]{norm},split=2[v0a][v0b];"
        f"[v0a]trim=0:{left_main:.6f},setpts=PTS-STARTPTS[vleft];"
        f"[v0b]trim={left_main:.6f}:{left_dur:.6f},setpts=PTS-STARTPTS[vlt];"
        f"[1:v]{norm},trim=0:{need_banner:.6f},setpts=PTS-STARTPTS[vb];"
        f"[2:v]{norm},split=2[v2a][v2b];"
        f"[v2a]trim={head_start:.6f}:{head_end:.6f},setpts=PTS-STARTPTS[vrh];"
        f"[v2b]trim={right_body_start:.6f}:{right_dur:.6f},setpts=PTS-STARTPTS[vright];"
        f"[vlt][vb]xfade=transition={ffmpeg_transition}:duration={xf:.6f}:offset={offset1:.6f}[vm1];"
        f"[vm1][vrh]xfade=transition={ffmpeg_transition}:duration={xf:.6f}:offset={offset2:.6f}[vmid];"
        f"[vmid]trim=0:{trans_dur:.6f},setpts=PTS-STARTPTS[vmid_t];"
        # concat requires identical WxH, SAR, pix_fmt, and time base on every input
        f"[vleft]{norm}[vleft_c];"
        f"[vmid_t]{norm}[vmid_c];"
        f"[vright]{norm}[vright_c];"
        f"[vleft_c][vmid_c][vright_c]concat=n=3:v=1:a=0[vout]"
    )


def _run_banner_merge_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def _merge_three_with_transitions(
    left_path: str,
    banner_path: str,
    right_path: str,
    output_path: str,
    ffmpeg_transition: str,
    timings: dict | None = None,
    canvas_w: int = 1920,
    canvas_h: int = 1080,
) -> str | None:
    """
    Legacy splice merge — rich xfade banner, extended hold on video.

    Incoming head in the xfade-out leg is trimmed from hold→hold+head on the right
    chunk so each overlay frame matches the continuous audio clock at that moment.
    """
    left_dur = _probe_duration(left_path)
    right_dur = _probe_duration(right_path)
    banner_dur = _probe_duration(banner_path)
    if min(left_dur, right_dur, banner_dur) <= 0:
        logging.warning(
            f"Banner merge skipped — invalid chunk duration "
            f"(left={left_dur:.3f}s right={right_dur:.3f}s banner={banner_dur:.3f}s)"
        )
        return None

    t = timings if timings else _compute_transition_timings(left_dur, right_dur)
    if not t:
        logging.warning(
            f"Banner merge skipped — chunks too short for transition "
            f"(left={left_dur:.3f}s right={right_dur:.3f}s)"
        )
        return None

    tail = t["tail"]
    head = t["head"]
    hold = t["hold"]
    xf = t["xfade"]
    left_main = t["left_main"]
    trans_dur = t["transition_window"]
    need_banner = t["banner_dur"]
    offset1 = max(0.0, tail - xf)
    offset2 = max(0.0, tail + hold)
    head_start, head_end, right_body_start = _head_window(hold, head, right_dur, xf)

    vfilt = _build_banner_video_filter(
        left_dur=left_dur,
        right_dur=right_dur,
        need_banner=need_banner,
        left_main=left_main,
        tail=tail,
        head_start=head_start,
        head_end=head_end,
        right_body_start=right_body_start,
        trans_dur=trans_dur,
        xf=xf,
        offset1=offset1,
        offset2=offset2,
        ffmpeg_transition=ffmpeg_transition,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
    )

    afilt, has_audio = _legacy_audio_filter(left_path, right_path, left_dur, right_dur, hold)
    if has_audio:
        vfilt = f"{vfilt};{afilt}"

    def _cmd_for_transition(filt: str) -> list[str]:
        cmd = [
            "ffmpeg", "-y",
            "-i", left_path,
            "-i", banner_path,
            "-i", right_path,
            "-filter_complex", filt,
            "-map", "[vout]",
        ]
        if has_audio:
            cmd.extend(["-map", "[aout]", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k"])
        else:
            cmd.append("-an")
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ])
        return cmd

    transitions_to_try = [ffmpeg_transition]
    if ffmpeg_transition not in ("fade", "fadeblack"):
        transitions_to_try.append("fade")

    used_transition = ffmpeg_transition
    for trans_name in transitions_to_try:
        filt = (
            vfilt
            if trans_name == ffmpeg_transition
            else vfilt.replace(
                f"transition={ffmpeg_transition}",
                f"transition={trans_name}",
            )
        )
        cmd = _cmd_for_transition(filt)
        try:
            _run_banner_merge_cmd(cmd)
            used_transition = trans_name
            if trans_name != ffmpeg_transition:
                logging.warning(
                    f"Banner merge succeeded with fallback transition={trans_name} "
                    f"(requested={ffmpeg_transition})"
                )
            break
        except subprocess.CalledProcessError as exc:
            logging.error(
                f"Banner boundary merge failed ({trans_name}) | "
                f"left={os.path.basename(left_path)} ({left_dur:.3f}s) "
                f"right={os.path.basename(right_path)} ({right_dur:.3f}s) | "
                f"tail={tail:.3f} xf={xf:.3f} head={head_start:.3f}-{head_end:.3f}\n"
                f"{(exc.stderr or '')[-1200:]}"
            )
    else:
        return None

    if not os.path.exists(output_path):
        return None

    out_dur = _probe_duration(output_path)
    logging.info(
        f"✅ Banner legacy splice | left={left_dur:.2f}s hold={hold:.2f}s "
        f"right={right_dur:.2f}s | head_trim={head_start:.2f}→{head_end:.2f}s "
        f"| out={out_dur:.2f}s (expected≈{t['video_total']:.2f}s) | "
        f"transition={used_transition}"
    )
    return output_path


# ---------------------------------------------------------------- pipeline --
def build_concat_chunk_types(
    vid_segments: list[dict],
    batch_groups: dict | None = None,
    talking_heads_mode: bool = False,
    has_clips: bool = False,
) -> list[bool]:
    """One is_clip flag per final concat chunk (mirrors batch composite walk)."""
    if not vid_segments:
        return []

    if talking_heads_mode and has_clips and batch_groups:
        seg_to_batch = {}
        for bidx, g in batch_groups.items():
            for seg_idx in g.get("seg_indices", []):
                seg_to_batch[seg_idx] = bidx

        types: list[bool] = []
        i = 0
        n = len(vid_segments)
        while i < n:
            if _is_clip_segment(vid_segments[i]):
                types.append(True)
                i += 1
                continue
            bidx = seg_to_batch.get(i)
            if bidx is None:
                types.append(False)
                i += 1
                continue
            types.append(False)
            i += len(batch_groups[bidx].get("seg_indices", [i]))
        return types

    return [_is_clip_segment(seg) for seg in vid_segments]


def find_type_boundaries(chunk_types: list[bool]) -> list[int]:
    boundaries = []
    for i in range(len(chunk_types) - 1):
        if chunk_types[i] != chunk_types[i + 1]:
            boundaries.append(i)
    return boundaries


# ------------------------------------------------------- windowed splice --
BANNER_SPLICE_MODE = os.getenv("BANNER_SPLICE_MODE", "window").strip().lower()
BANNER_SPLICE_WORKERS = max(1, int(os.getenv("BANNER_SPLICE_WORKERS", "4")))
_AUDIO_SR = 48000

_NVENC_ARGS = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "constqp", "-qp", "18",
               "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "60"]
_X264_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-g", "60"]
_NVENC_DISABLED = False
_ENC_USED: dict = {}


def _splice_encoders() -> list:
    """GPU first (unless USE_NVENC=0 or it already failed), CPU fallback."""
    use_nv = os.getenv("USE_NVENC", "1").strip().lower() not in ("0", "false", "no", "off")
    if use_nv and not _NVENC_DISABLED:
        return [("h264_nvenc", _NVENC_ARGS), ("libx264", _X264_ARGS)]
    return [("libx264", _X264_ARGS)]


def _run_piece(build_cmd, label: str) -> bool:
    """Run one ffmpeg piece: NVENC, falling back to libx264 only if NVENC fails."""
    global _NVENC_DISABLED
    last = None
    for name, enc in _splice_encoders():
        r = subprocess.run(build_cmd(enc), capture_output=True, text=True)
        if r.returncode == 0:
            _ENC_USED[name] = _ENC_USED.get(name, 0) + 1
            return True
        last = r
        if name == "h264_nvenc":
            _NVENC_DISABLED = True
            logging.warning(
                f"⚠️ Banner splice: NVENC failed on {label} — continuing on libx264:\n"
                f"{(r.stderr or '')[-600:]}"
            )
    logging.error(f"Banner splice piece failed ({label}):\n{(last.stderr if last else '')[-800:]}")
    return False


def _decoded_audio_seconds(path: str) -> float:
    """Exact decoded length of the first audio stream (sample count / rate)."""
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1",
             "-ar", str(_AUDIO_SR), "-f", "s16le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        total = 0
        while True:
            buf = proc.stdout.read(1 << 20)
            if not buf:
                break
            total += len(buf)
        proc.wait()
        return (total // 2) / float(_AUDIO_SR)
    except Exception:
        return 0.0


def _chunk_lengths(path: str) -> dict:
    """
    v: picture length (legacy estimate), a: exact audio length,
    L: timeline length — ALWAYS a whole number of output frames (L_f / fps),
       never shorter than the picture or the audio (padding never cuts a sample).

    L must sit exactly on the frame grid: the picture of a chunk is L_f frames
    and its sound is L_f / fps seconds of samples. If L were left off-grid
    (e.g. a 29.97 fps clip, v*30 = 300.3), the picture got rounded to 300
    frames while the sound kept 10.01 s — ~10 ms lost per chunk, always the same
    direction, so lips slowly drifted away from the voice over the video.
    """
    fps = BANNER_FPS
    v = _probe_duration(path)
    has_a = _has_audio_stream(path)
    a = _decoded_audio_seconds(path) if has_a else 0.0
    L_f = max(1, int(math.ceil(v * fps - 1e-6)))
    if has_a and a > 0:
        L_f = max(L_f, int(math.floor(a * fps + 1e-6)) + 1)
    return {"v": v, "a": a, "L": L_f / fps, "L_f": L_f, "has_audio": has_a}


def _count_video_frames(path: str) -> int:
    """Number of video packets (= frames) in a file, without decoding."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=300,
        )
        return int((r.stdout or "0").strip().split(",")[0] or 0)
    except Exception:
        return -1


def _vpad(L: float, v: float) -> str:
    """Hold the last frame so the picture always reaches L (audio is never cut)."""
    return f"tpad=stop_mode=clone:stop_duration={max(0.0, L - v) + 1.0:.6f},"


def _encode_body_video(src: str, out: str, f0: int, f1: int, L: float, v: float,
                       W: int, H: int) -> str | None:
    """Picture only, frames [f0, f1) of the chunk on the 30 fps grid."""
    norm = _banner_video_norm(W, H)
    filt = (
        f"[0:v]{norm},{_vpad(L, v)}trim=start_frame={f0}:end_frame={f1},setpts=PTS-STARTPTS,"
        f"format=yuv420p[v]"
    )
    ok = _run_piece(
        lambda enc: ["ffmpeg", "-y", "-i", src, "-filter_complex", filt, "-map", "[v]", "-an",
                     *enc, out],
        f"body {os.path.basename(src)} frames {f0}-{f1}",
    )
    return out if ok and os.path.exists(out) else None


def _encode_window_video(left: str, banner: str, right: str, out: str, *, pl: dict,
                         transition: str, W: int, H: int) -> str | None:
    """Outgoing tail ↔ banner hold ↔ incoming head — legacy filter, picture only."""
    norm = _banner_video_norm(W, H)
    vf = (
        f"[0:v]{norm},{_vpad(pl['L_left'], pl['v_left'])}"
        f"trim=start_frame={pl['lm_f']}:end_frame={pl['Ll_f']},setpts=PTS-STARTPTS[vlt];"
        f"[1:v]{norm},trim=end_frame={pl['nb_f']},setpts=PTS-STARTPTS[vb];"
        f"[2:v]{norm},{_vpad(pl['L_right'], pl['v_right'])}"
        f"trim=start_frame={pl['hs_f']}:end_frame={pl['he_f']},setpts=PTS-STARTPTS[vrh];"
        f"[vlt][vb]xfade=transition=@T@:duration={pl['xf']:.6f}:offset={pl['offset1']:.6f}[vm1];"
        f"[vm1][vrh]xfade=transition=@T@:duration={pl['xf']:.6f}:offset={pl['offset2']:.6f}[vmid];"
        f"[vmid]trim=end_frame={pl['n_f']},setpts=PTS-STARTPTS,{norm}[vout]"
    )
    tries = [transition] + (["fade"] if transition not in ("fade", "fadeblack") else [])
    for t in tries:
        filt = vf.replace("@T@", t)
        ok = _run_piece(
            lambda enc, filt=filt: ["ffmpeg", "-y", "-i", left, "-i", banner, "-i", right,
                                    "-filter_complex", filt, "-map", "[vout]", "-an", *enc, out],
            f"window {os.path.basename(left)} ({t})",
        )
        if ok and os.path.exists(out):
            if t != transition:
                logging.warning(f"Banner window used fallback transition={t} (requested={transition})")
            return out
    return None


def _plan_join(Ll: float, vl: float, Lr: float, vr: float, timings: dict) -> dict:
    """
    Legacy splice numbers (_merge_three_with_transitions), snapped to whole
    frames so the picture length always equals the sound length (no drift).
    """
    fps = BANNER_FPS
    tail, head, hold, xf = timings["tail"], timings["head"], timings["hold"], timings["xfade"]
    head_start, head_end, _ = _head_window(hold, head, Lr, xf)
    Ll_f, Lr_f = int(round(Ll * fps)), int(round(Lr * fps))
    lm_f = min(Ll_f - 1, max(1, int(round(timings["left_main"] * fps))))
    tail_f = Ll_f - lm_f
    hold_f = int(round(hold * fps))
    xf_f = max(1, min(int(round(xf * fps)), tail_f))
    hs_f = int(round(head_start * fps))
    he_f = min(Lr_f, max(hs_f + xf_f, int(round(head_end * fps))))
    xf_s = xf_f / fps
    return {
        "L_left": Ll, "v_left": vl, "L_right": Lr, "v_right": vr,
        "Ll_f": Ll_f, "Lr_f": Lr_f, "lm_f": lm_f, "hs_f": hs_f, "he_f": he_f,
        "nb_f": hold_f + 2 * xf_f, "n_f": tail_f + hold_f + (he_f - hs_f),
        "hold_f": hold_f, "xf": xf_s,
        "offset1": max(0.0, (tail_f - xf_f) / fps), "offset2": (tail_f + hold_f) / fps,
    }


def _splice_group(paths, lens, group, plans, banner_path, transition, W, H, chunk_folder):
    """
    One run of adjacent joins (chunks group[0] .. group[-1]+1) → one file.
    Picture: pieces encoded in parallel, stream-copied together.
    Sound:   every chunk's full audio, back to back (+silence only where the
             picture is longer), encoded once while muxing.
    """
    from concurrent.futures import ThreadPoolExecutor

    first, last = group[0], group[-1] + 1
    tag = uuid.uuid4().hex[:8]
    jobs = []
    frames = []  # planned frame count of every picture piece, in order
    for k in range(first, last + 1):
        f0 = plans[k - 1]["he_f"] if (k - 1) in plans else 0
        f1 = plans[k]["lm_f"] if k in plans else lens[k]["L_f"]
        if f1 > f0:
            out = os.path.join(chunk_folder, f"bnbody_{k}_{tag}.mkv")
            jobs.append((_encode_body_video, (paths[k], out, f0, f1, lens[k]["L"], lens[k]["v"], W, H), {}))
            frames.append(f1 - f0)
        elif f1 < f0:
            logging.warning(f"Banner splice: chunk {k} shorter than its joins — body skipped")
        if k in plans:
            out = os.path.join(chunk_folder, f"bnwin_{k}_{tag}.mkv")
            jobs.append((_encode_window_video, (paths[k], banner_path, paths[k + 1], out),
                         dict(pl=plans[k], transition=transition, W=W, H=H)))
            frames.append(plans[k]["n_f"])

    with ThreadPoolExecutor(max_workers=BANNER_SPLICE_WORKERS) as ex:
        futs = [ex.submit(fn, *a, **kw) for fn, a, kw in jobs]
        pieces = []
        for f in futs:
            try:
                pieces.append(f.result())
            except Exception as exc:
                logging.error(f"Banner splice piece crashed: {exc}")
                pieces.append(None)

    lst = os.path.join(chunk_folder, f"bnlist_{tag}.txt")
    out = os.path.join(chunk_folder, f"banner_merge_{first}_{last}_{tag}.mp4")
    try:
        if not pieces or any(p is None for p in pieces):
            return None
        # every piece must hold exactly the frames we planned
        for p, nf in zip(pieces, frames):
            got = _count_video_frames(p)
            if got != nf:
                logging.error(
                    f"Banner splice: {os.path.basename(p)} has {got} frames, planned {nf} — "
                    f"refusing to join (would break lip sync)"
                )
                return None
        # exact durations: the concat demuxer must not guess piece lengths from
        # container metadata (an off-by-a-frame guess per piece adds up to drift)
        with open(lst, "w", encoding="utf-8") as f:
            for p, nf in zip(pieces, frames):
                f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")
                f.write(f"duration {nf / BANNER_FPS:.9f}\n")
        total_f = sum(frames)

        # sound: chunk audio in order, padded (never trimmed) to its timeline length;
        # where a join could not swap the head (very short chunk), silence keeps sync.
        inputs, parts, labels = [], [], []
        n = 0
        for k in range(first, last + 1):
            Lk = lens[k]["L"]
            if lens[k]["has_audio"]:
                inputs += ["-i", paths[k]]
                n += 1
                parts.append(
                    f"[{n}:a:0]aresample={_AUDIO_SR},aformat=sample_fmts=fltp:channel_layouts=stereo,"
                    f"apad,atrim=end_sample={int(round(Lk * _AUDIO_SR))},asetpts=PTS-STARTPTS[a{k}]"
                )
            else:
                parts.append(f"anullsrc=r={_AUDIO_SR}:cl=stereo,atrim=duration={Lk:.6f}[a{k}]")
            labels.append(f"[a{k}]")
            pl = plans.get(k)
            if pl and pl["hs_f"] < pl["hold_f"]:
                gap = (pl["hold_f"] - pl["hs_f"]) / BANNER_FPS
                parts.append(f"anullsrc=r={_AUDIO_SR}:cl=stereo,atrim=duration={gap:.6f}[g{k}]")
                labels.append(f"[g{k}]")
        parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aout]")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, *inputs,
               "-filter_complex", ";".join(parts),
               "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy",
               "-c:a", "aac", "-ar", str(_AUDIO_SR), "-ac", "2", "-b:a", "192k",
               "-movflags", "+faststart", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            logging.error(f"Banner join mux failed:\n{(r.stderr or '')[-800:]}")
            return None
        # sync guard: picture and sound of the joined file must end together
        got_f = _count_video_frames(out)
        v_len = got_f / BANNER_FPS if got_f > 0 else _probe_duration(out)
        a_len = _decoded_audio_seconds(out)
        want = total_f / BANNER_FPS
        if got_f != total_f or abs(a_len - want) > 1.5 / BANNER_FPS:
            logging.error(
                f"Banner join sync check failed for chunks {first}-{last}: "
                f"picture {v_len:.3f}s ({got_f}/{total_f} frames) vs sound {a_len:.3f}s — "
                f"using the legacy splice instead"
            )
            return None
        return out
    finally:
        for p in pieces:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        try:
            os.remove(lst)
        except OSError:
            pass


def _apply_banners_windowed(paths, boundaries, config, transition, W, H,
                            chunk_folder, work_dir, static_root):
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    t0 = _time.time()
    _ENC_USED.clear()
    involved = sorted(set(boundaries) | {b + 1 for b in boundaries})
    for k in involved:
        if not paths[k] or not os.path.exists(paths[k]):
            logging.warning(f"Banner windowed splice: chunk {k} missing")
            return None
    with ThreadPoolExecutor(max_workers=8) as ex:
        lens = dict(zip(involved, ex.map(lambda k: _chunk_lengths(paths[k]), involved)))

    plans = {}
    for b in boundaries:
        timings = _compute_transition_timings(
            lens[b]["L"], lens[b + 1]["L"],
            hold_sec=config.get("holdSec"), xfade_sec=config.get("transitionSec"),
        )
        if not timings:
            logging.warning(
                f"Banner boundary {b} skipped — chunks too short "
                f"(left={lens[b]['L']:.2f}s right={lens[b + 1]['L']:.2f}s)"
            )
            continue
        plans[b] = _plan_join(lens[b]["L"], lens[b]["v"], lens[b + 1]["L"], lens[b + 1]["v"], timings)
    if not plans:
        return list(paths)

    need = max(pl["nb_f"] for pl in plans.values()) / BANNER_FPS
    banner_path = render_banner_video(
        config=config, canvas_w=W, canvas_h=H,
        output_path=os.path.join(chunk_folder, f"banner_{uuid.uuid4().hex[:8]}.mp4"),
        work_dir=work_dir, duration_sec=need + 0.2, static_root=static_root,
    )
    if not banner_path:
        logging.warning("Banner render failed")
        return None

    groups = []
    for b in sorted(plans):
        if groups and groups[-1][-1] == b - 1:
            groups[-1].append(b)
        else:
            groups.append([b])
    layer_count = len([l for l in config.get("layers", []) if l.get("visible", True)])
    logging.info(
        f"🎬 Banner joins (fast splice) | joins={len(plans)} groups={len(groups)} | "
        f"layers={layer_count} | type={config.get('transitionType')} | workers={BANNER_SPLICE_WORKERS}"
    )

    out_paths = list(paths)
    try:
        for group in reversed(groups):
            g0 = _time.time()
            merged = _splice_group(out_paths, lens, group, plans, banner_path, transition, W, H, chunk_folder)
            if not merged:
                return None
            first, last = group[0], group[-1] + 1
            audio_in = sum(lens[k]["a"] for k in range(first, last + 1))
            logging.info(
                f"✅ Banner joins {group} -> {os.path.basename(merged)} | "
                f"video {_probe_duration(merged):.2f}s | audio kept {audio_in:.2f}s of {audio_in:.2f}s | "
                f"{_time.time() - g0:.1f}s"
            )
            out_paths[first:last + 1] = [merged]
    finally:
        try:
            os.remove(banner_path)
        except OSError:
            pass
    logging.info(
        f"🎬 Transition banner complete | applied={len(plans)}/{len(boundaries)} | "
        f"{_time.time() - t0:.1f}s | encoders={dict(_ENC_USED)}"
    )
    return out_paths


def apply_banner_boundary_transitions(
    chunk_paths: list[str],
    vid_segments: list[dict],
    banner_state: dict,
    canvas_w: int,
    canvas_h: int,
    chunk_folder: str,
    talking_heads_mode: bool = False,
    has_clips: bool = False,
    batch_groups: dict | None = None,
    chunk_types: list[bool] | None = None,
    static_root: str | None = None,
) -> list[str]:
    if not banner_state.get("enabled"):
        return chunk_paths

    config = banner_state.get("config") or normalize_banner_config(None)
    transition = ffmpeg_transition_name(config.get("transitionType"))

    if chunk_types is None:
        chunk_types = build_concat_chunk_types(
            vid_segments,
            batch_groups=batch_groups,
            talking_heads_mode=talking_heads_mode,
            has_clips=has_clips,
        )

    if len(chunk_types) != len(chunk_paths):
        logging.warning(
            f"Banner skipped — chunk type count ({len(chunk_types)}) != "
            f"chunk path count ({len(chunk_paths)})"
        )
        return chunk_paths

    boundaries = find_type_boundaries(chunk_types)
    if not boundaries:
        logging.info("Transition banner enabled — no narration↔CLIP boundaries found")
        return chunk_paths

    work_dir = os.path.join(chunk_folder, "banner_work")
    os.makedirs(work_dir, exist_ok=True)
    paths = list(chunk_paths)
    applied = 0

    if BANNER_SPLICE_MODE != "legacy":
        fast = _apply_banners_windowed(
            paths, boundaries, config, transition, canvas_w, canvas_h,
            chunk_folder, work_dir, static_root,
        )
        if fast is not None:
            return fast
        logging.warning("⚠️ Fast banner splice failed — using legacy splice")

    logging.info(
        f"🎬 Transition banner | boundaries={boundaries} | transition={transition} | "
        f"tail/head≈{BANNER_SEGMENT_TAIL_SEC}/{BANNER_SEGMENT_HEAD_SEC}s | "
        f"hold≈{config.get('holdSec', BANNER_HOLD_SEC)}s | "
        f"xfade≈{config.get('transitionSec', BANNER_XFADE_SEC)}s | motion={config.get('motion', 'none')}"
    )

    for idx in reversed(boundaries):
        left = paths[idx]
        right = paths[idx + 1]
        if not left or not right or not os.path.exists(left) or not os.path.exists(right):
            logging.warning(f"Banner boundary {idx} skipped — missing chunk")
            continue

        left_dur = _probe_duration(left)
        right_dur = _probe_duration(right)
        timings = _compute_transition_timings(
            left_dur,
            right_dur,
            hold_sec=config.get("holdSec"),
            xfade_sec=config.get("transitionSec"),
        )
        if not timings:
            logging.warning(
                f"Banner boundary {idx} skipped — chunks too short "
                f"(left={left_dur:.2f}s right={right_dur:.2f}s)"
            )
            continue

        banner_out = os.path.join(
            chunk_folder,
            f"banner_{idx}_{uuid.uuid4().hex[:8]}.mp4",
        )
        banner_path = render_banner_video(
            config=config,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            output_path=banner_out,
            work_dir=work_dir,
            duration_sec=timings["banner_dur"] + 0.2,
            static_root=static_root,
        )
        if not banner_path:
            logging.warning(f"Banner render failed at boundary {idx}")
            continue

        layer_count = len([l for l in config.get("layers", []) if l.get("visible", True)])
        logging.info(
            f"🎬 Boundary {idx} banner | layers={layer_count} | "
            f"type={config.get('transitionType')} | dur={timings['banner_dur']:.2f}s"
        )
        logging.info(
            f"🎬 Boundary {idx} | tail={timings['tail']:.2f}s "
            f"hold={timings['hold']:.2f}s head={timings['head']:.2f}s "
            f"xfade={timings['xfade']:.2f}s "
            f"(video window={timings['transition_window']:.2f}s)"
        )

        merged_out = os.path.join(
            chunk_folder,
            f"banner_merge_{idx}_{uuid.uuid4().hex[:8]}.mp4",
        )
        merged = _merge_three_with_transitions(
            left_path=left,
            banner_path=banner_path,
            right_path=right,
            output_path=merged_out,
            ffmpeg_transition=transition,
            timings=timings,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )
        if not merged:
            logging.warning(f"Banner merge failed at boundary {idx}")
            continue

        paths[idx] = merged
        del paths[idx + 1]
        applied += 1
        logging.info(f"✅ Banner transition applied at boundary {idx}")

    logging.info(f"🎬 Transition banner complete | applied={applied}/{len(boundaries)}")
    return paths