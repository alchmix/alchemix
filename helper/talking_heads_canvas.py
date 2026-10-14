"""
Talking Heads canvas designer — backend renderer (layout_config style_version 2).

The Talking Heads Studio (frontend) can send a *designed* canvas on top of the
classic layout boxes.  Everything here is static per render, so it is drawn
once with PIL into a few PNGs and composited by ffmpeg:

    base (colour / blurred / image / video, built by the compositor)
      └─ UNDER png   gradient or solid bg, tint, pattern, vignette,
                     decorations placed "under", drop shadows of the frames
      └─ content / avatar / clip video  (alpha-masked → real rounded / circle corners)
      └─ OVER png    frame borders (solid / dashed / dotted / double),
                     decorations placed "over"

Payload additions (all optional, ignored by older renderers):

    layout_config = {
      ...legacy keys (layout_id, canvas, avatar_layer, content_layer, clip_layer,
                      background, content_fit, clip_fit)...,
      "style_version": 2,
      "canvas_id": "studio_split",
      "design": {
        "background": {
          "type": "solid" | "gradient" | "blurred" | "image" | "video",
          "color": "#0b1020",
          "gradient": {"type": "linear" | "radial", "angle": 135, "from": "#..", "to": "#.."},
          "url": "https://…"            (image / video),
          "tint": {"color": "#000000", "opacity": 0.35},
          "pattern": {"kind": "none" | "grid" | "dots" | "diagonal" | "scanlines",
                      "color": "#ffffff", "opacity": 0.06, "size": 64, "thickness": 2},
          "vignette": 0.0 – 1.0
        },
        "decor": [ banner-schema layers (fill / gradient / shape / text / image)
                   + "placement": "under" | "over", "scope": "both" | "narration" | "clip" ]
      }
    }
    each *_layer may carry:
      "frame":  {"width": px, "color": "#fff", "radius": px, "style": "solid|dashed|dotted|double",
                 "shape": "rect" | "circle", "opacity": 1}
      "shadow": {"enabled": true, "blur": px, "x": px, "y": px, "spread": px,
                 "color": "#000000", "opacity": 0.45}

All px values are canvas pixels (no preview scaling).  The browser preview
(ThCanvasScene.jsx) draws the same primitives with the same maths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os

from PIL import Image, ImageDraw, ImageFilter

try:  # numpy is used for exact CSS gradient / pattern maths
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from helper.banner_backend import (
    _apply_opacity,
    _gradient_patch,
    _hex_to_rgba,
    _image_patch,
    _layer_geometry,
    _num,
    _place_patch,
    _shape_patch,
    _text_patch,
)

STYLE_VERSION = 2
SUPERSAMPLE = 2
LAYER_KEYS = ("avatar", "content", "clip")


# ─────────────────────────────────────────────────────────────── helpers ──
def is_v2(layout_config) -> bool:
    return (
        isinstance(layout_config, dict)
        and int(_num(layout_config.get("style_version"), 1)) >= STYLE_VERSION
        and isinstance(layout_config.get("design"), dict)
    )


def _hex(color, default="#000000") -> str:
    c = str(color or "").strip()
    if len(c) == 7 and c.startswith("#"):
        try:
            int(c[1:], 16)
            return c
        except ValueError:
            pass
    return default


def _clamp01(v, default=1.0) -> float:
    return max(0.0, min(1.0, _num(v, default)))


def layer_box(layer) -> tuple[int, int, int, int] | None:
    if not isinstance(layer, dict):
        return None
    w = int(round(_num(layer.get("width"), 0)))
    h = int(round(_num(layer.get("height"), 0)))
    if w < 2 or h < 2:
        return None
    return int(round(_num(layer.get("x"), 0))), int(round(_num(layer.get("y"), 0))), w, h


def frame_of(layer) -> dict:
    f = (layer or {}).get("frame") or {}
    return {
        "width": max(0.0, _num(f.get("width"), 0.0)),
        "color": _hex(f.get("color"), "#ffffff"),
        "radius": max(0.0, _num(f.get("radius"), 0.0)),
        "style": str(f.get("style") or "solid").lower(),
        "shape": "circle" if str(f.get("shape") or "rect").lower() == "circle" else "rect",
        "opacity": _clamp01(f.get("opacity"), 1.0),
    }


def shadow_of(layer) -> dict | None:
    s = (layer or {}).get("shadow") or {}
    if not s or s.get("enabled") is False:
        return None
    if not s.get("enabled"):
        return None
    op = _clamp01(s.get("opacity"), 0.45)
    if op <= 0:
        return None
    return {
        "blur": max(0.0, _num(s.get("blur"), 40.0)),
        "x": _num(s.get("x"), 0.0),
        "y": _num(s.get("y"), 16.0),
        "spread": _num(s.get("spread"), 0.0),
        "color": _hex(s.get("color"), "#000000"),
        "opacity": op,
    }


def corner_radius(frame: dict, w: int, h: int) -> float:
    """Effective outer corner radius in px (CSS clamps to half the short side)."""
    if frame["shape"] == "circle":
        return min(w, h) / 2.0
    return max(0.0, min(frame["radius"], w / 2.0, h / 2.0))


def needs_mask(layer) -> bool:
    if not isinstance(layer, dict) or not layer.get("frame"):
        return False
    fr = frame_of(layer)
    return fr["shape"] == "circle" or fr["radius"] >= 0.5


def _shape_mask(w: int, h: int, frame: dict, ss: int = SUPERSAMPLE) -> Image.Image:
    """Anti-aliased L mask of the frame shape (ellipse for circle frames)."""
    W, H = max(1, w * ss), max(1, h * ss)
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    if frame["shape"] == "circle":
        d.ellipse((0, 0, W - 1, H - 1), fill=255)
    else:
        r = corner_radius(frame, w, h) * ss
        if r >= 1:
            d.rounded_rectangle((0, 0, W - 1, H - 1), radius=int(round(r)), fill=255)
        else:
            d.rectangle((0, 0, W - 1, H - 1), fill=255)
    return m.resize((max(1, w), max(1, h)), Image.LANCZOS) if ss > 1 else m


# ────────────────────────────────────────────────────────── background ──
def _bg_arrays(w: int, h: int):
    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.arange(h, dtype=np.float32) + 0.5
    return np.meshgrid(xs, ys)


def _pattern_layer(bg: dict, w: int, h: int) -> Image.Image | None:
    p = bg.get("pattern") or {}
    kind = str(p.get("kind") or "none").lower()
    if kind == "none" or np is None:
        return None
    op = _clamp01(p.get("opacity"), 0.06)
    if op <= 0:
        return None
    size = max(4.0, _num(p.get("size"), 64.0))
    t = max(0.5, _num(p.get("thickness"), 2.0))
    X, Y = _bg_arrays(w, h)
    if kind == "grid":
        mask = (np.mod(X, size) < t) | (np.mod(Y, size) < t)
    elif kind == "scanlines":
        mask = np.mod(Y, size) < t
    elif kind == "dots":
        cx = np.mod(X, size) - size / 2.0
        cy = np.mod(Y, size) - size / 2.0
        mask = (cx * cx + cy * cy) <= (t * 1.5) ** 2
    elif kind == "diagonal":
        # repeating-linear-gradient(45deg, c 0 t, transparent t size)
        a = math.radians(45.0)
        sa, ca = math.sin(a), math.cos(a)
        length = abs(w * sa) + abs(h * ca)
        pos = (X - w / 2.0) * sa - (Y - h / 2.0) * ca + length / 2.0
        mask = np.mod(pos, size) < t
    else:
        return None
    r, g, b, _ = _hex_to_rgba(_hex(p.get("color"), "#ffffff"))
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = r, g, b
    arr[..., 3] = (mask * round(255 * op)).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _vignette_layer(strength: float, w: int, h: int) -> Image.Image | None:
    """radial-gradient(ellipse farthest-corner, transparent 55%, rgba(0,0,0,s) 100%)."""
    s = _clamp01(strength, 0.0)
    if s <= 0 or np is None:
        return None
    X, Y = _bg_arrays(w, h)
    rx = w / 2.0 * math.sqrt(2)
    ry = h / 2.0 * math.sqrt(2)
    t = np.sqrt(((X - w / 2.0) / rx) ** 2 + ((Y - h / 2.0) / ry) ** 2)
    a = np.clip((t - 0.55) / 0.45, 0.0, 1.0) * s
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 3] = np.round(a * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def background_spec(design: dict) -> dict:
    bg = (design or {}).get("background") or {}
    btype = str(bg.get("type") or "blurred").lower()
    if btype not in ("solid", "gradient", "blurred", "image", "video"):
        btype = "blurred"
    return {**bg, "type": btype}


def paints_opaque_background(design: dict) -> bool:
    return background_spec(design)["type"] in ("solid", "gradient")


def _draw_background(canvas: Image.Image, design: dict) -> None:
    w, h = canvas.size
    bg = background_spec(design)
    if bg["type"] == "solid":
        canvas.alpha_composite(Image.new("RGBA", (w, h), _hex_to_rgba(_hex(bg.get("color"), "#0b1020"))))
    elif bg["type"] == "gradient":
        g = bg.get("gradient") or {}
        canvas.alpha_composite(_gradient_patch({
            "gradientFrom": _hex(g.get("from"), "#0f172a"),
            "gradientTo": _hex(g.get("to"), "#1e3a8a"),
            "gradientType": "radial" if str(g.get("type")).lower() == "radial" else "linear",
            "gradientAngle": _num(g.get("angle"), 135.0),
        }, w, h))
    tint = bg.get("tint") or {}
    t_op = _clamp01(tint.get("opacity"), 0.0)
    if t_op > 0 and bg["type"] in ("blurred", "image", "video"):
        canvas.alpha_composite(Image.new("RGBA", (w, h), _hex_to_rgba(_hex(tint.get("color"), "#000000"), t_op)))
    pat = _pattern_layer(bg, w, h)
    if pat is not None:
        canvas.alpha_composite(pat)
    vig = _vignette_layer(bg.get("vignette", 0), w, h)
    if vig is not None:
        canvas.alpha_composite(vig)


# ────────────────────────────────────────────────────────────── shadows ──
def _draw_shadow(canvas: Image.Image, box, frame: dict, shadow: dict) -> None:
    """CSS box-shadow: offset, spread, blur (σ = blur / 2)."""
    x, y, w, h = box
    spread = shadow["spread"]
    sw, sh = w + 2 * spread, h + 2 * spread
    if sw < 1 or sh < 1:
        return
    sigma = shadow["blur"] / 2.0
    pad = int(math.ceil(sigma * 3)) + 2
    pw, ph = int(round(sw)) + 2 * pad, int(round(sh)) + 2 * pad
    mask = Image.new("L", (pw, ph), 0)
    d = ImageDraw.Draw(mask)
    rect = (pad, pad, pad + int(round(sw)) - 1, pad + int(round(sh)) - 1)
    if frame["shape"] == "circle":
        d.ellipse(rect, fill=255)
    else:
        r = corner_radius(frame, w, h)
        r = max(0.0, r + spread) if r > 0 else 0.0
        if r >= 1:
            d.rounded_rectangle(rect, radius=int(round(min(r, sw / 2, sh / 2))), fill=255)
        else:
            d.rectangle(rect, fill=255)
    if sigma > 0.3:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=sigma))
    col = _hex_to_rgba(shadow["color"])
    patch = Image.new("RGBA", (pw, ph), col[:3] + (0,))
    patch.putalpha(mask.point(lambda p: int(round(p * shadow["opacity"]))))
    ox = int(round(x - spread + shadow["x"])) - pad
    oy = int(round(y - spread + shadow["y"])) - pad
    _paste(canvas, patch, ox, oy)


def _paste(canvas: Image.Image, patch: Image.Image, x: int, y: int) -> None:
    cw, ch = canvas.size
    pw, ph = patch.size
    l, t = max(0, x), max(0, y)
    r, b = min(cw, x + pw), min(ch, y + ph)
    if r <= l or b <= t:
        return
    canvas.alpha_composite(patch.crop((l - x, t - y, r - x, b - y)), (l, t))


# ─────────────────────────────────────────────────────────────── frames ──
def _path_points(cx0, cy0, cw, ch, r, circle, step=0.5):
    """Centre-line path, clockwise, starting where SVG <rect>/<ellipse> starts.

    rect  : (x + rx, y) → right along the top edge
    ellipse: (cx + rx, cy) → down (increasing angle)
    Returns [(x, y, cumulative_length), …].
    """
    pts = []
    if circle:
        rx, ry = cw / 2.0, ch / 2.0
        cx, cy = cx0 + rx, cy0 + ry
        perim = math.pi * (3 * (rx + ry) - math.sqrt((3 * rx + ry) * (rx + 3 * ry)))
        n = max(32, int(perim / step))
        prev = None
        total = 0.0
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            p = (cx + rx * math.cos(a), cy + ry * math.sin(a))
            if prev is not None:
                total += math.hypot(p[0] - prev[0], p[1] - prev[1])
            pts.append((p[0], p[1], total))
            prev = p
        return pts

    r = max(0.0, min(r, cw / 2.0, ch / 2.0))
    x0, y0, x1, y1 = cx0, cy0, cx0 + cw, cy0 + ch
    segs = []
    # top edge, TR arc, right edge, BR arc, bottom edge, BL arc, left edge, TL arc
    segs.append(("line", (x0 + r, y0), (x1 - r, y0)))
    segs.append(("arc", (x1 - r, y0 + r), -90, 0))
    segs.append(("line", (x1, y0 + r), (x1, y1 - r)))
    segs.append(("arc", (x1 - r, y1 - r), 0, 90))
    segs.append(("line", (x1 - r, y1), (x0 + r, y1)))
    segs.append(("arc", (x0 + r, y1 - r), 90, 180))
    segs.append(("line", (x0, y1 - r), (x0, y0 + r)))
    segs.append(("arc", (x0 + r, y0 + r), 180, 270))
    total = 0.0
    prev = None
    for seg in segs:
        if seg[0] == "line":
            (ax, ay), (bx, by) = seg[1], seg[2]
            L = math.hypot(bx - ax, by - ay)
            n = max(1, int(L / step))
            rng = range(n + 1)
            for i in rng:
                p = (ax + (bx - ax) * i / n, ay + (by - ay) * i / n)
                if prev is not None:
                    total += math.hypot(p[0] - prev[0], p[1] - prev[1])
                pts.append((p[0], p[1], total))
                prev = p
        else:
            if r < 0.01:
                continue
            (ccx, ccy), a0, a1 = seg[1], seg[2], seg[3]
            L = math.radians(a1 - a0) * r
            n = max(2, int(L / step))
            for i in range(n + 1):
                a = math.radians(a0 + (a1 - a0) * i / n)
                p = (ccx + r * math.cos(a), ccy + r * math.sin(a))
                if prev is not None:
                    total += math.hypot(p[0] - prev[0], p[1] - prev[1])
                pts.append((p[0], p[1], total))
                prev = p
    return pts


def _stroke_path(draw, pts, width, fill, dash=None, round_dots=False):
    """Draw a polyline path with an optional SVG-style dasharray."""
    if not pts:
        return
    if not dash:
        draw.line([(p[0], p[1]) for p in pts], fill=fill, width=max(1, int(round(width))), joint="curve")
        return
    on, off = dash
    period = on + off
    total = pts[-1][2]
    if round_dots:  # dasharray "0 gap" + round caps → dots
        s = 0.0
        rr = width / 2.0
        j = 0
        while s <= total + 1e-6:
            while j < len(pts) - 1 and pts[j + 1][2] < s:
                j += 1
            p0 = pts[j]
            p1 = pts[min(j + 1, len(pts) - 1)]
            seg = max(1e-6, p1[2] - p0[2])
            f = max(0.0, min(1.0, (s - p0[2]) / seg))
            x = p0[0] + (p1[0] - p0[0]) * f
            y = p0[1] + (p1[1] - p0[1]) * f
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=fill)
            s += period
        return
    current = []
    for p in pts:
        phase = math.fmod(p[2], period)
        if phase < on:
            current.append((p[0], p[1]))
        elif current:
            if len(current) > 1:
                draw.line(current, fill=fill, width=max(1, int(round(width))), joint="curve")
            current = []
    if len(current) > 1:
        draw.line(current, fill=fill, width=max(1, int(round(width))), joint="curve")


def frame_strokes(frame: dict, w: float, h: float):
    """[(inset, stroke_width, dash, round_dots)] — shared with the SVG preview."""
    bw = min(frame["width"], w / 2.0, h / 2.0)
    if bw <= 0:
        return []
    style = frame["style"]
    if style == "double" and bw >= 3:
        t = bw / 3.0
        return [(t / 2.0, t, None, False), (bw - t / 2.0, t, None, False)]
    if style == "dashed":
        return [(bw / 2.0, bw, (3 * bw, 2 * bw), False)]
    if style == "dotted":
        return [(bw / 2.0, bw, (0.0, 2 * bw), True)]
    return [(bw / 2.0, bw, None, False)]


def _draw_frame(canvas: Image.Image, box, frame: dict) -> None:
    x, y, w, h = box
    if frame["width"] <= 0 or frame["opacity"] <= 0:
        return
    ss = SUPERSAMPLE
    pad = 2
    pw, ph = (w + 2 * pad) * ss, (h + 2 * pad) * ss
    layer = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    col = _hex_to_rgba(frame["color"], frame["opacity"])
    r_outer = corner_radius(frame, w, h)
    circle = frame["shape"] == "circle"
    for inset, sw, dash, dots in frame_strokes(frame, w, h):
        cw, ch = w - 2 * inset, h - 2 * inset
        if cw <= 0 or ch <= 0:
            continue
        if not dash and not circle:
            # solid ring = outer rounded rect minus inner rounded rect (crisp, exact)
            ox0 = (pad + inset - sw / 2.0) * ss
            oy0 = (pad + inset - sw / 2.0) * ss
            ow, oh = (cw + sw) * ss, (ch + sw) * ss
            ro = max(0.0, r_outer - (inset - sw / 2.0)) * ss
            ri = max(0.0, r_outer - (inset + sw / 2.0)) * ss
            ring = Image.new("L", (pw, ph), 0)
            rd = ImageDraw.Draw(ring)
            box_o = (ox0, oy0, ox0 + ow - 1, oy0 + oh - 1)
            box_i = (ox0 + sw * ss, oy0 + sw * ss, ox0 + ow - 1 - sw * ss, oy0 + oh - 1 - sw * ss)
            _rr(rd, box_o, ro, 255)
            if box_i[2] > box_i[0] and box_i[3] > box_i[1]:
                _rr(rd, box_i, ri, 0)
            solid = Image.new("RGBA", (pw, ph), col)
            solid.putalpha(ring.point(lambda p: int(p * col[3] / 255)))
            layer.alpha_composite(solid)
            continue
        if not dash and circle:
            ring = Image.new("L", (pw, ph), 0)
            rd = ImageDraw.Draw(ring)
            o = (pad + inset - sw / 2.0) * ss
            rd.ellipse((o, o, o + (cw + sw) * ss - 1, o + (ch + sw) * ss - 1), fill=255)
            i = (pad + inset + sw / 2.0) * ss
            if cw - sw > 0 and ch - sw > 0:
                rd.ellipse((i, i, i + (cw - sw) * ss - 1, i + (ch - sw) * ss - 1), fill=0)
            solid = Image.new("RGBA", (pw, ph), col)
            solid.putalpha(ring.point(lambda p: int(p * col[3] / 255)))
            layer.alpha_composite(solid)
            continue
        rc = max(0.0, r_outer - inset)
        pts = _path_points((pad + inset) * ss, (pad + inset) * ss, cw * ss, ch * ss, rc * ss, circle, step=0.75)
        dash_ss = (dash[0] * ss, dash[1] * ss)
        tmp = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        _stroke_path(ImageDraw.Draw(tmp), pts, sw * ss, col[:3] + (255,), dash_ss, dots)
        if col[3] < 255:
            tmp = _apply_opacity(tmp, col[3] / 255.0)
        layer.alpha_composite(tmp)
    layer = layer.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)
    _paste(canvas, layer, x - pad, y - pad)


def _rr(draw, box, r, fill):
    x0, y0, x1, y1 = box
    r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    if r >= 1:
        draw.rounded_rectangle(box, radius=int(round(r)), fill=fill)
    else:
        draw.rectangle(box, fill=fill)


# ──────────────────────────────────────────────────────────── decor ──
def _decor_layers(design: dict, placement: str, mode: str) -> list[dict]:
    out = []
    for layer in (design or {}).get("decor") or []:
        if not isinstance(layer, dict) or layer.get("visible") is False:
            continue
        if str(layer.get("placement") or "over").lower() != placement:
            continue
        scope = str(layer.get("scope") or "both").lower()
        if scope != "both" and scope != mode:
            continue
        out.append(layer)
    return sorted(out, key=lambda l: _num(l.get("zIndex", 0), 0.0))


def _draw_decor(canvas: Image.Image, layers: list[dict], work_dir: str, static_root: str | None) -> None:
    cw, ch = canvas.size
    for layer in layers:
        try:
            opacity = _clamp01(layer.get("opacity", 1), 1.0)
            if opacity <= 0:
                continue
            ltype = layer.get("type")
            geom = _layer_geometry(layer, cw, ch)
            w = max(1, int(round(geom[2])))
            h = max(1, int(round(geom[3])))
            ox = oy = 0
            if ltype == "fill":
                patch = Image.new("RGBA", (w, h), _hex_to_rgba(layer.get("color", "#0f172a"), 1.0))
            elif ltype == "gradient":
                patch = _gradient_patch(layer, w, h)
            elif ltype == "shape":
                patch = _shape_patch(layer, w, h, ch)
            elif ltype == "image":
                patch = _image_patch(layer, w, h, work_dir, static_root, min(cw, ch) / 1080.0)
                if patch is None:
                    continue
            elif ltype == "text":
                res = _text_patch(layer, w, h, ch)
                if not res:
                    continue
                patch, ox, oy = res
            else:
                continue
            _place_patch(canvas, _apply_opacity(patch, opacity), geom, ox, oy)
        except Exception as exc:
            logging.warning(f"Canvas decor skipped ({layer.get('type', '?')}): {exc}")


# ──────────────────────────────────────────────────────── scene assets ──
def mode_layers(layout_config: dict, mode: str) -> dict:
    """Which boxes are visible in a scene mode → {key: layer dict}."""
    lid = layout_config.get("layout_id", "")
    if mode == "clip":
        cl = layout_config.get("clip_layer")
        return {"clip": cl} if layer_box(cl) else {}
    out = {}
    if lid != "full_avatar" and layer_box(layout_config.get("content_layer")):
        out["content"] = layout_config["content_layer"]
    if lid != "full_content" and layer_box(layout_config.get("avatar_layer")):
        out["avatar"] = layout_config["avatar_layer"]
    return out


def render_scene_pngs(
    layout_config: dict,
    canvas_w: int,
    canvas_h: int,
    mode: str,
    work_dir: str,
    static_root: str | None = None,
) -> dict:
    """Render UNDER / OVER pngs + per-layer masks for one scene mode (cached)."""
    os.makedirs(work_dir, exist_ok=True)
    design = layout_config.get("design") or {}
    layers = mode_layers(layout_config, mode)
    key_src = json.dumps(
        {"d": design, "l": layers, "m": mode, "w": canvas_w, "h": canvas_h, "v": 3},
        sort_keys=True, default=str,
    )
    key = hashlib.md5(key_src.encode("utf-8")).hexdigest()[:16]
    under_p = os.path.join(work_dir, f"thc_under_{mode}_{key}.png")
    over_p = os.path.join(work_dir, f"thc_over_{mode}_{key}.png")
    masks = {}

    for k, layer in layers.items():
        if not needs_mask(layer):
            continue
        x, y, w, h = layer_box(layer)
        # yuv420 filters (crop / pad) round odd boxes down to even sizes, so
        # the mask is built at that size and the layer is pinned to it.
        ew, eh = max(2, w - (w % 2)), max(2, h - (h % 2))
        mp = os.path.join(work_dir, f"thc_mask_{k}_{ew}x{eh}_{key}.png")
        if not os.path.exists(mp):
            _shape_mask(ew, eh, frame_of(layer)).save(mp)
        masks[k] = {"path": mp, "w": ew, "h": eh}

    if not (os.path.exists(under_p) and os.path.exists(over_p)):
        under = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        _draw_background(under, design)
        _draw_decor(under, _decor_layers(design, "under", mode), work_dir, static_root)
        for k in ("content", "avatar", "clip"):
            layer = layers.get(k)
            if not layer:
                continue
            sh = shadow_of(layer)
            if sh:
                _draw_shadow(under, layer_box(layer), frame_of(layer), sh)
        under.save(under_p)

        over = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        for k in ("content", "avatar", "clip"):
            layer = layers.get(k)
            if layer:
                _draw_frame(over, layer_box(layer), frame_of(layer))
        _draw_decor(over, _decor_layers(design, "over", mode), work_dir, static_root)
        over.save(over_p)

    under_img_alpha = Image.open(under_p).getchannel("A").getextrema()
    over_img_alpha = Image.open(over_p).getchannel("A").getextrema()
    return {
        "under": under_p if under_img_alpha[1] > 0 else None,
        "over": over_p if over_img_alpha[1] > 0 else None,
        "masks": masks,
        "opaque_bg": paints_opaque_background(design),
    }


# ───────────────────────────────────────────────────── ffmpeg glue ──
class CanvasFx:
    """Injects the designed canvas into an existing compositor filtergraph.

    Usage inside a compositor (after its own inputs are known):

        fx = CanvasFx.for_layout(layout_config, canvas_w, canvas_h, "narration",
                                 chunk_folder, first_input=count_inputs(inputs))
        if fx:
            current = fx.after_base(parts, current)
            av_label = fx.mask_layer(parts, "avatar", "av")   # before overlay
            ...
            current = fx.finish(parts, current)
            inputs += fx.input_args()
    """

    def __init__(self, assets: dict, first_input: int):
        self.assets = assets
        self._files: list[str] = []
        self._idx: dict[str, int] = {}
        for name in ("under", "over"):
            if assets.get(name):
                self._idx[name] = first_input + len(self._files)
                self._files.append(assets[name])
        self._mask_size: dict[str, tuple[int, int]] = {}
        for k, m in (assets.get("masks") or {}).items():
            self._idx[f"mask_{k}"] = first_input + len(self._files)
            self._files.append(m["path"])
            self._mask_size[k] = (int(m["w"]), int(m["h"]))

    @classmethod
    def for_layout(cls, layout_config, canvas_w, canvas_h, mode, work_dir, first_input,
                   static_root=None):
        if not is_v2(layout_config):
            return None
        try:
            assets = render_scene_pngs(layout_config, int(canvas_w), int(canvas_h), mode, work_dir, static_root)
        except Exception as exc:
            logging.warning(f"Canvas design render failed — falling back to classic layout: {exc}")
            return None
        logging.info(
            f"🎨 Canvas design ({mode}) | under={bool(assets['under'])} over={bool(assets['over'])} "
            f"masks={sorted(assets['masks'])}"
        )
        return cls(assets, first_input)

    # skip legacy drawbox borders — frames are drawn in the OVER png
    draws_borders = True

    def input_args(self) -> list[str]:
        # single still frames: overlay / alphamerge repeat the last secondary
        # frame until the main stream ends, so no duration bookkeeping is needed
        args: list[str] = []
        for p in self._files:
            args += ["-i", p]
        return args

    def after_base(self, parts: list[str], current: str) -> str:
        if "under" not in self._idx:
            return current
        parts.append(
            f"[{current}][{self._idx['under']}:v]overlay=0:0:format=auto,format=yuv420p[thc_base]"
        )
        return "thc_base"

    def mask_layer(self, parts: list[str], key: str, label: str) -> str:
        """Apply the rounded/circle alpha mask to an already-sized layer label."""
        name = f"mask_{key}"
        if name not in self._idx:
            return label
        safe = "".join(ch if ch.isalnum() else "_" for ch in label)
        out = f"{safe}_thcm"
        mw, mh = self._mask_size[key]
        parts.append(
            f"[{label}]scale={mw}:{mh},setsar=1,format=yuva420p[{safe}_thca];"
            f"[{self._idx[name]}:v]format=gray[{safe}_thmk];"
            f"[{safe}_thca][{safe}_thmk]alphamerge[{out}]"
        )
        return out

    def finish(self, parts: list[str], current: str) -> str:
        if "over" not in self._idx:
            return current
        parts.append(
            f"[{current}][{self._idx['over']}:v]overlay=0:0:format=auto,format=yuv420p[thc_final]"
        )
        return "thc_final"


def bg_scale_filter(layout_config, canvas_w, canvas_h) -> str:
    """Image / video backgrounds: v2 covers (like the preview); legacy stretches."""
    if is_v2(layout_config):
        return (
            f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={canvas_w}:{canvas_h}"
        )
    return f"scale={canvas_w}:{canvas_h}:flags=lanczos"


def count_inputs(args: list[str]) -> int:
    return sum(1 for a in args if a == "-i")