"""
Shared timeline FX math — keep frontend preview (timelineFx.js) in sync with
backend (editor.py).

v2 additions:
  - Per-overlay transitions (in/out): fade, slide-left/right/up/down, zoom.
      fade        → fade=...:alpha=1 on the rgba overlay stream
      slide-*     → time-based x/y expressions on the overlay filter
      zoom        → baked into the zoompan pre-render (images only; video
                    overlays downgrade to fade — mirrored in timelineFx.js)
  - Global FX: brightness / contrast / saturation / hue / blur / grayscale /
    sepia (color_fx_filters), whole-video fade in/out (fade_filters), plus
    the existing vignette + noise.

Parity notes (preview CSS filter ↔ ffmpeg):
  contrast, saturation, hue, grayscale, sepia map near-1:1.
  brightness: CSS is multiplicative, ffmpeg eq is additive — mapped
    brightness*0.5 which tracks closely in the usable range.
  blur: CSS radius ≈ 2×gaussian sigma → sigma = blur_px * 0.5.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Must match ZOOM_TRANS_SCALE in timelineFx.js
ZOOM_TRANS = 0.35

TRANSITION_TYPES = {
    "none",
    "fade",
    "slide-left",
    "slide-right",
    "slide-up",
    "slide-down",
    "zoom",
}

_SLIDE_SIGNS = {
    # axis, sign: offset applied to the centered overlay position
    "slide-left": ("x", -1),
    "slide-right": ("x", +1),
    "slide-up": ("y", +1),   # enters from the bottom
    "slide-down": ("y", -1),  # enters from the top
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


# ════════════════════════════════════════════════════════════════
# KEN BURNS (unchanged math)
# ════════════════════════════════════════════════════════════════

def ken_progress(mode: str, local_t: float, seg_len: float, speed: float = 8.0) -> float:
    mode = (mode or "auto").lower()
    seg_len = max(seg_len, 1e-6)
    local_t = max(0.0, local_t)
    if mode == "in":
        return min(1.0, local_t / seg_len)
    if mode == "out":
        return 1.0 - min(1.0, local_t / seg_len)
    phase = (local_t / max(0.001, speed)) * math.pi * 2
    return 0.5 * (1.0 - math.cos(phase))


def ken_scale(
    mode: str,
    local_t: float,
    seg_len: float,
    strength: float = 0.2,
    speed: float = 8.0,
) -> float:
    strength = clamp(strength, 0.0, 0.5)
    return 1.0 + strength * ken_progress(mode, local_t, seg_len, speed)


def parse_ken_burns(ken: dict[str, Any] | None) -> dict[str, Any]:
    ken = ken or {}
    return {
        "enabled": bool(ken.get("enabled", False)),
        "mode": str(ken.get("mode") or "auto"),
        "strength": clamp(ken.get("strength", 0.2), 0.0, 0.5),
        "speed": max(0.5, float(ken.get("speed", 8))),
    }


# ════════════════════════════════════════════════════════════════
# TRANSITIONS
# ════════════════════════════════════════════════════════════════

def parse_transition(t: dict[str, Any] | None, overlay_type: str = "image") -> dict[str, Any]:
    """Mirror of timelineFx.normalizeTransition (incl. zoom→fade for video)."""
    t = t or {}
    typ = str(t.get("type") or "none").lower()
    if typ not in TRANSITION_TYPES:
        typ = "none"
    if typ == "zoom" and overlay_type == "video":
        typ = "fade"
    dur = clamp(float(t.get("duration", 0.5) or 0.5), 0.05, 5.0)
    return {"type": typ, "duration": dur if typ != "none" else 0.0}


def transition_needs_zoom_prerender(
    tin: dict[str, Any], tout: dict[str, Any], overlay_type: str
) -> bool:
    return overlay_type == "image" and "zoom" in (tin["type"], tout["type"])


def overlay_alpha_fade_filters(
    start: float, end: float, tin: dict[str, Any], tout: dict[str, Any]
) -> list[str]:
    """
    fade filters (alpha channel) for the overlay stream. Stream PTS must
    already be shifted onto the main timeline (setpts +start) before these.
    fade and zoom both ramp opacity — matches the preview.
    """
    out: list[str] = []
    if tin["type"] in ("fade", "zoom"):
        d = tin["duration"]
        out.append(f"fade=t=in:st={start:.4f}:d={d:.4f}:alpha=1")
    if tout["type"] in ("fade", "zoom"):
        d = tout["duration"]
        out.append(f"fade=t=out:st={max(start, end - d):.4f}:d={d:.4f}:alpha=1")
    return out


def overlay_xy_exprs(
    start: float, end: float, tin: dict[str, Any], tout: dict[str, Any]
) -> tuple[str, str]:
    """
    x/y expressions for the overlay filter: centered fit-screen position plus
    slide offsets. p goes 0→1 entering, 1→0 exiting — offset = sign*DIM*(1-p),
    identical to the preview's translate(±(1-p)*100%).
    Commas escaped for filter_complex.
    """
    x_terms: list[str] = ["(W-w)/2"]
    y_terms: list[str] = ["(H-h)/2"]

    def add(trans: dict[str, Any], is_in: bool) -> None:
        axis_sign = _SLIDE_SIGNS.get(trans["type"])
        if not axis_sign:
            return
        axis, sign = axis_sign
        dim = "W" if axis == "x" else "H"
        d = trans["duration"]
        sgn = "-" if sign < 0 else "+"
        if is_in:
            # t in [start, start+d): offset = sign*DIM*(1 - (t-start)/d)
            term = (
                f"if(lt(t\\,{start + d:.4f})\\,"
                f"{sgn}{dim}*(1-(t-{start:.4f})/{d:.4f})\\,0)"
            )
        else:
            # t in (end-d, end]: offset = sign*DIM*(1 - (end-t)/d)
            term = (
                f"if(gt(t\\,{end - d:.4f})\\,"
                f"{sgn}{dim}*(1-({end:.4f}-t)/{d:.4f})\\,0)"
            )
        (x_terms if axis == "x" else y_terms).append(term)

    add(tin, True)
    add(tout, False)
    return "+".join(x_terms), "+".join(y_terms)


# ════════════════════════════════════════════════════════════════
# GLOBAL FX
# ════════════════════════════════════════════════════════════════

def parse_global_fx(
    fx: dict[str, Any] | None,
    legacy_vignette: dict[str, Any] | float | None = None,
    legacy_noise: float = 0.0,
) -> dict[str, Any]:
    """Normalize the metadata `fx` block; falls back to legacy fields."""
    fx = dict(fx or {})

    def g(key: str, default: float = 0.0) -> float:
        try:
            return float(fx.get(key, default) or 0.0)
        except (TypeError, ValueError):
            return default

    vignette_src = fx.get("vignette", legacy_vignette)
    noise_src = fx.get("noise", legacy_noise)
    try:
        noise_val = float(noise_src or 0.0)
    except (TypeError, ValueError):
        noise_val = 0.0

    return {
        "brightness": clamp(g("brightness"), -1.0, 1.0),
        "contrast": clamp(g("contrast"), -1.0, 1.0),
        "saturation": clamp(g("saturation"), -1.0, 1.0),
        "hue": clamp(g("hue"), -180.0, 180.0),
        "blur": clamp(g("blur"), 0.0, 50.0),
        "grayscale": clamp(g("grayscale"), 0.0, 1.0),
        "sepia": clamp(g("sepia"), 0.0, 1.0),
        "fade_in": clamp(g("fadeIn", g("fade_in")), 0.0, 10.0),
        "fade_out": clamp(g("fadeOut", g("fade_out")), 0.0, 10.0),
        "vignette": parse_vignette(vignette_src),
        "noise": clamp(noise_val, 0.0, 1.0),
    }


_SEPIA = ((0.393, 0.769, 0.189), (0.349, 0.686, 0.168), (0.272, 0.534, 0.131))
_IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def color_fx_filters(gfx: dict[str, Any]) -> list[str]:
    """FFmpeg color filters, in the same order the preview CSS filter applies."""
    filters: list[str] = []

    br, co, sa = gfx["brightness"], gfx["contrast"], gfx["saturation"]
    if br or co or sa:
        filters.append(
            f"eq=brightness={br * 0.5:.4f}"
            f":contrast={clamp(1 + co, 0.0, 3.0):.4f}"
            f":saturation={clamp(1 + sa, 0.0, 3.0):.4f}"
        )

    hue, gray = gfx["hue"], gfx["grayscale"]
    if hue or gray:
        filters.append(f"hue=h={hue:.2f}:s={clamp(1 - gray, 0.0, 1.0):.4f}")

    sep = gfx["sepia"]
    if sep > 0:
        m = [
            [(1 - sep) * i + sep * s for i, s in zip(irow, srow)]
            for irow, srow in zip(_IDENT, _SEPIA)
        ]
        filters.append(
            "colorchannelmixer="
            f"rr={m[0][0]:.4f}:rg={m[0][1]:.4f}:rb={m[0][2]:.4f}:"
            f"gr={m[1][0]:.4f}:gg={m[1][1]:.4f}:gb={m[1][2]:.4f}:"
            f"br={m[2][0]:.4f}:bg={m[2][1]:.4f}:bb={m[2][2]:.4f}"
        )

    if gfx["blur"] > 0:
        filters.append(f"gblur=sigma={gfx['blur'] * 0.5:.3f}")

    return filters


def fade_filters(duration: float, fade_in: float, fade_out: float) -> list[str]:
    out: list[str] = []
    if fade_in > 0:
        out.append(f"fade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0 and duration > fade_out:
        out.append(f"fade=t=out:st={duration - fade_out:.3f}:d={fade_out:.3f}")
    return out


# ════════════════════════════════════════════════════════════════
# VIGNETTE / NOISE (unchanged)
# ════════════════════════════════════════════════════════════════

def vignette_blend_opacity(strength: float) -> float:
    """Matches timelineFx.js radial gradient outer alpha (0.99 * strength)."""
    return clamp(strength, 0.0, 1.0) * 0.99


def noise_ffmpeg_level(strength: float) -> int:
    """Matches legacy editor.py noise mapping (0..1 → ffmpeg noise 0..50)."""
    return max(0, min(100, int(clamp(strength, 0.0, 1.0) * 50)))


def parse_vignette(vignette: dict[str, Any] | float | int | None) -> dict[str, Any]:
    if isinstance(vignette, dict):
        strength = float(vignette.get("strength", 0) or 0)
        enabled = bool(vignette.get("enabled", strength > 0))
        return {
            "enabled": enabled and strength > 0,
            "strength": clamp(strength, 0.0, 1.0),
            "opacity": vignette_blend_opacity(strength),
        }
    if isinstance(vignette, (int, float)):
        strength = clamp(vignette, 0.0, 1.0)
        return {
            "enabled": strength > 0,
            "strength": strength,
            "opacity": vignette_blend_opacity(strength),
        }
    return {"enabled": False, "strength": 0.0, "opacity": 0.0}


# ════════════════════════════════════════════════════════════════
# ZOOMPAN PRE-RENDER (Ken Burns + zoom transitions baked together)
# ════════════════════════════════════════════════════════════════

# Improbable key color used to fake transparency through the opaque
# zoompan pre-render (zoompan drops alpha). Keyed back out in the main graph.
PRERENDER_KEY_COLOR = "0x00FE01"


def _zoompan_chain(
    width: int,
    height: int,
    fps: float,
    num_frames: int,
    mode: str,
    strength: float,
    speed: float,
    trans_in: dict[str, Any] | None = None,
    trans_out: dict[str, Any] | None = None,
) -> str:
    """Upscale+zoompan portion of the pre-render (framing prefix excluded)."""
    w, h = int(width), int(height)
    fps_i = max(1, int(round(fps)))
    d = max(int(num_frames), 1)
    denom = max(d - 1, 1)
    sm = clamp(strength, 0.0, 0.5)
    sp = max(0.5, float(speed))
    mode = (mode or "auto").lower()

    if sm <= 0:
        z_ken = "1"
    elif mode == "in":
        z_ken = f"(1+{sm:.6f}*on/{denom})"
    elif mode == "out":
        z_ken = f"({1 + sm:.6f}-{sm:.6f}*on/{denom})"
    else:
        period = max(1, int(round(fps_i * sp)))
        z_ken = f"(1+{sm:.6f}*(0.5-0.5*cos(2*PI*on/{period})))"

    factors = [z_ken]
    if trans_in and trans_in.get("type") == "zoom":
        fin = max(1, int(round(trans_in["duration"] * fps_i)))
        factors.append(f"(1+{ZOOM_TRANS:.4f}*max(0\\,1-on/{fin}))")
    if trans_out and trans_out.get("type") == "zoom":
        fout = max(1, int(round(trans_out["duration"] * fps_i)))
        factors.append(f"(1+{ZOOM_TRANS:.4f}*max(0\\,(on-{denom - fout})/{fout}))")

    z_expr = "*".join(factors)
    smooth = int(os.getenv("TIMELINE_KB_UPSCALE", os.getenv("KENBURNS_FFMPEG_UPSCALE", "4000")))
    return (
        f"scale={smooth}:-1,"
        f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={fps_i}"
    )


def overlay_zoompan_vf(
    width: int,
    height: int,
    fps: float,
    num_frames: int,
    mode: str,
    strength: float,
    speed: float,
    trans_in: dict[str, Any] | None = None,
    trans_out: dict[str, Any] | None = None,
    pad_color: str = "black@0",
) -> str:
    """Legacy single-chain pre-render vf (contain fit, no background)."""
    w, h = int(width), int(height)
    zp = _zoompan_chain(w, h, fps, num_frames, mode, strength, speed, trans_in, trans_out)
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={pad_color},setsar=1,"
        f"{zp}"
    )


# ════════════════════════════════════════════════════════════════
# OVERLAY FRAMING (fit / focus) + BACKGROUND FILL
# ════════════════════════════════════════════════════════════════

BG_TYPES = {"none", "blur", "color", "gradient"}


def parse_fit(fit: Any) -> str:
    return "cover" if str(fit or "contain").lower() == "cover" else "contain"


def parse_focus(focus: dict[str, Any] | None) -> tuple[float, float]:
    focus = focus or {}

    def num(key: str) -> float:
        v = focus.get(key, 0.5)
        if v is None or v == "":
            return 0.5
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.5

    return clamp(num("x"), 0.0, 1.0), clamp(num("y"), 0.0, 1.0)


def _hex_to_ffmpeg(color: Any, default: str = "0x000000") -> str:
    c = str(color or "").strip()
    if c.startswith("#"):
        if len(c) == 4:
            c = "#" + "".join(ch * 2 for ch in c[1:])
        if len(c) == 7:
            return "0x" + c[1:].upper()
    if c.lower().startswith("0x") and len(c) == 8:
        return c.upper()
    return default


def parse_overlay_bg(bg: dict[str, Any] | None) -> dict[str, Any]:
    """Mirror of the frontend bg config. Matches timelineFx.defaultOverlayBg."""
    bg = bg or {}
    typ = str(bg.get("type") or "none").lower()
    if typ not in BG_TYPES:
        typ = "none"

    def num(key: str, default: float) -> float:
        v = bg.get(key, default)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    return {
        "type": typ,
        "blur": clamp(num("blur", 20.0), 1.0, 100.0),
        "color": _hex_to_ffmpeg(bg.get("color"), "0x000000"),
        "color2": _hex_to_ffmpeg(bg.get("color2"), "0x7C3AED"),
        "angle": clamp(num("angle", 180.0), 0.0, 360.0),
    }


def bg_blur_sigma(blur_px: float, base_w: int) -> float:
    """Resolution-relative: CSS blur(px) on the ~640px preview → gaussian
    sigma at export resolution (sigma ≈ radius/2, scaled by width)."""
    return max(1.0, float(blur_px) * (base_w / 640.0) * 0.5)


def fg_frame_filters(w: int, h: int, fit: str, focus: tuple[float, float]) -> list[str]:
    """Foreground framing: contain (letterbox) or cover (crop at focus)."""
    if fit == "cover":
        fx, fy = focus
        return [
            f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=fast_bilinear",
            f"crop={w}:{h}:(in_w-out_w)*{fx:.4f}:(in_h-out_h)*{fy:.4f}",
        ]
    return [f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=fast_bilinear"]


def cover_blur_filters(w: int, h: int, focus: tuple[float, float], sigma: float) -> list[str]:
    """Blurred-image background: cover-crop at the same focus point, then blur."""
    fx, fy = focus
    return [
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=fast_bilinear",
        f"crop={w}:{h}:(in_w-out_w)*{fx:.4f}:(in_h-out_h)*{fy:.4f}",
        f"gblur=sigma={sigma:.2f}",
    ]


def gradient_source(w: int, h: int, c1: str, c2: str, angle_deg: float, duration: float) -> str:
    """FFmpeg gradients source matching CSS linear-gradient(angle, c1, c2):
    0deg = to top, 90deg = to right, first color at the start point."""
    a = math.radians(angle_deg)
    s, c = math.sin(a), math.cos(a)
    length = abs(w * s) + abs(h * c)
    x0 = w / 2 - s * length / 2
    y0 = h / 2 + c * length / 2
    x1 = w / 2 + s * length / 2
    y1 = h / 2 - c * length / 2
    def px(v: float, hi: int) -> int:
        return max(0, min(hi - 1, int(round(v))))
    return (
        f"gradients=s={w}x{h}:c0={c1}:c1={c2}"
        f":x0={px(x0, w)}:y0={px(y0, h)}:x1={px(x1, w)}:y1={px(y1, h)}"
        f":d={duration:.3f}"
    )


def overlay_prerender_graph(
    w: int,
    h: int,
    fps: float,
    num_frames: int,
    ken: dict[str, Any],
    tin: dict[str, Any],
    tout: dict[str, Any],
    fit: str,
    focus: tuple[float, float],
    bg: dict[str, Any],
    duration: float,
    aspect_mismatch: bool,
) -> tuple[str, bool]:
    """
    Full filter_complex for the animated-image pre-render (input [0:v] is the
    looped still). Returns (graph, needs_colorkey_in_main_graph).

    Ken Burns / zoom transitions animate the FOREGROUND image only; the
    background (blur / color / gradient) stays static — matches the preview.
    """
    strength = ken["strength"] if ken.get("enabled") else 0.0
    zp = _zoompan_chain(w, h, fps, num_frames, ken.get("mode", "auto"), strength,
                        ken.get("speed", 8.0), tin, tout)

    if fit == "cover":
        vf = ",".join(fg_frame_filters(w, h, "cover", focus) + ["setsar=1", zp])
        return f"[0:v]{vf}[vout]", False

    # contain
    if bg["type"] == "none":
        color = PRERENDER_KEY_COLOR if aspect_mismatch else "black"
        vf = ",".join([
            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={color}",
            "setsar=1", zp,
        ])
        return f"[0:v]{vf}[vout]", aspect_mismatch

    if bg["type"] == "color":
        vf = ",".join([
            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={bg['color']}",
            "setsar=1", zp,
        ])
        return f"[0:v]{vf}[vout]", False

    # blur / gradient: composite fg over bg inside the pre-render.
    # fg keeps transparency via key-pad → colorkey, so the bg shows through.
    if aspect_mismatch:
        fg_chain = ",".join([
            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={PRERENDER_KEY_COLOR}",
            "setsar=1", zp,
            "format=rgba",
            f"colorkey={PRERENDER_KEY_COLOR}:0.08:0.05",
        ])
    else:
        fg_chain = ",".join([
            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            "setsar=1", zp, "format=rgba",
        ])

    if bg["type"] == "blur":
        sigma = bg_blur_sigma(bg["blur"], w)
        lines = [
            "[0:v]split[bga][fga]",
            "[bga]" + ",".join(cover_blur_filters(w, h, focus, sigma)) + "[bgb]",
            f"[fga]{fg_chain}[fgk]",
            "[bgb][fgk]overlay=x=0:y=0:shortest=1[vout]",
        ]
    else:  # gradient
        lines = [
            gradient_source(w, h, bg["color"], bg["color2"], bg["angle"], duration + 1) + "[bgg]",
            f"[0:v]{fg_chain}[fgk]",
            "[bgg][fgk]overlay=x=0:y=0:shortest=1[vout]",
        ]
    return ";".join(lines), False