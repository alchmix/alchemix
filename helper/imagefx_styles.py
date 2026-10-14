"""Space-documentary ImageFX style — used only when ai_prompt is \"space\".

Reference: Proof-style YouTube space explainers — impasto oil B-roll (NOT CGI/stock/photos).
Thematic illustrations visualize abstract concepts beat-by-beat in a fixed cosmic palette.
"""

from __future__ import annotations

SPACE_STYLE_KEY = "space"

# ── Impasto / palette (always locked) ────────────────────────────────────────
IMPASTO_ART_SPEC = (
    "rich textured oil-painting impasto style on visible canvas weave; thick layered paint with "
    "heavy visible brushstrokes and paint buildup on highlights; hand-painted fine-art aesthetic — "
    "absolutely NOT standard CGI, NOT 3D render, NOT stock footage, NOT photograph, NOT digital smoothness; "
    "dramatic cosmic color palette: deep purples and violets in nebula clouds, midnight blues and navy "
    "void, fiery golds and glowing oranges on rim-lit edges, warm amber highlights on metal and atmosphere; "
    "high contrast between dark void and luminous subject; sfumato soft edges in deep background, sharper "
    "impasto strokes on foreground subject; cinematic widescreen 16:9; awe mystery and scientific wonder."
)

SPACE_STYLE_DESCRIPTION = (
    f"{IMPASTO_ART_SPEC} "
    "Premium YouTube space-documentary thematic illustration; scientifically plausible subjects rendered "
    "as painterly art; epic scale; consistent series visual DNA on every frame; "
    "NO text, NO labels, NO logos, NO watermarks, NO UI, NO typography, "
    "NO flat vector, NO anime, NO cartoon."
)

# ── Thematic illustration catalog (what this niche visualizes) ─────────────────
THEMATIC_ILLUSTRATION_CATALOG = """
THEMATIC ILLUSTRATIONS — dynamically visualize what the narrator discusses (pick per beat):

SPACECRAFT & HARDWARE (accurate, painterly):
- Voyager 1 / deep-space probes: white parabolic high-gain dish, golden thermal blankets, RTG housings,
  instrument booms, golden record — structurally correct, impasto texture on metal
- Satellite buses, antenna arrays, solar panels, instrument close-ups, golden disc, RTG cylinders
- Clean-room assembly silhouettes, rolled gold foil, wiring harnesses (no readable text on panels)

WORLDS & COSMIC STRUCTURES:
- Planets: Jupiter with Great Red Spot, Saturn rings, Earth blue marble, Mars rust surface
- Moons, asteroid fields, comet tails, ring gaps, storm systems, polar auroras on gas giants
- Spiral galaxies, elliptical galaxies, barred spirals, galactic cores, star clusters
- Nebulae, supernova remnants, pulsar beams, black hole accretion glow (stylized, not textbook diagram)

PHENOMENA & ABSTRACT SCIENCE (visualize the concept):
- Solar wind: streaming golden-orange particle rivers flowing from the Sun into deep blue space
- Heliopause / heliosphere boundary: glowing shock-wave membrane, Sun-side bubble, interstellar side dark
- Magnetosphere, radiation belts, cosmic rays as painterly light streaks
- Gravity wells as subtle canvas distortion, orbital paths as luminous arcs
- Light-speed / distance: ladder of time, fading signal waves crossing starfields
- Vacuum, cold of space, interstellar medium as purple-blue mist

LAUNCH & GROUND SYSTEMS:
- Rocket on launchpad at night or dawn, painterly exhaust flame (gold-orange), gantry silhouettes
- Ascending rocket trail cutting through purple-blue sky, staging implied by light bursts
- Deep Space Network dishes in desert at sunset, radio beams as impasto gold strokes toward sky

TECH & MISSION OPS (when line is about engineering/data/teams):
- Glowing computer circuit boards and telemetry paths (abstract green-gold traces, no readable text)
- NASA mission control: dark room, faces lit by orange-gold CRT/screen glow, star charts on displays
- Oscilloscope-style waveforms as pure light art, data streams as flowing gold lines in navy void
- Vintage 1970s computing hardware for period missions (painterly, warm amber monitor glow)

HUMAN PRESENCE (rare — only when narration focuses on people):
- Small figures in mission control or at observatory; always secondary to screens, craft, or sky
- Never portrait close-ups; always impasto environment dominates frame
""".strip()

CRAFT_ACCURACY_HINT = (
    "Voyager 1 accuracy when shown: dominant white parabolic high-gain antenna dish; rectangular bus with "
    "gold Mylar thermal blankets; twin RTG cylinders; long magnetometer and instrument booms; golden record "
    "on mounting plate — all rendered with thick impasto brushwork, not slick CGI."
)

SPACE_SCENE_RULES = f"""
CHANNEL FORMAT (Proof-style space explainer B-roll):
- Every image is an IMPASTO OIL PAINTING — never CGI, stock photo, or photoreal render.
- ~80% cosmic/engineering/thematic subjects; ~20% human or ground ops — only when the line requires it.
- Beat-literal: illustrate THIS line's concept; vary composition each beat (no identical Voyager center-shot).

{THEMATIC_ILLUSTRATION_CATALOG}

HOW TO MATCH ABSTRACT NARRATION:
- If the line describes a process (solar wind, signal travel, crossing a boundary) → paint the PHENOMENON
  as a luminous impasto visual metaphor, not a textbook diagram with labels.
- If the line names an object (Voyager, Jupiter, heliopause) → that object is the hero, accurate and clear.
- If the line is about engineering/data/computers → glowing circuit traces, consoles, hardware — painterly.
- If the line is about launch → launchpad and rocket as hero, gold-orange flame against midnight blue.

DO NOT:
- Use photoreal NASA photography look, slick 3D, or flat vector infographic style.
- Default to human faces when the line is about space, craft, or phenomena.
- Repeat the same composition; ignore named worlds or concepts in the script.

DO:
- Deep purples, midnight blues, fiery golds, glowing oranges on every frame.
- Visible thick brushstrokes and canvas texture always.
- Generator-safe; no real names; no text in image.
""".strip()


def is_space_style(style_spec: str | None) -> bool:
    return (style_spec or "").strip().lower() == SPACE_STYLE_KEY


def generate_space_visual_bible(script: str, client) -> str:
    """Extract hero craft, worlds, abstract concepts, and style lock from script."""
    trimmed_script = " ".join((script or "").split()[:800])

    system = f"""
You are a visual story analyst for impasto oil-painting space-documentary videos (Proof-style YouTube).

All B-roll is thematic illustration — thick brushstrokes, cosmic palette (deep purples, midnight blues,
fiery golds, glowing oranges) — NOT CGI or stock footage. Mostly space/craft/phenomena; humans rare.

Output format (plain text, use these exact section headers):
TOPIC: one sentence
HERO CRAFT OR MISSION: name + accurate visual features for impasto illustration (antenna, body, instruments)
WORLDS & LOCATIONS: every planet, moon, galaxy, or place named in script
PHENOMENA & ABSTRACT CONCEPTS: solar wind, heliopause, magnetosphere, signals, distance, radiation, etc. from script — how to paint each
ENGINEERING & OPS MOMENTS: launches, circuit boards, mission control, antennas — only if script mentions them
PAINTERLY STYLE LOCK: impasto brushwork + palette (purple/blue/gold/orange) — same on every frame
WHEN TO SHOW HUMANS: short list of lines that truly need mission control/scientists (keep minimal)
MOOD: emotional tone
AVOID: photoreal CGI, human faces on most beats, wrong craft shape, text in images

Reference catalog of visual types this niche uses:
{THEMATIC_ILLUSTRATION_CATALOG[:1800]}

Rules: generator-safe; no celebrity names; no dialogue quotes.
""".strip()

    user = f"""
Script (first ~800 words):
{trimmed_script}

Write the visual bible — map script concepts to impasto thematic illustrations.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.35,
    )
    return resp.choices[0].message.content.strip()


def generate_space_image_fx_prompts(
    text_segment: str,
    script_context: str,
    prev_segments: list[str] | None,
    *,
    client,
    append_style_fn,
    override_prompt: str | None = None,
) -> list[str]:
    """Thematic impasto illustration per beat — abstract concepts painted, craft-accurate when named."""
    if prev_segments is None:
        prev_segments = []

    continuity_text = (
        "\n".join(prev_segments[-2:]) if prev_segments else "None — opening scene."
    )
    style_description = override_prompt or SPACE_STYLE_DESCRIPTION
    global_context = (script_context or "").strip() or "No global visual bible provided."

    system = """
You are the illustration director for an impasto oil-painting space-documentary (Proof-style YouTube B-roll).

Every image is a THEMATIC ILLUSTRATION — rich textured oil painting with visible thick brushstrokes —
visualizing the specific concept the narrator just said. Never CGI, never stock photo, never 3D render.

Decision order:
1. What concept, object, or phenomenon does THIS line discuss? (Voyager, heliopause, solar wind, launch, etc.)
2. Choose the matching illustration type from the thematic catalog — paint that concept in impasto style.
3. Use deep purples, midnight blues, fiery golds, glowing oranges in the palette.
4. Space/craft/phenomena first; humans/mission control only if the line is clearly about people or ops.
5. If Voyager or hero craft appears → structurally accurate dish, booms, golden blankets.
6. Vary angle and scale from previous beat — never duplicate the last composition.

Hard rules: one frozen painterly moment; no text/logos; generator-safe; return only scene description.
""".strip()

    user = f"""
GLOBAL VISUAL BIBLE (craft accuracy, worlds, abstract concepts, style lock):
{global_context[:2200]}

CRAFT ACCURACY:
{CRAFT_ACCURACY_HINT}

PREVIOUS NARRATION (do not repeat same composition):
{continuity_text}

CURRENT NARRATION (paint THIS concept as impasto thematic illustration):
"{text_segment.strip()}"

THEMATIC ILLUSTRATION RULES & CATALOG:
{SPACE_SCENE_RULES}

MANDATORY IMPASTO STYLE (never CGI/photo/stock):
{style_description}

TASK:
Write one dense paragraph (80-200 words).
- Describe the thematic illustration for THIS line: subject, impasto brushwork, cosmic palette colors,
  lighting (gold-orange rim on midnight blue), scale, mood.
- Match narration literally (named world, craft, phenomenon, launch, or abstract process).
- Thick visible brushstrokes and canvas texture throughout.

Return ONLY the scene description.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.55,
    )

    scene = resp.choices[0].message.content.strip()
    final_prompt = append_style_fn(
        scene,
        "16:9 widescreen",
        style_description,
        3200,
    )
    return [final_prompt]
