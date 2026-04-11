"""
Whisk Character Handler Module
===============================
Handles Whisk character generation and refinement for video production.

Key Features:
- Full pipeline via whisk_generate.mjs Node.js script (module API, not CLI)
- The CLI "generate" has no --subject flag — module API is required
- Automatic recovery on ALL error types with cookie rotation
- Process-safe with multiprocessing.Manager support

Correct flow (per segment):
    Python subprocess calls whisk_generate.mjs "full" mode which does:
        1. whisk.newProject(name)
        2. project.uploadReference(imagePath, "SUBJECT")
        3. project.generateImage({prompt, aspectRatio})
        4. generated.refine(refinePrompt)
        → prints [+] REFINED_PATH: <path>  (parsed by Python)

    Cookie passed via WHISK_COOKIE environment variable (avoids Windows
    shell escaping issues with long cookie strings).
"""

import os
import re
import json
import shutil
import subprocess
import logging
import traceback
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, List, Any
from helper.telegram_notification_engine import send_telegram_notification, send_telegram_error

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    openai_client = None
    logging.warning("⚠️ OpenAI not installed. Whisk prompt generation will use fallback.")

try:
    from cookie_reader import read_active_cookie, signal_force_rotate
except ImportError:
    logging.error("❌ Cannot import cookie_reader. Whisk integration will fail.")
    read_active_cookie = None
    signal_force_rotate = None


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

DEFAULT_IMAGE_DURATION  = 5.0   # seconds
WHISK_UPLOAD_TIMEOUT    = 180   # seconds (used during init verify upload)
WHISK_GENERATE_TIMEOUT  = 300   # seconds (full pipeline: upload+generate+refine)

# Path to the Node.js helper script — must be in same directory as this file
WHISK_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "whisk_generate.mjs"
)


# ═══════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════

class Whisk401Error(Exception):
    """
    Raised on ANY Whisk failure that should trigger retry.
    Covers: 401, 400, bad cookie, generation/refinement failure,
    no output file, timeout, and any unexpected error.
    """
    pass


class WhiskCharacterError(Exception):
    """Raised on fatal character initialization errors."""
    pass


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def find_whisk_cmd() -> str:
    """Find whisk executable across platforms."""
    found = shutil.which("whisk")
    if found:
        return found

    appdata     = os.environ.get("APPDATA", "")
    userprofile = os.environ.get("USERPROFILE", "")

    candidates = [
        os.path.join(appdata, "npm", "whisk.cmd"),
        os.path.join(appdata, "npm", "whisk"),
        os.path.join(userprofile, ".bun", "bin", "whisk"),
        os.path.join(userprofile, ".bun", "bin", "whisk.exe"),
        "/usr/local/bin/whisk",
        os.path.expanduser("~/.bun/bin/whisk"),
        os.path.expanduser("~/.npm-global/bin/whisk"),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    return "whisk"


def find_node_cmd() -> str:
    """Find node.exe on Windows and node on Linux/macOS."""
    found = shutil.which("node")
    if found:
        return found

    userprofile   = os.environ.get("USERPROFILE", "")
    program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")

    candidates = [
        os.path.join(program_files, "nodejs", "node.exe"),
        os.path.join(program_files + " (x86)", "nodejs", "node.exe"),
        os.path.join(userprofile, "AppData", "Roaming", "nvm", "current", "node.exe"),
        "C:\\Program Files\\nodejs\\node.exe",
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    return "node"


def extract_whisk_media_id(stdout: str) -> Optional[str]:
    """Extract media ID from whisk CLI output. Format: [+] ID: CAMa..."""
    m = re.search(r'\[\+\]\s+ID:\s+(\S+)', stdout)
    if m and len(m.group(1)) >= 20:
        return m.group(1)
    m = re.search(r'(CAM[a-zA-Z0-9+/=_\-]{20,})', stdout)
    return m.group(1) if m else None


def extract_whisk_project_id(stdout: str) -> Optional[str]:
    """Extract project ID from whisk project output."""
    m = re.search(r'\[\+\]\s+Project ID:\s+([a-f0-9\-]{36})', stdout)
    return m.group(1).strip() if m else None


def extract_script_value(stdout: str, key: str) -> Optional[str]:
    """
    Extract a tagged value from whisk_generate.mjs output.
    e.g. extract_script_value(stdout, "REFINED_PATH") → "C:\\...\\image.png"
    """
    m = re.search(rf'\[\+\]\s+{re.escape(key)}:\s+(.+)', stdout)
    return m.group(1).strip() if m else None


def handle_whisk_401(stdout: str) -> None:
    """Signal cookie rotation daemon on any auth failure."""
    if signal_force_rotate:
        logging.warning("🔐 Cookie invalid/expired. Signalling daemon to force rotate...")
        signal_force_rotate()
        logging.info("✅ Force rotate signal sent. Daemon will rotate immediately.")
    else:
        logging.warning("🔐 Cookie invalid/expired but signal_force_rotate not available.")


def check_stdout_for_errors(stdout: str, context: str) -> None:
    """
    Inspect stdout for ALL known Whisk error patterns regardless of return code.
    Raises Whisk401Error for any error that should trigger retry.
    """
    stdout_lower = stdout.lower()

    # Auth failures
    if (
        "401"                       in stdout
        or "unauthenticated"        in stdout_lower
        or "new cookie is required" in stdout_lower
        or "cookie is required"     in stdout_lower
        or "invalid cookie"         in stdout_lower
        or "cookie expired"         in stdout_lower
        or "auth failure"           in stdout_lower
        or "unauthorized"           in stdout_lower
    ):
        handle_whisk_401(stdout)
        raise Whisk401Error(
            f"[{context}] Whisk auth failure — cookie invalid/expired, rotating"
        )

    # Stale media_id
    if "400" in stdout or "invalid_argument" in stdout_lower:
        raise Whisk401Error(
            f"[{context}] Whisk 400 INVALID_ARGUMENT — stale ID, retrying"
        )

    # Script-level errors from whisk_generate.mjs
    if "[!] error:" in stdout_lower:
        raise Whisk401Error(f"[{context}] Script error: {stdout[:300]}")

    # Legacy CLI failure messages
    if "[!] refinement failed" in stdout_lower:
        raise Whisk401Error(f"[{context}] Whisk refinement failed in stdout")

    if "[!] generation failed" in stdout_lower:
        raise Whisk401Error(f"[{context}] Whisk generation failed in stdout")

    if "[!] upload failed" in stdout_lower:
        raise Whisk401Error(f"[{context}] Whisk upload failed in stdout")

    if "[!] project" in stdout_lower and "failed" in stdout_lower:
        raise Whisk401Error(f"[{context}] Whisk project creation failed in stdout")


# ═══════════════════════════════════════════════════════════════
# INIT HELPER — UPLOAD CHARACTER (verify image + cookie on startup)
# ═══════════════════════════════════════════════════════════════

def whisk_create_project(
    cookie: str,
    project_name: str = "Whisk Character Project",
    timeout: int = WHISK_UPLOAD_TIMEOUT,
) -> Optional[str]:
    """Create a Whisk project via CLI. Used during initialization only."""
    try:
        whisk_cmd = find_whisk_cmd()
        use_shell  = os.name == "nt" and whisk_cmd.lower().endswith(".cmd")

        cmd = [
            whisk_cmd, "project",
            "--name",   project_name,
            "--cookie", cookie,
        ]

        logging.info(f"📁 Creating Whisk project: {project_name}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            shell=use_shell,
        )

        stdout = result.stdout or ""
        logging.info(f"Whisk project rc={result.returncode}")
        logging.info(f"Whisk project stdout:\n{stdout[:600]}")

        check_stdout_for_errors(stdout, "whisk_create_project")

        if result.returncode != 0:
            raise Whisk401Error(f"whisk project non-zero exit (rc={result.returncode})")

        project_id = extract_whisk_project_id(stdout)
        if not project_id:
            logging.error("❌ Could not extract project ID from output")
            return None

        logging.info(f"✅ Project created: {project_id}")
        return project_id

    except subprocess.TimeoutExpired:
        raise Whisk401Error(f"whisk project timed out after {timeout}s")
    except Whisk401Error:
        raise
    except Exception as e:
        logging.error(f"❌ Whisk project creation exception: {e}")
        raise Whisk401Error(f"whisk project unexpected exception: {e}")


def whisk_upload_character(
    image_path: str,
    cookie: str,
    timeout: int = WHISK_UPLOAD_TIMEOUT,
) -> Optional[str]:
    """
    Upload character image as SUBJECT via CLI.
    Used during initialization to verify the image and cookie are valid.
    Per-segment generation is handled entirely by whisk_generate.mjs.
    """
    try:
        if not os.path.exists(image_path):
            logging.error(f"❌ Image file not found: {image_path}")
            return None

        if os.path.getsize(image_path) < 1000:
            logging.error(f"❌ Image file too small")
            return None

        project_name = f"char_{uuid.uuid4().hex[:8]}"
        project_id   = whisk_create_project(cookie, project_name, timeout)

        if not project_id:
            logging.error("❌ Failed to create Whisk project")
            return None

        whisk_cmd = find_whisk_cmd()
        use_shell  = os.name == "nt" and whisk_cmd.lower().endswith(".cmd")

        cmd = [
            whisk_cmd, "upload", image_path,
            "--category", "SUBJECT",
            "--project",  project_id,
            "--cookie",   cookie,
        ]

        logging.info(f"☁️  Uploading: {os.path.basename(image_path)}")
        logging.info(f"   Project: {project_id}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            shell=use_shell,
        )

        stdout = result.stdout or ""
        logging.info(f"Whisk upload rc={result.returncode}")
        logging.info(f"Whisk upload stdout:\n{stdout[:600]}")

        check_stdout_for_errors(stdout, "whisk_upload_character")

        if result.returncode != 0:
            raise Whisk401Error(f"whisk upload non-zero exit (rc={result.returncode})")

        subject_ref_id = extract_whisk_media_id(stdout)
        if not subject_ref_id:
            logging.error("❌ Failed to extract subject_ref_id from upload output")
            return None

        logging.info(f"✅ Character uploaded successfully: {subject_ref_id}")
        return subject_ref_id

    except subprocess.TimeoutExpired:
        raise Whisk401Error(f"whisk upload timed out after {timeout}s")
    except Whisk401Error:
        raise
    except Exception as e:
        logging.error(f"❌ Whisk upload exception: {e}")
        raise Whisk401Error(f"whisk upload unexpected exception: {e}")


# ═══════════════════════════════════════════════════════════════
# CORE — FULL PIPELINE VIA NODE.JS SCRIPT
# ═══════════════════════════════════════════════════════════════

def whisk_full_pipeline(
    character_image_path: str,
    generate_prompt: str,
    refine_prompt: str,
    cookie: str,
    output_dir: Path,
    timeout: int = WHISK_GENERATE_TIMEOUT,
) -> str:
    """
    Run the full Whisk pipeline via whisk_generate.mjs:
        1. newProject()
        2. project.uploadReference(imagePath, "SUBJECT")
        3. project.generateImage({prompt: generate_prompt, aspectRatio: LANDSCAPE})
        4. generated.refine(refine_prompt)

    WHY Node.js script and not CLI:
        The CLI "whisk generate" has no --subject flag.
        Subject references only work via the JS module API,
        scoped to the project they were uploaded into.

    Cookie is passed via WHISK_COOKIE env var to avoid
    Windows PowerShell escaping issues with long cookie strings.

    generate_prompt and refine_prompt are passed via env vars for the
    same reason — avoids shell escaping issues with long/special strings.

    Returns:
        Path to final refined (or generated) image file.

    Raises:
        Whisk401Error on ANY failure.
    """
    try:
        if not os.path.exists(WHISK_SCRIPT_PATH):
            raise Whisk401Error(
                f"whisk_generate.mjs not found at: {WHISK_SCRIPT_PATH}\n"
                f"Place whisk_generate.mjs in the same directory as whisk_character_handler.py"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        node_cmd = find_node_cmd()

        cmd = [
            node_cmd,
            WHISK_SCRIPT_PATH,
            "full",
            character_image_path,
            str(output_dir),
            # prompts are passed via env vars — see below
        ]

        # Pass cookie and prompts via environment variables —
        # avoids shell escaping issues on Windows with long/special strings
        env = os.environ.copy()
        env["WHISK_COOKIE"]          = cookie
        env["WHISK_GENERATE_PROMPT"] = generate_prompt
        env["WHISK_REFINE_PROMPT"]   = refine_prompt

        logging.info(f"🚀 Running Whisk pipeline (project→upload→generate→refine)...")
        logging.info(f"   Image:          {character_image_path}")
        logging.info(f"   Output:         {output_dir}")
        logging.info(f"   Generate prompt: {generate_prompt[:100]}...")
        logging.info(f"   Refine prompt:   {refine_prompt[:80]}...")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
            # Do NOT use shell=True on Windows — node.exe is a real binary
        )

        stdout = result.stdout or ""
        logging.info(f"Whisk pipeline rc={result.returncode}")
        logging.info(f"Whisk pipeline stdout:\n{stdout[:1500]}")

        # Check for errors before checking rc
        check_stdout_for_errors(stdout, "whisk_full_pipeline")

        if result.returncode != 0:
            raise Whisk401Error(
                f"whisk_generate.mjs non-zero exit (rc={result.returncode})"
            )

        # Parse output path
        refined_path = extract_script_value(stdout, "REFINED_PATH")

        if not refined_path:
            # Fallback to generated path if refine step didn't produce output tag
            refined_path = extract_script_value(stdout, "GENERATED_PATH")
            if refined_path:
                logging.warning("⚠️ Using GENERATED_PATH as fallback (no REFINED_PATH found)")

        if not refined_path:
            raise Whisk401Error(
                "whisk_generate.mjs output contained no REFINED_PATH or GENERATED_PATH"
            )

        # Normalize path separators for Windows
        refined_path = str(Path(refined_path))

        if not os.path.exists(refined_path):
            raise Whisk401Error(f"Output file not found on disk: {refined_path}")

        if os.path.getsize(refined_path) < 1000:
            raise Whisk401Error(f"Output file too small: {refined_path}")

        logging.info(f"✅ Pipeline complete: {os.path.basename(refined_path)}")
        return refined_path

    except subprocess.TimeoutExpired:
        raise Whisk401Error(f"whisk_generate.mjs timed out after {timeout}s")
    except Whisk401Error:
        raise
    except Exception as e:
        raise Whisk401Error(f"whisk_generate.mjs unexpected exception: {e}")


# ═══════════════════════════════════════════════════════════════
# MASTER PROMPT BUILDER — TWO-STEP PIPELINE
# ═══════════════════════════════════════════════════════════════

def build_whisk_refine_prompt(
    segment_text: str,
    character_descriptor: str,
    style_lock: str,
    negative_anchors: str = "",
    client=None,
) -> dict:
    """
    Master two-step Whisk prompt pipeline for a narration segment.

    STEP 1 — Classify (temp 0.3, json_object):
        Determines character state, role, and emotional condition.

    STEP 2 — Scene Design (temp 0.65):
        Builds cinematic scene around the classified character state.
        Outputs both generate_prompt (for whisk.generateImage) and
        refine_prompt (for generated.refine).

    Args:
        segment_text         — raw narration segment text
        character_descriptor — full appearance description of the character
        style_lock           — visual style every element must obey
        negative_anchors     — elements to explicitly avoid in generation
        client               — initialised OpenAI client instance;
                               if None, returns structured hard fallback

    Returns dict:
        generate_prompt  → full scene prompt for whisk.generateImage()
        refine_prompt    → short directive for generated.refine()
        state            → character form: adult|child|baby|silhouette|symbolic|absent
        role             → scene role: protagonist|witness|symbolic|background|absent
        emotion          → compound emotional state string
        reason           → one-sentence justification for state/role choice
        analysis         → full scene analysis from Step 2
        fallback_used    → True if GPT pipeline failed at any step
    """

    result = {
        "generate_prompt": "",
        "refine_prompt":   "",
        "state":           "adult",
        "role":            "protagonist",
        "emotion":         "",
        "reason":          "",
        "analysis":        "",
        "fallback_used":   False,
    }

    # ── HARD FALLBACK (no OpenAI client) ─────────────────────────────
    if not client:
        fallback = (
            f"{character_descriptor} "
            f"Scene: {segment_text.strip()}. "
            f"Style: {style_lock}."
            + (f" Avoid: {negative_anchors}." if negative_anchors else "")
        )
        result["generate_prompt"] = fallback
        result["refine_prompt"]   = segment_text.strip()[:120]
        result["fallback_used"]   = True
        logging.warning("⚠️  No OpenAI client — using hard fallback prompt.")
        return result

    # ══════════════════════════════════════════════════════════════
    # STEP 1 — CLASSIFY: STATE + ROLE + EMOTION
    # ══════════════════════════════════════════════════════════════

    CLASSIFIER_SYSTEM = """You are a character director for an animated story series.

Read the narration segment and make three decisions about the character.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE — what is the character doing in this scene?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
protagonist   → character drives the scene, center of attention
witness       → character is present but reacting to something else
symbolic      → character represents an idea, not themselves literally
background    → character is distant, part of the environment
absent        → scene works better with no character at all
               (use for: pure environment shots, crowd scenes,
                object close-ups, abstract concept scenes)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATE — what form does the character take?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
adult         → canonical form (default)
child         → younger version (age 7–9), same facial DNA
baby          → infant version (6–12 months), inherited features
silhouette    → shadow or outline only, no face detail
symbolic      → abstracted form: glowing figure, archetypal shape, light
absent        → does not appear in the scene

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMOTION — what is the character feeling?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always compound: primary emotion + suppressed undercurrent + physical expression.
Be precise — not "sad" but "hollow resignation masking old grief,
eyes unfocused, hands loose at sides, weight forward on one foot."
Leave as empty string ONLY if state is "absent."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — valid JSON only, no preamble, no markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "role": "protagonist|witness|symbolic|background|absent",
  "state": "adult|child|baby|silhouette|symbolic|absent",
  "emotion": "compound emotion string, or empty string if absent",
  "reason": "one sentence — why this combination serves this scene"
}"""

    try:
        clf_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user",   "content": f'Narration segment:\n"""{segment_text.strip()}"""'},
            ],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        clf = _safe_json(clf_response.choices[0].message.content)
        result["state"]   = clf.get("state",   "adult")
        result["role"]    = clf.get("role",    "protagonist")
        result["emotion"] = clf.get("emotion", "")
        result["reason"]  = clf.get("reason",  "")

        logging.info(
            f"[WHISK CLASSIFY] state={result['state']} | "
            f"role={result['role']} | emotion={result['emotion'][:80]}"
        )

    except Exception as e:
        logging.warning(f"⚠️  Step 1 classification failed: {e} — defaulting adult/protagonist")
        # result keeps default values; Step 2 still runs

    # ══════════════════════════════════════════════════════════════
    # STEP 2 — SCENE DESIGN: GENERATE + REFINE PROMPTS
    # ══════════════════════════════════════════════════════════════

    # Resolve character anchor based on classified state
    state = result["state"]

    if state == "absent":
        character_anchor = "NO CHARACTER — environment and atmosphere carry the full scene"

    elif state == "silhouette":
        character_anchor = (
            f"Dark silhouette of the character — outline and posture only, "
            f"no face detail, no colour. "
            f"Shape derived from: {character_descriptor}"
        )

    elif state == "symbolic":
        character_anchor = (
            f"Abstracted symbolic form of the character — glowing figure, "
            f"archetypal shape, no literal face or clothing. "
            f"Essence derived from: {character_descriptor}"
        )

    elif state == "child":
        character_anchor = (
            f"Younger child version (age 7–9) of the character. "
            f"Same face structure, eye shape, skin tone, and hair texture as the adult — "
            f"smaller, rounder features; distinguishing marks faintly present. "
            f"Adult descriptor for reference: {character_descriptor}"
        )

    elif state == "baby":
        character_anchor = (
            f"Infant version (6–12 months) of the character. "
            f"Round face, wide eyes, same skin tone and hair texture inherited from adult form. "
            f"Adult descriptor for reference: {character_descriptor}"
        )

    else:  # adult (default)
        character_anchor = character_descriptor

    SCENE_SYSTEM = f"""You are a visual scene director for an animated story series.

You receive a narration segment and three already-made decisions:
the character's role, state, and emotional condition.

Your job: design the world around the character so the scene is
cinematic, alive, and emotionally true to the narration.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE ONLY RULE THAT NEVER BREAKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHARACTER APPEARANCE = COMPLETELY FROZEN.
Never alter face, hair, skin tone, clothing, or distinguishing marks.
Only the world changes. Only the emotion changes. Only the pose changes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU CHANGE EVERY SINGLE SCENE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Environment       → indoor / outdoor / abstract / surreal / cosmic /
                    microscopic / historical / dreamlike — never repeat settings

Time & light      → dawn, midday, dusk, midnight, storm, candlelight,
                    neon, fire, moonlight — light defines mood

Camera            → vary radically scene to scene:
                    extreme close-up, tight portrait, medium, wide,
                    establishing, bird's eye, worm's eye, over-the-shoulder,
                    dutch angle, silhouette against light

Scale             → character filling frame vs tiny in vast environment

Pose & gesture    → never static. Use verbs:
                    stumbling, reaching, turning, collapsing, sprinting,
                    crouching, throwing head back, pressing hands to chest

Micro-expression  → one specific face detail that sells the emotion:
                    jaw clenched, one eye squinting, lip caught between teeth,
                    nostrils flared, brow furrowed asymmetrically

Secondary world   → crowds, shadows, other figures, objects, symbols,
                    weather effects, particles, reflections, fire, water

Color palette     → warm vs cool, saturated vs muted, high contrast vs soft —
                    must match the emotional temperature of the scene

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAKING SCENES FEEL ALIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every scene must be mid-action — capture the "just did / is doing / about to" moment.

At least one thing in the environment must be REACTING to the character
or to the emotional weight of the moment:
  → hair or fabric mid-movement
  → crowd turning to look or turning away
  → light shifting or flickering
  → dust, petals, embers, rain responding to movement
  → reflections, shadows, ripples

Emotion lives in BOTH face AND body simultaneously.

Camera angle must serve the emotion:
  vulnerability or defeat    → high angle, character looks small
  power or defiance          → low angle, character looms
  intimacy or internal pain  → extreme close-up, details matter
  isolation or vastness      → wide, character tiny in environment
  disorientation or tension  → dutch angle or fragmented framing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSENT / SYMBOLIC / SILHOUETTE SCENES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If state is "absent":
  No character whatsoever. Build a pure environment or object scene.
  The world itself carries the full emotional weight of the narration.

If state is "silhouette" or "symbolic":
  Use the anchor descriptor provided exactly.
  Scene feels archetypal — universal, not literal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT SCENE ASSETS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHARACTER STATE:     {state}
CHARACTER ANCHOR:    {character_anchor}
EMOTION:             {result["emotion"] or "neutral"}
ROLE IN SCENE:       {result["role"]}
STYLE LOCK:          {style_lock}
NEVER INCLUDE:       {negative_anchors or "nothing specified"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (use these headers exactly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ANALYSIS]
Segment meaning:       what is the core idea or emotion of this narration?
Visual metaphor:       the single strongest image that represents this idea
Environment:           setting, time, atmosphere, light source
Action:                what is the character doing at the peak moment?
Camera:                angle, scale, framing — and why this serves the scene
Secondary elements:    what else exists in this frame?
Emotional temperature: warm/cool, saturated/muted, heavy/light — and why

[GENERATE PROMPT]
The full image generation prompt. 100–130 words. Continuous prose, no bullets.
Structure: CHARACTER ANCHOR first (if present) → environment → action →
camera → secondary elements → close with style lock and negative anchors.
If state is absent: begin with environment directly.
Every sentence must earn its place. No filler. No repetition.

[REFINE PROMPT]
20–40 words. Action and mood only.
No character appearance. No style language.
This nudges the already-generated image toward the right emotional truth.
One moment. One feeling. One environmental detail.
Write it like a director's note whispered mid-take."""

    try:
        scene_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SCENE_SYSTEM},
                {"role": "user",   "content": (
                    f'Narration segment:\n"""{segment_text.strip()}"""\n\n'
                    f"State: {result['state']} | "
                    f"Role: {result['role']} | "
                    f"Emotion: {result['emotion'] or 'neutral'}\n\n"
                    "Design the scene."
                )},
            ],
            temperature=0.65,
            max_tokens=1200,
        )

        full_response = scene_response.choices[0].message.content.strip()
        result["analysis"]        = full_response
        result["generate_prompt"] = _extract_block(full_response, "GENERATE PROMPT")
        result["refine_prompt"]   = _extract_block(full_response, "REFINE PROMPT")

        # Safety net: compress generate_prompt → refine if refine came back empty
        if not result["refine_prompt"] and result["generate_prompt"]:
            result["refine_prompt"] = " ".join(result["generate_prompt"].split()[:35])
            logging.warning("⚠️  REFINE PROMPT missing — compressed from GENERATE PROMPT.")

        # Safety net: generate_prompt empty entirely
        if not result["generate_prompt"]:
            raise ValueError("generate_prompt extracted as empty string")

        logging.debug(f"[WHISK FULL ANALYSIS]\n{full_response}")
        logging.info(f"[WHISK GENERATE] {result['generate_prompt'][:120]}...")
        logging.info(f"[WHISK REFINE]   {result['refine_prompt'][:80]}...")

    except Exception as e:
        logging.error(f"❌ Step 2 scene generation failed: {e} — using fallback prompt")
        result["generate_prompt"] = (
            f"{character_descriptor} "
            f"Scene: {segment_text.strip()}. "
            f"Style: {style_lock}."
            + (f" Avoid: {negative_anchors}." if negative_anchors else "")
        )
        result["refine_prompt"] = segment_text.strip()[:120]
        result["fallback_used"] = True

    return result


# ═══════════════════════════════════════════════════════════════
# PROMPT HELPERS
# ═══════════════════════════════════════════════════════════════

def _extract_block(text: str, header: str) -> str:
    """
    Extract the content under a [HEADER] block,
    stopping at the next [HEADER] or end of string.
    Strips surrounding quotes and whitespace.
    """
    tag = f"[{header}]"
    if tag not in text:
        return ""
    after = text.split(tag)[-1].strip()
    next_header = re.search(r'\[[A-Z ]+\]', after)
    if next_header:
        after = after[:next_header.start()].strip()
    return after.strip().strip('"').strip()


def _safe_json(raw: str) -> dict:
    """
    Parse JSON safely.
    Falls back to regex extraction on parse failure.
    Returns empty dict if all attempts fail.
    """
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r'\{[\s\S]+\}', raw)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


# ═══════════════════════════════════════════════════════════════
# HIGH-LEVEL API — INITIALIZE CHARACTER
# ═══════════════════════════════════════════════════════════════

def initialize_whisk_character(
    character_id: str,
    db,
    cookie: str = None,
) -> Optional[Dict[str, Any]]:
    """
    Initialize Whisk character for video generation.

    Loads character from MongoDB, verifies image exists, does one
    test upload to confirm cookie is valid. Returns context dict.

    Per-segment generation is handled by whisk_full_pipeline() which
    calls whisk_generate.mjs — it does its own upload internally.

    Retries up to 3 times on auth failures with cookie rotation.

    Returns:
        Dict with character context or None on failure.
    """
    try:
        from bson.objectid import ObjectId

        logging.info(f"🎭 Initializing Whisk character: {character_id}")

        character = db.characters.find_one({"_id": ObjectId(character_id)})
        if not character:
            raise WhiskCharacterError(f"Character {character_id} not found in database")

        logging.info(f"📦 Loaded character: {character['name']}")

        char_folder = character.get("character_folder")
        if not char_folder:
            raise WhiskCharacterError(f"Character {character_id} has no character_folder")

        character_image_path = os.path.join(
            "character_uploads",
            char_folder,
            character.get("base_image")
        )

        if not os.path.exists(character_image_path):
            raise WhiskCharacterError(f"Character image not found: {character_image_path}")

        logging.info(f"🖼️  Character image: {character_image_path}")
        logging.info("☁️  Uploading character to Whisk cloud...")

        subject_ref_id = None
        max_attempts   = 3

        for attempt in range(1, max_attempts + 1):
            try:
                current_cookie = cookie if (cookie and attempt == 1) else None
                if current_cookie is None:
                    if read_active_cookie:
                        current_cookie = read_active_cookie()
                    else:
                        raise WhiskCharacterError(
                            "Cannot read cookie: read_active_cookie not available"
                        )

                if not current_cookie:
                    logging.warning(f"⚠️ No cookie available on attempt {attempt}")
                    time.sleep(3)
                    continue

                logging.info(f"☁️  Upload attempt {attempt}/{max_attempts}...")
                subject_ref_id = whisk_upload_character(character_image_path, current_cookie)

                if subject_ref_id:
                    break

                logging.warning(
                    f"⚠️ Upload attempt {attempt}/{max_attempts} returned no subject_ref_id"
                )

            except Whisk401Error as e:
                logging.warning(
                    f"⚠️ Upload attempt {attempt}/{max_attempts} failed (auth): {e}"
                )
                if attempt < max_attempts:
                    logging.info("⏳ Waiting 5s for cookie daemon to rotate...")
                    time.sleep(5)
                else:
                    logging.error("❌ All upload attempts exhausted after auth failures")

            except WhiskCharacterError:
                raise

            except Exception as e:
                logging.error(f"❌ Upload attempt {attempt} unexpected error: {e}")
                if attempt < max_attempts:
                    time.sleep(3)

        if not subject_ref_id:
            raise WhiskCharacterError(
                f"Failed to upload character to Whisk cloud after {max_attempts} attempts"
            )

        logging.info(f"✅ Character uploaded successfully: {subject_ref_id}")
        logging.info(f"🎭 Whisk character initialized: {character['name']}")

        return {
            "media_id":             subject_ref_id,
            "character_id":         str(character_id),
            "character_name":       character["name"],
            "character_descriptor": character.get("character_descriptor", ""),
            "style_lock":           character.get("style_lock", ""),
            "negative_anchors":     character.get("negative_anchors", ""),
            "character_image_path": character_image_path,
        }

    except Exception as e:
        logging.error(f"❌ Failed to initialize Whisk character: {e}")
        logging.error(traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════════════════
# HIGH-LEVEL API — GENERATE IMAGE FOR SEGMENT
# ═══════════════════════════════════════════════════════════════

def whisk_generate_for_segment(
    segment: Dict[str, Any],
    whisk_shared_state: Dict[str, Any],
    media_lock,
    duration_needed: float,
    segment_index: int,
) -> List[Dict[str, Any]]:
    """
    Generate one Whisk character image for a video segment.

    Calls whisk_full_pipeline() → whisk_generate.mjs → Node.js module API:
        project → upload SUBJECT → generateImage(generate_prompt) → refine(refine_prompt)

    build_whisk_refine_prompt() returns a dict — generate_prompt and refine_prompt
    are unpacked separately before being passed to whisk_full_pipeline().

    On failure, rotates cookie and retries once.

    Returns single-item list with full duration, or [] on failure.
    """

    if not whisk_shared_state:
        logging.error(f"❌ [Seg {segment_index}] No Whisk context available")
        return []

    if not read_active_cookie:
        logging.error(f"❌ [Seg {segment_index}] Cookie reader not available")
        return []

    current_cookie = read_active_cookie()
    if not current_cookie:
        logging.error(f"❌ [Seg {segment_index}] No active cookie")
        return []

    character_descriptor = whisk_shared_state.get("character_descriptor", "")
    style_lock           = whisk_shared_state.get("style_lock", "")
    negative_anchors     = whisk_shared_state.get("negative_anchors", "")
    character_name       = whisk_shared_state.get("character_name", "Unknown")
    character_image_path = whisk_shared_state.get("character_image_path")

    if not character_image_path or not os.path.exists(character_image_path):
        logging.error(f"❌ [Seg {segment_index}] Character image not found: {character_image_path}")
        return []

    # ── Build scene prompts (returns dict) ────────────────────────
    prompt_data = build_whisk_refine_prompt(
        segment_text=segment.get("text", ""),
        character_descriptor=character_descriptor,
        style_lock=style_lock,
        negative_anchors=negative_anchors,
        client=openai_client,  # pass the module-level OpenAI client explicitly
    )

    # Unpack the two prompts from the dict
    generate_prompt = prompt_data["generate_prompt"]
    refine_prompt   = prompt_data["refine_prompt"]

    if prompt_data.get("fallback_used"):
        logging.warning(f"⚠️  [Seg {segment_index}] Prompt fallback was used")

    logging.info(f"🎨 [Seg {segment_index}] Generate: {generate_prompt[:80]}...")
    logging.info(f"🎨 [Seg {segment_index}] Refine:   {refine_prompt[:60]}...")
    logging.info(
        f"🎨 [Seg {segment_index}] State={prompt_data['state']} | "
        f"Role={prompt_data['role']}"
    )

    output_dir = Path("whisk_outputs") / f"seg_{segment_index:03d}"

    def _build_result(path: str) -> List[Dict[str, Any]]:
        return [
            {
                "media": {
                    "type":      "image",
                    "path":      path,
                    "source":    "whisk_character",
                    "duration":  duration_needed,
                    "character": character_name,
                    "tag":       None,
                    "embedding": None,
                },
                "duration":  duration_needed,
                "has_audio": False,
            }
        ]

    def _run(cookie: str) -> str:
        return whisk_full_pipeline(
            character_image_path=character_image_path,
            generate_prompt=generate_prompt,
            refine_prompt=refine_prompt,
            cookie=cookie,
            output_dir=output_dir,
            timeout=WHISK_GENERATE_TIMEOUT,
        )

    # ── First attempt ──────────────────────────────────────────
    try:
        refined_image_path = _run(current_cookie)
        logging.info(
            f"✅ [Seg {segment_index}] Whisk image generated: "
            f"{os.path.basename(refined_image_path)}"
        )
        return _build_result(refined_image_path)

    except Whisk401Error as e:
        logging.warning(f"⚠️ [Seg {segment_index}] First attempt failed: {e}")

        # Rotate cookie if auth-related
        if any(k in str(e).lower() for k in ["auth", "cookie", "401", "unauthenticated", "unauthorized"]):
            if signal_force_rotate:
                logging.warning("🔐 Signalling cookie rotation before retry...")
                signal_force_rotate()
                logging.info("⏳ Waiting 5s for daemon to rotate...")
                time.sleep(5)

        # ── Retry with fresh cookie ────────────────────────────
        try:
            fresh_cookie = read_active_cookie()
            if not fresh_cookie:
                logging.error(f"❌ [Seg {segment_index}] No cookie for retry")
                return []

            refined_image_path = _run(fresh_cookie)
            logging.info(f"✅ [Seg {segment_index}] Retry successful")
            return _build_result(refined_image_path)

        except Whisk401Error as retry_error:
            logging.error(f"❌ [Seg {segment_index}] Retry also failed: {retry_error}")
            send_telegram_notification(
                f"🚨 Whisk DOUBLE FAILURE\n"
                f"Segment: {segment_index}\n"
                f"First: {e}\n"
                f"Retry: {retry_error}",
                "Whisk Critical",
                "https://claude.ai/error",
                "system"
            )
            return []

        except Exception as retry_error:
            logging.error(f"❌ [Seg {segment_index}] Unexpected retry error: {retry_error}")
            logging.error(traceback.format_exc())
            return []

    except Exception as e:
        logging.error(f"❌ [Seg {segment_index}] Unexpected Whisk error: {e}")
        logging.error(traceback.format_exc())
        return []


# ═══════════════════════════════════════════════════════════════
# MODULE METADATA
# ═══════════════════════════════════════════════════════════════

__all__ = [
    "initialize_whisk_character",
    "whisk_generate_for_segment",
    "build_whisk_refine_prompt",
    "Whisk401Error",
    "WhiskCharacterError",
]

__version__ = "5.0.0"