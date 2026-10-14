"""
Black dissolve transitions between consecutive CLIP segments in the final concat list.

Back-to-back [CLIP:…] chunks are merged with FFmpeg xfade (fadeblack) so each join
dissolves through black — no bright flash (photosensitive-safe).

Pipeline toggles live in the module constants below (future: wire to env).
"""

from __future__ import annotations

import logging
import os
import subprocess
import uuid

from helper.banner_backend import (
    _has_audio_stream,
    _probe_duration,
    build_concat_chunk_types,
)

# --- Pipeline config (future env: CLIP_TO_CLIP_TRANSITIONS=1|0) ---
CLIP_TO_CLIP_TRANSITIONS_ENABLED = True

# --- Dissolve length in seconds (future env: CLIP_TO_CLIP_DISSOLVE_SEC) ---
CLIP_TO_CLIP_DISSOLVE_SEC = 0.8
# Max dissolve if env override is added later (future env: CLIP_TO_CLIP_DISSOLVE_SEC_MAX)
CLIP_TO_CLIP_DISSOLVE_SEC_MAX = 1.5
CLIP_TO_CLIP_DISSOLVE_SEC_MIN = 0.06

DEFAULT_FPS = 30


def clip_segment_transitions_enabled() -> bool:
    return bool(CLIP_TO_CLIP_TRANSITIONS_ENABLED)


def _requested_dissolve_sec() -> float:
    return max(
        CLIP_TO_CLIP_DISSOLVE_SEC_MIN,
        min(float(CLIP_TO_CLIP_DISSOLVE_SEC), CLIP_TO_CLIP_DISSOLVE_SEC_MAX),
    )


def _dissolve_video_norm(canvas_w: int, canvas_h: int, fps: int = DEFAULT_FPS) -> str:
    """Canvas + CFR for xfade — omit setpts (xfade requires stable frame rate)."""
    w, h = int(canvas_w), int(canvas_h)
    r = int(fps)
    return (
        f"scale={w}:{h}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={r},format=yuv420p"
    )


def _compute_dissolve_xfade(
    left_dur: float, right_dur: float, requested: float
) -> float | None:
    """Pick xfade duration that fits both clips."""
    frame = 1.0 / float(DEFAULT_FPS)
    min_seg = max(0.12, 2.0 * frame)
    if left_dur < min_seg or right_dur < min_seg:
        return None
    xf = min(requested, left_dur * 0.5, right_dur * 0.5)
    xf = max(0.06, xf)
    if xf >= left_dur * 0.98 or xf >= right_dur * 0.98:
        return None
    return xf


def merge_clip_pair_black_dissolve(
    left_path: str,
    right_path: str,
    output_path: str,
    *,
    canvas_w: int,
    canvas_h: int,
    dissolve_sec: float | None = None,
) -> str | None:
    """Merge two clip chunks with a fadeblack (through-black) dissolve."""
    left_dur = _probe_duration(left_path)
    right_dur = _probe_duration(right_path)
    if min(left_dur, right_dur) <= 0:
        return None

    requested = dissolve_sec if dissolve_sec is not None else _requested_dissolve_sec()
    xf = _compute_dissolve_xfade(left_dur, right_dur, requested)
    if not xf:
        logging.warning(
            "Clip black dissolve skipped — chunks too short "
            f"(left={left_dur:.3f}s right={right_dur:.3f}s)"
        )
        return None

    offset = max(0.0, left_dur - xf)
    out_dur = left_dur + right_dur - xf
    norm = _dissolve_video_norm(canvas_w, canvas_h, DEFAULT_FPS)

    left_a = _has_audio_stream(left_path)
    right_a = _has_audio_stream(right_path)

    vf = (
        f"[0:v]{norm}[v0];"
        f"[1:v]{norm}[v1];"
        f"[v0][v1]xfade=transition=fadeblack:duration={xf:.6f}:offset={offset:.6f}[vout]"
    )

    if left_a and right_a:
        af = f"[0:a][1:a]acrossfade=d={xf:.6f}:c1=tri:c2=tri[aout]"
        filt = f"{vf};{af}"
        maps = ["-map", "[vout]", "-map", "[aout]"]
    else:
        filt = (
            f"{vf};"
            f"anullsrc=r=44100:cl=stereo:d={out_dur:.6f}[aout]"
        )
        maps = ["-map", "[vout]", "-map", "[aout]"]

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        left_path,
        "-i",
        right_path,
        "-filter_complex",
        filt,
        *maps,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False
        )
    except Exception as exc:
        logging.warning(f"Clip black dissolve ffmpeg failed to start: {exc}")
        return None

    if r.returncode != 0 or not os.path.exists(output_path):
        err = (r.stderr or r.stdout or "")[-800:]
        logging.warning(f"Clip black dissolve failed ({r.returncode}): {err}")
        return None

    probed = _probe_duration(output_path)
    if probed <= 0:
        logging.warning(f"Clip black dissolve output has no duration: {output_path}")
        return None
    return output_path


def apply_clip_to_clip_transitions(
    chunk_paths: list,
    vid_segments: list[dict],
    *,
    chunk_folder: str,
    canvas_w: int,
    canvas_h: int,
    talking_heads_mode: bool = False,
    has_clips: bool = False,
    batch_groups: dict | None = None,
) -> list:
    """
    Merge consecutive CLIP concat units with fadeblack dissolves.

    Returns a new list (often shorter than chunk_paths). On mismatch or errors,
    returns the original chunk_paths unchanged.
    """
    if not chunk_paths or not vid_segments:
        return chunk_paths

    chunk_types = build_concat_chunk_types(
        vid_segments,
        batch_groups=batch_groups,
        talking_heads_mode=talking_heads_mode,
        has_clips=has_clips,
    )
    if len(chunk_types) != len(chunk_paths):
        logging.warning(
            "Clip-to-clip transitions skipped — chunk type count "
            f"({len(chunk_types)}) != chunk path count ({len(chunk_paths)})"
        )
        return chunk_paths

    work_dir = os.path.join(chunk_folder, "clip_transition_work")
    os.makedirs(work_dir, exist_ok=True)
    dissolve_sec = _requested_dissolve_sec()

    paths = list(chunk_paths)
    types = list(chunk_types)
    merged_count = 0
    i = 0
    while i < len(paths):
        if i + 1 >= len(paths) or not (types[i] and types[i + 1]):
            i += 1
            continue

        left = paths[i]
        right = paths[i + 1]
        if not left or not right or not os.path.exists(str(left)) or not os.path.exists(
            str(right)
        ):
            logging.warning(
                f"Clip-to-clip dissolve skipped at boundary {i} — missing chunk file"
            )
            i += 1
            continue

        out_path = os.path.join(
            work_dir,
            f"clip_dissolve_{i}_{uuid.uuid4().hex[:8]}.mp4",
        )
        merged = merge_clip_pair_black_dissolve(
            str(left),
            str(right),
            out_path,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            dissolve_sec=dissolve_sec,
        )
        if not merged:
            i += 1
            continue

        paths[i : i + 2] = [merged]
        types[i : i + 2] = [True]
        merged_count += 1
        # Do not advance i — try chaining A+B with C when three clips in a row

    if merged_count:
        logging.info(
            f"Clip-to-clip transitions | black dissolve merges={merged_count} "
            f"(requested xfade ~{dissolve_sec:.3f}s, fadeblack)"
        )
    else:
        logging.info("Clip-to-clip transitions enabled — no adjacent CLIP pairs merged")

    return paths if merged_count else chunk_paths


def concat_chunk_paths_copy(
    chunk_paths: list,
    output_path: str,
    *,
    concat_list_path: str | None = None,
) -> str | None:
    """Concatenate video chunks with FFmpeg concat demuxer (stream copy)."""
    paths = [p for p in chunk_paths if p and os.path.exists(str(p))]
    if not paths:
        return None

    list_path = concat_list_path or output_path + ".txt"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    except Exception as exc:
        logging.warning(f"Concat failed to start: {exc}")
        return None
    if r.returncode != 0 or not os.path.exists(output_path):
        return None
    return output_path
