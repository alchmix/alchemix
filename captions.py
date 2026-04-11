# captions.py - Whisper + TS-style caption chunking + ASS writer
from pathlib import Path
from typing import List, Tuple
import whisper

# (start, end, text)
Caption = Tuple[float, float, str]

# Simple global model cache
_WHISPER_MODELS = {}


def get_whisper_model(model_size: str = "base"):
    """Lazy-load and cache Whisper models."""
    global _WHISPER_MODELS
    if model_size not in _WHISPER_MODELS:
        print(f"🔊 Loading Whisper model ({model_size})...")
        _WHISPER_MODELS[model_size] = whisper.load_model(model_size)
        print("✅ Whisper model loaded")
    return _WHISPER_MODELS[model_size]


def generate_captions(
    video_path: Path,
    model_size: str = "base",
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    max_gap: float = 0.4,
) -> List[Caption]:
    """
    High-level API (same name as before, but TS-style logic):

    1. Transcribe with Whisper
    2. Merge short segments
    3. Wrap into 1–2 lines with max width
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = get_whisper_model(model_size)

    print(f"🎤 Transcribing audio from {video_path.name}...")
    try:
        result = model.transcribe(str(video_path), word_timestamps=False)
    except Exception as e:
        print(f"⚠️ Whisper transcription error: {e}")
        return []

    raw_segments = result.get("segments", [])
    if not raw_segments:
        print("⚠️ No segments returned from Whisper.")
        return []

    # 1) Convert to simple list
    temp: List[Caption] = []
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        temp.append((start, end, text))

    print(f"✅ Whisper returned {len(temp)} raw segments")

    # 2) Merge segments with tiny gaps (so we don't get spammy subtitles)
    merged: List[Caption] = []
    if not temp:
        return merged

    cur_start, cur_end, cur_text = temp[0]
    for s, e, t in temp[1:]:
        gap = s - cur_end
        if gap <= max_gap:
            # same caption block
            cur_end = e
            cur_text = (cur_text + " " + t).strip()
        else:
            merged.append((cur_start, cur_end, cur_text))
            cur_start, cur_end, cur_text = s, e, t
    merged.append((cur_start, cur_end, cur_text))

    print(f"✅ Merged to {len(merged)} blocks (gap <= {max_gap}s)")

    # 3) Wrap each merged text into up to max_lines lines
    final: List[Caption] = []
    for start, end, text in merged:
        chunks = _split_caption_text(text, max_chars_per_line, max_lines)
        if not chunks:
            continue

        total = len(chunks)
        duration = max(0.01, end - start)
        per = duration / total

        # Distribute chunks evenly inside original segment duration
        for i, chunk_text in enumerate(chunks):
            c_start = start + i * per
            c_end = start + (i + 1) * per
            final.append((c_start, c_end, chunk_text))

    print(f"✅ Final caption chunks: {len(final)}")
    return final


def _split_caption_text(
    text: str,
    max_chars_per_line: int,
    max_lines: int,
) -> List[str]:
    """
    Split text into up to `max_lines` lines with ~max_chars_per_line,
    trying to break at word boundaries.
    Returns list of strings where '\n' is inserted between lines.
    """
    text = " ".join(text.split())
    if not text:
        return []

    words = text.split(" ")
    lines: List[str] = []
    current = ""

    for w in words:
        candidate = (current + " " + w).strip()
        if len(candidate) <= max_chars_per_line:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w

    if current:
        lines.append(current)

    # Now group lines into up to max_lines per caption "block"
    blocks: List[str] = []
    buf: List[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) >= max_lines:
            blocks.append("\\N".join(buf))  # ASS line break
            buf = []
    if buf:
        blocks.append("\\N".join(buf))

    return blocks


def save_captions_srt(captions: List[Caption], output_path: Path):
    """Keep SRT export (unchanged API)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(captions, 1):
            f.write(f"{i}\n")
            f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
            # Convert ASS line breaks to SRT
            f.write(text.replace("\\N", "\n") + "\n\n")
    print(f"✅ Saved SRT: {output_path}")


def _format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_ass_file(
    captions: List[Caption],
    output_path: Path,
    playres_x: int = 1920,
    playres_y: int = 1080,
    fontsize: int = 56,
    bottom_margin: int = 80,
):
    """
    Write a styled ASS subtitle file (TS-style look):
    - Bottom center
    - Bold white with black outline
    - Safe vertical margins
    """
    header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: {playres_x}
PlayResY: {playres_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
; PrimaryColour/OutlineColour are in &HAABBGGRR format
Style: Default, Inter, {fontsize}, &H00FFFFFF, &H000000FF, &H00000000, &H64000000, -1, 0, 0, 0, 100,100,0,0,1,4,0,2,40,40,{bottom_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for (start, end, text) in captions:
            start_ass = _format_ass_time(start)
            end_ass = _format_ass_time(end)
            f.write(
                f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}\n"
            )

    print(f"✅ ASS subtitles written: {output_path}")


def _format_ass_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centi = int((seconds % 1) * 100)  # 1/100s
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centi:02d}"
