import os
import time
import json
import uuid
import base64
import logging
import asyncio
import threading
from pathlib import Path

import websockets
from pydub import AudioSegment, effects

# =========================
# Cartesia Constants
# =========================
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "sk_car_yMdWHU97k3KvtYat2VywcS")
CARTESIA_WS_URL  = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2025-04-16"
CARTESIA_MODEL   = "sonic-3"

CARTESIA_VOICES = {
    "carson": "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
}

TARGET_SR_HZ       = 48000
TARGET_CH          = 2
TARGET_SW          = 2
GLOBAL_HEADROOM_DB = 3.0
FINAL_PAD_MS       = 100
FINAL_PAD_SEC      = FINAL_PAD_MS / 1000.0


# =========================
# Voice resolution
# =========================
def resolve_cartesia_voice(raw_voice: str) -> str | None:
    v = raw_voice.strip()
    if v in CARTESIA_VOICES:
        resolved = CARTESIA_VOICES[v]
        logging.info(f"Cartesia voice resolved: '{v}' → '{resolved}'")
        return resolved
    return None


# =========================
# PCM → AudioSegment
# =========================
def _pcm_to_audiosegment(pcm_bytes: bytes) -> AudioSegment:
    audio = AudioSegment(
        data         = pcm_bytes,
        sample_width = 2,
        frame_rate   = TARGET_SR_HZ,
        channels     = 1,
    )
    return (audio
            .set_channels(TARGET_CH)
            .set_sample_width(TARGET_SW)
            .set_frame_rate(TARGET_SR_HZ)
            .high_pass_filter(30))


# =========================
# Build nested chunk for SRT
# =========================
def _build_nested_chunk(
    segment_text:    str,
    word_marks:      list[dict],
    char_offset:     int,
    audio_offset_ms: float,
) -> dict | None:
    if not word_marks:
        return None

    chunks   = []
    char_pos = char_offset

    for wm in word_marks:
        word_text = wm["word"]
        chunks.append({
            "start":      char_pos,
            "end":        char_pos + len(word_text),
            "start_time": wm["start"] * 1000.0 + audio_offset_ms,
            "end_time":   wm["end"]   * 1000.0 + audio_offset_ms,
            "value":      word_text,
        })
        char_pos += len(word_text) + 1

    if not chunks:
        return None

    return {
        "start":      char_offset,
        "end":        char_offset + len(segment_text),
        "start_time": chunks[0]["start_time"],
        "end_time":   chunks[-1]["end_time"],
        "value":      segment_text,
        "chunks":     chunks,
    }


# =========================
# Async WebSocket core
# =========================
async def _cartesia_ws_generate(
    spoken_segments: list[tuple[int, str]],
    voice_id:        str,
    max_retries:     int = 12,
    window_size:     int = 5,
) -> dict[int, dict]:
    """
    Streams all segments over a single WebSocket connection.

    Single context_id → prosody maintained across entire script.

    Uses a sliding window semaphore so up to window_size segments
    are in-flight at once — fast enough for large scripts but
    controlled enough to avoid overwhelming Cartesia.

    Segment boundary tracking:
    - flush_done marks the boundary between segments
    - current_seg_pos advances on each flush_done received
    - Word timestamps are relative to each segment's own audio
      (they reset to 0.0 at each flush boundary)
    """
    n          = len(spoken_segments)
    context_id = str(uuid.uuid4())

    ws_url = (
        f"{CARTESIA_WS_URL}"
        f"?cartesia_version={CARTESIA_VERSION}"
        f"&api_key={CARTESIA_API_KEY}"
    )

    base_payload = {
        "model_id": CARTESIA_MODEL,
        "voice":    {"mode": "id", "id": voice_id},
        "output_format": {
            "container":   "raw",
            "encoding":    "pcm_s16le",
            "sample_rate": TARGET_SR_HZ,
        },
        "context_id":                context_id,
        "add_timestamps":            True,
        "use_normalized_timestamps": False,
    }

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            async with websockets.connect(
                ws_url,
                ping_interval = 20,
                max_size      = 10 * 1024 * 1024,
            ) as ws:
                logging.info(
                    f"✅ WebSocket connected | "
                    f"context={context_id} | "
                    f"{n} segments | "
                    f"window={window_size}"
                )

                seg_audio_chunks : dict[int, list[bytes]] = {}
                seg_word_marks   : dict[int, list[dict]]  = {}
                current_seg_pos  = 0

                # Sliding window: allows up to window_size segments in-flight.
                # Sender acquires a slot before sending each segment.
                # Receiver releases a slot on each flush_done.
                window = asyncio.Semaphore(window_size)

                # ─── SEND ─────────────────────────────────────────────
                async def _send_all():
                    for pos, (orig_idx, text) in enumerate(spoken_segments):
                        is_last = (pos == n - 1)

                        # Acquire window slot — blocks if window_size
                        # segments are already in-flight
                        await window.acquire()

                        await ws.send(json.dumps({
                            **base_payload,
                            "transcript": text,
                            "continue":   not is_last,
                        }))
                        logging.debug(
                            f"→ Sent seg pos={pos} "
                            f"orig={orig_idx} "
                            f"continue={not is_last} "
                            f"'{text[:40]}…'"
                        )

                        await ws.send(json.dumps({
                            **base_payload,
                            "transcript": "",
                            "flush":      True,
                            "continue":   not is_last,
                        }))
                        logging.debug(f"→ Sent flush after pos={pos}")

                # ─── RECEIVE ──────────────────────────────────────────
                async def _receive_all():
                    nonlocal current_seg_pos

                    async for raw_msg in ws:
                        try:
                            event = json.loads(raw_msg)
                        except json.JSONDecodeError:
                            continue

                        etype = event.get("type")

                        if etype == "chunk":
                            data = event.get("data")
                            if data:
                                seg_audio_chunks.setdefault(
                                    current_seg_pos, []
                                ).append(base64.b64decode(data))

                        elif etype == "timestamps":
                            ts     = event.get("word_timestamps") or {}
                            words  = ts.get("words",  [])
                            starts = ts.get("start",  [])
                            ends   = ts.get("end",    [])
                            marks  = [
                                {
                                    "word":  w,
                                    "start": float(s),
                                    "end":   float(e),
                                }
                                for w, s, e in zip(words, starts, ends)
                            ]
                            seg_word_marks.setdefault(
                                current_seg_pos, []
                            ).extend(marks)

                        elif etype == "flush_done":
                            flush_id = event.get("flush_id", "?")
                            logging.info(
                                f"← flush_done flush_id={flush_id} | "
                                f"sealed pos={current_seg_pos} | "
                                f"chunks={len(seg_audio_chunks.get(current_seg_pos, []))} | "
                                f"marks={len(seg_word_marks.get(current_seg_pos, []))}"
                            )
                            current_seg_pos += 1
                            # Release window slot — sender can send next segment
                            window.release()

                        elif etype == "done" and event.get("done"):
                            logging.info("← done: all generation complete")
                            break

                        elif etype == "error":
                            raise RuntimeError(
                                f"Cartesia error: {event.get('error')}"
                            )

                await asyncio.gather(_send_all(), _receive_all())

                # ─── Map pos → orig_idx → result ──────────────────────
                results = {}
                for pos, (orig_idx, text) in enumerate(spoken_segments):
                    audio_chunks = seg_audio_chunks.get(pos, [])
                    word_marks   = seg_word_marks.get(pos, [])

                    if not audio_chunks:
                        logging.error(
                            f"Segment {orig_idx} (pos={pos}): "
                            f"no audio received"
                        )
                        results[orig_idx] = {
                            "success": False,
                            "text":    text,
                        }
                        continue

                    audio = _pcm_to_audiosegment(b"".join(audio_chunks))

                    results[orig_idx] = {
                        "success":    True,
                        "audio":      audio,
                        "duration":   audio.duration_seconds,
                        "word_marks": word_marks,
                        "seg_start":  word_marks[0]["start"] if word_marks else 0.0,
                        "seg_end":    word_marks[-1]["end"]  if word_marks else audio.duration_seconds,
                        "text":       text,
                    }

                    logging.info(
                        f"✅ Segment {orig_idx} (pos={pos}): "
                        f"{audio.duration_seconds:.3f}s | "
                        f"{len(word_marks)} words | "
                        f"{results[orig_idx]['seg_start']:.3f}s → "
                        f"{results[orig_idx]['seg_end']:.3f}s"
                    )

                return results

        except Exception as e:
            logging.error(
                f"WebSocket attempt {attempt}/{max_retries} failed: {e}"
            )
            if attempt >= max_retries:
                raise
            wait = min(60.0, 2.0 ** attempt)
            logging.info(f"Retrying in {wait:.1f}s…")
            await asyncio.sleep(wait)

    return {}


# =========================
# Run async in dedicated thread
# Safe to call from Flask/any sync context
# =========================
def _run_async_in_thread(coro) -> dict:
    """
    Runs an async coroutine in a brand new event loop
    in a dedicated thread.

    Safe to call from any context including threads that
    already have a running event loop (Flask, multiprocessing
    workers etc). Avoids 'cannot schedule new futures after
    shutdown' error from asyncio.run() in worker threads.
    """
    result    = {}
    exception = {}

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
        except Exception as e:
            exception["value"] = e
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()

    if "value" in exception:
        raise exception["value"]

    return result.get("value", {})


# =========================
# Main orchestrator
# =========================
def generate_audio_segments_cartesia(
    segments:     list[dict],
    voice:        str,
    full_script:  str,
    voice_id:     str,
    output_dir:   str = "static/audio",
    max_retries:  int = 12,
    current_user  = None,
) -> tuple:
    """
    Cartesia audio generation via WebSocket.

    Why WebSocket:
    - Single context_id maintains prosody across all segments
      (natural pitch/tone/emotion throughout the script)
    - flush boundaries give exact per-segment audio
    - Word timestamps reset to 0.0 per segment (exact, no mapping)
    - No tokenization guessing — each flush = one segment

    Timestamp contract:
    - seg_start = FINAL_PAD_SEC + running_offset + word_marks[0].start
    - seg_end   = FINAL_PAD_SEC + running_offset + word_marks[-1].end
    - running_offset accumulates previous segment durations
    - final_audio = FINAL_PAD_MS + combined + FINAL_PAD_MS
    """
    from helper.telegram_notification_engine import send_telegram_error

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    ts_stamp   = int(time.time())
    base_name  = f"narration_{ts_stamp}"

    logging.info("")
    logging.info("=" * 60)
    logging.info("🚀 CARTESIA WEBSOCKET GENERATION START")
    logging.info("=" * 60)
    logging.info(f"   Voice ID : '{voice_id}'")
    logging.info(f"   Segments : {len(segments)}")
    logging.info("=" * 60)

    # Identify spoken segments in order
    spoken = [
        (idx, (seg.get("text") or "").strip())
        for idx, seg in enumerate(segments)
        if (seg.get("text") or "").strip()
        and not (seg.get("text") or "").strip().startswith("[CLIP:")
    ]

    if not spoken:
        logging.error("No spoken segments found")
        return ([], None, None, [], [], None)

    logging.info(f"Spoken segments: {len(spoken)}/{len(segments)}")

    # ─── WebSocket generation ──────────────────────────────────────────
    try:
        results = _run_async_in_thread(
            _cartesia_ws_generate(
                spoken_segments = spoken,
                voice_id        = voice_id,
                max_retries     = max_retries,
                window_size     = 5,
            )
        )
    except Exception as e:
        logging.error(f"❌ WebSocket generation failed: {e}")
        try:
            send_telegram_error(
                f"🎙️ ❌ Cartesia WebSocket Failed\n\n"
                f"Voice ID: `{voice_id}`\n"
                f"Error: {e}",
                current_user,
            )
        except Exception:
            pass
        return ([], None, None, [], [], None)

    # ─── Retry any failures ────────────────────────────────────────────
    failed_spoken = [
        (idx, text) for idx, text in spoken
        if not results.get(idx, {}).get("success")
    ]

    if failed_spoken:
        logging.warning(
            f"⚠️ {len(failed_spoken)} segments failed — "
            f"retrying individually via WebSocket…"
        )
        for idx, text in failed_spoken:
            try:
                retry_result = _run_async_in_thread(
                    _cartesia_ws_generate(
                        spoken_segments = [(idx, text)],
                        voice_id        = voice_id,
                        max_retries     = 3,
                        window_size     = 1,
                    )
                )
                if retry_result.get(idx, {}).get("success"):
                    results[idx] = retry_result[idx]
                    logging.info(f"✅ Retry succeeded: segment {idx}")
                else:
                    logging.error(f"❌ Retry failed: segment {idx}")
            except Exception as e:
                logging.error(f"❌ Retry exception segment {idx}: {e}")

    # ─── Build outputs ─────────────────────────────────────────────────
    segment_audio_paths = []
    segment_timestamps  = []
    nested_chunks       = []
    combined_audio      = AudioSegment.silent(duration=0)
    char_offset         = 0
    running_offset_sec  = 0.0
    last_successful_idx = None

    for idx in range(len(segments)):
        text   = (segments[idx].get("text") or "").strip()
        result = results.get(idx)

        if not text or text.startswith("[CLIP:") or not result or not result.get("success"):
            segment_audio_paths.append(None)
            segment_timestamps.append((0.0, 0.0))
            if text:
                char_offset += len(text) + 1
            continue

        # Save segment WAV
        seg_wav = os.path.join(output_dir, f"{ts_stamp}_segment_{idx}.wav")
        try:
            result["audio"].export(seg_wav, format="wav")
        except Exception as e:
            logging.error(f"❌ Failed to export segment {idx} WAV: {e}")
            segment_audio_paths.append(None)
            segment_timestamps.append((0.0, 0.0))
            char_offset += len(text) + 1
            continue

        seg_start = FINAL_PAD_SEC + running_offset_sec + result["seg_start"]
        seg_end   = FINAL_PAD_SEC + running_offset_sec + result["seg_end"]

        segment_audio_paths.append(seg_wav)
        segment_timestamps.append((seg_start, seg_end))
        last_successful_idx = idx

        seg_audio_offset_ms = (FINAL_PAD_SEC + running_offset_sec) * 1000.0

        nested = _build_nested_chunk(
            segment_text    = text,
            word_marks      = result["word_marks"],
            char_offset     = char_offset,
            audio_offset_ms = seg_audio_offset_ms,
        )
        if nested:
            nested_chunks.append(nested)

        combined_audio     += result["audio"]
        running_offset_sec += result["duration"]
        char_offset        += len(text) + 1

    # Tail pad on last segment
    if last_successful_idx is not None:
        s, e = segment_timestamps[last_successful_idx]
        segment_timestamps[last_successful_idx] = (s, e + FINAL_PAD_SEC)

    # ─── Export final audio ────────────────────────────────────────────
    if len(combined_audio) == 0:
        logging.error("No audio to export")
        return ([], None, None, [], [], None)

    final_wav_path = os.path.join(output_dir, f"{base_name}.wav")
    final_mp3_path = os.path.join(output_dir, f"{base_name}.mp3")

    try:
        polished = effects.normalize(combined_audio, headroom=GLOBAL_HEADROOM_DB)
        polished = (polished
                    .fade_in(100).fade_out(100)
                    .set_frame_rate(TARGET_SR_HZ)
                    .set_channels(TARGET_CH)
                    .set_sample_width(TARGET_SW))
        final_audio = (
            AudioSegment.silent(duration=FINAL_PAD_MS)
            + polished
            + AudioSegment.silent(duration=FINAL_PAD_MS)
        )
        final_audio.export(final_wav_path, format="wav")
        final_audio.export(final_mp3_path, format="mp3", bitrate="320k")
        logging.info(
            f"✅ Final WAV: {final_wav_path} "
            f"({final_audio.duration_seconds:.3f}s)"
        )
        logging.info(f"✅ Final MP3: {final_mp3_path}")
    except Exception as e:
        logging.error(f"❌ Export failed: {e}")
        return ([], None, None, [], [], None)

    # ─── SRT ───────────────────────────────────────────────────────────
    from test2 import nested_list_to_srt

    srt_path = None
    if nested_chunks:
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        if not nested_list_to_srt(nested_chunks, srt_path):
            srt_path = None

    final_mp3_url = (
        f"http://localhost:5000/static/audio/{Path(final_mp3_path).name}"
        if final_mp3_path else None
    )

    successful = sum(1 for p in segment_audio_paths if p is not None)
    total_time = time.time() - start_time

    logging.info("")
    logging.info("=" * 60)
    logging.info(
        f"🎉 DONE — {successful}/{len(segments)} segments, "
        f"{total_time:.1f}s"
    )
    logging.info("=" * 60)

    return (
        segment_audio_paths,
        final_wav_path,
        final_mp3_url,
        segment_timestamps,
        nested_chunks,
        srt_path,
    )