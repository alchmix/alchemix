# import os
# import re
# import gc
# import time
# import json
# import base64
# import random
# import logging
# from pathlib import Path
# from threading import Semaphore

# from dotenv import load_dotenv
# import requests
# from pydub import AudioSegment, effects
# from helper.cartesia import (
#     CARTESIA_API_KEY,
#     CARTESIA_VERSION,
#     resolve_cartesia_voice,
#     generate_audio_segments_cartesia,
# )
# from helper.telegram_notification_engine import send_telegram_notification, send_telegram_error

# # =========================
# # Env + Logging
# # =========================
# print("Loading environment variables...")
# load_dotenv()

# OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
# SPEECHIFY_TOKEN = os.getenv("SPEECHIFY_API_KEY")
# API_BASE        = os.getenv("OPENAI_BASE_URL")
# UPLOAD_FOLDER   = os.getenv("UPLOAD_FOLDER", "static/audio")

# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# # =========================
# # Constants
# # =========================
# SPEECHIFY_API_URL = "https://api.sws.speechify.com/v1/audio/speech"

# SPEECHIFY_VOICES = {
#     "cweb":         "5dfd8793-8f49-4205-aebb-5a9b47c18f76",
#     "andrew":       "affbf842-63c1-4979-8994-70deea0a38a6",
#     "stoic":        "940642fc-de60-4531-b77d-b03f0a38a6e6",
#     "uncensored":   "c9065edf-b025-4542-ab6a-50e1bdf475ee",
#     "jennifer":     "jennifer",
#     "liam":         "944a6cef-9a26-41f5-a2cd-ef29cac74bb2",
#     "nora":         "34872d14-035e-4416-b213-b5669a868091",
#     "explainer":    "06b97011-e9a3-4aa3-b2d8-a7696c24380a",
#     "war":          "fd8226b3-22ff-4739-b0ce-22e50adf95bb",
#     "health":       "8d16ac9d-94d6-46bc-8cf4-4a5ed04c5947",
#     "santa":        "ad5d0175-2a9e-4428-ba83-aca4a3220daa",
#     "dark":         "c091248b-9814-452d-ac94-eb71ea646a7b",
#     "noel":         "793d5cc2-4ddb-4e5d-bd44-5a406d0c0984",
#     "trice-reason": "ea8fc708-8054-4140-bf71-130292136a90",
#     "g-finance":    "3232a1d7-97ca-4952-a65c-0416d888d91e",
#     "park-stoic":   "0383c304-6173-482a-adb2-8989ffface25",
#     "emma":         "ec60fb61-5981-4cef-9b0d-82c0fb7c295a",
#     "edna":         "c44d297c-81eb-4c9b-8d4e-e833f632bbbb",
#     "g-fitness":    "ebbef645-600e-40e1-ac41-032d232294b9",
# }

# # Flat set of all known Speechify UUIDs for fast membership check
# SPEECHIFY_VOICE_IDS = set(SPEECHIFY_VOICES.values())

# OPENAI_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}

# TARGET_SR_HZ       = 48000
# TARGET_CH          = 2
# TARGET_SW          = 2
# SEG_FADE_MS        = 15
# GLOBAL_HEADROOM_DB = 3.0
# FINAL_PAD_MS       = 100
# FINAL_PAD_SEC      = FINAL_PAD_MS / 1000.0

# GLOBAL_API_SEMAPHORE = Semaphore(15)

# # In-process cache: voice_id → provider string
# # Avoids repeated API probes within the same process lifetime
# _provider_cache: dict[str, str] = {}


# # =========================
# # Custom exceptions
# # =========================
# class SpeechifyBadVoiceError(Exception):
#     """400 from Speechify — permanent, never retry."""
#     pass


# # =========================
# # UUID helpers
# # =========================
# _UUID_RE = re.compile(
#     r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
#     re.IGNORECASE,
# )


# def _is_uuid(v: str) -> bool:
#     return bool(_UUID_RE.fullmatch(v))


# def _is_cartesia_uuid(v: str):
#     """
#     Try to classify a raw UUID as Cartesia or Speechify.

#     Returns:
#         True  — confirmed Cartesia
#         False — confirmed Speechify
#         None  — ambiguous, needs API probe
#     """
#     if not _is_uuid(v):
#         return False

#     # Known Speechify UUID
#     if v in SPEECHIFY_VOICE_IDS:
#         return False

#     # Known Cartesia UUID (values of the alias map)
#     from helper.cartesia import CARTESIA_VOICES
#     if v in CARTESIA_VOICES.values():
#         return True

#     # Ambiguous — same UUID4 format used by both providers
#     return None


# # =========================
# # Provider probe
# # Used only for old channels that pre-date voice_provider storage.
# # Calls metadata endpoints (no audio generated, no credits used).
# # =========================
# def _probe_voice_provider(voice_id: str) -> str:
#     """
#     Probe both APIs to determine which provider owns this voice ID.
#     Uses metadata-only endpoints — no audio is generated, no credits used.

#     Returns: 'cartesia' | 'speechify'
#     """
#     logging.info(
#         f"🔍 Probing voice provider for unknown UUID: '{voice_id[:8]}…' "
#         f"(old channel — no voice_provider stored)"
#     )

#     # ── Try Cartesia first: GET /voices/{id} ──────────────────────────
#     try:
#         resp = requests.get(
#             f"https://api.cartesia.ai/voices/{voice_id}",
#             headers={
#                 "Authorization":    f"Bearer {CARTESIA_API_KEY}",
#                 "Cartesia-Version": CARTESIA_VERSION,
#             },
#             timeout=10,
#         )
#         if resp.ok:
#             name = resp.json().get("name", voice_id)
#             logging.info(
#                 f"✅ Voice '{voice_id[:8]}…' confirmed Cartesia "
#                 f"(name: '{name}')"
#             )
#             return "cartesia"
#         logging.debug(
#             f"Cartesia probe returned {resp.status_code} for '{voice_id[:8]}…'"
#         )
#     except Exception as e:
#         logging.debug(f"Cartesia probe exception: {e}")

#     # ── Try Speechify: GET /voices/{id} ───────────────────────────────
#     try:
#         resp = requests.get(
#             f"https://api.sws.speechify.com/v1/voices/{voice_id}",
#             headers={
#                 "Authorization": f"Bearer {SPEECHIFY_TOKEN}",
#                 "Content-Type":  "application/json",
#             },
#             timeout=10,
#         )
#         if resp.ok:
#             logging.info(
#                 f"✅ Voice '{voice_id[:8]}…' confirmed Speechify"
#             )
#             return "speechify"
#         logging.debug(
#             f"Speechify probe returned {resp.status_code} for '{voice_id[:8]}…'"
#         )
#     except Exception as e:
#         logging.debug(f"Speechify probe exception: {e}")

#     logging.warning(
#         f"⚠️ Could not confirm provider for '{voice_id[:8]}…' "
#         f"— defaulting to Speechify"
#     )
#     return "speechify"


# # =========================
# # Voice resolution
# # =========================
# def resolve_voice(raw_voice: str, provider_hint: str = None) -> tuple[str, str]:
#     """
#     Resolve a raw voice value to (voice_id, provider).

#     Resolution order:
#     1. Explicit provider_hint  (new channels / recently updated channels)
#     2. Named alias maps        (e.g. "carson" → Cartesia, "stoic" → Speechify)
#     3. OpenAI voice name set
#     4. UUID classification     (known IDs checked against both maps)
#     5. API probe               (ambiguous UUID — old channels only)
#        └─ result cached in _provider_cache + persisted to channel document
#     """
#     v = raw_voice.strip()

#     # ── 1. Trust explicit provider hint ──────────────────────────────
#     if provider_hint == "cartesia":
#         logging.info(f"Voice '{v[:8]}…' resolved as Cartesia (provider hint)")
#         return v, "cartesia"

#     if provider_hint == "speechify":
#         logging.info(f"Voice '{v[:8]}…' resolved as Speechify (provider hint)")
#         return v, "speechify"

#     if provider_hint == "openai":
#         logging.info(f"Voice '{v}' resolved as OpenAI (provider hint)")
#         return v, "openai"

#     # ── 2. Named alias maps ───────────────────────────────────────────
#     cartesia_id = resolve_cartesia_voice(v)
#     if cartesia_id is not None:
#         return cartesia_id, "cartesia"

#     if v in SPEECHIFY_VOICES:
#         resolved = SPEECHIFY_VOICES[v]
#         logging.info(f"Voice resolved via Speechify name map: '{v}' → '{resolved}'")
#         return resolved, "speechify"

#     # ── 3. OpenAI voice name ──────────────────────────────────────────
#     if v in OPENAI_VOICES:
#         logging.info(f"Voice resolved as OpenAI: '{v}'")
#         return v, "openai"

#     # ── 4. UUID classification ────────────────────────────────────────
#     is_cartesia = _is_cartesia_uuid(v)

#     if is_cartesia is True:
#         logging.info(f"Voice '{v[:8]}…' classified as Cartesia (known UUID)")
#         return v, "cartesia"

#     if is_cartesia is False:
#         # Confirmed Speechify (known UUID or not a UUID at all)
#         logging.info(f"Voice '{v[:8]}…' classified as Speechify (known UUID or non-UUID)")
#         return v, "speechify"

#     # ── 5. Ambiguous UUID — probe APIs ────────────────────────────────
#     # is_cartesia is None here: UUID format but not in either known set
#     # This path only hits for old channels without voice_provider stored

#     if v in _provider_cache:
#         cached = _provider_cache[v]
#         logging.info(
#             f"Voice '{v[:8]}…' provider from in-process cache: {cached}"
#         )
#         return v, cached

#     provider = _probe_voice_provider(v)
#     _provider_cache[v] = provider

#     return v, provider


# # =========================
# # Retry helper
# # =========================
# def with_retries(fn, *args,
#                  retry_forever=False,
#                  max_retries=4,
#                  initial_delay=1.0,
#                  backoff_base=2.0,
#                  max_delay=60.0,
#                  jitter_ratio=0.25,
#                  exceptions=(Exception,),
#                  desc="operation",
#                  **kwargs):
#     attempt = 0
#     while True:
#         try:
#             attempt += 1
#             return fn(*args, **kwargs)
#         except SpeechifyBadVoiceError:
#             raise
#         except exceptions as e:
#             if not retry_forever and attempt >= max_retries:
#                 raise

#             is_rate_limit = False
#             if hasattr(e, 'response'):
#                 is_rate_limit = getattr(e.response, 'status_code', None) == 429
#             elif '429' in str(e) or 'rate limit' in str(e).lower():
#                 is_rate_limit = True

#             if is_rate_limit:
#                 delay = min(max_delay, initial_delay * (backoff_base ** (attempt + 2)))
#                 logging.warning(f"⚠️ Rate limit hit for {desc}, waiting longer...")
#             else:
#                 delay = min(max_delay, initial_delay * (backoff_base ** (attempt - 1)))

#             jitter = delay * jitter_ratio * random.random()
#             wait   = delay + jitter
#             logging.warning(
#                 f"Retry #{attempt} for {desc} after error: {e}\n"
#                 f"→ Waiting {wait:.1f}s ..."
#             )
#             time.sleep(wait)


# # =========================
# # OpenAI client
# # =========================
# try:
#     from openai import OpenAI
#     openai_client = (
#         OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE, timeout=60.0)
#         if OPENAI_API_KEY else None
#     )
# except Exception:
#     openai_client = None


# # =========================
# # Audio utilities
# # =========================
# def _standardize_wav(path_in: Path, path_out: Path, add_fades=True):
#     audio = AudioSegment.from_file(path_in)
#     audio = (audio
#              .set_frame_rate(TARGET_SR_HZ)
#              .set_channels(TARGET_CH)
#              .set_sample_width(TARGET_SW))
#     if add_fades:
#         audio = audio.fade_in(SEG_FADE_MS).fade_out(SEG_FADE_MS)
#     audio = audio.high_pass_filter(30)
#     path_out.parent.mkdir(parents=True, exist_ok=True)
#     audio.export(path_out, format="wav")
#     return path_out


# # =========================
# # Extract word chunks
# # =========================
# def extract_word_chunks(timestamps_data):
#     if not timestamps_data:
#         return None

#     if isinstance(timestamps_data, dict):
#         if 'chunks' in timestamps_data and isinstance(timestamps_data['chunks'], list):
#             return timestamps_data['chunks']
#         if 'value' in timestamps_data and 'start_time' in timestamps_data:
#             return [timestamps_data]

#     elif isinstance(timestamps_data, list):
#         return timestamps_data

#     logging.error(f"Unexpected timestamps format: {type(timestamps_data)}")
#     return None


# # =========================
# # Core TTS function
# # =========================
# def generate_speech(
#     text: str,
#     output_mp3: str,
#     output_wav: str,
#     voice: str,
#     retry_forever=True,
#     max_retries=12,
#     include_timestamps: bool = True,
#     current_user=None,
# ):
#     mp3_path = Path(output_mp3)
#     wav_path = Path(output_wav)
#     wav_path.parent.mkdir(parents=True, exist_ok=True)

#     word_marks = None
#     voice_id, provider = resolve_voice(voice)

#     # ─── SPEECHIFY ────────────────────────────────────────────────────
#     if provider == "speechify":
#         headers = {
#             "Authorization": f"Bearer {SPEECHIFY_TOKEN}",
#             "Content-Type":  "application/json",
#         }

#         def _speechify_call(fmt: str):
#             payload = {
#                 "input":        text,
#                 "voice_id":     voice_id,
#                 "audio_format": fmt,
#                 "model":        "simba-english",
#             }
#             if include_timestamps:
#                 payload["include_timestamps"] = True
#                 payload["timestamp_type"]     = "word"

#             resp = requests.post(
#                 SPEECHIFY_API_URL, json=payload,
#                 headers=headers, timeout=90,
#             )

#             if not resp.ok:
#                 if resp.status_code == 429:
#                     raise RuntimeError(f"Speechify rate limit (429): {resp.text}")

#                 if resp.status_code == 400:
#                     msg = (
#                         f"Speechify 400 — voice ID '{voice_id}' is invalid, "
#                         f"deleted, or not accessible. Response: {resp.text}"
#                     )
#                     logging.error(f"❌ {msg}")
#                     try:
#                         send_telegram_error(
#                             f"🎙️ ❌ TTS Aborted — Bad Voice ID\n\n"
#                             f"Voice ID: `{voice_id}`\n"
#                             f"Status: Speechify 400 Bad Request\n"
#                             f"Reason: Voice may be deleted or belong to a different account.\n\n"
#                             f"⚠️ Action needed: Go to channel settings and reassign a valid voice.",
#                             current_user,
#                         )
#                     except Exception as tg_err:
#                         logging.warning(f"Telegram notification failed: {tg_err}")
#                     raise SpeechifyBadVoiceError(msg)

#                 raise RuntimeError(
#                     f"Speechify API error: {resp.status_code} - {resp.text}"
#                 )

#             return resp.json()

#         try:
#             data = with_retries(
#                 _speechify_call, "wav",
#                 retry_forever=retry_forever,
#                 max_retries=max_retries,
#                 desc=f"Speechify WAV (voice={voice_id[:8]}…)",
#             )
#             audio_b64 = data.get("audio_data")
#             if not audio_b64:
#                 raise ValueError("Speechify returned no audio_data")
#             tmp = wav_path.with_suffix(".tmp.wav")
#             tmp.write_bytes(base64.b64decode(audio_b64))
#             _standardize_wav(tmp, wav_path, add_fades=True)
#             tmp.unlink(missing_ok=True)

#             raw_ts = data.get("timestamps") or data.get("speech_marks")
#             if raw_ts:
#                 word_marks = extract_word_chunks(raw_ts)

#         except SpeechifyBadVoiceError:
#             raise

#         except Exception as e:
#             logging.warning(f"WAV flow failed ({e}), falling back to MP3…")
#             data = with_retries(
#                 lambda: _speechify_call("mp3"),
#                 retry_forever=retry_forever,
#                 max_retries=max_retries,
#                 desc=f"Speechify MP3 (voice={voice_id[:8]}…)",
#             )
#             audio_b64 = data.get("audio_data")
#             if not audio_b64:
#                 raise ValueError("Speechify returned no audio_data")
#             mp3_path.write_bytes(base64.b64decode(audio_b64))
#             _standardize_wav(mp3_path, wav_path, add_fades=True)
#             raw_ts = data.get("timestamps") or data.get("speech_marks")
#             if raw_ts:
#                 word_marks = extract_word_chunks(raw_ts)

#     # ─── OPENAI ───────────────────────────────────────────────────────
#     else:
#         if openai_client is None:
#             raise RuntimeError(
#                 "OpenAI client not available — set OPENAI_API_KEY"
#             )

#         def _openai_call():
#             resp = openai_client.audio.speech.create(
#                 model="tts-1", voice=voice_id, input=text,
#             )
#             mp3_path.parent.mkdir(parents=True, exist_ok=True)
#             resp.stream_to_file(mp3_path)

#         with_retries(
#             _openai_call,
#             retry_forever=retry_forever,
#             max_retries=max_retries,
#             desc=f"OpenAI TTS (voice={voice_id})",
#         )
#         _standardize_wav(mp3_path, wav_path, add_fades=True)
#         word_marks = None

#     def _export_mp3():
#         AudioSegment.from_wav(wav_path).export(
#             mp3_path, format="mp3", bitrate="320k"
#         )

#     with_retries(
#         _export_mp3,
#         retry_forever=retry_forever,
#         max_retries=max_retries,
#         desc="Export MP3",
#     )

#     return word_marks


# # =========================
# # Subtitle helpers
# # =========================
# def build_nested_chunk(segment_text, word_marks, segment_start_ms, char_offset):
#     if not word_marks:
#         return None

#     valid = [
#         w for w in word_marks
#         if isinstance(w, dict) and
#         all(k in w for k in ['value', 'start_time', 'end_time', 'start', 'end'])
#     ]
#     if not valid:
#         return None

#     return {
#         "start":      char_offset,
#         "end":        char_offset + len(segment_text),
#         "start_time": segment_start_ms + valid[0]["start_time"],
#         "end_time":   segment_start_ms + valid[-1]["end_time"],
#         "value":      segment_text,
#         "chunks": [{
#             "start":      char_offset + w["start"],
#             "end":        char_offset + w["end"],
#             "start_time": segment_start_ms + w["start_time"],
#             "end_time":   segment_start_ms + w["end_time"],
#             "value":      w["value"],
#         } for w in valid],
#     }


# def _format_srt_time(ms: float) -> str:
#     total = ms / 1000.0
#     h = int(total // 3600)
#     m = int((total % 3600) // 60)
#     s = total % 60
#     return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


# def nested_list_to_srt(nested_list, output_path):
#     if not nested_list:
#         logging.error("nested_list is empty")
#         return False

#     words = []
#     for nested in nested_list:
#         if not nested or 'chunks' not in nested:
#             continue
#         for w in nested["chunks"]:
#             if isinstance(w, dict) and all(
#                 k in w for k in ['value', 'start_time', 'end_time']
#             ):
#                 words.append(w)

#     if not words:
#         logging.error("No valid words found in nested_list")
#         return False

#     words.sort(key=lambda w: w["start_time"])

#     try:
#         with open(output_path, "w", encoding="utf-8") as f:
#             for idx, w in enumerate(words, 1):
#                 f.write(
#                     f"{idx}\n"
#                     f"{_format_srt_time(w['start_time'])} --> "
#                     f"{_format_srt_time(w['end_time'])}\n"
#                     f"{w['value'].upper()}\n\n"
#                 )
#         logging.info(f"Wrote {len(words)} subtitle entries to {output_path}")
#         return True
#     except Exception as e:
#         logging.error(f"Failed to write SRT: {e}")
#         return False


# # =========================
# # Worker helpers
# # =========================
# def get_optimal_audio_workers(num_segments, voice, num_concurrent_videos=1):
#     _, provider = resolve_voice(voice)
#     if provider == "speechify":
#         base = 5 if num_segments < 10 else (15 if num_segments < 50 else 20)
#     else:
#         base = 3 if num_segments < 20 else (5 if num_segments < 50 else 8)

#     if num_concurrent_videos >= 3:
#         return max(5, base // 2)
#     elif num_concurrent_videos == 2:
#         return max(8, int(base * 0.6))
#     return base


# # =========================
# # Persist discovered provider back to channel
# # So old channels only ever get probed once
# # =========================
# def _persist_voice_provider(channel_id: str, provider: str):
#     try:
#         from mongodb import mongo_instance
#         from bson import ObjectId
#         db = mongo_instance.get_db()
#         db.channels.update_one(
#             {"_id": ObjectId(channel_id)},
#             {"$set": {"voice_provider": provider}},
#         )
#         logging.info(
#             f"✅ Persisted voice_provider='{provider}' "
#             f"for channel {channel_id}"
#         )
#     except Exception as e:
#         logging.warning(f"Failed to persist voice_provider: {e}")


# # =========================
# # Parallel audio generation
# # =========================
# def generate_audio_segments(
#     segments,
#     voice,
#     full_script=None,
#     output_dir=UPLOAD_FOLDER,
#     retry_forever=True,
#     max_retries=12,
#     max_workers=None,
#     num_concurrent_videos=1,
#     retry_failed_segments=True,
#     current_user=None,
#     voice_provider=None,    # explicit hint from channel.voice_provider
#     channel_id=None,        # passed so we can persist discovered provider
# ):
#     from concurrent.futures import ThreadPoolExecutor, as_completed

#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = time.strftime("%Y%m%d%H%M%S")

#     # ── Resolve voice + provider ──────────────────────────────────────
#     _voice_id, _provider = resolve_voice(voice, provider_hint=voice_provider)

#     # If this was an old channel (no hint) and we just probed,
#     # persist the result so it never needs probing again
#     if voice_provider is None and channel_id:
#         _persist_voice_provider(channel_id, _provider)

#     # ── Cartesia early-exit ───────────────────────────────────────────
#     if _provider == "cartesia":
#         if not full_script:
#             logging.error("Cartesia requires full_script argument")
#             return ([], None, None, [], [], None)
#         return generate_audio_segments_cartesia(
#             segments     = segments,
#             voice        = voice,
#             full_script  = full_script,
#             voice_id     = _voice_id,
#             output_dir   = output_dir,
#             max_retries  = max_retries,
#             current_user = current_user,
#         )

#     # ── Auto-calculate workers ────────────────────────────────────────
#     if max_workers is None:
#         max_workers = get_optimal_audio_workers(
#             len(segments), voice, num_concurrent_videos
#         )

#     voice_id, provider = _voice_id, _provider

#     logging.info("")
#     logging.info("=" * 60)
#     logging.info("🚀 AUDIO GENERATION START")
#     logging.info("=" * 60)
#     logging.info(f"   Raw voice value : '{voice}'")
#     logging.info(f"   Resolved ID     : '{voice_id}'")
#     logging.info(f"   Provider        : {provider}")
#     logging.info(f"   Segments        : {len(segments)}")
#     logging.info(f"   Workers         : {max_workers}")
#     logging.info(f"   Concurrent vids : {num_concurrent_videos}")
#     logging.info(f"   Channel ID      : {channel_id or 'not provided'}")
#     logging.info("=" * 60)

#     # ─── Per-segment worker ───────────────────────────────────────────
#     def generate_single_segment(seg_data):
#         idx, segment = seg_data
#         text = (segment.get("text") or "").strip()

#         if not text or text.startswith("[CLIP:"):
#             return {
#                 "idx": idx, "audio_path": None, "duration": 0.0,
#                 "word_marks": None, "text": text,
#                 "success": False, "skipped": True,
#             }

#         base     = f"{timestamp}_segment_{idx}"
#         mp3_path = os.path.join(output_dir, f"{base}.mp3")
#         wav_path = os.path.join(output_dir, f"{base}.wav")

#         try:
#             with GLOBAL_API_SEMAPHORE:
#                 local_marks = generate_speech(
#                     text, mp3_path, wav_path, voice,
#                     retry_forever=retry_forever,
#                     max_retries=max_retries,
#                     include_timestamps=True,
#                     current_user=current_user,
#                 )

#             seg_dur = AudioSegment.from_wav(wav_path).duration_seconds
#             logging.info(f"✅ Segment {idx}: {seg_dur:.2f}s")

#             return {
#                 "idx": idx, "audio_path": wav_path, "duration": seg_dur,
#                 "word_marks": local_marks, "text": text,
#                 "success": True, "skipped": False,
#             }

#         except SpeechifyBadVoiceError:
#             raise

#         except Exception as e:
#             logging.error(f"❌ Segment {idx} failed: {e}")
#             return {
#                 "idx": idx, "audio_path": None, "duration": 0.0,
#                 "word_marks": None, "text": text,
#                 "success": False, "skipped": False, "error": str(e),
#             }

#     # ─── Phase 1: Parallel generation ────────────────────────────────
#     start_time      = time.time()
#     segment_results = {}
#     bad_voice_abort = False

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(generate_single_segment, (i, seg)): i
#             for i, seg in enumerate(segments)
#         }
#         completed = 0
#         for future in as_completed(futures):
#             try:
#                 result = future.result()
#                 segment_results[result["idx"]] = result
#                 if result["success"]:
#                     completed += 1
#                     if completed % 10 == 0:
#                         elapsed = time.time() - start_time
#                         logging.info(
#                             f"⚡ {completed}/{len(segments)} "
#                             f"({completed/elapsed:.1f}/sec)"
#                         )

#             except SpeechifyBadVoiceError:
#                 bad_voice_abort = True
#                 for f in futures:
#                     f.cancel()
#                 logging.error(
#                     "🚨 Bad voice ID — cancelling all remaining segments"
#                 )
#                 try:
#                     send_telegram_error(
#                         f"🎙️ ❌ Audio Generation Aborted\n\n"
#                         f"Voice ID: `{voice_id}`\n"
#                         f"Error: Speechify 400 Bad Request\n"
#                         f"Completed before abort: {completed}/{len(segments)}\n\n"
#                         f"⚠️ Action needed: Reassign a valid voice in channel settings.",
#                         current_user,
#                     )
#                 except Exception as tg_err:
#                     logging.warning(f"Telegram notification failed: {tg_err}")
#                 break

#     if bad_voice_abort:
#         return ([], None, None, [], [], None)

#     successful = sum(1 for r in segment_results.values() if r["success"])
#     skipped    = sum(1 for r in segment_results.values() if r.get("skipped"))
#     failed     = len(segments) - successful - skipped

#     logging.info(
#         f"Phase 1: ✅{successful} ⏭️{skipped} ❌{failed} "
#         f"in {time.time()-start_time:.1f}s"
#     )

#     # ─── Phase 1.5: Retry failures ───────────────────────────────────
#     if retry_failed_segments and failed > 0:
#         failed_segs = [
#             (idx, segments[idx])
#             for idx, r in segment_results.items()
#             if not r["success"] and not r.get("skipped")
#         ]
#         logging.warning(
#             f"Retrying {len(failed_segs)} failed segments with 5 workers…"
#         )

#         with ThreadPoolExecutor(max_workers=5) as executor:
#             retry_futures = {
#                 executor.submit(generate_single_segment, sd): sd[0]
#                 for sd in failed_segs
#             }
#             for future in as_completed(retry_futures):
#                 try:
#                     result = future.result()
#                     if result["success"]:
#                         segment_results[result["idx"]] = result
#                         logging.info(
#                             f"✅ Retry succeeded: segment {result['idx']}"
#                         )
#                     else:
#                         logging.error(
#                             f"❌ Retry failed: segment {result['idx']}"
#                         )
#                 except SpeechifyBadVoiceError:
#                     logging.error(
#                         "🚨 Bad voice ID in retry phase — stopping"
#                     )
#                     break

#         successful = sum(1 for r in segment_results.values() if r["success"])
#         failed     = len(segments) - successful - skipped

#     if failed > 0:
#         failed_idx = [
#             idx for idx, r in segment_results.items()
#             if not r["success"] and not r.get("skipped")
#         ]
#         logging.error(f"🚨 {failed} segments permanently failed: {failed_idx}")
#         try:
#             send_telegram_error(
#                 f"⚠️ Audio Generation — Partial Failure\n\n"
#                 f"Voice ID: `{voice_id}`\n"
#                 f"Failed segments: {failed_idx}\n"
#                 f"Successful: {successful}/{len(segments)}\n\n"
#                 f"Video will have {failed} silent gap(s).",
#                 current_user,
#             )
#         except Exception as tg_err:
#             logging.warning(f"Telegram notification failed: {tg_err}")

#     # ─── Phase 2: Build timestamps ────────────────────────────────────
#     segment_audio_paths = []
#     segment_timestamps  = []
#     nested_chunks       = []
#     total_duration      = FINAL_PAD_SEC
#     char_offset         = 0
#     last_successful_idx = None

#     for idx in range(len(segments)):
#         result = segment_results.get(idx)

#         if not result or not result["success"]:
#             segment_audio_paths.append(None)
#             segment_timestamps.append((0.0, 0.0))
#             if result and result.get("text"):
#                 char_offset += len(result["text"]) + 1
#             continue

#         segment_audio_paths.append(result["audio_path"])
#         seg_start = total_duration
#         seg_end   = seg_start + result["duration"]
#         segment_timestamps.append((seg_start, seg_end))
#         total_duration      = seg_end
#         last_successful_idx = idx

#         if result["word_marks"]:
#             nested = build_nested_chunk(
#                 result["text"],
#                 result["word_marks"],
#                 seg_start * 1000.0,
#                 char_offset,
#             )
#             if nested:
#                 nested_chunks.append(nested)

#         char_offset += len(result["text"]) + 1

#     if last_successful_idx is not None:
#         s, e = segment_timestamps[last_successful_idx]
#         segment_timestamps[last_successful_idx] = (s, e + FINAL_PAD_SEC)

#     # ─── Phase 3: Combine audio ───────────────────────────────────────
#     logging.info("Combining audio segments…")
#     combined = AudioSegment.silent(duration=0)

#     for idx, path in enumerate(segment_audio_paths):
#         if not path:
#             continue
#         try:
#             seg = AudioSegment.from_wav(path)
#             combined += seg.fade_in(15).fade_out(15).high_pass_filter(30)
#         except Exception as e:
#             logging.error(f"Failed to load segment {idx}: {e}")

#     if len(combined) == 0:
#         logging.error("No audio to combine!")
#         return ([], None, None, [], [], None)

#     combined = effects.normalize(combined, headroom=GLOBAL_HEADROOM_DB)
#     combined = (
#         combined.fade_in(100).fade_out(100)
#         .set_frame_rate(TARGET_SR_HZ)
#         .set_channels(TARGET_CH)
#         .set_sample_width(TARGET_SW)
#     )
#     final_audio = (
#         AudioSegment.silent(duration=FINAL_PAD_MS)
#         + combined
#         + AudioSegment.silent(duration=FINAL_PAD_MS)
#     )

#     base_name      = f"narration_{int(time.time())}"
#     final_wav_path = os.path.join(output_dir, f"{base_name}.wav")
#     final_mp3_path = os.path.join(output_dir, f"{base_name}.mp3")

#     try:
#         final_audio.export(final_wav_path, format="wav")
#         final_audio.export(final_mp3_path, format="mp3", bitrate="320k")
#         logging.info(
#             f"✅ WAV: {final_wav_path} "
#             f"({final_audio.duration_seconds:.1f}s)"
#         )
#         logging.info(f"✅ MP3: {final_mp3_path}")
#     except Exception as e:
#         logging.error(f"Export failed: {e}")
#         final_wav_path = final_mp3_path = None

#     final_mp3_url = (
#         f"http://localhost:5000/{UPLOAD_FOLDER}/{Path(final_mp3_path).name}"
#         if final_mp3_path else None
#     )

#     # ─── SRT ─────────────────────────────────────────────────────────
#     srt_path = None
#     if nested_chunks and final_wav_path:
#         srt_path = os.path.join(output_dir, f"{base_name}.srt")
#         if not nested_list_to_srt(nested_chunks, srt_path):
#             srt_path = None

#     total_time = time.time() - start_time
#     logging.info("")
#     logging.info("=" * 60)
#     logging.info(
#         f"🎉 DONE — {successful}/{len(segments)} segments, "
#         f"{total_time:.1f}s total"
#     )
#     logging.info("=" * 60)

#     return (
#         segment_audio_paths,
#         final_wav_path,
#         final_mp3_url,
#         segment_timestamps,
#         nested_chunks,
#         srt_path,
#     )



import os
import re
import gc
import time
import json
import base64
import random
import logging
import subprocess
import shutil
import uuid
from pathlib import Path
from threading import Semaphore

from dotenv import load_dotenv
import requests
from pydub import AudioSegment, effects
from helper.cartesia import (
    CARTESIA_API_KEY,
    CARTESIA_VERSION,
    resolve_cartesia_voice,
    generate_audio_segments_cartesia,
)
from helper.telegram_notification_engine import send_telegram_notification, send_telegram_error

# =========================
# Env + Logging
# =========================
print("Loading environment variables...")
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
SPEECHIFY_TOKEN = os.getenv("SPEECHIFY_API_KEY")
API_BASE        = os.getenv("OPENAI_BASE_URL")
UPLOAD_FOLDER   = os.getenv("UPLOAD_FOLDER", "static/audio")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =========================
# Constants
# =========================
SPEECHIFY_API_URL = "https://api.sws.speechify.com/v1/audio/speech"

SPEECHIFY_VOICES = {
    "cweb":         "5dfd8793-8f49-4205-aebb-5a9b47c18f76",
    "andrew":       "affbf842-63c1-4979-8994-70deea0a38a6",
    "stoic":        "940642fc-de60-4531-b77d-b03f0a38a6e6",
    "uncensored":   "c9065edf-b025-4542-ab6a-50e1bdf475ee",
    "jennifer":     "jennifer",
    "liam":         "944a6cef-9a26-41f5-a2cd-ef29cac74bb2",
    "nora":         "34872d14-035e-4416-b213-b5669a868091",
    "explainer":    "06b97011-e9a3-4aa3-b2d8-a7696c24380a",
    "war":          "fd8226b3-22ff-4739-b0ce-22e50adf95bb",
    "health":       "8d16ac9d-94d6-46bc-8cf4-4a5ed04c5947",
    "santa":        "ad5d0175-2a9e-4428-ba83-aca4a3220daa",
    "dark":         "c091248b-9814-452d-ac94-eb71ea646a7b",
    "noel":         "793d5cc2-4ddb-4e5d-bd44-5a406d0c0984",
    "trice-reason": "ea8fc708-8054-4140-bf71-130292136a90",
    "g-finance":    "3232a1d7-97ca-4952-a65c-0416d888d91e",
    "park-stoic":   "0383c304-6173-482a-adb2-8989ffface25",
    "emma":         "ec60fb61-5981-4cef-9b0d-82c0fb7c295a",
    "edna":         "c44d297c-81eb-4c9b-8d4e-e833f632bbbb",
    "g-fitness":    "ebbef645-600e-40e1-ac41-032d232294b9",
}

SPEECHIFY_VOICE_IDS = set(SPEECHIFY_VOICES.values())
OPENAI_VOICES       = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}

TARGET_SR_HZ       = 48000
TARGET_CH          = 2
TARGET_SW          = 2
SEG_FADE_MS        = 15
GLOBAL_HEADROOM_DB = 3.0
FINAL_PAD_MS       = 100
FINAL_PAD_SEC      = FINAL_PAD_MS / 1000.0

# 9 minutes — safe batch limit for external avatar platforms
DREAMFACE_BATCH_MAX_SEC = 9 * 60  # 540 seconds

GLOBAL_API_SEMAPHORE = Semaphore(15)

_provider_cache: dict[str, str] = {}


# =========================
# Custom exceptions
# =========================
class SpeechifyBadVoiceError(Exception):
    """400 from Speechify — permanent, never retry."""
    pass


# =========================
# UUID helpers
# =========================
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


def _is_uuid(v: str) -> bool:
    return bool(_UUID_RE.fullmatch(v))


def _is_cartesia_uuid(v: str):
    if not _is_uuid(v):
        return False
    if v in SPEECHIFY_VOICE_IDS:
        return False
    from helper.cartesia import CARTESIA_VOICES
    if v in CARTESIA_VOICES.values():
        return True
    return None


# =========================
# Provider probe
# =========================
def _probe_voice_provider(voice_id: str) -> str:
    logging.info(
        f"🔍 Probing voice provider for unknown UUID: '{voice_id[:8]}…'"
    )

    try:
        resp = requests.get(
            f"https://api.cartesia.ai/voices/{voice_id}",
            headers={
                "Authorization":    f"Bearer {CARTESIA_API_KEY}",
                "Cartesia-Version": CARTESIA_VERSION,
            },
            timeout=10,
        )
        if resp.ok:
            name = resp.json().get("name", voice_id)
            logging.info(f"✅ Voice '{voice_id[:8]}…' confirmed Cartesia (name: '{name}')")
            return "cartesia"
        logging.debug(f"Cartesia probe returned {resp.status_code} for '{voice_id[:8]}…'")
    except Exception as e:
        logging.debug(f"Cartesia probe exception: {e}")

    try:
        resp = requests.get(
            f"https://api.sws.speechify.com/v1/voices/{voice_id}",
            headers={
                "Authorization": f"Bearer {SPEECHIFY_TOKEN}",
                "Content-Type":  "application/json",
            },
            timeout=10,
        )
        if resp.ok:
            logging.info(f"✅ Voice '{voice_id[:8]}…' confirmed Speechify")
            return "speechify"
        logging.debug(f"Speechify probe returned {resp.status_code} for '{voice_id[:8]}…'")
    except Exception as e:
        logging.debug(f"Speechify probe exception: {e}")

    logging.warning(
        f"⚠️ Could not confirm provider for '{voice_id[:8]}…' — defaulting to Speechify"
    )
    return "speechify"


# =========================
# Voice resolution
# =========================
def resolve_voice(raw_voice: str, provider_hint: str = None) -> tuple[str, str]:
    v = raw_voice.strip()

    if provider_hint == "cartesia":
        return v, "cartesia"
    if provider_hint == "speechify":
        return v, "speechify"
    if provider_hint == "openai":
        return v, "openai"

    cartesia_id = resolve_cartesia_voice(v)
    if cartesia_id is not None:
        return cartesia_id, "cartesia"

    if v in SPEECHIFY_VOICES:
        resolved = SPEECHIFY_VOICES[v]
        logging.info(f"Voice resolved via Speechify name map: '{v}' → '{resolved}'")
        return resolved, "speechify"

    if v in OPENAI_VOICES:
        logging.info(f"Voice resolved as OpenAI: '{v}'")
        return v, "openai"

    is_cartesia = _is_cartesia_uuid(v)

    if is_cartesia is True:
        return v, "cartesia"
    if is_cartesia is False:
        return v, "speechify"

    if v in _provider_cache:
        return v, _provider_cache[v]

    provider = _probe_voice_provider(v)
    _provider_cache[v] = provider
    return v, provider


# =========================
# Retry helper
# =========================
def with_retries(fn, *args,
                 retry_forever=False,
                 max_retries=4,
                 initial_delay=1.0,
                 backoff_base=2.0,
                 max_delay=60.0,
                 jitter_ratio=0.25,
                 exceptions=(Exception,),
                 desc="operation",
                 **kwargs):
    attempt = 0
    while True:
        try:
            attempt += 1
            return fn(*args, **kwargs)
        except SpeechifyBadVoiceError:
            raise
        except exceptions as e:
            if not retry_forever and attempt >= max_retries:
                raise

            is_rate_limit = False
            if hasattr(e, 'response'):
                is_rate_limit = getattr(e.response, 'status_code', None) == 429
            elif '429' in str(e) or 'rate limit' in str(e).lower():
                is_rate_limit = True

            if is_rate_limit:
                delay = min(max_delay, initial_delay * (backoff_base ** (attempt + 2)))
            else:
                delay = min(max_delay, initial_delay * (backoff_base ** (attempt - 1)))

            jitter = delay * jitter_ratio * random.random()
            wait   = delay + jitter
            logging.warning(
                f"Retry #{attempt} for {desc} after error: {e}\n"
                f"→ Waiting {wait:.1f}s ..."
            )
            time.sleep(wait)


# =========================
# OpenAI client
# =========================
try:
    from openai import OpenAI
    openai_client = (
        OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE, timeout=60.0)
        if OPENAI_API_KEY else None
    )
except Exception:
    openai_client = None


# =========================
# Audio utilities
# =========================
def _standardize_wav(path_in: Path, path_out: Path, add_fades=True):
    audio = AudioSegment.from_file(path_in)
    audio = (audio
             .set_frame_rate(TARGET_SR_HZ)
             .set_channels(TARGET_CH)
             .set_sample_width(TARGET_SW))
    if add_fades:
        audio = audio.fade_in(SEG_FADE_MS).fade_out(SEG_FADE_MS)
    audio = audio.high_pass_filter(30)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    audio.export(path_out, format="wav")
    return path_out


# =========================
# Extract word chunks
# =========================
def extract_word_chunks(timestamps_data):
    if not timestamps_data:
        return None

    if isinstance(timestamps_data, dict):
        if 'chunks' in timestamps_data and isinstance(timestamps_data['chunks'], list):
            return timestamps_data['chunks']
        if 'value' in timestamps_data and 'start_time' in timestamps_data:
            return [timestamps_data]

    elif isinstance(timestamps_data, list):
        return timestamps_data

    logging.error(f"Unexpected timestamps format: {type(timestamps_data)}")
    return None


# =========================
# Core TTS function
# =========================
def generate_speech(
    text: str,
    output_mp3: str,
    output_wav: str,
    voice: str,
    retry_forever=True,
    max_retries=12,
    include_timestamps: bool = True,
    current_user=None,
):
    mp3_path = Path(output_mp3)
    wav_path = Path(output_wav)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    word_marks = None
    voice_id, provider = resolve_voice(voice)

    # ─── SPEECHIFY ───────────────────────────────────────────
    if provider == "speechify":
        headers = {
            "Authorization": f"Bearer {SPEECHIFY_TOKEN}",
            "Content-Type":  "application/json",
        }

        def _speechify_call(fmt: str):
            payload = {
                "input":        text,
                "voice_id":     voice_id,
                "audio_format": fmt,
                "model":        "simba-english",
            }
            if include_timestamps:
                payload["include_timestamps"] = True
                payload["timestamp_type"]     = "word"

            resp = requests.post(
                SPEECHIFY_API_URL, json=payload,
                headers=headers, timeout=90,
            )

            if not resp.ok:
                if resp.status_code == 429:
                    raise RuntimeError(f"Speechify rate limit (429): {resp.text}")

                if resp.status_code == 400:
                    msg = (
                        f"Speechify 400 — voice ID '{voice_id}' is invalid, "
                        f"deleted, or not accessible. Response: {resp.text}"
                    )
                    logging.error(f"❌ {msg}")
                    try:
                        send_telegram_error(
                            f"🎙️ ❌ TTS Aborted — Bad Voice ID\n\n"
                            f"Voice ID: `{voice_id}`\n"
                            f"Status: Speechify 400 Bad Request\n"
                            f"Reason: Voice may be deleted or belong to a different account.\n\n"
                            f"⚠️ Action needed: Go to channel settings and reassign a valid voice.",
                            current_user,
                        )
                    except Exception as tg_err:
                        logging.warning(f"Telegram notification failed: {tg_err}")
                    raise SpeechifyBadVoiceError(msg)

                raise RuntimeError(
                    f"Speechify API error: {resp.status_code} - {resp.text}"
                )

            return resp.json()

        try:
            data = with_retries(
                _speechify_call, "wav",
                retry_forever=retry_forever,
                max_retries=max_retries,
                desc=f"Speechify WAV (voice={voice_id[:8]}…)",
            )
            audio_b64 = data.get("audio_data")
            if not audio_b64:
                raise ValueError("Speechify returned no audio_data")
            tmp = wav_path.with_suffix(".tmp.wav")
            tmp.write_bytes(base64.b64decode(audio_b64))
            _standardize_wav(tmp, wav_path, add_fades=True)
            tmp.unlink(missing_ok=True)

            raw_ts = data.get("timestamps") or data.get("speech_marks")
            if raw_ts:
                word_marks = extract_word_chunks(raw_ts)

        except SpeechifyBadVoiceError:
            raise

        except Exception as e:
            logging.warning(f"WAV flow failed ({e}), falling back to MP3…")
            data = with_retries(
                lambda: _speechify_call("mp3"),
                retry_forever=retry_forever,
                max_retries=max_retries,
                desc=f"Speechify MP3 (voice={voice_id[:8]}…)",
            )
            audio_b64 = data.get("audio_data")
            if not audio_b64:
                raise ValueError("Speechify returned no audio_data")
            mp3_path.write_bytes(base64.b64decode(audio_b64))
            _standardize_wav(mp3_path, wav_path, add_fades=True)
            raw_ts = data.get("timestamps") or data.get("speech_marks")
            if raw_ts:
                word_marks = extract_word_chunks(raw_ts)

    # ─── OPENAI ──────────────────────────────────────────────
    else:
        if openai_client is None:
            raise RuntimeError("OpenAI client not available — set OPENAI_API_KEY")

        def _openai_call():
            resp = openai_client.audio.speech.create(
                model="tts-1", voice=voice_id, input=text,
            )
            mp3_path.parent.mkdir(parents=True, exist_ok=True)
            resp.stream_to_file(mp3_path)

        with_retries(
            _openai_call,
            retry_forever=retry_forever,
            max_retries=max_retries,
            desc=f"OpenAI TTS (voice={voice_id})",
        )
        _standardize_wav(mp3_path, wav_path, add_fades=True)
        word_marks = None

    def _export_mp3():
        AudioSegment.from_wav(wav_path).export(
            mp3_path, format="mp3", bitrate="320k"
        )

    with_retries(
        _export_mp3,
        retry_forever=retry_forever,
        max_retries=max_retries,
        desc="Export MP3",
    )

    return word_marks


# =========================
# Subtitle helpers
# =========================
def build_nested_chunk(segment_text, word_marks, segment_start_ms, char_offset):
    if not word_marks:
        return None

    valid = [
        w for w in word_marks
        if isinstance(w, dict) and
        all(k in w for k in ['value', 'start_time', 'end_time', 'start', 'end'])
    ]
    if not valid:
        return None

    return {
        "start":      char_offset,
        "end":        char_offset + len(segment_text),
        "start_time": segment_start_ms + valid[0]["start_time"],
        "end_time":   segment_start_ms + valid[-1]["end_time"],
        "value":      segment_text,
        "chunks": [{
            "start":      char_offset + w["start"],
            "end":        char_offset + w["end"],
            "start_time": segment_start_ms + w["start_time"],
            "end_time":   segment_start_ms + w["end_time"],
            "value":      w["value"],
        } for w in valid],
    }


def _format_srt_time(ms: float) -> str:
    total = ms / 1000.0
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


def nested_list_to_srt(nested_list, output_path):
    if not nested_list:
        logging.error("nested_list is empty")
        return False

    words = []
    for nested in nested_list:
        if not nested or 'chunks' not in nested:
            continue
        for w in nested["chunks"]:
            if isinstance(w, dict) and all(
                k in w for k in ['value', 'start_time', 'end_time']
            ):
                words.append(w)

    if not words:
        logging.error("No valid words found in nested_list")
        return False

    words.sort(key=lambda w: w["start_time"])

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, w in enumerate(words, 1):
                f.write(
                    f"{idx}\n"
                    f"{_format_srt_time(w['start_time'])} --> "
                    f"{_format_srt_time(w['end_time'])}\n"
                    f"{w['value'].upper()}\n\n"
                )
        logging.info(f"Wrote {len(words)} subtitle entries to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to write SRT: {e}")
        return False


# =========================
# Worker helpers
# =========================
def get_optimal_audio_workers(num_segments, voice, num_concurrent_videos=1):
    _, provider = resolve_voice(voice)
    if provider == "speechify":
        base = 5 if num_segments < 10 else (15 if num_segments < 50 else 20)
    else:
        base = 3 if num_segments < 20 else (5 if num_segments < 50 else 8)

    if num_concurrent_videos >= 3:
        return max(5, base // 2)
    elif num_concurrent_videos == 2:
        return max(8, int(base * 0.6))
    return base


# =========================
# Persist discovered provider
# =========================
def _persist_voice_provider(channel_id: str, provider: str):
    try:
        from mongodb import mongo_instance
        from bson import ObjectId
        db = mongo_instance.get_db()
        db.channels.update_one(
            {"_id": ObjectId(channel_id)},
            {"$set": {"voice_provider": provider}},
        )
        logging.info(f"✅ Persisted voice_provider='{provider}' for channel {channel_id}")
    except Exception as e:
        logging.warning(f"Failed to persist voice_provider: {e}")


# =========================
# Talking Heads batch builder
#
# Groups narration audio segments into batches of max 9 minutes
# respecting CLIP boundaries. Uses FFmpeg concat for bit-perfect
# joining. Final output is MP3 @ 192kbps for lightweight downloads.
#
# Batch files saved to: {output_dir}/batches/
#
# Each segment entry in "segments_data" carries:
#   seg_index    — original index in vid_segments
#   duration_sec — actual WAV duration for that segment
#   avatar_start — cumulative offset within the batch audio file
#                  used by encode_video_segment to seek the avatar
#                  video to the correct position without trimming
# =========================
def build_dreamface_batches(
    segments,
    segment_audio_paths,
    segment_results,
    output_dir,
    timestamp,
):
    """
    Returns list of batch dicts:
    [
        {
            "batch_index":     0,
            "audio_path":      "static/audio/batches/dreamface_batch_0_xxx.mp3",
            "duration_sec":    487.3,
            "segment_indices": [0, 1, 2],   # kept for backward compat
            "segments_data": [              # ← NEW: per-segment timing
                {
                    "seg_index":    0,
                    "duration_sec": 4.2,
                    "avatar_start": 0.0,    # seek position in avatar video
                },
                {
                    "seg_index":    1,
                    "duration_sec": 3.8,
                    "avatar_start": 4.2,
                },
                {
                    "seg_index":    2,
                    "duration_sec": 6.1,
                    "avatar_start": 8.0,
                },
            ],
        },
        ...
    ]
    """
    # ── Dedicated subfolder ───────────────────────────────────
    batches_dir = os.path.join(output_dir, "batches")
    os.makedirs(batches_dir, exist_ok=True)

    batches     = []
    batch_index = 0

    current_indices  = []
    current_duration = 0.0

    def _flush_batch(indices):
        nonlocal batch_index

        if not indices:
            return

        # Collect valid WAV paths for these indices
        paths = [
            segment_audio_paths[i]
            for i in indices
            if segment_audio_paths[i] and os.path.exists(segment_audio_paths[i])
        ]

        if not paths:
            logging.warning(f"⚠️ Batch {batch_index}: no valid audio paths — skipping")
            return

        batch_id  = f"dreamface_batch_{batch_index}_{timestamp}_{uuid.uuid4().hex[:8]}"
        batch_wav = os.path.join(batches_dir, f"{batch_id}.wav")
        batch_mp3 = os.path.join(batches_dir, f"{batch_id}.mp3")

        # ── Step 1: Concatenate WAVs (bit-perfect, -c copy) ───
        if len(paths) == 1:
            shutil.copy2(paths[0], batch_wav)
            logging.info(f"✅ Batch {batch_index}: single segment copied → {batch_wav}")
        else:
            concat_list = os.path.join(
                batches_dir,
                f"concat_list_{batch_index}_{timestamp}.txt"
            )
            try:
                with open(concat_list, "w", encoding="utf-8") as f:
                    for p in paths:
                        safe = os.path.abspath(p).replace("\\", "/")
                        f.write(f"file '{safe}'\n")

                cmd_concat = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list,
                    "-c", "copy",
                    batch_wav,
                ]
                result = subprocess.run(cmd_concat, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(
                        f"❌ Batch {batch_index} WAV concat failed: "
                        f"{result.stderr[-500:]}"
                    )
                    return
                logging.info(
                    f"✅ Batch {batch_index}: {len(paths)} segments "
                    f"concatenated → {batch_wav}"
                )
            finally:
                if os.path.exists(concat_list):
                    os.remove(concat_list)

        # ── Step 2: Convert WAV → MP3 @ 192kbps ──────────────
        cmd_mp3 = [
            "ffmpeg", "-y",
            "-i", batch_wav,
            "-codec:a", "libmp3lame",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            batch_mp3,
        ]
        result_mp3 = subprocess.run(cmd_mp3, capture_output=True, text=True)

        if result_mp3.returncode == 0 and os.path.exists(batch_mp3):
            try:
                os.remove(batch_wav)
            except Exception as e:
                logging.warning(f"⚠️ Could not remove batch WAV: {e}")
            final_path = batch_mp3
            mp3_size   = os.path.getsize(batch_mp3) / (1024 * 1024)
            logging.info(
                f"✅ Batch {batch_index}: MP3 ready "
                f"({mp3_size:.1f}MB) → {batch_mp3}"
            )
        else:
            logging.warning(
                f"⚠️ Batch {batch_index}: MP3 conversion failed — keeping WAV. "
                f"Error: {result_mp3.stderr[-300:]}"
            )
            final_path = batch_wav

        # ── Step 3: Build per-segment timing data ─────────────
        # avatar_start = cumulative offset within THIS batch's
        # audio file. This is the seek position encode_video_segment
        # uses to read the avatar video without trimming it.
        # Resets to 0.0 for every new batch.
        segments_data    = []
        cumulative_start = 0.0

        for i in indices:
            result      = segment_results.get(i)
            seg_duration = result["duration"] if (result and result.get("success")) else 0.0

            segments_data.append({
                "seg_index":    i,
                "duration_sec": round(seg_duration, 4),
                "avatar_start": round(cumulative_start, 4),
            })

            logging.debug(
                f"   seg {i}: avatar_start={cumulative_start:.3f}s "
                f"duration={seg_duration:.3f}s"
            )

            cumulative_start += seg_duration

        # ── Step 4: Total batch duration ──────────────────────
        # batch_duration = sum(s["duration_sec"] for s in segments_data)

        batch_duration = sum(
    segment_results[i]["duration"]
    for i in indices
    if segment_results.get(i) and segment_results[i].get("success")
)

        batches.append({
            "batch_index":     batch_index,
            "audio_path":      final_path,
            "duration_sec":    round(batch_duration, 3),
            "segment_indices": list(indices),       # backward compat
            "segments_data":   segments_data,       # ← NEW: timing per segment
        })

        file_size = os.path.getsize(final_path) / (1024 * 1024)
        logging.info(
            f"📦 Batch {batch_index}: {len(indices)} segments | "
            f"{batch_duration:.1f}s | {file_size:.1f}MB"
        )
        logging.info(
            f"   Avatar seek offsets: "
            f"{[(s['seg_index'], s['avatar_start']) for s in segments_data]}"
        )
        batch_index += 1

    # ── Main grouping loop ────────────────────────────────────
    for idx, segment in enumerate(segments):
        text    = (segment.get("text") or "").strip()
        is_clip = text.startswith("[CLIP:") and text.endswith("]")

        if is_clip:
            if current_indices:
                _flush_batch(current_indices)
                current_indices  = []
                current_duration = 0.0
            continue

        result = segment_results.get(idx)
        if not result or not result.get("success"):
            current_indices.append(idx)
            continue

        seg_dur = result["duration"]

        if current_indices and (current_duration + seg_dur) > DREAMFACE_BATCH_MAX_SEC:
            _flush_batch(current_indices)
            current_indices  = []
            current_duration = 0.0

        current_indices.append(idx)
        current_duration += seg_dur

    if current_indices:
        _flush_batch(current_indices)

    logging.info(
        f"🎬 Talking Heads batching complete: "
        f"{len(batches)} batch(es) created"
    )
    return batches


# =========================
# Parallel audio generation
# =========================


def generate_audio_segments(
    segments,
    voice,
    full_script=None,
    output_dir=UPLOAD_FOLDER,
    retry_forever=True,
    max_retries=12,
    max_workers=None,
    num_concurrent_videos=1,
    retry_failed_segments=True,
    current_user=None,
    voice_provider=None,
    channel_id=None,
    use_talking_heads=False,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")

    # ── Resolve voice + provider ──────────────────────────────
    _voice_id, _provider = resolve_voice(voice, provider_hint=voice_provider)

    if voice_provider is None and channel_id:
        _persist_voice_provider(channel_id, _provider)

    # ── Cartesia early-exit ───────────────────────────────────
    if _provider == "cartesia":
        if not full_script:
            logging.error("Cartesia requires full_script argument")
            return ([], None, None, [], [], None, [])

        (
            segment_audio_paths,
            final_wav_path,
            final_mp3_url,
            segment_timestamps,
            nested_chunks,
            srt_path,
        ) = generate_audio_segments_cartesia(
            segments     = segments,
            voice        = voice,
            full_script  = full_script,
            voice_id     = _voice_id,
            output_dir   = output_dir,
            max_retries  = max_retries,
            current_user = current_user,
        )

        if not segment_audio_paths:
            return ([], None, None, [], [], None, [])

        dreamface_batches = []
        if use_talking_heads:
            segment_results = {}
            for idx, seg in enumerate(segments):
                text    = (seg.get("text") or "").strip()
                is_clip = text.startswith("[CLIP:") and text.endswith("]")
                path    = segment_audio_paths[idx] if idx < len(segment_audio_paths) else None
                if is_clip or not path:
                    segment_results[idx] = {"success": False, "skipped": True,  "duration": 0.0}
                elif path and os.path.exists(path):
                    dur = AudioSegment.from_wav(path).duration_seconds
                    segment_results[idx] = {"success": True,  "skipped": False, "duration": dur}
                else:
                    segment_results[idx] = {"success": False, "skipped": False, "duration": 0.0}

            logging.info("🎬 Building Talking Heads audio batches for Cartesia…")
            try:
                dreamface_batches = build_dreamface_batches(
                    segments            = segments,
                    segment_audio_paths = segment_audio_paths,
                    segment_results     = segment_results,
                    output_dir          = output_dir,
                    timestamp           = timestamp,
                )
                logging.info(f"✅ Talking Heads batches ready: {len(dreamface_batches)} batch(es)")
            except Exception as e:
                logging.error(f"❌ Talking Heads batch building failed: {e}")

        return (
            segment_audio_paths,
            final_wav_path,
            final_mp3_url,
            segment_timestamps,
            nested_chunks,
            srt_path,
            dreamface_batches,
        )

    # ── Auto-calculate workers ────────────────────────────────
    if max_workers is None:
        max_workers = get_optimal_audio_workers(
            len(segments), voice, num_concurrent_videos
        )

    voice_id, provider = _voice_id, _provider

    logging.info("")
    logging.info("=" * 60)
    logging.info("🚀 AUDIO GENERATION START")
    logging.info("=" * 60)
    logging.info(f"   Raw voice value   : '{voice}'")
    logging.info(f"   Resolved ID       : '{voice_id}'")
    logging.info(f"   Provider          : {provider}")
    logging.info(f"   Segments          : {len(segments)}")
    logging.info(f"   Workers           : {max_workers}")
    logging.info(f"   Concurrent vids   : {num_concurrent_videos}")
    logging.info(f"   Channel ID        : {channel_id or 'not provided'}")
    logging.info(f"   Talking Heads     : {use_talking_heads}")
    logging.info("=" * 60)

    # ─── Per-segment worker ───────────────────────────────────
    def generate_single_segment(seg_data):
        idx, segment = seg_data
        text = (segment.get("text") or "").strip()

        if not text or text.startswith("[CLIP:"):
            return {
                "idx": idx, "audio_path": None, "duration": 0.0,
                "word_marks": None, "text": text,
                "success": False, "skipped": True,
            }

        base     = f"{timestamp}_segment_{idx}"
        mp3_path = os.path.join(output_dir, f"{base}.mp3")
        wav_path = os.path.join(output_dir, f"{base}.wav")

        try:
            with GLOBAL_API_SEMAPHORE:
                local_marks = generate_speech(
                    text, mp3_path, wav_path, voice,
                    retry_forever=retry_forever,
                    max_retries=max_retries,
                    include_timestamps=True,
                    current_user=current_user,
                )

            seg_dur = AudioSegment.from_wav(wav_path).duration_seconds
            logging.info(f"✅ Segment {idx}: {seg_dur:.2f}s")

            return {
                "idx": idx, "audio_path": wav_path, "duration": seg_dur,
                "word_marks": local_marks, "text": text,
                "success": True, "skipped": False,
            }

        except SpeechifyBadVoiceError:
            raise

        except Exception as e:
            logging.error(f"❌ Segment {idx} failed: {e}")
            return {
                "idx": idx, "audio_path": None, "duration": 0.0,
                "word_marks": None, "text": text,
                "success": False, "skipped": False, "error": str(e),
            }

    # ─── Phase 1: Parallel generation ────────────────────────
    start_time      = time.time()
    segment_results = {}
    bad_voice_abort = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_single_segment, (i, seg)): i
            for i, seg in enumerate(segments)
        }
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                segment_results[result["idx"]] = result
                if result["success"]:
                    completed += 1
                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        logging.info(
                            f"⚡ {completed}/{len(segments)} "
                            f"({completed/elapsed:.1f}/sec)"
                        )

            except SpeechifyBadVoiceError:
                bad_voice_abort = True
                for f in futures:
                    f.cancel()
                logging.error("🚨 Bad voice ID — cancelling all remaining segments")
                try:
                    send_telegram_error(
                        f"🎙️ ❌ Audio Generation Aborted\n\n"
                        f"Voice ID: `{voice_id}`\n"
                        f"Error: Speechify 400 Bad Request\n"
                        f"Completed before abort: {completed}/{len(segments)}\n\n"
                        f"⚠️ Action needed: Reassign a valid voice in channel settings.",
                        current_user,
                    )
                except Exception as tg_err:
                    logging.warning(f"Telegram notification failed: {tg_err}")
                break

    if bad_voice_abort:
        return ([], None, None, [], [], None, [])

    successful = sum(1 for r in segment_results.values() if r["success"])
    skipped    = sum(1 for r in segment_results.values() if r.get("skipped"))
    failed     = len(segments) - successful - skipped

    logging.info(
        f"Phase 1: ✅{successful} ⏭️{skipped} ❌{failed} "
        f"in {time.time()-start_time:.1f}s"
    )

    # ─── Phase 1.5: Retry failures ───────────────────────────
    if retry_failed_segments and failed > 0:
        failed_segs = [
            (idx, segments[idx])
            for idx, r in segment_results.items()
            if not r["success"] and not r.get("skipped")
        ]
        logging.warning(f"Retrying {len(failed_segs)} failed segments with 5 workers…")

        with ThreadPoolExecutor(max_workers=5) as executor:
            retry_futures = {
                executor.submit(generate_single_segment, sd): sd[0]
                for sd in failed_segs
            }
            for future in as_completed(retry_futures):
                try:
                    result = future.result()
                    if result["success"]:
                        segment_results[result["idx"]] = result
                        logging.info(f"✅ Retry succeeded: segment {result['idx']}")
                    else:
                        logging.error(f"❌ Retry failed: segment {result['idx']}")
                except SpeechifyBadVoiceError:
                    logging.error("🚨 Bad voice ID in retry phase — stopping")
                    break

        successful = sum(1 for r in segment_results.values() if r["success"])
        failed     = len(segments) - successful - skipped

    if failed > 0:
        failed_idx = [
            idx for idx, r in segment_results.items()
            if not r["success"] and not r.get("skipped")
        ]
        logging.error(f"🚨 {failed} segments permanently failed: {failed_idx}")
        try:
            send_telegram_error(
                f"⚠️ Audio Generation — Partial Failure\n\n"
                f"Voice ID: `{voice_id}`\n"
                f"Failed segments: {failed_idx}\n"
                f"Successful: {successful}/{len(segments)}\n\n"
                f"Video will have {failed} silent gap(s).",
                current_user,
            )
        except Exception as tg_err:
            logging.warning(f"Telegram notification failed: {tg_err}")

    # ─── Phase 2: Build timestamps ────────────────────────────
    segment_audio_paths = []
    segment_timestamps  = []
    nested_chunks       = []
    total_duration      = FINAL_PAD_SEC
    char_offset         = 0
    last_successful_idx = None

    for idx in range(len(segments)):
        result = segment_results.get(idx)

        if not result or not result["success"]:
            segment_audio_paths.append(None)
            segment_timestamps.append((0.0, 0.0))
            if result and result.get("text"):
                char_offset += len(result["text"]) + 1
            continue

        segment_audio_paths.append(result["audio_path"])
        seg_start = total_duration
        seg_end   = seg_start + result["duration"]
        segment_timestamps.append((seg_start, seg_end))
        total_duration      = seg_end
        last_successful_idx = idx

        if result["word_marks"]:
            nested = build_nested_chunk(
                result["text"],
                result["word_marks"],
                seg_start * 1000.0,
                char_offset,
            )
            if nested:
                nested_chunks.append(nested)

        char_offset += len(result["text"]) + 1

    if last_successful_idx is not None:
        s, e = segment_timestamps[last_successful_idx]
        segment_timestamps[last_successful_idx] = (s, e + FINAL_PAD_SEC)

    # ─── Phase 3: Combine full narration audio ────────────────
    logging.info("Combining audio segments…")

    valid_paths = [p for p in segment_audio_paths if p and os.path.exists(p)]

    final_wav_path = None
    final_mp3_path = None
    final_mp3_url  = None

    if valid_paths:
        base_name      = f"narration_{int(time.time())}"
        final_wav_path = os.path.join(output_dir, f"{base_name}.wav")
        final_mp3_path = os.path.join(output_dir, f"{base_name}.mp3")

        concat_list = os.path.join(output_dir, f"narration_concat_{timestamp}.txt")
        try:
            with open(concat_list, "w", encoding="utf-8") as f:
                for p in valid_paths:
                    safe = os.path.abspath(p).replace("\\", "/")
                    f.write(f"file '{safe}'\n")

            cmd_wav = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                final_wav_path,
            ]
            result = subprocess.run(cmd_wav, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(
                    f"❌ Full narration concat failed: {result.stderr[-500:]}"
                )
                final_wav_path = None
            else:
                wav_size = os.path.getsize(final_wav_path) / (1024 * 1024)
                logging.info(f"✅ WAV: {final_wav_path} ({wav_size:.1f}MB)")

                cmd_mp3 = [
                    "ffmpeg", "-y",
                    "-i", final_wav_path,
                    "-codec:a", "libmp3lame",
                    "-b:a", "320k",
                    final_mp3_path,
                ]
                result = subprocess.run(cmd_mp3, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"❌ MP3 export failed: {result.stderr[-500:]}")
                    final_mp3_path = None
                else:
                    mp3_size = os.path.getsize(final_mp3_path) / (1024 * 1024)
                    logging.info(f"✅ MP3: {final_mp3_path} ({mp3_size:.1f}MB)")
                    final_mp3_url = (
                        f"http://localhost:5000/{UPLOAD_FOLDER}/"
                        f"{Path(final_mp3_path).name}"
                    )

        finally:
            if os.path.exists(concat_list):
                os.remove(concat_list)

    else:
        logging.error("No audio to combine!")
        return ([], None, None, [], [], None, [])

    # ─── SRT ─────────────────────────────────────────────────
    srt_path = None
    if nested_chunks and final_wav_path:
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        if not nested_list_to_srt(nested_chunks, srt_path):
            srt_path = None

    total_time = time.time() - start_time
    logging.info("")
    logging.info("=" * 60)
    logging.info(
        f"🎉 DONE — {successful}/{len(segments)} segments, "
        f"{total_time:.1f}s total"
    )
    logging.info("=" * 60)

    # ─── Phase 4: Talking Heads batch building ────────────────
    dreamface_batches = []
    if use_talking_heads:
        logging.info("🎬 Building Talking Heads audio batches…")
        try:
            dreamface_batches = build_dreamface_batches(
                segments            = segments,
                segment_audio_paths = segment_audio_paths,
                segment_results     = segment_results,
                output_dir          = output_dir,
                timestamp           = timestamp,
            )
            logging.info(
                f"✅ Talking Heads batches ready: "
                f"{len(dreamface_batches)} batch(es)"
            )
        except Exception as e:
            logging.error(f"❌ Talking Heads batch building failed: {e}")
            dreamface_batches = []

    return (
        segment_audio_paths,   # [0] per-segment WAV paths
        final_wav_path,        # [1] full combined narration WAV
        final_mp3_url,         # [2] MP3 URL string
        segment_timestamps,    # [3] list of (start, end) tuples
        nested_chunks,         # [4] word-level timing data
        srt_path,              # [5] subtitle file path
        dreamface_batches,     # [6] batch list with segments_data
    )



# def generate_audio_segments(
#     segments,
#     voice,
#     full_script=None,
#     output_dir=UPLOAD_FOLDER,
#     retry_forever=True,
#     max_retries=12,
#     max_workers=None,
#     num_concurrent_videos=1,
#     retry_failed_segments=True,
#     current_user=None,
#     voice_provider=None,
#     channel_id=None,
#     use_talking_heads=False,
# ):
#     from concurrent.futures import ThreadPoolExecutor, as_completed

#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = time.strftime("%Y%m%d%H%M%S")

#     # ── Resolve voice + provider ──────────────────────────────
#     _voice_id, _provider = resolve_voice(voice, provider_hint=voice_provider)

#     if voice_provider is None and channel_id:
#         _persist_voice_provider(channel_id, _provider)

#     # ── Cartesia early-exit ───────────────────────────────────
    

#     # ── Auto-calculate workers ────────────────────────────────
#     if max_workers is None:
#         max_workers = get_optimal_audio_workers(
#             len(segments), voice, num_concurrent_videos
#         )

#     voice_id, provider = _voice_id, _provider

#     logging.info("")
#     logging.info("=" * 60)
#     logging.info("🚀 AUDIO GENERATION START")
#     logging.info("=" * 60)
#     logging.info(f"   Raw voice value   : '{voice}'")
#     logging.info(f"   Resolved ID       : '{voice_id}'")
#     logging.info(f"   Provider          : {provider}")
#     logging.info(f"   Segments          : {len(segments)}")
#     logging.info(f"   Workers           : {max_workers}")
#     logging.info(f"   Concurrent vids   : {num_concurrent_videos}")
#     logging.info(f"   Channel ID        : {channel_id or 'not provided'}")
#     logging.info(f"   Talking Heads     : {use_talking_heads}")
#     logging.info("=" * 60)

#     # ─── Per-segment worker ───────────────────────────────────
#     def generate_single_segment(seg_data):
#         idx, segment = seg_data
#         text = (segment.get("text") or "").strip()

#         if not text or text.startswith("[CLIP:"):
#             return {
#                 "idx": idx, "audio_path": None, "duration": 0.0,
#                 "word_marks": None, "text": text,
#                 "success": False, "skipped": True,
#             }

#         base     = f"{timestamp}_segment_{idx}"
#         mp3_path = os.path.join(output_dir, f"{base}.mp3")
#         wav_path = os.path.join(output_dir, f"{base}.wav")

#         try:
#             with GLOBAL_API_SEMAPHORE:
#                 local_marks = generate_speech(
#                     text, mp3_path, wav_path, voice,
#                     retry_forever=retry_forever,
#                     max_retries=max_retries,
#                     include_timestamps=True,
#                     current_user=current_user,
#                 )

#             seg_dur = AudioSegment.from_wav(wav_path).duration_seconds
#             logging.info(f"✅ Segment {idx}: {seg_dur:.2f}s")

#             return {
#                 "idx": idx, "audio_path": wav_path, "duration": seg_dur,
#                 "word_marks": local_marks, "text": text,
#                 "success": True, "skipped": False,
#             }

#         except SpeechifyBadVoiceError:
#             raise

#         except Exception as e:
#             logging.error(f"❌ Segment {idx} failed: {e}")
#             return {
#                 "idx": idx, "audio_path": None, "duration": 0.0,
#                 "word_marks": None, "text": text,
#                 "success": False, "skipped": False, "error": str(e),
#             }

#     # ─── Phase 1: Parallel generation ────────────────────────
#     start_time      = time.time()
#     segment_results = {}
#     bad_voice_abort = False

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(generate_single_segment, (i, seg)): i
#             for i, seg in enumerate(segments)
#         }
#         completed = 0
#         for future in as_completed(futures):
#             try:
#                 result = future.result()
#                 segment_results[result["idx"]] = result
#                 if result["success"]:
#                     completed += 1
#                     if completed % 10 == 0:
#                         elapsed = time.time() - start_time
#                         logging.info(
#                             f"⚡ {completed}/{len(segments)} "
#                             f"({completed/elapsed:.1f}/sec)"
#                         )

#             except SpeechifyBadVoiceError:
#                 bad_voice_abort = True
#                 for f in futures:
#                     f.cancel()
#                 logging.error("🚨 Bad voice ID — cancelling all remaining segments")
#                 try:
#                     send_telegram_error(
#                         f"🎙️ ❌ Audio Generation Aborted\n\n"
#                         f"Voice ID: `{voice_id}`\n"
#                         f"Error: Speechify 400 Bad Request\n"
#                         f"Completed before abort: {completed}/{len(segments)}\n\n"
#                         f"⚠️ Action needed: Reassign a valid voice in channel settings.",
#                         current_user,
#                     )
#                 except Exception as tg_err:
#                     logging.warning(f"Telegram notification failed: {tg_err}")
#                 break

#     if bad_voice_abort:
#         return ([], None, None, [], [], None, [])

#     successful = sum(1 for r in segment_results.values() if r["success"])
#     skipped    = sum(1 for r in segment_results.values() if r.get("skipped"))
#     failed     = len(segments) - successful - skipped

#     logging.info(
#         f"Phase 1: ✅{successful} ⏭️{skipped} ❌{failed} "
#         f"in {time.time()-start_time:.1f}s"
#     )

#     # ─── Phase 1.5: Retry failures ───────────────────────────
#     if retry_failed_segments and failed > 0:
#         failed_segs = [
#             (idx, segments[idx])
#             for idx, r in segment_results.items()
#             if not r["success"] and not r.get("skipped")
#         ]
#         logging.warning(f"Retrying {len(failed_segs)} failed segments with 5 workers…")

#         with ThreadPoolExecutor(max_workers=5) as executor:
#             retry_futures = {
#                 executor.submit(generate_single_segment, sd): sd[0]
#                 for sd in failed_segs
#             }
#             for future in as_completed(retry_futures):
#                 try:
#                     result = future.result()
#                     if result["success"]:
#                         segment_results[result["idx"]] = result
#                         logging.info(f"✅ Retry succeeded: segment {result['idx']}")
#                     else:
#                         logging.error(f"❌ Retry failed: segment {result['idx']}")
#                 except SpeechifyBadVoiceError:
#                     logging.error("🚨 Bad voice ID in retry phase — stopping")
#                     break

#         successful = sum(1 for r in segment_results.values() if r["success"])
#         failed     = len(segments) - successful - skipped

#     if failed > 0:
#         failed_idx = [
#             idx for idx, r in segment_results.items()
#             if not r["success"] and not r.get("skipped")
#         ]
#         logging.error(f"🚨 {failed} segments permanently failed: {failed_idx}")
#         try:
#             send_telegram_error(
#                 f"⚠️ Audio Generation — Partial Failure\n\n"
#                 f"Voice ID: `{voice_id}`\n"
#                 f"Failed segments: {failed_idx}\n"
#                 f"Successful: {successful}/{len(segments)}\n\n"
#                 f"Video will have {failed} silent gap(s).",
#                 current_user,
#             )
#         except Exception as tg_err:
#             logging.warning(f"Telegram notification failed: {tg_err}")

#     # ─── Phase 2: Build timestamps ────────────────────────────
#     segment_audio_paths = []
#     segment_timestamps  = []
#     nested_chunks       = []
#     total_duration      = FINAL_PAD_SEC
#     char_offset         = 0
#     last_successful_idx = None

#     for idx in range(len(segments)):
#         result = segment_results.get(idx)

#         if not result or not result["success"]:
#             segment_audio_paths.append(None)
#             segment_timestamps.append((0.0, 0.0))
#             if result and result.get("text"):
#                 char_offset += len(result["text"]) + 1
#             continue

#         segment_audio_paths.append(result["audio_path"])
#         seg_start = total_duration
#         seg_end   = seg_start + result["duration"]
#         segment_timestamps.append((seg_start, seg_end))
#         total_duration      = seg_end
#         last_successful_idx = idx

#         if result["word_marks"]:
#             nested = build_nested_chunk(
#                 result["text"],
#                 result["word_marks"],
#                 seg_start * 1000.0,
#                 char_offset,
#             )
#             if nested:
#                 nested_chunks.append(nested)

#         char_offset += len(result["text"]) + 1

#     if last_successful_idx is not None:
#         s, e = segment_timestamps[last_successful_idx]
#         segment_timestamps[last_successful_idx] = (s, e + FINAL_PAD_SEC)

#     # ─── Phase 3: Combine full narration audio ────────────────
#     logging.info("Combining audio segments…")

#     valid_paths = [p for p in segment_audio_paths if p and os.path.exists(p)]

#     final_wav_path = None
#     final_mp3_path = None
#     final_mp3_url  = None

#     if valid_paths:
#         base_name      = f"narration_{int(time.time())}"
#         final_wav_path = os.path.join(output_dir, f"{base_name}.wav")
#         final_mp3_path = os.path.join(output_dir, f"{base_name}.mp3")

#         concat_list = os.path.join(output_dir, f"narration_concat_{timestamp}.txt")
#         try:
#             with open(concat_list, "w", encoding="utf-8") as f:
#                 for p in valid_paths:
#                     safe = os.path.abspath(p).replace("\\", "/")
#                     f.write(f"file '{safe}'\n")

#             cmd_wav = [
#                 "ffmpeg", "-y",
#                 "-f", "concat", "-safe", "0",
#                 "-i", concat_list,
#                 "-c", "copy",
#                 final_wav_path,
#             ]
#             result = subprocess.run(cmd_wav, capture_output=True, text=True)
#             if result.returncode != 0:
#                 logging.error(
#                     f"❌ Full narration concat failed: {result.stderr[-500:]}"
#                 )
#                 final_wav_path = None
#             else:
#                 wav_size = os.path.getsize(final_wav_path) / (1024 * 1024)
#                 logging.info(f"✅ WAV: {final_wav_path} ({wav_size:.1f}MB)")

#                 cmd_mp3 = [
#                     "ffmpeg", "-y",
#                     "-i", final_wav_path,
#                     "-codec:a", "libmp3lame",
#                     "-b:a", "320k",
#                     final_mp3_path,
#                 ]
#                 result = subprocess.run(cmd_mp3, capture_output=True, text=True)
#                 if result.returncode != 0:
#                     logging.error(f"❌ MP3 export failed: {result.stderr[-500:]}")
#                     final_mp3_path = None
#                 else:
#                     mp3_size = os.path.getsize(final_mp3_path) / (1024 * 1024)
#                     logging.info(f"✅ MP3: {final_mp3_path} ({mp3_size:.1f}MB)")
#                     final_mp3_url = (
#                         f"http://localhost:5000/{UPLOAD_FOLDER}/"
#                         f"{Path(final_mp3_path).name}"
#                     )

#         finally:
#             if os.path.exists(concat_list):
#                 os.remove(concat_list)

#     else:
#         logging.error("No audio to combine!")
#         return ([], None, None, [], [], None, [])

#     # ─── SRT ─────────────────────────────────────────────────
#     srt_path = None
#     if nested_chunks and final_wav_path:
#         srt_path = os.path.join(output_dir, f"{base_name}.srt")
#         if not nested_list_to_srt(nested_chunks, srt_path):
#             srt_path = None

#     total_time = time.time() - start_time
#     logging.info("")
#     logging.info("=" * 60)
#     logging.info(
#         f"🎉 DONE — {successful}/{len(segments)} segments, "
#         f"{total_time:.1f}s total"
#     )
#     logging.info("=" * 60)

#     # ─── Phase 4: Talking Heads batch building ────────────────
#     dreamface_batches = []
#     if use_talking_heads:
#         logging.info("🎬 Building Talking Heads audio batches…")
#         try:
#             dreamface_batches = build_dreamface_batches(
#                 segments            = segments,
#                 segment_audio_paths = segment_audio_paths,
#                 segment_results     = segment_results,
#                 output_dir          = output_dir,
#                 timestamp           = timestamp,
#             )
#             logging.info(
#                 f"✅ Talking Heads batches ready: "
#                 f"{len(dreamface_batches)} batch(es)"
#             )
#         except Exception as e:
#             logging.error(f"❌ Talking Heads batch building failed: {e}")
#             dreamface_batches = []

#     return (
#         segment_audio_paths,   # [0] per-segment WAV paths
#         final_wav_path,        # [1] full combined narration WAV
#         final_mp3_url,         # [2] MP3 URL string
#         segment_timestamps,    # [3] list of (start, end) tuples
#         nested_chunks,         # [4] word-level timing data
#         srt_path,              # [5] subtitle file path
#         dreamface_batches,     # [6] batch list with segments_data
#     )
