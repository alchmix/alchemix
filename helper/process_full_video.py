# import os
# import time
# import logging
# import subprocess
# import tempfile
# import gc
# from datetime import datetime
# import traceback
# import uuid

# from tests import create_video
# from test2 import generate_audio_segments
# from bson.objectid import ObjectId
# from dotenv import load_dotenv
# from mongodb import mongo_instance
# from helper.telegram_notification_engine import send_telegram_notification, send_telegram_error

# load_dotenv()

# FRONTEND_URL = os.getenv("FRONTEND_URL")
# PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:5000")
# db = mongo_instance.get_db()
# OUTPUT = "outputs"


# def standardize_video_for_concat(input_path, output_path):
#     try:
#         cmd = [
#             "ffmpeg", "-y",
#             "-i", input_path,
#             "-c:v", "libx264",
#             "-preset", "fast",
#             "-crf", "18",
#             "-pix_fmt", "yuv420p",
#             "-r", "30",
#             "-g", "60",
#             "-keyint_min", "60",
#             "-sc_threshold", "0",
#             "-c:a", "aac",
#             "-ar", "48000",
#             "-ac", "2",
#             "-b:a", "128k",
#             "-movflags", "+faststart",
#             "-vsync", "cfr",
#             output_path
#         ]
#         subprocess.run(cmd, check=True)
#         logging.info(f"✅ Standardized video: {output_path}")
#     except subprocess.CalledProcessError as e:
#         logging.error(f"❌ Failed to standardize video {input_path}: {e}")
#         raise


# def sanitize_path(path):
#     return os.path.abspath(path).replace("\\", "/")


# def safe_concatenate_videos(video_paths, output_path):
#     try:
#         concat_file = os.path.join(tempfile.gettempdir(), f"concat_{time.time()}.txt")

#         with open(concat_file, 'w', encoding='utf-8') as f:
#             for video_path in video_paths:
#                 f.write(f"file '{sanitize_path(video_path)}'\n")

#         try:
#             ffmpeg_cmd = [
#                 "ffmpeg", "-y",
#                 "-f", "concat",
#                 "-safe", "0",
#                 "-protocol_whitelist", "file,pipe",
#                 "-i", concat_file,
#                 "-c", "copy",
#                 "-movflags", "+faststart",
#                 output_path
#             ]
#             subprocess.run(ffmpeg_cmd, check=True)
#             logging.info("✅ Direct concatenation successful")

#         except subprocess.CalledProcessError:
#             logging.warning("❗ Direct copy failed. Re-encoding...")
#             ffmpeg_cmd = [
#                 "ffmpeg", "-y",
#                 "-f", "concat",
#                 "-safe", "0",
#                 "-protocol_whitelist", "file,pipe",
#                 "-i", concat_file,
#                 "-c:v", "libx264",
#                 "-preset", "fast",
#                 "-crf", "18",
#                 "-pix_fmt", "yuv420p",
#                 "-r", "30",
#                 "-g", "60",
#                 "-keyint_min", "60",
#                 "-sc_threshold", "0",
#                 "-c:a", "aac",
#                 "-ar", "48000",
#                 "-ac", "2",
#                 "-b:a", "128k",
#                 "-movflags", "+faststart",
#                 "-vsync", "cfr",
#                 output_path
#             ]
#             subprocess.run(ffmpeg_cmd, check=True)
#             logging.info("✅ Re-encoded concatenation successful")

#         finally:
#             if os.path.exists(concat_file):
#                 os.remove(concat_file)

#     except Exception as e:
#         logging.error(f"❌ Concatenation failed: {e}")
#         raise


# def _extract_audio_paths(result):
#     if not result:
#         return None
#     if isinstance(result, (list, tuple)):
#         return result[0]
#     return result


# def _extract_full_audio_path(result):
#     if not result:
#         return None
#     if isinstance(result, (list, tuple)):
#         return result[1]
#     return result


# def process_full_video(user_id, data, vid_data_id, asset_folder=None):
#     try:
#         logging.info(f"👤 Processing full video for user: {user_id}")
#         db.video_data.update_one(
#             {"_id": ObjectId(vid_data_id)},
#             {"$set": {"status": "processing"}}
#         )

#         # ------- inputs & flags -------
#         sections      = (data or {}).get('data', {}) or {}
#         intro_section = (sections.get('intro') or {}) or {}
#         body_section  = (sections.get('body')  or {}) or {}

#         intro_segments = intro_section.get('segment', []) or []
#         body_segments  = body_section.get('segment',  []) or []

#         intro_script = intro_section.get("section_script", "")
#         body_script  = body_section.get("section_script",  "")

#         has_body = (
#             body_segments is not None
#             and isinstance(body_segments, list)
#             and any(
#                 seg.get("text")
#                 for seg in body_segments
#                 if isinstance(seg, dict)
#             )
#         )

#         avatar            = data.get('avatarImage')
#         background_video  = data.get('backgroundVideo')
#         voice             = data.get("voice", "alloy")
#         background        = data.get("backgroundMusic")
#         bg_music          = data.get("bgMusic")
#         niche             = data.get('niche', 'original')
#         phase_media       = data.get('mediaPlan')
#         kenburns          = data.get('kenBurns')
#         storyboard        = data.get('storyboardAnimation')
#         orientation       = data.get('orientation')
#         avatar_active     = data.get('avatar')
#         caption_settings  = data.get('captionSettings')
#         captions_enabled  = data.get('captionsEnabled')
#         transition_config = data.get('transition_config')
#         use_talking_heads = data.get('useTalkingHeads', False)  # ← NEW

#         if bg_music:
#             background = bg_music

#         if not orientation:
#             orientation = 'landscape'

#         # Mode selection
#         mode = "create"
#         if niche == "commentary":
#             mode = "commentary"
#         elif niche == "original":
#             mode = "original"

#         if not intro_segments:
#             raise ValueError("❌ Missing intro segment data")

#         # ------- narration (intro) -------
#         logging.info("🔊 Generating narration (intro)...")
#         intro_result = generate_audio_segments(
#             intro_segments,
#             voice,
#             intro_script,
#             use_talking_heads=use_talking_heads,  # ← NEW
#         )
#         intro_segment_audio_paths = _extract_audio_paths(intro_result)
#         full_intro_audio_path     = _extract_full_audio_path(intro_result)
#         full_intro_srt_path       = intro_result[5]
#         full_intro_nested_chunks  = intro_result[4]
#         dreamface_batches         = intro_result[6]  # ← NEW

#         if not intro_segment_audio_paths:
#             raise RuntimeError("❌ Intro narration generation failed")

#         # ------- TALKING HEADS INTERCEPT -------
#         if use_talking_heads and dreamface_batches:
#             logging.info("🎬 Talking heads mode — intercepting pipeline...")

#             base_url = PUBLIC_BASE_URL.rstrip("/")
#             for batch in dreamface_batches:
#                 filename = os.path.basename(batch["audio_path"])
#                 batch["download_url"] = f"{base_url}/api/dreamface/audio/batches/{filename}"
#                 batch["avatar_url"]   = None

#             job = {
#                 "user_id":    user_id,
#                 "vid_data_id": vid_data_id,
#                 "status":     "awaiting_dreamface",
#                 "created_at": datetime.utcnow(),

#                 "create_video_payload": {
#                     "phase_media":       phase_media,
#                     "mode":              mode,
#                     "avatar":            avatar,
#                     "background_video":  background_video,
#                     "background":        background,
#                     "orientation":       orientation,
#                     "storyboard":        storyboard,
#                     "kenburns":          kenburns,
#                     "avatar_active":     avatar_active,
#                     "caption_settings":  caption_settings,
#                     "captions_enabled":  captions_enabled,
#                     "transition_config": transition_config,
#                     "clip_data_list":    intro_section.get('clips', []),
#                     "image_data_list":   intro_section.get('imageData', []),
#                     "vid_segments":      intro_segments,
#                     "characters":        intro_section.get('section_characters', []),
#                     "narration_text":    intro_script,
#                     "asset_folder":      asset_folder,
#                     "section_type":      "intro",
#                     "title":             data.get('title', ''),
#                     "niche":             niche,
#                 },

#                 "audio_payload": {
#                     "segment_audio_paths": intro_segment_audio_paths,
#                     "full_wav_path":       full_intro_audio_path,
#                     "srt_path":            full_intro_srt_path,
#                 },

#                 "batches": dreamface_batches,
#             }

#             result = db.dreamface_jobs.insert_one(job)
#             job_id = str(result.inserted_id)

#             logging.info(f"✅ DreamFace job saved: {job_id}")

#             db.video_data.update_one(
#                 {"_id": ObjectId(vid_data_id)},
#                 {"$set": {
#                     "status": "awaiting_dreamface",
#                     "dreamface_job_id": job_id
#                 }}
#             )

#             batch_links = "\n".join([
#                 f"Batch {b['batch_index']} ({b['duration_sec']:.1f}s):\n{b['download_url']}"
#                 for b in dreamface_batches
#             ])

#             send_telegram_notification(
#                 f"🎬 DreamFace Audio Ready!\n\n"
#                 f"Job: {job_id}\n"
#                 f"Batches: {len(dreamface_batches)}\n\n"
#                 f"{batch_links}\n\n"
#                 f"Download, process in DreamFace, then submit avatar links.",
#                 "View Job",
#                 f"{FRONTEND_URL}/dreamface/{job_id}",
#                 user_id
#             )

#             logging.info("✅ Pipeline intercepted — awaiting DreamFace avatar links")
#             return  # ← Pipeline stops here

#         # ------- narration (body) — legacy -------
#         body_segment_audio_paths = None
#         full_body_audio_path     = None
#         full_body_srt_path       = None

#         if has_body:
#             logging.info("🔊 Generating narration (body)...")
#             body_result = generate_audio_segments(
#                 body_segments,
#                 voice,
#                 body_script,
#             )
#             body_segment_audio_paths = _extract_audio_paths(body_result)
#             full_body_audio_path     = _extract_full_audio_path(body_result)
#             full_body_srt_path       = body_result[5]
#             if not body_segment_audio_paths:
#                 raise RuntimeError("❌ Body narration generation failed")

#         # ------- render intro video -------
#         logging.info("🎬 Generating intro video...")
#         intro_path, intro_folder, _ = create_video(
#             phase_media,
#             full_audio=full_intro_audio_path,
#             mode=mode,
#             avatar=avatar,
#             background_video=background_video,
#             background=background,
#             user_id=user_id,
#             section_type='intro',
#             orientation=orientation,
#             storyboard=storyboard,
#             kenburns=kenburns,
#             srt=full_intro_srt_path,
#             avatar_active=avatar_active,
#             caption_settings=caption_settings,
#             captions_enabled=captions_enabled,
#             transition_config=transition_config,
#             script=intro_script,
#             clip_data_list=intro_section.get('clips', []),
#             image_data_list=intro_section.get('imageData', []),
#             segment_audio_paths=intro_segment_audio_paths,
#             narration_text=intro_section.get('section_script', ''),
#             characters=intro_section.get('section_characters', []),
#             vid_segments=intro_segments,
#             asset_folder=asset_folder
#         )

#         # ------- render body video — legacy -------
#         body_path = body_folder = None
#         if has_body:
#             logging.info("🎬 Generating body video...")
#             body_path, body_folder, _ = create_video(
#                 phase_media,
#                 full_audio=full_body_audio_path,
#                 mode=mode,
#                 avatar=avatar,
#                 background_video=background_video,
#                 background=background,
#                 user_id=user_id,
#                 section_type='body',
#                 orientation=orientation,
#                 storyboard=storyboard,
#                 kenburns=kenburns,
#                 srt=full_body_srt_path,
#                 avatar_active=avatar_active,
#                 caption_settings=caption_settings,
#                 captions_enabled=captions_enabled,
#                 transition_config=transition_config,
#                 script=body_script,
#                 clip_data_list=body_section.get('clips', []),
#                 image_data_list=body_section.get('imageData', []),
#                 segment_audio_paths=body_segment_audio_paths,
#                 narration_text=body_section.get('section_script', ''),
#                 characters=body_section.get('section_characters', []),
#                 vid_segments=body_segments,
#                 asset_folder=asset_folder
#             )

#         if not intro_path or not os.path.exists(intro_path):
#             raise RuntimeError("❌ Intro video generation failed")

#         # ------- finalize -------
#         os.makedirs(OUTPUT, exist_ok=True)
#         final_name = f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#         final_path = sanitize_path(os.path.join(OUTPUT, final_name))

#         # logging.info("🔧 Standardizing for finalization...")
#         # intro_std = intro_path.replace(".mp4", "_std.mp4")
#         # standardize_video_for_concat(intro_path, intro_std)

#         logging.info("➡️ Using intro as final (stream copy)...")
#         subprocess.run(
#     ["ffmpeg", "-y", "-i", intro_path, "-c", "copy", "-movflags", "+faststart", final_path],
#     check=True,)
        
#         if not os.path.exists(final_path):
#             raise RuntimeError("❌ Final video was not created")
        
        

#         # body_std = None
#         # if has_body and body_path:
#         #     body_std = body_path.replace(".mp4", "_std.mp4")
#         #     standardize_video_for_concat(body_path, body_std)

#         # if has_body and body_std:
#         #     logging.info("🧩 Concatenating intro + body...")
#         #     safe_concatenate_videos([intro_std, body_std], final_path)
#         # else:
#         #     logging.info("➡️ No body — using intro as final...")
#         #     subprocess.run(
#         #         ["ffmpeg", "-y", "-i", intro_std, "-c", "copy", final_path],
#         #         check=True
#         #     )

#         if not os.path.exists(final_path):
#             raise RuntimeError("❌ Final video was not created")

#         # ------- DB persist -------
#         logging.info("💾 Saving metadata to database...")
#         combined_asset_folder = (
#             intro_folder if not body_folder
#             else f"{intro_folder}-{body_folder}"
#         )
#         result = db.completed_videos.insert_one({
#             "title":        data.get('title', ''),
#             "sections":     sections,
#             "video_path":   final_name,
#             "asset_folder": combined_asset_folder,
#             "created_at":   datetime.utcnow(),
#             "niche":        niche,
#             "user_id":      user_id
#         })

#         # ------- cleanup intro only-------
#         logging.info("🧹 Cleaning intermediate files...")
#         for p in [intro_path]:
#             try:
#                 if os.path.exists(p):
#                     os.remove(p)
#             except Exception as ce:
#                 logging.warning(f"Could not delete {p}: {ce}")

#         db.video_data.update_one(
#             {"_id": ObjectId(vid_data_id)},
#             {"$set": {"status": "completed"}}
#         )

#         send_telegram_notification(
#             "🎉 Video Completed!",
#             "View",
#             f"{FRONTEND_URL}/completed/{str(result.inserted_id)}",
#             user_id
#         )
#         logging.info(f"✅ Video processing complete for user {user_id}")

#     except Exception as e:
#         db.video_data.update_one(
#             {"_id": ObjectId(vid_data_id)},
#             {"$set": {"status": "draft"}}
#         )
#         logging.error(f"❌ Failed to process video: {e}")
#         send_telegram_error(f"❌ Video processing failed: {e}", user_id)
#     finally:
#         gc.collect()


# def resume_dreamface_pipeline(job_id, user_id):
#     """
#     Resumes create_video after avatar links are submitted.

#     The avatar video is treated as completely separate from the
#     content pipeline. It never touches clip_data_list, media_pool,
#     tags, or download_video_clips_in_parallel.

#     Flow:
#       1. Download avatar video(s) locally
#       2. Build avatar_segment_map { seg_idx: { path, avatar_start, duration } }
#       3. Call create_video with the ORIGINAL payload untouched +
#          layout_config + avatar_segment_map
#       4. encode_video_segment runs normally for content,
#          overlay_with_layout composites avatar on top at the end
#     """
#     try:
#         job = db.dreamface_jobs.find_one({"_id": ObjectId(job_id)})
#         if not job:
#             raise ValueError(f"Job {job_id} not found")

#         payload       = job["create_video_payload"]
#         audio_payload = job["audio_payload"]
#         batches       = job["batches"]
#         vid_data_id   = job["vid_data_id"]
#         layout_config = job.get("layout_config", None)

#         logging.info(f"▶️  Resuming Talking Heads pipeline for job {job_id}")
#         logging.info(
#             f"📐 Layout config: "
#             f"{layout_config.get('layout_id') if layout_config else 'None'}"
#         )

#         if not layout_config:
#             raise ValueError(
#                 "layout_config missing from job — "
#                 "was it saved when avatar links were submitted?"
#             )

#         # ── Update job status ─────────────────────────────────
#         db.dreamface_jobs.update_one(
#             {"_id": ObjectId(job_id)},
#             {"$set": {"status": "processing"}}
#         )

#         # ── Download avatars + build avatar_segment_map ───────
#         # avatar_segment_map = {
#         #     seg_idx: {
#         #         "path":         local mp4 path,
#         #         "batch_index":  int,
#         #         "avatar_start": float (seek offset in avatar file),
#         #         "duration":     float (this segment's audio duration),
#         #     }
#         # }
#         # This is the ONLY mechanism for passing avatar info
#         # to encode_video_segment. Nothing else is touched.
#         avatar_segment_map = {}

#         for batch in batches:
#             batch_index   = batch["batch_index"]
#             raw_url       = batch["avatar_url"]
#             segments_data = batch.get("segments_data", [])

#             logging.info(
#                 f"📦 Processing batch {batch_index}: "
#                 f"{len(segments_data)} segments | "
#                 f"URL: {raw_url[:60]}…"
#             )

#             # ── Download / extract avatar video ───────────────
#             local_path = _resolve_avatar_url(
#                 raw_url     = raw_url,
#                 batch_index = batch_index,
#                 job_id      = job_id,
#             )

#             if not local_path:
#                 raise RuntimeError(
#                     f"❌ Could not obtain avatar video "
#                     f"for batch {batch_index}. URL: {raw_url}"
#                 )

#             size_mb = os.path.getsize(local_path) / (1024 * 1024)
#             logging.info(
#                 f"✅ Batch {batch_index} avatar ready: "
#                 f"{local_path} ({size_mb:.1f}MB)"
#             )

#             # ── Map each segment to its seek offset ───────────
#             # avatar_start = cumulative offset within THIS batch's
#             # avatar file. Pre-computed in build_dreamface_batches
#             # from actual WAV durations. Resets to 0.0 per batch.
#             for seg_data in segments_data:
#                 avatar_segment_map[seg_data["seg_index"]] = {
#                     "path":         local_path,
#                     "batch_index":  batch_index,
#                     "avatar_start": seg_data["avatar_start"],
#                     "duration":     seg_data["duration_sec"],
#                 }

#             logging.info(
#                 f"✅ Batch {batch_index}: "
#                 f"{len(segments_data)} segments mapped | "
#                 f"seek offsets: "
#                 f"{[(s['seg_index'], s['avatar_start']) for s in segments_data]}"
#             )

#         logging.info(
#             f"🗺️  avatar_segment_map complete: "
#             f"{len(avatar_segment_map)} segments mapped"
#         )

#         # ── Call create_video with ORIGINAL payload ───────────
#         # vid_segments and clip_data_list are taken directly from
#         # the saved payload — completely untouched.
#         # The avatar reaches encode_video_segment via
#         # avatar_segment_map only. Nothing else changes.
#         logging.info(
#             f"🎬 Calling create_video | "
#             f"layout={layout_config.get('layout_id')} | "
#             f"avatar_segments={len(avatar_segment_map)}"
#         )

#         intro_path, intro_folder, _ = create_video(
#             payload["phase_media"],
#             full_audio          = audio_payload["full_wav_path"],
#             mode                = payload["mode"],
#             avatar              = payload["avatar"],
#             background_video    = payload["background_video"],
#             background          = payload["background"],
#             user_id             = user_id,
#             section_type        = payload["section_type"],
#             orientation         = payload["orientation"],
#             storyboard          = payload["storyboard"],
#             kenburns            = payload["kenburns"],
#             srt                 = audio_payload["srt_path"],
#             avatar_active       = payload["avatar_active"],
#             caption_settings    = payload["caption_settings"],
#             captions_enabled    = payload["captions_enabled"],
#             transition_config   = payload["transition_config"],
#             script              = payload["narration_text"],
#             clip_data_list      = payload["clip_data_list"],    # ← untouched
#             image_data_list     = payload["image_data_list"],   # ← untouched
#             segment_audio_paths = audio_payload["segment_audio_paths"],
#             narration_text      = payload["narration_text"],
#             characters          = payload["characters"],
#             vid_segments        = payload["vid_segments"],      # ← untouched
#             asset_folder        = payload["asset_folder"],
#             layout_config       = layout_config,                # ← canvas coords
#             avatar_segment_map  = avatar_segment_map,           # ← avatar only
#         )

#         if not intro_path or not os.path.exists(intro_path):
#             raise RuntimeError("❌ create_video returned no output path")

#         # ── Finalize ──────────────────────────────────────────
#         os.makedirs(OUTPUT, exist_ok=True)
#         final_name = f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#         final_path = sanitize_path(os.path.join(OUTPUT, final_name))

#         # logging.info("🔧 Standardizing for finalization…")
#         # intro_std = intro_path.replace(".mp4", "_std.mp4")
#         # standardize_video_for_concat(intro_path, intro_std)

#         logging.info("➡️ Using intro as final (stream copy)...")
#         subprocess.run(
#     ["ffmpeg", "-y", "-i", intro_path, "-c", "copy", "-movflags", "+faststart", final_path],
#     check=True,
# )

#         if not os.path.exists(final_path):
#             raise RuntimeError(
#                 "❌ Final video was not created"
#             )

#         # ── Save to completed_videos ──────────────────────────
#         result = db.completed_videos.insert_one({
#             "title":        payload.get("title", ""),
#             "video_path":   final_name,
#             "asset_folder": intro_folder,
#             "created_at":   datetime.utcnow(),
#             "niche":        payload.get("niche", ""),
#             "user_id":      user_id,
#         })

#         # ── Cleanup temp files ────────────────────────────────
#         for p in [intro_path]:
#             try:
#                 if os.path.exists(p):
#                     os.remove(p)
#             except Exception as ce:
#                 logging.warning(f"Could not delete {p}: {ce}")

#         # ── Mark job complete ─────────────────────────────────
#         db.dreamface_jobs.update_one(
#             {"_id": ObjectId(job_id)},
#             {"$set": {"status": "completed"}}
#         )
#         db.video_data.update_one(
#             {"_id": ObjectId(vid_data_id)},
#             {"$set": {"status": "completed"}}
#         )

#         send_telegram_notification(
#             "🎉 Talking Heads Video Completed!",
#             "View",
#             f"{FRONTEND_URL}/completed/{str(result.inserted_id)}",
#             user_id,
#         )
#         logging.info(
#             f"✅ Talking Heads pipeline complete — job {job_id}"
#         )

#     except Exception as e:
#         full_trace = traceback.format_exc()
#         logging.error(
#             f"❌ Talking Heads pipeline resume failed: "
#             f"{e}\n{full_trace}"
#         )

#         db.dreamface_jobs.update_one(
#             {"_id": ObjectId(job_id)},
#             {"$set": {"status": "failed", "error": str(e)}}
#         )

#         send_telegram_error(
#             f"❌ Talking Heads resume failed for job {job_id}\n\n"
#             f"Error: {e}\n\n"
#             f"Trace: {full_trace[:400]}",
#             user_id,
#         )
#     finally:
#         gc.collect()

# def _resolve_avatar_url(raw_url: str, batch_index: int, job_id: str) -> str | None:
#     """
#     Resolves a user-supplied avatar URL to a local file path.

#     Handles three cases:
#       1. DreamFace share link  → Playwright extraction + download
#       2. Direct video URL      → requests download
#       3. Already a local path  → return as-is (shouldn't happen normally)
#     """
#     import re

#     AVATAR_DIR = os.getenv("AVATAR_DOWNLOAD_DIR", "static/avatars")
#     os.makedirs(AVATAR_DIR, exist_ok=True)

#     # ── Case 3: already a local path ─────────────────────────
#     if not raw_url.startswith("http"):
#         if os.path.exists(raw_url):
#             logging.info(f"📁 Batch {batch_index}: using existing local path")
#             return raw_url
#         else:
#             logging.error(f"❌ Batch {batch_index}: local path not found: {raw_url}")
#             return None

#     # ── Case 1: DreamFace share link ─────────────────────────
#     share_link_patterns = [
#         r"dreamfaceapp\.com/creation",
#         r"dreamfaceapp\.com/.+previewId",
#     ]
#     is_share_link = any(re.search(p, raw_url) for p in share_link_patterns)

#     if is_share_link:
#         logging.info(
#             f"🔍 Batch {batch_index}: detected share link — "
#             f"launching Playwright extractor…"
#         )
#         try:
#             from helper.avatar_extractor import extract_avatar_from_share_link
#             local_path = extract_avatar_from_share_link(raw_url, output_dir=AVATAR_DIR)
#             if local_path and os.path.exists(local_path):
#                 size_mb = os.path.getsize(local_path) / (1024 * 1024)
#                 logging.info(
#                     f"✅ Batch {batch_index}: extracted → {local_path} ({size_mb:.1f}MB)"
#                 )
#                 return local_path
#             else:
#                 logging.error(
#                     f"❌ Batch {batch_index}: Playwright extraction returned nothing"
#                 )
#                 return None
#         except Exception as e:
#             logging.error(
#                 f"❌ Batch {batch_index}: share link extraction failed: {e}"
#             )
#             return None

#     # ── Case 2: Direct video URL ──────────────────────────────
#     logging.info(f"⬇️  Batch {batch_index}: direct URL download…")
#     try:
#         import requests as req
#         filename  = f"avatar_batch{batch_index}_{job_id[:8]}_{uuid.uuid4().hex[:8]}.mp4"
#         save_path = os.path.join(AVATAR_DIR, filename)

#         headers = {
#             "User-Agent": (
#                 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                 "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
#             ),
#             "Referer": "https://www.dreamfaceapp.com/",
#         }

#         response = req.get(raw_url, headers=headers, stream=True, timeout=180)
#         response.raise_for_status()

#         with open(save_path, "wb") as f:
#             for chunk in response.iter_content(chunk_size=1024 * 1024):
#                 if chunk:
#                     f.write(chunk)

#         file_size = os.path.getsize(save_path)
#         if file_size < 100_000:
#             os.remove(save_path)
#             raise ValueError(f"Downloaded file too small: {file_size} bytes")

#         size_mb = file_size / (1024 * 1024)
#         logging.info(
#             f"✅ Batch {batch_index}: downloaded → {save_path} ({size_mb:.1f}MB)"
#         )
#         return save_path

#     except Exception as e:
#         logging.error(f"❌ Batch {batch_index}: direct download failed: {e}")
#         return None

# def process_regenerated_video(user_id, data, asset_folder):
#     try:
#         logging.info(f"👤 Regenerating full video for user: {user_id}")
#         avatar           = data.get('avatarImage')
#         background_video = data.get('backgroundVideo')
#         voice            = data.get("voice", "alloy")
#         background       = data.get("backgroundMusic")
#         niche            = data.get("niche", "original")
#         phase_media      = data.get('mediaPlan')

#         mode = "regenerate_commentary" if niche == "commentary" else "regenerate"

#         sections       = (data or {}).get("sections", {}) or {}
#         intro_segments = (sections.get("intro") or {}).get("segment", []) or []
#         body_segments  = (sections.get("body")  or {}).get("segment", []) or []

#         has_body = bool(body_segments)

#         if not intro_segments:
#             raise ValueError("❌ Missing intro segment data")

#         logging.info("🔊 Generating narration (intro)...")
#         intro_result = generate_audio_segments(intro_segments, voice)
#         intro_segment_audio_paths = _extract_audio_paths(intro_result)
#         full_intro_audio_path     = _extract_full_audio_path(intro_result)

#         if not intro_segment_audio_paths:
#             raise RuntimeError("Intro narration generation failed.")

#         body_segment_audio_paths = None
#         full_body_audio_path     = None

#         if has_body:
#             logging.info("🔊 Generating narration (body)...")
#             body_result = generate_audio_segments(body_segments, voice)
#             body_segment_audio_paths = _extract_audio_paths(body_result)
#             full_body_audio_path     = _extract_full_audio_path(body_result)
#             if not body_segment_audio_paths:
#                 raise RuntimeError("Body narration generation failed.")

#         logging.info("🎬 Generating intro video...")
#         intro_video_result = create_video(
#             phase_media,
#             full_audio=full_intro_audio_path,
#             mode=mode,
#             avatar=avatar,
#             background_video=background_video,
#             background=background,
#             user_id=user_id,
#             section_type="intro",
#             segment_audio_paths=intro_segment_audio_paths,
#             vid_segments=intro_segments,
#             asset_folder=asset_folder
#         )

#         logging.info(f"📽️ create_video returned for intro: {intro_video_result}")

#         if intro_video_result:
#             intro_video_path, intro_folder = intro_video_result[0], intro_video_result[1]
#         else:
#             intro_video_path, intro_folder = None, None
#             logging.error("❌ create_video for intro returned None")

#         body_video_path = body_folder = None
#         if has_body:
#             logging.info("🎬 Generating body video...")
#             body_video_result = create_video(
#                 phase_media,
#                 full_audio=full_body_audio_path,
#                 mode=mode,
#                 avatar=avatar,
#                 background_video=background_video,
#                 background=background,
#                 user_id=user_id,
#                 section_type="body",
#                 segment_audio_paths=body_segment_audio_paths,
#                 vid_segments=body_segments,
#                 asset_folder=asset_folder
#             )

#             logging.info(f"📽️ create_video returned for body: {body_video_result}")

#             if body_video_result:
#                 body_video_path, body_folder = body_video_result[0], body_video_result[1]
#             else:
#                 body_video_path, body_folder = None, None
#                 logging.error("❌ create_video for body returned None")

#         os.makedirs(OUTPUT, exist_ok=True)
#         final_name = f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#         final_path = sanitize_path(os.path.join(OUTPUT, final_name))

#         logging.info("🔧 Standardizing for finalization...")
#         intro_std = intro_video_path.replace(".mp4", "_std.mp4")
#         standardize_video_for_concat(intro_video_path, intro_std)

#         body_std = None
#         if has_body and body_video_path:
#             body_std = body_video_path.replace(".mp4", "_std.mp4")
#             standardize_video_for_concat(body_video_path, body_std)

#         if has_body and body_std:
#             logging.info("🧩 Concatenating intro + body...")
#             safe_concatenate_videos([intro_std, body_std], final_path)
#         else:
#             logging.info("➡️ No body — using intro as final...")
#             subprocess.run(
#                 ["ffmpeg", "-y", "-i", intro_std, "-c", "copy", final_path],
#                 check=True
#             )

#         if not os.path.exists(final_path):
#             raise RuntimeError("Final video creation failed")

#         combined_asset_folder = (
#             intro_folder if not body_folder
#             else f"{intro_folder}-{body_folder}"
#         )
#         result = db.completed_videos.insert_one({
#             "title":        data.get('title', ''),
#             "sections":     sections,
#             "video_path":   final_name,
#             "asset_folder": combined_asset_folder,
#             "created_at":   datetime.utcnow(),
#             "niche":        niche,
#             "user_id":      user_id,
#         })

#         for path in filter(None, [intro_video_path, body_video_path, intro_std, body_std]):
#             try:
#                 if os.path.exists(path):
#                     os.remove(path)
#                     logging.info(f"🧼 Deleted: {path}")
#             except Exception as e:
#                 logging.warning(f"Could not delete {path}: {e}")

#         send_telegram_notification(
#             "🎉 Video Regeneration Completed!",
#             "View",
#             f"{FRONTEND_URL}/completed/{str(result.inserted_id)}",
#             user_id
#         )
#         logging.info(f"🎬 Full video job complete for user {user_id}.")

#     except Exception as e:
#         logging.error(f"❌ Failed to process Regenerated video: {e}")
#         send_telegram_error(
#             f"🎉 ❌ Video regeneration failed: {e}",
#             user_id
#         )
#     finally:
#         gc.collect()


import os
import time
import logging
import subprocess
import tempfile
import gc
from datetime import datetime
import traceback
import uuid

import env_loader  # noqa: F401

from segment_pool_worker import get_create_video


def create_video(*args, **kwargs):
    return get_create_video()(*args, **kwargs)


from test2 import generate_audio_segments
from pipeline_fair_share import audio_max_workers
from bson.objectid import ObjectId
from mongodb import mongo_instance
from helper.telegram_notification_engine import send_telegram_notification, send_telegram_error

FRONTEND_URL = os.getenv("FRONTEND_URL")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:5000")
db = mongo_instance.get_db()
OUTPUT = "outputs"


def _generate_audio_fair(*args, **kwargs):
    """TTS with fair-share worker cap when multiple videos run in parallel."""
    workers = audio_max_workers()
    kwargs.setdefault("max_workers", workers)
    kwargs.setdefault("workers", workers)
    try:
        return generate_audio_segments(*args, **kwargs)
    except TypeError:
        if "max_workers" in kwargs or "workers" in kwargs:
            kwargs.pop("max_workers", None)
            kwargs.pop("workers", None)
            logging.warning(
                "test2.generate_audio_segments ignores max_workers — "
                "add fair-share in test2.py (see pipeline_fair_share.py)"
            )
        return generate_audio_segments(*args, **kwargs)


def standardize_video_for_concat(input_path, output_path):
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-g", "60",
            "-keyint_min", "60",
            "-sc_threshold", "0",
            "-c:a", "aac",
            "-ar", "48000",
            "-ac", "2",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-vsync", "cfr",
            output_path
        ]
        subprocess.run(cmd, check=True)
        logging.info(f"✅ Standardized video: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to standardize video {input_path}: {e}")
        raise

def sanitize_path(path):
    return os.path.abspath(path).replace("\\", "/")

def safe_concatenate_videos(video_paths, output_path):
    try:
        concat_file = os.path.join(tempfile.gettempdir(), f"concat_{time.time()}.txt")

        with open(concat_file, 'w', encoding='utf-8') as f:
            for video_path in video_paths:
                f.write(f"file '{sanitize_path(video_path)}'\n")

        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-protocol_whitelist", "file,pipe",
                "-i", concat_file,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            logging.info("✅ Direct concatenation successful")

        except subprocess.CalledProcessError:
            logging.warning("❗ Direct copy failed. Re-encoding...")
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-protocol_whitelist", "file,pipe",
                "-i", concat_file,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-r", "30",
                "-g", "60",
                "-keyint_min", "60",
                "-sc_threshold", "0",
                "-c:a", "aac",
                "-ar", "48000",
                "-ac", "2",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-vsync", "cfr",
                output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            logging.info("✅ Re-encoded concatenation successful")

        finally:
            if os.path.exists(concat_file):
                os.remove(concat_file)

    except Exception as e:
        logging.error(f"❌ Concatenation failed: {e}")
        raise

def _extract_audio_paths(result):
    if not result:
        return None
    if isinstance(result, (list, tuple)):
        return result[0]
    return result

def _extract_full_audio_path(result):
    if not result:
        return None
    if isinstance(result, (list, tuple)):
        return result[1]
    return result

def process_full_video(user_id, data, vid_data_id, asset_folder=None):
    try:
        logging.info(f"👤 Processing full video for user: {user_id}")
        db.video_data.update_one(
            {"_id": ObjectId(vid_data_id)},
            {"$set": {"status": "processing"}}
        )

        sections      = (data or {}).get('data', {}) or {}
        intro_section = (sections.get('intro') or {}) or {}
        body_section  = (sections.get('body')  or {}) or {}

        intro_segments = intro_section.get('segment', []) or []
        body_segments  = body_section.get('segment',  []) or []

        intro_script = intro_section.get("section_script", "")
        body_script  = body_section.get("section_script",  "")

        has_body = (
            body_segments is not None
            and isinstance(body_segments, list)
            and any(
                seg.get("text")
                for seg in body_segments
                if isinstance(seg, dict)
            )
        )

        avatar            = data.get('avatarImage')
        background_video  = data.get('backgroundVideo')
        voice             = data.get("voice", "alloy")
        background        = data.get("backgroundMusic")
        bg_music          = data.get("bgMusic")
        niche             = data.get('niche', 'original')
        phase_media       = data.get('mediaPlan')
        kenburns          = data.get('kenBurns')
        storyboard        = data.get('storyboardAnimation')
        orientation       = data.get('orientation')
        avatar_active     = data.get('avatar')
        caption_settings  = data.get('captionSettings')
        captions_enabled  = data.get('captionsEnabled')
        transition_config = data.get('transition_config')
        use_talking_heads = data.get('useTalkingHeads', False)

        # ── NEW: teaser + narration images from frontend ─────────
        intro_teaser_config      = data.get("introTeaser")
        narration_images_config  = data.get("narrationImages")
        transition_banner_config = data.get("transitionBanner")

        from helper.teaser_backend import normalize_teaser_config
        _teaser_preview = normalize_teaser_config(intro_teaser_config)
        logging.info(
            f"📺 Intro teaser from client | text={_teaser_preview.get('chyronText')!r} | "
            f"layout={_teaser_preview.get('layoutMode')} | bg={_teaser_preview.get('stripColor')}"
        )

        if bg_music:
            background = bg_music

        if not orientation:
            orientation = 'landscape'

        mode = "create"
        if niche == "commentary":
            mode = "commentary"
        elif niche == "original":
            mode = "original"

        if not intro_segments:
            raise ValueError("❌ Missing intro segment data")

        logging.info("🔊 Generating narration (intro)...")
        intro_result = _generate_audio_fair(
            intro_segments,
            voice,
            intro_script,
            use_talking_heads=use_talking_heads,
        )
        intro_segment_audio_paths = _extract_audio_paths(intro_result)
        full_intro_audio_path     = _extract_full_audio_path(intro_result)
        full_intro_srt_path       = intro_result[5]
        full_intro_nested_chunks  = intro_result[4]
        dreamface_batches         = intro_result[6]

        if not intro_segment_audio_paths:
            raise RuntimeError("❌ Intro narration generation failed")

        # ── TALKING HEADS INTERCEPT ─────────────────────────────
        if use_talking_heads and dreamface_batches:
            logging.info("🎬 Talking heads mode — intercepting pipeline...")

            base_url = PUBLIC_BASE_URL.rstrip("/")
            for batch in dreamface_batches:
                filename = os.path.basename(batch["audio_path"])
                batch["download_url"] = f"{base_url}/api/dreamface/audio/batches/{filename}"
                batch["avatar_url"]   = None

            job = {
                "user_id":    user_id,
                "vid_data_id": vid_data_id,
                "status":     "awaiting_dreamface",
                "title":      data.get("title", ""),
                "created_at": datetime.utcnow(),

                "create_video_payload": {
                    "phase_media":       phase_media,
                    "mode":              mode,
                    "avatar":            avatar,
                    "background_video":  background_video,
                    "background":        background,
                    "orientation":       orientation,
                    "storyboard":        storyboard,
                    "kenburns":          kenburns,
                    "avatar_active":     avatar_active,
                    "caption_settings":  caption_settings,
                    "captions_enabled":  captions_enabled,
                    "transition_config": transition_config,
                    "clip_data_list":    intro_section.get('clips', []),
                    "image_data_list":   intro_section.get('imageData', []),
                    "vid_segments":      intro_segments,
                    "characters":        intro_section.get('section_characters', []),
                    "narration_text":    intro_script,
                    "asset_folder":      asset_folder,
                    "section_type":      "intro",
                    "title":             data.get('title', ''),
                    "niche":             niche,
                    "intro_teaser":      intro_teaser_config,
                    "narration_images":  narration_images_config,
                    "transition_banner": transition_banner_config,
                },

                "audio_payload": {
                    "segment_audio_paths": intro_segment_audio_paths,
                    "full_wav_path":       full_intro_audio_path,
                    "srt_path":            full_intro_srt_path,
                },

                "batches": dreamface_batches,
            }

            result = db.dreamface_jobs.insert_one(job)
            job_id = str(result.inserted_id)

            logging.info(f"✅ DreamFace job saved: {job_id}")

            db.video_data.update_one(
                {"_id": ObjectId(vid_data_id)},
                {"$set": {
                    "status": "awaiting_dreamface",
                    "dreamface_job_id": job_id
                }}
            )

            batch_links = "\n".join([
                f"Batch {b['batch_index']} ({b['duration_sec']:.1f}s):\n{b['download_url']}"
                for b in dreamface_batches
            ])

            send_telegram_notification(
                f"🎬 DreamFace Audio Ready!\n\n"
                f"Job: {job_id}\n"
                f"Batches: {len(dreamface_batches)}\n\n"
                f"{batch_links}\n\n"
                f"Download, process in DreamFace, then submit avatar links.",
                "View Job",
                f"{FRONTEND_URL}/dreamface/{job_id}",
                user_id
            )

            logging.info("✅ Pipeline intercepted — awaiting DreamFace avatar links")
            return

        body_segment_audio_paths = None
        full_body_audio_path     = None
        full_body_srt_path       = None

        if has_body:
            logging.info("🔊 Generating narration (body)...")
            body_result = _generate_audio_fair(
                body_segments,
                voice,
                body_script,
            )
            body_segment_audio_paths = _extract_audio_paths(body_result)
            full_body_audio_path     = _extract_full_audio_path(body_result)
            full_body_srt_path       = body_result[5]
            if not body_segment_audio_paths:
                raise RuntimeError("❌ Body narration generation failed")

        logging.info("🎬 Generating intro video...")
        intro_path, intro_folder, _ = create_video(
            phase_media,
            full_audio=full_intro_audio_path,
            mode=mode,
            avatar=avatar,
            background_video=background_video,
            background=background,
            user_id=user_id,
            section_type='intro',
            orientation=orientation,
            storyboard=storyboard,
            kenburns=kenburns,
            srt=full_intro_srt_path,
            avatar_active=avatar_active,
            caption_settings=caption_settings,
            captions_enabled=captions_enabled,
            transition_config=transition_config,
            script=intro_script,
            clip_data_list=intro_section.get('clips', []),
            image_data_list=intro_section.get('imageData', []),
            segment_audio_paths=intro_segment_audio_paths,
            narration_text=intro_section.get('section_script', ''),
            characters=intro_section.get('section_characters', []),
            vid_segments=intro_segments,
            asset_folder=asset_folder,
            intro_teaser_config=intro_teaser_config,
            narration_images_config=narration_images_config,
            transition_banner_config=transition_banner_config,
        )

        body_path = body_folder = None
        if has_body:
            logging.info("🎬 Generating body video...")
            body_path, body_folder, _ = create_video(
                phase_media,
                full_audio=full_body_audio_path,
                mode=mode,
                avatar=avatar,
                background_video=background_video,
                background=background,
                user_id=user_id,
                section_type='body',
                orientation=orientation,
                storyboard=storyboard,
                kenburns=kenburns,
                srt=full_body_srt_path,
                avatar_active=avatar_active,
                caption_settings=caption_settings,
                captions_enabled=captions_enabled,
                transition_config=transition_config,
                script=body_script,
                clip_data_list=body_section.get('clips', []),
                image_data_list=body_section.get('imageData', []),
                segment_audio_paths=body_segment_audio_paths,
                narration_text=body_section.get('section_script', ''),
                characters=body_section.get('section_characters', []),
                vid_segments=body_segments,
                asset_folder=asset_folder,
                intro_teaser_config=intro_teaser_config,
                narration_images_config=narration_images_config,
                transition_banner_config=transition_banner_config,
            )

        if not intro_path or not os.path.exists(intro_path):
            raise RuntimeError("❌ Intro video generation failed")

        os.makedirs(OUTPUT, exist_ok=True)
        final_name = f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        final_path = sanitize_path(os.path.join(OUTPUT, final_name))

        logging.info("➡️ Using intro as final (stream copy)...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", intro_path, "-c", "copy", "-movflags", "+faststart", final_path],
            check=True,
        )

        if not os.path.exists(final_path):
            raise RuntimeError("❌ Final video was not created")

        logging.info("💾 Saving metadata to database...")
        combined_asset_folder = (
            intro_folder if not body_folder
            else f"{intro_folder}-{body_folder}"
        )
        result = db.completed_videos.insert_one({
            "title":        data.get('title', ''),
            "sections":     sections,
            "video_path":   final_name,
            "asset_folder": combined_asset_folder,
            "created_at":   datetime.utcnow(),
            "niche":        niche,
            "user_id":      user_id
        })

        logging.info("🧹 Cleaning intermediate files...")
        for p in [intro_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as ce:
                logging.warning(f"Could not delete {p}: {ce}")

        db.video_data.update_one(
            {"_id": ObjectId(vid_data_id)},
            {"$set": {"status": "completed"}}
        )

        send_telegram_notification(
            "🎉 Video Completed!",
            "View",
            f"{FRONTEND_URL}/completed/{str(result.inserted_id)}",
            user_id
        )
        logging.info(f"✅ Video processing complete for user {user_id}")

    except Exception as e:
        db.video_data.update_one(
            {"_id": ObjectId(vid_data_id)},
            {"$set": {"status": "draft"}}
        )
        logging.error(f"❌ Failed to process video: {e}")
        send_telegram_error(f"❌ Video processing failed: {e}", user_id)
    finally:
        gc.collect()

def resume_dreamface_pipeline(job_id, user_id):
    try:
        job = db.dreamface_jobs.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise ValueError(f"Job {job_id} not found")

        payload       = job["create_video_payload"]
        audio_payload = job["audio_payload"]
        batches       = job["batches"]
        vid_data_id   = job["vid_data_id"]
        layout_config = job.get("layout_config", None)

        intro_teaser_config     = payload.get("intro_teaser")
        narration_images_config = payload.get("narration_images")
        transition_banner_config = payload.get("transition_banner")

        logging.info(f"▶️  Resuming Talking Heads pipeline for job {job_id}")

        if not layout_config:
            raise ValueError(
                "layout_config missing from job — "
                "was it saved when avatar links were submitted?"
            )

        db.dreamface_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {
                "status": "processing",
                "processing_started_at": datetime.utcnow(),
            }}
        )

        avatar_segment_map = {}

        for batch in batches:
            batch_index   = batch["batch_index"]
            raw_url       = (batch.get("avatar_url") or "").strip()
            segments_data = batch.get("segments_data", [])

            logging.info(
                f"🎭 Batch {batch_index}: resolving avatar "
                f"(url={raw_url[:120]}{'…' if len(raw_url) > 120 else ''})"
            )

            local_path = _resolve_avatar_url(
                raw_url=raw_url,
                batch_index=batch_index,
                job_id=job_id,
            )

            if not local_path:
                raise RuntimeError(
                    f"❌ Could not obtain avatar video for batch {batch_index}. URL: {raw_url}"
                )

            for seg_data in segments_data:
                avatar_segment_map[seg_data["seg_index"]] = {
                    "path":         local_path,
                    "batch_index":  batch_index,
                    "avatar_start": seg_data["avatar_start"],
                    "duration":     seg_data["duration_sec"],
                }

        intro_path, intro_folder, _ = create_video(
            payload["phase_media"],
            full_audio          = audio_payload["full_wav_path"],
            mode                = payload["mode"],
            avatar              = payload["avatar"],
            background_video    = payload["background_video"],
            background          = payload["background"],
            user_id             = user_id,
            section_type        = payload["section_type"],
            orientation         = payload["orientation"],
            storyboard          = payload["storyboard"],
            kenburns            = payload["kenburns"],
            srt                 = audio_payload["srt_path"],
            avatar_active       = payload["avatar_active"],
            caption_settings    = payload["caption_settings"],
            captions_enabled    = payload["captions_enabled"],
            transition_config   = payload["transition_config"],
            script              = payload["narration_text"],
            clip_data_list      = payload["clip_data_list"],
            image_data_list     = payload["image_data_list"],
            segment_audio_paths = audio_payload["segment_audio_paths"],
            narration_text      = payload["narration_text"],
            characters          = payload["characters"],
            vid_segments        = payload["vid_segments"],
            asset_folder        = payload["asset_folder"],
            layout_config       = layout_config,
            avatar_segment_map  = avatar_segment_map,
            intro_teaser_config = intro_teaser_config,
            narration_images_config = narration_images_config,
            transition_banner_config = transition_banner_config,
        )

        if not intro_path or not os.path.exists(intro_path):
            raise RuntimeError("❌ create_video returned no output path")

        os.makedirs(OUTPUT, exist_ok=True)
        final_name = f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        final_path = sanitize_path(os.path.join(OUTPUT, final_name))

        subprocess.run(
            ["ffmpeg", "-y", "-i", intro_path, "-c", "copy", "-movflags", "+faststart", final_path],
            check=True,
        )

        if not os.path.exists(final_path):
            raise RuntimeError("❌ Final video was not created")

        result = db.completed_videos.insert_one({
            "title":        payload.get("title", ""),
            "video_path":   final_name,
            "asset_folder": intro_folder,
            "created_at":   datetime.utcnow(),
            "niche":        payload.get("niche", ""),
            "user_id":      user_id,
        })

        for p in [intro_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as ce:
                logging.warning(f"Could not delete {p}: {ce}")

        db.dreamface_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "completed"}}
        )
        db.video_data.update_one(
            {"_id": ObjectId(vid_data_id)},
            {"$set": {"status": "completed"}}
        )

        send_telegram_notification(
            "🎉 Talking Heads Video Completed!",
            "View",
            f"{FRONTEND_URL}/completed/{str(result.inserted_id)}",
            user_id,
        )
        logging.info(f"✅ Talking Heads pipeline complete — job {job_id}")

    except Exception as e:
        full_trace = traceback.format_exc()
        logging.error(f"❌ Talking Heads pipeline resume failed: {e}\n{full_trace}")

        db.dreamface_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "failed", "error": str(e)}}
        )

        send_telegram_error(
            f"❌ Talking Heads resume failed for job {job_id}\n\n"
            f"Error: {e}\n\n"
            f"Trace: {full_trace[:400]}",
            user_id,
        )
    finally:
        gc.collect()

def _resolve_avatar_url(raw_url: str, batch_index: int, job_id: str) -> str | None:
    import hashlib
    import re

    raw_url = (raw_url or "").strip()
    if not raw_url:
        logging.error(f"❌ Batch {batch_index}: empty avatar_url")
        return None

    AVATAR_DIR = os.getenv("AVATAR_DOWNLOAD_DIR", "static/avatars")
    os.makedirs(AVATAR_DIR, exist_ok=True)

    url_key = hashlib.sha256(raw_url.encode("utf-8")).hexdigest()[:12]

    if not raw_url.startswith("http"):
        logging.error(
            f"❌ Batch {batch_index}: avatar_url must be an http(s) DreamFace link "
            f"(got local path or invalid value: {raw_url!r})"
        )
        return None

    share_link_patterns = [
        r"dreamfaceapp\.com/creation",
        r"dreamfaceapp\.com/.+previewId",
    ]
    is_share_link = any(re.search(p, raw_url) for p in share_link_patterns)

    if is_share_link:
        try:
            from helper.avatar_extractor import extract_avatar_from_share_link
            local_path = extract_avatar_from_share_link(
                raw_url, output_dir=AVATAR_DIR, url_key=url_key,
            )
            if local_path and os.path.exists(local_path):
                logging.info(
                    f"✅ Batch {batch_index}: share link → {local_path} "
                    f"(url_key={url_key})"
                )
                return local_path
            return None
        except Exception as e:
            logging.error(f"❌ Batch {batch_index}: share link extraction failed: {e}")
            return None

    try:
        import requests as req
        filename  = f"avatar_batch{batch_index}_{url_key}.mp4"
        save_path = os.path.join(AVATAR_DIR, filename)

        if os.path.exists(save_path) and os.path.getsize(save_path) >= 100_000:
            logging.info(
                f"♻️ Batch {batch_index}: reusing cached direct download "
                f"{save_path} (url_key={url_key})"
            )
            from helper.dreamface_tag_strip import maybe_strip_dreamface_avatar
            return maybe_strip_dreamface_avatar(save_path)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.dreamfaceapp.com/",
        }

        response = req.get(raw_url, headers=headers, stream=True, timeout=180)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(save_path)
        if file_size < 100_000:
            os.remove(save_path)
            raise ValueError(f"Downloaded file too small: {file_size} bytes")

        logging.info(
            f"✅ Batch {batch_index}: direct download → {save_path} "
            f"({file_size / (1024 * 1024):.1f}MB, url_key={url_key})"
        )
        from helper.dreamface_tag_strip import maybe_strip_dreamface_avatar
        return maybe_strip_dreamface_avatar(save_path)

    except Exception as e:
        logging.error(f"❌ Batch {batch_index}: direct download failed: {e}")
        return None


def process_regenerated_video(user_id, data, asset_folder):
    try:
        logging.info(f"👤 Regenerating full video for user: {user_id}")
        avatar           = data.get('avatarImage')
        background_video = data.get('backgroundVideo')
        voice            = data.get("voice", "alloy")
        background       = data.get("backgroundMusic")
        niche            = data.get("niche", "original")
        phase_media      = data.get('mediaPlan')

        intro_teaser_config     = data.get("introTeaser")
        narration_images_config = data.get("narrationImages")
        transition_banner_config = data.get("transitionBanner")

        mode = "regenerate_commentary" if niche == "commentary" else "regenerate"

        sections       = (data or {}).get("sections", {}) or {}
        intro_segments = (sections.get("intro") or {}).get("segment", []) or []
        body_segments  = (sections.get("body")  or {}).get("segment", []) or []

        has_body = bool(body_segments)

        if not intro_segments:
            raise ValueError("❌ Missing intro segment data")

        logging.info("🔊 Generating narration (intro)...")
        intro_result = _generate_audio_fair(intro_segments, voice)
        intro_segment_audio_paths = _extract_audio_paths(intro_result)
        full_intro_audio_path     = _extract_full_audio_path(intro_result)

        if not intro_segment_audio_paths:
            raise RuntimeError("Intro narration generation failed.")

        body_segment_audio_paths = None
        full_body_audio_path     = None

        if has_body:
            logging.info("🔊 Generating narration (body)...")
            body_result = _generate_audio_fair(body_segments, voice)
            body_segment_audio_paths = _extract_audio_paths(body_result)
            full_body_audio_path     = _extract_full_audio_path(body_result)
            if not body_segment_audio_paths:
                raise RuntimeError("Body narration generation failed.")

        logging.info("🎬 Generating intro video...")
        intro_video_result = create_video(
            phase_media,
            full_audio=full_intro_audio_path,
            mode=mode,
            avatar=avatar,
            background_video=background_video,
            background=background,
            user_id=user_id,
            section_type="intro",
            segment_audio_paths=intro_segment_audio_paths,
            vid_segments=intro_segments,
            asset_folder=asset_folder,
            intro_teaser_config=intro_teaser_config,
            narration_images_config=narration_images_config,
            transition_banner_config=transition_banner_config,
        )

        logging.info(f"📽️ create_video returned for intro: {intro_video_result}")

        if intro_video_result:
            intro_video_path, intro_folder = intro_video_result[0], intro_video_result[1]
        else:
            intro_video_path, intro_folder = None, None
            logging.error("❌ create_video for intro returned None")

        body_video_path = body_folder = None
        if has_body:
            logging.info("🎬 Generating body video...")
            body_video_result = create_video(
                phase_media,
                full_audio=full_body_audio_path,
                mode=mode,
                avatar=avatar,
                background_video=background_video,
                background=background,
                user_id=user_id,
                section_type="body",
                segment_audio_paths=body_segment_audio_paths,
                vid_segments=body_segments,
                asset_folder=asset_folder,
                intro_teaser_config=intro_teaser_config,
                narration_images_config=narration_images_config,
                transition_banner_config=transition_banner_config,
            )

            logging.info(f"📽️ create_video returned for body: {body_video_result}")

            if body_video_result:
                body_video_path, body_folder = body_video_result[0], body_video_result[1]
            else:
                body_video_path, body_folder = None, None
                logging.error("❌ create_video for body returned None")

        os.makedirs(OUTPUT, exist_ok=True)
        final_name = f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        final_path = sanitize_path(os.path.join(OUTPUT, final_name))

        logging.info("🔧 Standardizing for finalization...")
        intro_std = intro_video_path.replace(".mp4", "_std.mp4")
        standardize_video_for_concat(intro_video_path, intro_std)

        body_std = None
        if has_body and body_video_path:
            body_std = body_video_path.replace(".mp4", "_std.mp4")
            standardize_video_for_concat(body_video_path, body_std)

        if has_body and body_std:
            logging.info("🧩 Concatenating intro + body...")
            safe_concatenate_videos([intro_std, body_std], final_path)
        else:
            logging.info("➡️ No body — using intro as final...")
            subprocess.run(
                ["ffmpeg", "-y", "-i", intro_std, "-c", "copy", final_path],
                check=True
            )

        if not os.path.exists(final_path):
            raise RuntimeError("Final video creation failed")

        combined_asset_folder = (
            intro_folder if not body_folder
            else f"{intro_folder}-{body_folder}"
        )
        result = db.completed_videos.insert_one({
            "title":        data.get('title', ''),
            "sections":     sections,
            "video_path":   final_name,
            "asset_folder": combined_asset_folder,
            "created_at":   datetime.utcnow(),
            "niche":        niche,
            "user_id":      user_id,
        })

        for path in filter(None, [intro_video_path, body_video_path, intro_std, body_std]):
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logging.info(f"🧼 Deleted: {path}")
            except Exception as e:
                logging.warning(f"Could not delete {path}: {e}")

        send_telegram_notification(
            "🎉 Video Regeneration Completed!",
            "View",
            f"{FRONTEND_URL}/completed/{str(result.inserted_id)}",
            user_id
        )
        logging.info(f"🎬 Full video job complete for user {user_id}.")

    except Exception as e:
        logging.error(f"❌ Failed to process Regenerated video: {e}")
        send_telegram_error(
            f"🎉 ❌ Video regeneration failed: {e}",
            user_id
        )
    finally:
        gc.collect()