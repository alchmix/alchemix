import os
import time
import logging
import subprocess
import tempfile
import gc
from datetime import datetime

from tests import create_video
from test2 import generate_audio_segments
from bson.objectid import ObjectId
from dotenv import load_dotenv
from mongodb import mongo_instance
from helper.telegram_notification_engine import send_telegram_notification, send_telegram_error

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL")
db = mongo_instance.get_db()
OUTPUT = "outputs"


def standardize_video_for_concat(input_path, output_path):
    """
    Standardize video encoding parameters for smooth concatenation.
    This ensures all videos have identical stream characteristics.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", "30",                    # Fixed frame rate
            "-g", "60",                    # GOP size (2 seconds at 30fps)
            "-keyint_min", "60",           # Minimum keyframe interval
            "-sc_threshold", "0",          # Disable scene change detection
            "-c:a", "aac",
            "-ar", "48000",                # Sample rate
            "-ac", "2",                    # Stereo
            "-b:a", "128k",                # Audio bitrate
            "-movflags", "+faststart",
            "-vsync", "cfr",               # Constant frame rate
            output_path
        ]
        subprocess.run(cmd, check=True)
        logging.info(f"✅ Standardized video: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to standardize video {input_path}: {e}")
        raise


def sanitize_path(path):
    """Ensure ffmpeg-compatible forward-slash paths."""
    return os.path.abspath(path).replace("\\", "/")


def safe_concatenate_videos(video_paths, output_path):
    """
    Safely concatenate videos with proper error handling and fallback.
    """
    try:
        # Create temporary concat file
        concat_file = os.path.join(tempfile.gettempdir(), f"concat_{time.time()}.txt")

        with open(concat_file, 'w', encoding='utf-8') as f:
            for video_path in video_paths:
                f.write(f"file '{sanitize_path(video_path)}'\n")

        # First attempt: Direct concatenation
        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-protocol_whitelist", "file,pipe",
                "-i", concat_file,
                "-c", "copy",              # Copy both video and audio
                "-movflags", "+faststart",
                output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            logging.info("✅ Direct concatenation successful")

        except subprocess.CalledProcessError:
            logging.warning("❗ Direct copy failed. Re-encoding with standardized parameters...")

            # Fallback: Re-encode on concat
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
            # Clean up concat file
            if os.path.exists(concat_file):
                os.remove(concat_file)

    except Exception as e:
        logging.error(f"❌ Concatenation failed: {e}")
        raise


def _extract_audio_paths(result):
    """
    Accepts either:
      - [segment_paths, narration_path, narration_url, timestamps]
      - or just [segment_paths]
      - or tuple variants
    Returns segment_paths or None.
    """
    if not result:
        return None
    if isinstance(result, (list, tuple)):
        # First item should be the list of per-segment audio paths
        return result[0]
    # Fallback: if the function ever returns just the list
    return result

def _extract_full_audio_path(result):
    """
    Accepts either:
      - [segment_paths, narration_path, narration_url, timestamps]
      - or just [segment_paths]
      - or tuple variants
    Returns segment_paths or None.
    """
    if not result:
        return None
    if isinstance(result, (list, tuple)):
        # First item should be the list of per-segment audio paths
        return result[1]
    # Fallback: if the function ever returns just the list
    return result

def process_full_video(user_id, data, vid_data_id, asset_folder=None):
    try:
        logging.info(f"👤 Processing full video for user: {user_id}")
        db.video_data.update_one({"_id": ObjectId(vid_data_id)}, {"$set": {"status": "processing"}})

        # ------- inputs & flags -------
        sections = (data or {}).get('data', {}) or {}
        intro_section = (sections.get('intro') or {}) or {}
        body_section  = (sections.get('body')  or {}) or {}

        intro_segments = intro_section.get('segment', []) or []
        body_segments  = body_section.get('segment', []) or []

        intro_script =intro_section.get("section_script","")
        body_script =body_section.get("section_script","")

        has_body = (
    body_segments is not None 
    and isinstance(body_segments, list) 
    and any(seg.get("text") for seg in body_segments if isinstance(seg, dict))
)
  # only true if list has content

        avatar           = data.get('avatarImage')
        background_video = data.get('backgroundVideo')
        voice            = data.get("voice", "alloy")
        background       = data.get("backgroundMusic")
        bg_music         = data.get("bgMusic")
        niche            = data.get('niche', 'original')
        phase_media      = data.get('mediaPlan')
        kenburns         = data.get('kenBurns')
        storyboard       = data.get('storyboardAnimation')
        orientation      = data.get('orientation')
        avatar_active    = data.get('avatar')
        caption_settings = data.get('captionSettings')
        captions_enabled = data.get('captionsEnabled')
        transition_config = data.get('transition_config')



        if bg_music:
            background=bg_music

        if not orientation:
           orientation='landscape'

        # Mode selection
        mode = "create"
        if niche == "commentary":
            mode = "commentary"
        elif niche == "original":
            mode = "original"

        if not intro_segments:
            raise ValueError("❌ Missing intro segment data")

        # ------- narration -------
        logging.info("🔊 Generating narration (intro)...")
        intro_result = generate_audio_segments(intro_segments, voice,intro_script)
        intro_segment_audio_paths = _extract_audio_paths(intro_result)
        full_intro_audio_path=_extract_full_audio_path(intro_result)
        full_intro_srt_path=intro_result[5]
        print("INTRO SRT PATH", full_intro_srt_path)
        full_intro_nested_chunks_path=intro_result[4]
        if not intro_segment_audio_paths:
            raise RuntimeError("❌ Intro narration generation failed")

        body_segment_audio_paths = None
        if has_body:
            logging.info("🔊 Generating narration (body)...")
            body_result = generate_audio_segments(body_segments, voice)
            body_segment_audio_paths = _extract_audio_paths(body_result)
            full_body_audio_path=_extract_full_audio_path(body_result)
            full_body_srt_path=body_result[5]
            full_body_nested_chunks_path=body_result[4]
            if not body_segment_audio_paths:
                raise RuntimeError("❌ Body narration generation failed")

        # ------- render videos -------
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
            orientation=orientation,     # NEW
            storyboard=storyboard,            # NEW
            kenburns=kenburns ,               # NEW
            srt=full_intro_srt_path,
            avatar_active=avatar_active,
            caption_settings=caption_settings,
            captions_enabled=captions_enabled,
            transition_config=transition_config,
            script =intro_section.get("section_script",""),
            clip_data_list=intro_section.get('clips', []),
            image_data_list=intro_section.get('imageData', []),
            segment_audio_paths=intro_segment_audio_paths,
            narration_text=intro_section.get('section_script', ''),
            characters=intro_section.get('section_characters', []),
            vid_segments=intro_segments,
            asset_folder=asset_folder
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
                orientation=orientation,     # NEW
                storyboard=storyboard,            # NEW
                kenburns=kenburns ,               # NEW
                srt=full_intro_srt_path,
                avatar_active=avatar_active,
                caption_settings=caption_settings,
                captions_enabled=captions_enabled,
                script =body_section.get("section_script",""),
                clip_data_list=body_section.get('clips', []),
                image_data_list=body_section.get('imageData', []),
                segment_audio_paths=body_segment_audio_paths,
                narration_text=body_section.get('section_script', ''),
                characters=body_section.get('section_characters', []),
                vid_segments=body_segments,
                asset_folder=asset_folder
            )

        # ------- finalize (concat if body exists) -------
        os.makedirs(OUTPUT, exist_ok=True)
        final_name = f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        final_path = sanitize_path(os.path.join(OUTPUT, final_name))

        logging.info("🔧 Standardizing for finalization...")
        intro_std = intro_path.replace(".mp4", "_std.mp4")
        standardize_video_for_concat(intro_path, intro_std)

        body_std = None
        if has_body and body_path:
            body_std = body_path.replace(".mp4", "_std.mp4")
            standardize_video_for_concat(body_path, body_std)

        if has_body and body_std:
            logging.info("🧩 Concatenating intro + body...")
            safe_concatenate_videos([intro_std, body_std], final_path)
        else:
            logging.info("➡️ No body provided — using intro as final...")
            # Copy standardized intro to final container without re-encode
            subprocess.run(["ffmpeg", "-y", "-i", intro_std, "-c", "copy", final_path], check=True)

        if not os.path.exists(final_path):
            raise RuntimeError("❌ Final video was not created")

        # ------- DB persist -------
        logging.info("💾 Saving metadata to database...")
        combined_asset_folder = intro_folder if not body_folder else f"{intro_folder}-{body_folder}"
        result = db.completed_videos.insert_one({
            "title": data.get('title', ''),
            "sections": sections,
            "video_path": final_name,
            "asset_folder": combined_asset_folder,
            "created_at": datetime.utcnow(),
            "niche": niche,
            "user_id": user_id
        })

        # ------- cleanup -------
        logging.info("🧹 Cleaning intermediate files...")
        for p in filter(None, [intro_path, intro_std, body_path, body_std]):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as ce:
                logging.warning(f"Could not delete {p}: {ce}")

        db.video_data.update_one({"_id": ObjectId(vid_data_id)}, {"$set": {"status": "completed"}})

        send_telegram_notification(
            "🎉 Video Completed!",
            "View",
            f"{FRONTEND_URL}/completed/{str(result.inserted_id)}",
            user_id
        )
        logging.info(f"✅ Video processing complete for user {user_id}")

    except Exception as e:
        db.video_data.update_one({"_id": ObjectId(vid_data_id)}, {"$set": {"status": "draft"}})
        logging.error(f"❌ Failed to process video: {e}")
        send_telegram_error(f"❌ Video processing failed: {e}", user_id)
    finally:
        gc.collect()


def process_regenerated_video(user_id, data, asset_folder):
    try:
        logging.info(f"👤 Regenerating full video for user: {user_id}")
        avatar           = data.get('avatarImage')
        background_video = data.get('backgroundVideo')
        voice            = data.get("voice", "alloy")
        background       = data.get("backgroundMusic")
        niche            = data.get("niche", "original")
        phase_media      = data.get('mediaPlan')

        mode = "regenerate_commentary" if niche == "commentary" else "regenerate"

        sections = (data or {}).get("sections", {}) or {}
        intro_segments = (sections.get("intro") or {}).get("segment", []) or []
        body_segments  = (sections.get("body")  or {}).get("segment", []) or []

        has_body = bool(body_segments)

        if not intro_segments:
            raise ValueError("❌ Missing intro segment data")

        # ------- narration -------
        logging.info("🔊 Generating narration (intro)...")
        intro_result = generate_audio_segments(intro_segments, voice)
        intro_segment_audio_paths = _extract_audio_paths(intro_result)
        full_intro_audio_path=_extract_full_audio_path(intro_result)
        
        if not intro_segment_audio_paths:
            raise RuntimeError("Intro narration generation failed.")

        body_segment_audio_paths = None
        if has_body:
            logging.info("🔊 Generating narration (body)...")
            body_result = generate_audio_segments(body_segments, voice)
            body_segment_audio_paths = _extract_audio_paths(body_result)
            full_body_audio_path=_extract_full_audio_path(body_result)
            if not body_segment_audio_paths:
                raise RuntimeError("Body narration generation failed.")

        # ------- render videos -------
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
            asset_folder=asset_folder
        )
        
        # Log the returned value from create_video for intro
        logging.info(f"📽️ create_video returned for intro: {intro_video_result}")

        # Safely unpack with fallback values
        if intro_video_result:
            intro_video_path, intro_folder = intro_video_result[0], intro_video_result[1]
        else:
            intro_video_path, intro_folder = None, None
            logging.error("❌ create_video for intro returned an unexpected result (None or incomplete).")

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
                asset_folder=asset_folder
            )
            
            # Log the returned value from create_video for body
            logging.info(f"📽️ create_video returned for body: {body_video_result}")

            # Safely unpack with fallback values
            if body_video_result:
                body_video_path, body_folder = body_video_result[0], body_video_result[1]
            else:
                body_video_path, body_folder = None, None
                logging.error("❌ create_video for body returned an unexpected result (None or incomplete).")

        # ------- finalize -------
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
            logging.info("➡️ No body provided — using intro as final...")
            # Copy standardized intro to final container without re-encode
            subprocess.run(["ffmpeg", "-y", "-i", intro_std, "-c", "copy", final_path], check=True)

        if not os.path.exists(final_path):
            raise RuntimeError("Final video creation failed")

        # ------- DB -------
        combined_asset_folder = intro_folder if not body_folder else f"{intro_folder}-{body_folder}"
        result = db.completed_videos.insert_one({
            "title": data.get('title', ''),
            "sections": sections,
            "video_path": final_name,
            "asset_folder": combined_asset_folder,
            "created_at": datetime.utcnow(),
            "niche": niche,
            "user_id": user_id,
        })

        # ------- cleanup -------
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
