import yt_dlp
import requests
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips, AudioFileClip
from tempfile import TemporaryDirectory
from PIL import Image
import torch
import numpy as np
import cv2
import openai
from dotenv import load_dotenv
from functools import lru_cache
from video_helper.segment_narration import segment_narration

# Import YOLO
from ultralytics import YOLO

# Import CLIP model and processor
from transformers import CLIPProcessor, CLIPModel

# ----------------- Configuration -----------------
DEFAULT_DURATION = 5  # Default duration for image clips
MATCH_THRESHOLD = 0.2  # Threshold for media matching
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_WORKERS = 8  # For thread pool

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO
)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ----------------- Model Initialization -----------------
logging.info("Initializing models...")
# Initialize CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Initialize YOLO
yolo_model = YOLO("yolov8n.pt")

# ----------------- Utility Functions -----------------

def time_to_seconds(timestamp):
    """Convert timestamp string (HH:MM:SS) to seconds."""
    try:
        h, m, s = map(float, timestamp.split(":"))
        return h * 3600 + m * 60 + s
    except Exception as e:
        logging.error(f"Invalid timestamp format {timestamp}: {e}")
        return 0

def progress_hook(d):
    """Progress hook for yt-dlp downloads."""
    if d['status'] == 'downloading':
        logging.info(f"📥 Downloading: {d.get('_percent_str', '')} | Speed: {d.get('_speed_str', '')} | ETA: {d.get('_eta_str', '')}")

def download_video_segment(url, output_path, start_time, end_time):
    """Download a specific segment of a YouTube video."""
    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'quiet': True,
        'progress_hooks': [progress_hook],  # Ensure progress_hook function is defined
        'download_ranges': lambda _, __: [{'start_time': start_time, 'end_time': end_time}],
        'force_keyframes_at_cuts': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        logging.error(f"Error downloading video segment from {url}: {e}")
        return None
    
def download_image(img_url, img_path):
    """Download an image from a URL."""
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        with open(img_path, 'wb') as f:
            f.write(response.content)
        return img_path
    except Exception as e:
        logging.error(f"Error downloading image {img_url}: {e}")
        return None

# ----------------- AI Processing Functions -----------------

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

@lru_cache(maxsize=128)
def encode_text(text):
    """Encode text using CLIP."""
    try:
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        return text_features[0].cpu().numpy()
    except Exception as e:
        logging.error(f"Error encoding text: {e}")
        return np.zeros((512,))  # Fallback vector

def encode_image_from_pil(image: Image.Image):
    """Encode an image using CLIP from a PIL image."""
    try:
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features[0].cpu().numpy()
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return np.zeros((512,))  # Fallback vector

def extract_representative_frame(video_clip):
    """Extract a frame from the middle of the clip."""
    try:
        duration = video_clip.duration
        frame_time = duration / 2.0
        frame = video_clip.get_frame(frame_time)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return pil_image
    except Exception as e:
        logging.error(f"Error extracting representative frame: {e}")
        return None

def run_yolo_on_image(pil_image):
    """Run YOLO on a PIL image and return detected labels."""
    try:
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        results = yolo_model(cv_image)
        labels = set()
        for result in results:
            for detection in result.boxes.cls.cpu().numpy():
                label = yolo_model.names[int(detection)]
                labels.add(label)
        return "scene contains: " + ", ".join(labels) if labels else "no significant objects detected"
    except Exception as e:
        logging.error(f"Error running YOLO on image: {e}")
        return "error detecting scene"

def evaluate_media(pil_image, narration_text):
    """Evaluate how well an image matches the narration."""
    try:
        image_embedding = encode_image_from_pil(pil_image)
        text_embedding = encode_text(narration_text)
        score_image = cosine_similarity(image_embedding, text_embedding)

        scene_description = run_yolo_on_image(pil_image)
        scene_embedding = encode_text(scene_description)
        score_scene = cosine_similarity(scene_embedding, text_embedding)

        return (score_image + score_scene) / 2.0
    except Exception as e:
        logging.error(f"Error evaluating media: {e}")
        return 0.0

# # ----------------- Narration Processing -----------------

# def segment_narration(narration_text,characters):
#     """Split narration into logical segments using GPT-4."""
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "Split this narration into logical visual segments. Return only the segments separated by '|||' with no additional text."},
#                 {"role": "user", "content": narration_text}
#             ]
#         )
#         segments = response.choices[0].message.content.split("|||")
#         return [s.strip() for s in segments if s.strip()]
#     except Exception as e:
#         logging.error(f"Error segmenting narration: {e}")
#         # Fallback: split by sentences or fixed intervals
#         return [narration_text]


def time_segments(audio_path, segments):
    """Estimate time boundaries for each segment based on total audio duration."""
    try:
        if not isinstance(audio_path, str):
            raise ValueError("audio_path must be a valid file path (str), but received: " + str(type(audio_path)))

        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        num_segments = len(segments)

        if num_segments == 0:
            raise ValueError("No segments found for timing calculation.")

        segment_duration = total_duration / num_segments
        return [(i * segment_duration, (i + 1) * segment_duration) for i in range(num_segments)]

    except Exception as e:
        logging.error(f"Error calculating time segments: {e}")
        return []

# ----------------- Media Selection -----------------

def find_best_media_combination(media_pool, text_segment, character_name, duration_needed):
    """
    Select the best media (videos or images) for a character during narration.
    Ensures media appears while narration is playing and rotates character images.
    
    Args:
        media_pool (list): Available images and videos.
        text_segment (str): The narration segment.
        character_name (str): The character associated with the segment.
        duration_needed (float): Duration required for the segment.
    
    Returns:
        list: Selected media items with durations and attributes.
    """
    text_embedding = encode_text(text_segment)
    selected_media = []
    remaining_duration = duration_needed
    used_media = set()

    # Filter media pool for only the character's media
    relevant_media = [
        media for media in media_pool if media['character'] == character_name and not media['used']
    ]

    # Rank media by similarity to text (ensuring relevance)
    media_candidates = []
    for media in relevant_media:
        score = cosine_similarity(text_embedding, media['embedding'])
        media_candidates.append((media, score))

    # Sort media by highest similarity
    media_candidates.sort(key=lambda x: x[1], reverse=True)

    # Track used images to ensure rotation
    used_images = set()

    # Select media
    for media, score in media_candidates:
        if remaining_duration <= 0:
            break  # Stop when segment duration is filled

        if media['type'] == 'video':
            # Play full video or trim to fit the narration segment
            duration = min(media['duration'], remaining_duration)
            selected_media.append({
                'media': media,
                'duration': duration,
                'score': score,
                'has_audio': not media.get('mute', False)
            })
            remaining_duration -= duration
            used_media.add(id(media))

        elif media['type'] == 'image' and media['path'] not in used_images:
            # Show each image for 4 seconds, then rotate to the next one
            image_duration = min(4, remaining_duration)
            selected_media.append({
                'media': media,
                'duration': image_duration,
                'score': score,
                'has_audio': False
            })
            remaining_duration -= image_duration
            used_media.add(id(media))
            used_images.add(media['path'])  # Ensure a different image is used next

    return selected_media

# ----------------- Video Creation -----------------
def create_video(clip_data_list, image_data_list, voice_over_path, narration_text, characters, output_dir="outputs"):
    """Create a video by aligning media with narration segments."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")

    with TemporaryDirectory() as temp_dir:
        # Segment narration and prepare audio
        logging.info("🔍 Analyzing narration structure...")
        segments = segment_narration(narration_text, characters)
        audio = AudioFileClip(voice_over_path)
        time_ranges = time_segments(voice_over_path, segments)

        # Prepare media pool
        media_pool = []

        # Process video clips
        logging.info("📦 Preparing video clips...")
        

    for clip_idx, clip_data in enumerate(clip_data_list):
        character_name = clip_data.get("characterName", "Unknown")

      # Ensure "videos" key exists before proceeding
        videos = clip_data.get("videos", [])
        if not videos:
           logging.warning(f"⚠️ No videos found for {character_name}")
           continue

        for video_data in videos:  # Loop through multiple videos per character
            yt_url = video_data.get("url")
            if not yt_url:
               logging.error(f"⚠️ Skipping video entry with missing URL: {video_data}")
               continue

            vidclips = video_data.get("vidclips", [])
            if not vidclips:
               logging.warning(f"⚠️ No vidclips found for video: {yt_url}")
               continue
  
            for vidclip_idx, vidclip in enumerate(vidclips):
                start = time_to_seconds(vidclip.get("startTime", "0:00:00"))
                end = time_to_seconds(vidclip.get("endTime", "0:00:05"))
                clip_filename = f"clip_{clip_idx}_{vidclip_idx}.mp4"
                clip_path = os.path.join(temp_dir, clip_filename)

                downloaded_path = download_video_segment(yt_url, clip_path, start, end)
                if not downloaded_path:
                    continue

                try:
                    clip = VideoFileClip(downloaded_path)
                except Exception as e:
                    logging.error(f"Error opening video clip {downloaded_path}: {e}")
                    continue

                mute = vidclip.get("mute", False)
                if mute:
                    clip = clip.without_audio()

                rep_frame = extract_representative_frame(clip)
                if rep_frame is None:
                    clip.close()
                    continue

                media_pool.append({
                    "type": "video",
                    "clip": clip,
                    "character": character_name,
                    "duration": clip.duration,
                    "embedding": encode_image_from_pil(rep_frame),
                    "used": False,
                    "mute": mute
                 })

        # Process images with ThreadPoolExecutor for speed
        logging.info("🖼️ Preparing images...")
        image_paths = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_character = {}

            for image_data in image_data_list:
                character_name = image_data["characterName"]
                character_folder = os.path.join(temp_dir, character_name.replace(" ", "_"))
                os.makedirs(character_folder, exist_ok=True)

                for img_url in image_data["images"]:
                    img_filename = os.path.basename(img_url)
                    img_path = os.path.join(character_folder, img_filename)

                    future = executor.submit(download_image, img_url, img_path)
                    future_to_character[future] = (character_name, img_path)

            # Process completed downloads
            for future in tqdm(as_completed(future_to_character), total=len(future_to_character), desc="Downloading Images"):
                character, img_path = future_to_character[future]
                try:
                    result = future.result()
                    if result:
                        if character not in image_paths:
                            image_paths[character] = []
                        image_paths[character].append(result)
                except Exception as e:
                    logging.error(f"⚠️ Error downloading image for {character}: {e}")

        # Store processed images in media pool
        for character, img_list in image_paths.items():
            for img_path in img_list:
                try:
                    img = Image.open(img_path).convert("RGB")
                    media_pool.append({
                        "type": "image",
                        "path": img_path,
                        "character": character,
                        "embedding": encode_image_from_pil(img),
                        "used": False
                    })
                except Exception as e:
                    logging.error(f"⚠️ Error processing image {img_path}: {e}")

        # Build enhanced timeline
        logging.info("🎞️ Building timeline...")
        final_clips = []

        for seg_idx, (segment, (start_time, end_time)) in enumerate(zip(segments, time_ranges)):
            duration_needed = end_time - start_time
            character_name = segment["character"]

            selected_media = find_best_media_combination(media_pool, segment["text"], character_name, duration_needed)
            if not selected_media:
                logging.warning(f"⚠️ No suitable media found for segment {seg_idx + 1} ({character_name})")
                continue

            segment_clips = []
            try:
                audio_segment = audio.subclipped(start_time, end_time)
            except Exception as e:
                logging.error(f"Error subclipping audio from {start_time} to {end_time}: {e}")
                continue

            for media_item in selected_media:
                media = media_item["media"]
                try:
                    if media_item["has_audio"]:
                        if segment_clips:
                            combined = concatenate_videoclips(segment_clips, method="compose")
                            combined = combined.with_audio(audio_segment)
                            final_clips.append(combined)
                            segment_clips = []
                        final_clips.append(media["clip"])
                    else:
                        if media["type"] == "video":
                            clip = media["clip"].subclipped(0, media_item["duration"])
                            if clip.duration > 0:
                                segment_clips.append(clip)
                        else:
                            img_clip = ImageClip(media["path"]).with_duration(media_item["duration"])
                            if segment_clips:
                                img_clip = img_clip.resized(width=segment_clips[0].w, height=segment_clips[0].h)
                            segment_clips.append(img_clip)
                except Exception as e:
                    logging.error(f"Error processing media item: {e}")

            if segment_clips:
                try:
                    combined = concatenate_videoclips(segment_clips, method="compose")
                    combined = combined.with_audio(audio_segment)
                    final_clips.append(combined)
                except Exception as e:
                    logging.error(f"Error concatenating clips for segment {seg_idx + 1}: {e}")

        # Final assembly
        if not final_clips:
            raise ValueError("❌ No valid media found for any segments!")

        try:
            final_video = concatenate_videoclips(final_clips, method="compose")
        except Exception as e:
            logging.error(f"Error concatenating final clips: {e}")
            return None

        logging.info("📤 Exporting final video...")
        try:
            final_video.write_videofile(
                output_path,
                codec="libx264",
                fps=24,
                threads=4,
                preset="ultrafast",
                audio_codec="aac"
            )
        except Exception as e:
            logging.error(f"Error writing final video to file: {e}")
            return None
        finally:
            final_video.close()
            audio.close()
            for media in media_pool:
                if media["type"] == "video":
                    media["clip"].close()

    logging.info(f"✅ Video saved: {output_path}")
    return output_path







# # ----------------- Example Data -----------------

clip_data_list = [
    {
      "characterName": "Larry David",
      "videos": [
        {
          "url": "https://www.youtube.com/watch?v=AvwEzqXL5aI",
          "vidclips": [
            { "startTime": "00:00:20.44", "endTime": "00:00:23.19", "mute": True },
            { "startTime": "00:00:44.59", "endTime": "00:00:53.11", "mute": True }
          ]
        },
        {
          "url": "https://www.youtube.com/watch?v=Gz0hOMvMDaY",
          "vidclips": [
            { "startTime": "00:00:17.59", "endTime": "00:00:21.51", "mute": True },
            { "startTime": "00:00:23.80", "endTime": "00:00:28.67", "mute": True }
          ]
        },
        {
          "url": "https://www.youtube.com/watch?v=FxiTzMeDCDM",
          "vidclips": [
            { "startTime": "00:00:11.37", "endTime": "00:00:19.22", "mute": True }
          ]
        }
      ]
    },
    {
      "characterName": "Dick Ebersol",
      "videos": [
        {
          "url": "https://www.youtube.com/watch?v=KNeNtWniPns",
          "vidclips": [
            { "startTime": "00:00:11.79", "endTime": "00:00:15.68", "mute": True }
          ]
        }
      ]
    },
    {
      "characterName": "Julia Louis-Dreyfus",
      "videos": [
        {
          "url": "https://www.youtube.com/watch?v=J5byMntW-w4",
          "vidclips": [
            { "startTime": "00:01:37.50", "endTime": "00:01:43.39", "mute": True },
            { "startTime": "00:02:28.71", "endTime": "00:02:33.44", "mute": True }
          ]
        },
        {
          "url": "https://www.youtube.com/watch?v=1hXpyyt5tWU",
          "vidclips": [
            { "startTime": "00:00:36.13", "endTime": "00:00:45.58", "mute": True },
            { "startTime": "00:01:07.38", "endTime": "00:01:17.04", "mute": True }
          ]
        },
        {
          "url": "https://www.youtube.com/watch?v=Nfe2MoPDdkM",
          "vidclips": [
            { "startTime": "00:00:04.56", "endTime": "00:00:09.12", "mute": True },
            { "startTime": "00:00:11.39", "endTime": "00:00:20.90", "mute": True },
            { "startTime": "00:00:27.99", "endTime": "00:00:42.02", "mute": True }
          ]
        }
      ]
    }

]

image_data_list = [
    {
        "characterName": "Larry David",
        "images":[
  "https://live.staticflickr.com/2694/4361196470_c730c9c033_b.jpg",
  "https://live.staticflickr.com/4067/4361196246_2c2ba6b697_b.jpg",
  "https://live.staticflickr.com/3767/12582244745_a6b58586ff_o.jpg",
  "https://live.staticflickr.com/5193/5828930859_fe83eb264a_o.jpg",
  "https://live.staticflickr.com/91/281158848_745c3b1d75_o.jpg",
  "https://live.staticflickr.com/2603/3935516131_175f2571c3_b.jpg",
  "https://live.staticflickr.com/4068/4433865581_d30d348a43_o.jpg",
  "https://live.staticflickr.com/3063/2491711563_a8a9400c80_b.jpg",
  "https://live.staticflickr.com/8324/8122158132_4d9cfd8d60_o.jpg",
  "https://live.staticflickr.com/3255/2797713079_0630eb7aff_o.jpg",
  "https://live.staticflickr.com/1725/42784135891_ac1c5f4f6c_o.jpg",
  "https://live.staticflickr.com/173/438016270_887ff19f71_o.jpg",
  "https://live.staticflickr.com/2902/14269382433_29968425fe_o.jpg",
  "https://live.staticflickr.com/8120/8694129895_4f2c3e4971_o.jpg",
  "https://live.staticflickr.com/6111/7002070499_3711f700b2_o.jpg",
  "https://live.staticflickr.com/2635/3834963572_b20b97990c_o.jpg",
  "https://live.staticflickr.com/3270/2644633889_4e6faaba21_o.jpg",
  "https://live.staticflickr.com/8471/8124828915_b687dea66b_o.jpg",
  "https://live.staticflickr.com/3023/2347646053_db2a096291_o.jpg",
  "https://live.staticflickr.com/6016/6010748608_3bd6e89b24_o.jpg",
  "https://live.staticflickr.com/112/303272148_2ee775e347_o.jpg",
  "https://live.staticflickr.com/2346/2259326563_091a58d08b_o.jpg",
  "https://live.staticflickr.com/4080/4771758456_432937b191_o.jpg",
  "https://live.staticflickr.com/133/347326484_d8834ad0fb_o.jpg",
  "https://live.staticflickr.com/2372/2473331259_5069b7aa43_b.jpg",
  "https://live.staticflickr.com/8471/8117060992_8b9cbaa1df_o.jpg",
  "https://live.staticflickr.com/1082/823271136_2d49dea77d_b.jpg",
  "https://farm1.staticflickr.com/7/8805975_2db729dcff_b.jpg",
  "https://live.staticflickr.com/6001/6010749552_d3be3f53fc_o.jpg",
  "https://live.staticflickr.com/7166/6665858997_0e132aa3f9_o.jpg"
]

    },
    {
        "characterName": "Dick Ebersol",
        "images":[
  "https://live.staticflickr.com/3388/3339982810_33f4768b2d_o.jpg",
  "https://live.staticflickr.com/2760/4516722164_4d312ddeb5_b.jpg",
  "https://live.staticflickr.com/3099/5738939282_007902c94e_o.jpg",
  "https://live.staticflickr.com/2734/4365476520_3e1b18368e_o.jpg",
  "https://live.staticflickr.com/8345/8182134752_bbb9ff2b2e_o.jpg",
  "https://live.staticflickr.com/8347/8182138810_95a42c5f25_o.jpg",
  "https://live.staticflickr.com/2834/33192817320_09c854d2eb_o.jpg",
  "https://live.staticflickr.com/8198/8182138124_48cf081efe_o.jpg",
  "https://live.staticflickr.com/8348/8182136734_0405be3bd7_o.jpg",
  "https://live.staticflickr.com/8206/8182136062_62ce5911bb_o.jpg",
  "https://live.staticflickr.com/8490/8182133348_a8cf7e185a_o.jpg",
  "https://live.staticflickr.com/8345/8182131668_593fd94104_o.jpg",
  "https://live.staticflickr.com/8206/8182102467_bd81521f5c_o.jpg",
  "https://live.staticflickr.com/8349/8182132446_0a3772cc3a_o.jpg",
  "https://live.staticflickr.com/8488/8182096065_2a55c3346f_o.jpg",
  "https://live.staticflickr.com/8482/8182099383_f029c088c9_o.jpg",
  "https://live.staticflickr.com/7115/8166863906_2a5725bd7e_o.jpg",
  "https://live.staticflickr.com/8486/8182135416_be473a2b71_o.jpg",
  "https://live.staticflickr.com/8347/8182092971_5f8a3d6d94_o.jpg",
  "https://live.staticflickr.com/7250/8166833821_bfa1d15704_o.jpg",
  "https://live.staticflickr.com/7116/8166863296_7b35ed57ff_o.jpg",
  "https://live.staticflickr.com/4046/4372745188_11b9a61037_b.jpg",
  "https://live.staticflickr.com/7248/8166932203_45efff2ec6_o.jpg",
  "https://live.staticflickr.com/5260/5388193127_6369ba7d66_o.jpg",
  "https://live.staticflickr.com/5140/5388800476_6d4cb54a00_o.jpg",
  "https://live.staticflickr.com/8341/8166944750_c788a04664_o.jpg",
  "https://live.staticflickr.com/7116/8166912375_465739d482_o.jpg",
  "https://live.staticflickr.com/8207/8182112043_15e7124b2f_o.jpg",
  "https://live.staticflickr.com/7123/8166920815_233de0cb53_o.jpg",
  "https://live.staticflickr.com/7276/8166937103_d29b8bba36_o.jpg"
]

    },
    {
        "characterName": "Julia Louis-Dreyfus",
        "images": [
  "https://live.staticflickr.com/120/251547225_a78693e6f5_o.jpg",
  "https://live.staticflickr.com/110/251549691_e90414bdd5_o.jpg",
  "https://live.staticflickr.com/104/252688593_a9704854ae_o.jpg",
  "https://live.staticflickr.com/4201/34211093323_a4de5b6d44_o.jpg",
  "https://live.staticflickr.com/4202/34211109103_5296edbe78_o.jpg",
  "https://live.staticflickr.com/4272/35021742515_999bc3ff75_o.jpg",
  "https://live.staticflickr.com/4219/34211103603_88b20e76bd_o.jpg",
  "https://live.staticflickr.com/4244/34889723921_a2ea280ccb_o.jpg",
  "https://live.staticflickr.com/4201/34889742551_1c025d0303_o.jpg",
  "https://live.staticflickr.com/4275/34889733481_27bb38ce0d_o.jpg",
  "https://live.staticflickr.com/4198/34889746851_2974af513b_o.jpg",
  "https://live.staticflickr.com/4196/35021725805_41a6143759_o.jpg",
  "https://live.staticflickr.com/4225/35021795305_6fb8490507_o.jpg",
  "https://live.staticflickr.com/4227/34889719841_f243f30b16_o.jpg",
  "https://live.staticflickr.com/4252/34211095953_ec707b6e94_o.jpg",
  "https://live.staticflickr.com/4228/34634539560_a3326024e6_o.jpg",
  "https://live.staticflickr.com/4274/34211044773_685a9f4f02_o.jpg",
  "https://live.staticflickr.com/4247/34211106993_7c89b50392_o.jpg",
  "https://live.staticflickr.com/4268/34178086664_61306caec1_o.jpg",
  "https://live.staticflickr.com/4204/35021728305_a3472c0a52_o.jpg",
  "https://live.staticflickr.com/4247/34211048523_c5a5f848e9_o.jpg",
  "https://live.staticflickr.com/4224/34981800956_c0f5d48d5b_o.jpg",
  "https://live.staticflickr.com/4248/34889727621_6169fc2c00_o.jpg",
  "https://live.staticflickr.com/4248/34178261844_f72ffeefc4_o.jpg",
  "https://live.staticflickr.com/4198/34981841146_9e1b6f94aa_o.jpg",
  "https://live.staticflickr.com/4195/34634536500_c0954f90fa_o.jpg",
  "https://live.staticflickr.com/7006/6784034483_b0938039df_o.jpg",
  "https://live.staticflickr.com/660/20685836923_5549221345_o.jpg",
  "https://live.staticflickr.com/608/21118843000_a669e40ff4_o.jpg",
  "https://live.staticflickr.com/5759/21315085841_df51ea7ff0_o.jpg"
]
    }
]

narration_text = """
Today, we're diving into the hilariously turbulent world of Saturday Night Live, and guess what? It involves some of the biggest celebs that you might've totally forgotten once graced Studio 8H. But first, let's talk about a legendary meltdown that could have cost a career, but instead, became a footnote in the saga of one of comedy’s most iconic figures.

Yes, I’m talking about none other than Larry David. Picture this: It's the mid-1980s, the air is thick with hairspray and comedy sketches, and Larry David is a writer for SNL. Now, Larry, as you might know from his later work on "Curb Your Enthusiasm," has a bit of a... let’s say, unique personality. He’s brilliant but not exactly the poster boy for patience, and SNL was putting that to the test.

Week after week, Larry’s sketches smash it at the readthroughs, everyone’s laughing, the atmosphere's electric, and then, bam! Cut. His sketches are axed from the live show repeatedly. Can you imagine the frustration building up? It’s like baking the world's best chocolate cake, and then, just as you're about to serve it, someone tosses it straight into the trash - every single week!

Now let's fast forward to one infamous night. It's nearly showtime, mere minutes left, and another Larry David original gets the chop. This time, he snaps. Right there in the chaos of pre-live madness, he storms up to the executive producer, Dick Ebersol, and unleashes a torrent of fury. "This f--king show stinks! It’s s--t! I’m done! I’m gone! F--k this! I’m out!” Yes, he actually said that! And off he went, striding out into the frigid New York night, probably fuming and steaming more than the manholes he was stepping over.

But here comes the twist—by the next morning, with his anger cooled off and the reality of a missing paycheck settling in, Larry does something unimaginably bold. He strolls back into SNL the following week as if his explosive exit was just a figment of everyone’s imagination. Imagine walking into the writers' meeting, all eyes on him, whispers all around, and he just casually pitches a new sketch about a circus. 

And they let him stay! He only lasted about a year after that, but during his time there, he met Julia Louis-Dreyfus, setting the stage for future collaborations that we all know and love.

So, not only did Larry David manage to keep his job after declaring the show absolute garbage, but he went on to become a legendary figure in comedy. Stick around as we uncover more shocking stories of celebs and their SNL escapades that you’ve probably forgotten. You won't believe who else has walked through those studio doors!

But that's just one slice of the saga. The SNL vault is deep with juicy tidbits and forgotten stars who once walked its notorious halls. Let's crank up the way-back machine because next up is a true Hollywood A-lister whose SNL stint might just have you saying, "Wait, what?"

Enter Robert Downey Jr. Yes, the Iron Man himself! It's hard to imagine RDJ, the suave, quip-dropping superhero, as a cast member on SNL. But cast your mind back to the 1984-1985 season. That's right, before the days of Marvel dominance and blockbuster spotlights, Robert was a fresh-faced, 20-year-old sketch comedy actor on SNL.

Now picture this young, eager RDJ, thrown into the competitive ring that is SNL. It was a season known for its dramatic overhaul, with a completely new cast plucked straight from the fringes of fame. But despite the electricity of new beginnings, it wasn't all smooth sailing. The season itself faced harsh criticism; it was deemed one of the less successful in the show's run, and poor RDJ, well, let’s just say not many of his characters are etched in the annals of comedy glory.

Imagine the frustration of a young artist, ready to make his mark, only to find that the platform just isn't fitting right. Week after week, sketches fell flat, audiences were tough, and the spotlight seemed just out of reach. It's a year of potential that never fully materialized on the SNL stage. And yet, for RDJ, it was but a brief stop on his meteoric rise to stardom.

Moving on from Iron Man to a woman of steel, did you know that Julia Louis-Dreyfus—yes, Elaine from "Seinfeld" and Selina Meyer from "Veep"—was also a bright-eyed player on the SNL stage? Joining in the early 1980s, like RDJ and Larry David, she found herself amidst the chaos and charm of the SNL family. Rumor has it her time at SNL was packed with potential but hampered by less-than-stellar scripts and a show in a transition period.

Julia, the comedic genius who would later snag countless Emmys, once walked through Studio 8H, likely pondering sketch ideas and dodging those notorious last-minute cuts. Can you feel her exasperation? There she was, a future queen of comedy, jesting and jiving in bits that often didn’t make it past dress rehearsal.

Fast forward and add into the mix, none other than Joan Cusack. That's right, before she was handling the shenanigans of the Gallagher family in "Shameless," Joan was trying to find her comedic timing on SNL during that same tumultuous 1984-1985 season as RDJ and Julia. It's said her time on SNL was challenging, facing stiff competition from established comedies of the time, like SCTV. The frustration of creating—crafting characters, hoping for laughs, all under the pressure cooker of live television—is something Joan faced head-on in Studio 8H.

Each of these stars—RDJ, Julia, Joan, and yes, our man Larry David—shared that same space, breathed that hallowed SNL air, and faced their own battles with the beast of live sketch comedy. Some nights felt like a knock-out punch of disappointment as sketches were axed and characters flopped, while other nights promised glimmers of the brilliance for which they'd later be universally celebrated.

So next time you see RDJ zipping through the sky in a high-tech suit, or Julia Louis-Dreyfus ruling the Veep's office with a razor-sharp wit, or even Joan Cusack managing a raucous household on-screen, remember where they once stood. They stood where every iconic SNL cast member has stood—under the bright lights, in front of the live cameras, facing down the exhilarating, maddening, and utterly unique challenge that is Saturday Night Live.

And what about Larry? Well, despite that explosive, near-career-ending frustration, his story reminds us all that sometimes, the path to legendary status is fraught with moments of doubt, battles with the establishment, and, yes, sheer comedic chaos.

Instead of fading away, these stars used their frustrations as fuel, propelling them onto paths that would one day make them household names. And isn't that the ultimate SNL silver lining? The frustration, the fight, the raw unfiltered challenge of it all—it didn't define them. It merely forged them in the fire of live television, readying them for the great stages yet to come.

Stay tuned, and keep laughing, because in the world of comedy, you never know which frustrated newcomer today might be tomorrow’s legendary star.

As we've peeled back the layers of SNL's star-studded history, it's clear that the spotlight has illuminated some surprising faces over the years. From Robert Downey Jr.'s early comic attempts to Larry David’s explosive walkout, which, frankly, might have been his funniest sketch ever. These tales are not just footnotes in their careers but pivotal moments that shaped these icons in ways we're only beginning to understand. The question remains: what other secrets does the stage of SNL hold? Who else walked through those doors, only to become the mega-stars of today?

If you loved diving into these forgotten chapters of SNL history and crave more juicy revelations, don’t forget to smash that like button. Dive into the comments and let us know which SNL alum surprised you the most, and who you want us to uncover next. Be sure to subscribe and ring the bell for more gossip that you didn’t know you needed, right here, where the past meets the scandalous present! Keep guessing, keep laughing, and until next time, keep tuning in!
"""

characters = ["Larry David", "Dick Ebersol", "Julia Louis-Dreyfus"]

voice_over_path = "narration_20250219_171843.mp3"  # Update with actual path

# Call the function to generate the video
output_video = create_video(clip_data_list, image_data_list, voice_over_path, narration_text, characters, output_dir="outputs")

print(f"Generated video: {output_video}")
