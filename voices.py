# import moviepy.editor as mp

# size = (1920, 1080)

# img = 'https://www.colorado.edu/cumuseum/sites/default/files/styles/widescreen/public/slider/coachwhip2_1.jpg'

# slide = mp.ImageClip(img).set_fps(25).set_duration(10).resize(size)

# slide = slide.resize(lambda t: 1 + 0.04 * t)  # Zoom-in effect
# slide = slide.set_position(('center', 'center'))
# slide = mp.CompositeVideoClip([slide], size=size)

# slide.write_videofile('zoom-test-1.mp4')



# import requests
# import os
# import logging
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By

# # Set up logging
# logging.basicConfig(level=logging.INFO)

# def setup_selenium():
#     """Set up Selenium WebDriver in headless mode."""
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")  # Run without opening browser
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")

#     service = Service(ChromeDriverManager().install())
#     return webdriver.Chrome(service=service, options=chrome_options)

# def download_image(img_url, img_path):
#     """Download an image using requests, with fallback to Selenium."""
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#         "Referer": img_url  # Some sites check the referer
#     }

#     try:
#         response = requests.get(img_url, headers=headers, timeout=10, allow_redirects=True)
#         response.raise_for_status()  # Raise error for bad responses

#         # Ensure response is an image
#         if "image" not in response.headers.get("Content-Type", ""):
#             logging.warning(f"URL is not an image: {img_url}. Trying Selenium...")
#             return download_image_with_selenium(img_url, img_path)

#         with open(img_path, "wb") as f:
#             f.write(response.content)
#         logging.info(f"Downloaded: {img_url}")
#         return img_path

#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error downloading {img_url}: {e}. Trying Selenium...")
#         return download_image_with_selenium(img_url, img_path)

# def download_image_with_selenium(img_url, img_path):
#     """Use Selenium to extract the actual image source and download it."""
#     try:
#         driver = setup_selenium()
#         driver.get(img_url)

#         # Find all image elements
#         img_elements = driver.find_elements(By.TAG_NAME, "img")
#         img_src = None

#         for img in img_elements:
#             src = img.get_attribute("src")
#             if src and src.startswith("http"):
#                 img_src = src
#                 break  # Take the first valid image

#         driver.quit()

#         if img_src:
#             return download_image(img_src, img_path)
#         else:
#             logging.error(f"Failed to extract image URL using Selenium for: {img_url}")
#             return None

#     except Exception as e:
#         logging.error(f"Selenium error for {img_url}: {e}")
#         return None

# # List of images to download
# image_urls = [
#     "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSN1j0OMRLLNcVlOmZUsj40Lo1Y1y6B5qN_HQ&s",
#     "https://www.usatoday.com/gcdn/presto/2021/11/23/USAT/22706a3e-ed5a-4013-af16-a55da3f1d1f8-Grammynoms012.JPG?width=700&height=582&fit=crop&format=pjpg&auto=webp",
#     "https://i.guim.co.uk/img/static/sys-images/Music/Pix/pictures/2012/7/2/1341221128749/Justin-Bieber-in-2012-008.jpg?width=465&dpr=1&s=none&crop=none",
#     "https://lemaclinic.com/wp-content/uploads/Justin-bieber-badass-smile-makeover.webp",
#     "https://live.staticflickr.com/2054/2397566428_1d2f3c70f2_o.jpg"
# ]

# # Create folder for saving images
# save_folder = "downloaded_images"
# os.makedirs(save_folder, exist_ok=True)

# # Download images
# for i, url in enumerate(image_urls):
#     save_path = os.path.join(save_folder, f"image_{i}.jpg")
#     download_image(url, save_path)


# def create_blurred_background_image(image_path, duration):
#     """Creates a video clip from an image with a blurred background if necessary."""
#     try:
#         # Open the image and convert to RGB
#         img = Image.open(image_path).convert("RGB")
#         img_width, img_height = img.size
#         aspect_ratio = img_width / img_height

#         target_width = 1920
#         target_height = 1080
#         target_aspect_ratio = target_width / target_height

#         needs_background = (img_width < target_width or img_height < target_height or
#                             abs(aspect_ratio - target_aspect_ratio) > 0.1)

#         if needs_background:
#             blurred_bg = img.resize((target_width, target_height)).filter(ImageFilter.GaussianBlur(40))

#             if img_width < target_width and img_height < target_height:
#                 scale_factor = min(target_width / img_width, target_height / img_height) * 0.8
#                 new_width = int(img_width * scale_factor)
#                 new_height = int(img_height * scale_factor)
#                 resized_img = img.resize((new_width, new_height))
#             elif aspect_ratio < target_aspect_ratio:
#                 new_height = int(target_height * 0.9)
#                 new_width = int(new_height * aspect_ratio)
#                 resized_img = img.resize((new_width, new_height))
#             else:
#                 new_width = int(target_width * 0.9)
#                 new_height = int(new_width / aspect_ratio)
#                 resized_img = img.resize((new_width, new_height))

#             x_offset = (target_width - resized_img.width) // 2
#             y_offset = (target_height - resized_img.height) // 2

#             composite = blurred_bg.copy()
#             composite.paste(resized_img, (x_offset, y_offset))

#             # Determine the file extension and save the composite image
#             composite_path = image_path.rsplit('.', 1)[0] + "_composite.png"
#             composite.save(composite_path)

#             clip = ImageClip(composite_path).set_duration(duration)
#         else:
#             clip = ImageClip(image_path).set_duration(duration)

#         clip = clip.resize(1.1)

#         effects = [
#             lambda clip: CompositeVideoClip([clip.resize(lambda t: 1 + 0.05 * (t / clip.duration))]),
#             lambda clip: CompositeVideoClip([clip.resize(lambda t: 1 - 0.03 * (t / clip.duration))]),
#             lambda clip: CompositeVideoClip([clip.set_position(lambda t: ("center", -min(150, 10 * t)))]),
#             lambda clip: CompositeVideoClip([clip.set_position(lambda t: ("center", min(150, 10 * t)))]),
#             lambda clip: CompositeVideoClip([clip.set_position(lambda t: (min(150, 10 * t), "center"))]),
#             lambda clip: CompositeVideoClip([clip.set_position(lambda t: (-min(150, 10 * t), "center"))]),
#             lambda clip: CompositeVideoClip([
#                 clip.set_position(lambda t: (min(100, 8 * t), -min(100, 8 * t)))
#             ]),
#             lambda clip: CompositeVideoClip([
#                 clip.set_position(lambda t: (-min(100, 8 * t), min(100, 8 * t)))
#             ]),
#             lambda clip: CompositeVideoClip([
#                 clip.resize(lambda t: 1 + 0.04 * (t / clip.duration)).set_position(lambda t: (min(80, 5 * t), "center"))
#             ]),
#         ]

#         chosen_effect = random.choice(effects)
#         final_clip = chosen_effect(clip)
#         final_clip = final_clip.set_duration(duration)

#         return final_clip

#     except Exception as e:
#         logging.error(f"Error processing image {image_path}: {e}")
#         return None
# def time_segments(audio_path, segments):
#     """Estimate time boundaries for each segment based on word count proportionality, ignoring empty segments."""
#     try:
#         if not isinstance(audio_path, str):
#             raise ValueError(f"audio_path must be a valid file path (str), but received: {type(audio_path)}")
#         if not segments or not isinstance(segments, list):
#             raise ValueError("Segments must be a non-empty list of strings.")

#         # Extract text if segments contain dictionaries
#         if all(isinstance(seg, dict) and "text" in seg for seg in segments):
#             segments = [seg["text"] for seg in segments]

#         if not all(isinstance(seg, str) for seg in segments):
#             raise ValueError("Segments must be a list of strings.")

#         # Filter out empty segments
#         non_empty_segments = [seg for seg in segments if seg.strip()]

#         if not non_empty_segments:
#             raise ValueError("All segments are empty; cannot allocate time.")

#         # Get total duration of the audio file
#         with AudioFileClip(audio_path) as audio:
#             total_duration = audio.duration

#         # Compute segment weights based on word count
#         segment_word_counts = [len(seg.split()) for seg in non_empty_segments]
#         total_words = sum(segment_word_counts)

#         if total_words == 0:
#             raise ValueError("All segments are empty; cannot allocate time.")

#         # Calculate proportional time allocation
#         time_allocations = [(word_count / total_words) * total_duration for word_count in segment_word_counts]

#         # Generate time boundaries
#         time_boundaries = []
#         start_time = 0.0

#         for duration in time_allocations:
#             end_time = start_time + duration
#             time_boundaries.append((round(start_time, 2), round(end_time, 2)))
#             start_time = end_time

#         return time_boundaries

#     except Exception as e:
#         raise RuntimeError(f"Error in time_segments function: {str(e)}")
# def setup_selenium():
#     """Set up Selenium WebDriver in headless mode."""
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")  # Run without opening browser
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")

#     service = Service(ChromeDriverManager().install())
#     return webdriver.Chrome(service=service, options=chrome_options)
# def clean_filename(filename):
#     """Sanitize filenames by removing invalid characters."""
#     filename = re.sub(r'[<>:"/\\|?*]', '', filename)  # Remove Windows invalid characters
#     filename = filename.split('?')[0]  # Remove query parameters
#     return filename.strip()
# def download_image(img_url, img_path):
#     """Download an image using requests, with fallback to Selenium."""
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#         "Referer": img_url  # Some sites check the referer
#     }

#     try:
#         response = requests.get(img_url, headers=headers, timeout=10, allow_redirects=True)
#         response.raise_for_status()  # Raise error for bad responses

#         # Ensure response is an image
#         if "image" not in response.headers.get("Content-Type", ""):
#             logging.warning(f"URL is not an image: {img_url}. Trying Selenium...")
#             return download_image_with_selenium(img_url, img_path)  # One-time fallback
        
#         # Clean the filename before saving
#         img_path = os.path.join(os.path.dirname(img_path), clean_filename(os.path.basename(img_path)))

#         with open(img_path, "wb") as f:
#             f.write(response.content)
#         logging.info(f"Downloaded: {img_url}")
#         return img_path

#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error downloading {img_url}: {e}. Trying Selenium...")
#         return download_image_with_selenium(img_url, img_path)  # One-time fallback

# def download_image_with_selenium(img_url, img_path):
#     """Use Selenium to extract the actual image source and download it without recursion."""
#     try:
#         driver = setup_selenium()
#         driver.get(img_url)

#         # Find all image elements
#         img_elements = driver.find_elements(By.TAG_NAME, "img")
#         img_src = None

#         for img in img_elements:
#             src = img.get_attribute("src")
#             if src and src.startswith("http"):
#                 img_src = src
#                 break  # Take the first valid image

#         driver.quit()

#         if img_src:
#             try:
#                 headers = {
#                     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#                     "Referer": img_url
#                 }
#                 response = requests.get(img_src, headers=headers, timeout=10)
#                 response.raise_for_status()

#                 with open(img_path, "wb") as f:
#                     f.write(response.content)
#                 logging.info(f"Downloaded via Selenium: {img_src}")
#                 return img_path

#             except requests.exceptions.RequestException as e:
#                 logging.error(f"Failed to download extracted image {img_src}: {e}")
#                 return None
#         else:
#             logging.error(f"Failed to extract image URL using Selenium for: {img_url}")
#             return None

#     except Exception as e:
#         logging.error(f"Selenium error for {img_url}: {e}")
#         return None


# import re
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By













# def find_best_media_combination(media_pool, text_segment, character_name, duration_needed):
#     """Dynamically select media items to exactly match segment duration from available pool."""
#     text_embedding = encode_text(text_segment)
#     relevant_media = [m for m in media_pool if m['character'] == character_name]
    
#     selected = []
#     remaining = duration_needed
#     iteration = 0  # Track selection passes
    
#     while remaining > 0 and iteration < 2:
#         # First pass: use unused media, second pass: reuse if needed
#         candidates = [
#             m for m in relevant_media 
#             if (iteration == 0 and not m.get('used', False)) or 
#                (iteration == 1 and m.get('used', False))
#         ]
        
#         # Sort candidates by relevance
#         scored = sorted(
#             [(m, cosine_similarity(text_embedding, m['embedding'])) for m in candidates],
#             key=lambda x: x[1], 
#             reverse=True
#         )
        
#         for media, score in scored:
#             if remaining <= 0:
#                 break
                
#             available_duration = media['duration']
#             use_duration = min(available_duration, remaining)
            
#             selected.append({
#                 'media': media,
#                 'duration': use_duration,
#                 'has_audio': not media.get('mute', False) if media['type'] == 'video' else False
#             })
            
#             if iteration == 0:  # Mark as used only in first pass
#                 media['used'] = True
#             remaining -= use_duration
        
#         # Move to reuse pass if needed
#         if remaining > 0 and iteration == 0 and not any(not m.get('used', False) for m in relevant_media):
#             iteration = 1
#         else:
#             break
            
#     # Final fallback: reuse media if still remaining
#     if remaining > 0:
#         scored = sorted(
#             [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#             key=lambda x: x[1], 
#             reverse=True
#         )
        
#         while remaining > 0 and scored:
#             for media, score in scored:
#                 if remaining <= 0:
#                     break
#                 use_duration = min(media['duration'], remaining)
#                 selected.append({
#                     'media': media,
#                     'duration': use_duration,
#                     'has_audio': not media.get('mute', False) if media['type'] == 'video' else False
#                 })
#                 remaining -= use_duration
    
#     return selected

# # ----------------- Video Creation -----------------

# def create_video(clip_data_list, image_data_list, voice_over_path, narration_text, characters, vid_segments, output_dir="outputs"):
#     """Create a video with dynamic media selection, guaranteed duration matching, and no placeholders."""
#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")
#     temp_dir = os.path.join(TEMP_DIR, timestamp)
#     os.makedirs(temp_dir, exist_ok=True)

#     try:
#         media_pool = []

#         logging.info("📦 Preparing video clips...")
#         for clip_data in clip_data_list:
#             character_name = clip_data.get("characterName", "Unknown")
#             videos = clip_data.get("videos", [])

#             for video_data in videos:
#                 yt_url = video_data.get("url")
#                 vidclips = video_data.get("vidclips", [])

#                 for vidclip in vidclips:
#                     try:
#                         start = time_to_seconds(vidclip.get("startTime", "0:00:00"))
#                         end = time_to_seconds(vidclip.get("endTime", "0:00:05"))
#                         clip_path = os.path.join(temp_dir, f"{character_name}_{start}_{end}.mp4")

#                         downloaded_path = download_video_segment(yt_url, clip_path, start, end)
#                         if not downloaded_path:
#                             continue

#                         clip = VideoFileClip(downloaded_path).resize((1920, 1080))

#                         mute = vidclip.get("mute", False)
#                         if mute or not clip.audio:
#                             clip = clip.without_audio()

#                         rep_frame = extract_representative_frame(clip)
#                         media_pool.append({
#                             "type": "video",
#                             "clip": clip,
#                             "character": character_name,
#                             "embedding": encode_image_from_pil(rep_frame),
#                             "mute": mute,
#                             "duration": clip.duration,
#                             "used": False,
#                             "tag": vidclip.get("tag")
#                         })
#                     except Exception as e:
#                         logging.error(f"Error processing video clip: {e}")

#         logging.info("🖼️ Preparing images...")
#         image_paths = {}

#         with ThreadPoolExecutor(max_workers=4) as executor:
#             future_to_character = {}

#             for image_data in image_data_list:
#                 character_name = image_data["characterName"]
#                 character_folder = os.path.join(temp_dir, character_name.replace(" ", "_"))
#                 os.makedirs(character_folder, exist_ok=True)

#                 for img_url in image_data["images"]:
#                     img_filename = os.path.basename(img_url)
#                     img_path = os.path.join(character_folder, img_filename)
#                     future = executor.submit(download_image, img_url, img_path)
#                     future_to_character[future] = (character_name, img_path)

#             for future in tqdm(as_completed(future_to_character), total=len(future_to_character), desc="Downloading Images"):
#                 character, img_path = future_to_character[future]
#                 try:
#                     result = future.result()
#                     if result:
#                         image_paths.setdefault(character, []).append(result)
#                 except Exception as e:
#                     logging.error(f"⚠️ Error downloading image for {character}: {e}")

#         for character, img_list in image_paths.items():
#             for img_path in img_list:
#                 try:
#                     img = Image.open(img_path).convert("RGB")
#                     media_pool.append({
#                         "type": "image",
#                         "path": img_path,
#                         "character": character,
#                         "embedding": encode_image_from_pil(img),
#                         "duration": DEFAULT_DURATION,
#                         "used": False,
#                         "tag": None
#                     })
#                 except Exception as e:
#                     logging.error(f"⚠️ Error processing image {img_path}: {e}")

#         logging.info("🎞️ Building timeline...")
#         segments = vid_segments
#         main_audio = AudioFileClip(voice_over_path)
#         time_ranges = time_segments(voice_over_path, segments)
#         final_clips = []

#         for seg_idx, (segment, (seg_start, seg_end)) in enumerate(zip(segments, time_ranges)):
#             duration_needed = seg_end - seg_start
#             character_name = segment["character"]

#             selected_media = find_best_media_combination(
#                 media_pool,
#                 segment["text"],
#                 segment.get("tag"),
#                 character_name,
#                 duration_needed
#             )

#             segment_clips = []
#             total_duration_used = 0

#             for item in selected_media:
#                 media = item['media']
#                 if media['type'] == 'video':
#                     # Ensure videos that are not muted always play full length
#                     if not media.get('mute', False):
#                         clip = media['clip']
#                         total_duration_used += clip.duration
#                     else:
#                         clip = media['clip'].subclip(0, min(item['duration'], media['clip'].duration))
#                         total_duration_used += item['duration']

#                 else:  # Handle images
#                     clip = create_blurred_background_image(media['path'], item['duration'])
#                     total_duration_used += item['duration']

#                 # Check if clip is valid before appending
#                 if clip:
#                     segment_clips.append(clip)

#             if segment_clips:
#                 segment_video = concatenate_videoclips(segment_clips, method="compose")
#                 segment_video = segment_video.set_audio(main_audio.subclip(seg_start, seg_end))
#                 final_clips.append(segment_video)

#         total_video = concatenate_videoclips(final_clips, method="compose")
#         total_video_duration = total_video.duration
#         total_audio_duration = main_audio.duration

#         if total_video_duration < total_audio_duration:
#             duration_diff = total_audio_duration - total_video_duration
#             logging.info(f"Adding {duration_diff:.2f}s to match audio duration")

#             last_character = segments[-1]["character"]
#             last_text = segments[-1]["text"]

#             additional_media = find_best_media_combination(
#                 media_pool,
#                 last_text,
#                 None,
#                 last_character,
#                 duration_diff
#             )

#             additional_clips = []
#             for item in additional_media:
#                 media = item['media']
#                 if media['type'] == 'video':
#                     clip = media['clip'].subclip(0, min(item['duration'], media['clip'].duration))
#                     if media.get('mute', False):
#                         clip = clip.without_audio()
#                 else:
#                     clip = create_blurred_background_image(media['path'], item['duration'])
#                     clip = clip.fx(vfx.resize, width=clip.w * 1.1)
#                 if clip:
#                     additional_clips.append(clip)

#             if additional_clips:
#                 additional_video = concatenate_videoclips(additional_clips, method="compose")
#                 total_video = concatenate_videoclips([total_video, additional_video], method="compose")

#         total_video.write_videofile(
#             output_path,
#             codec="libx264",
#             fps=24,
#             preset="fast",
#             audio_codec="aac",
#             threads=MAX_WORKERS,
#             ffmpeg_params=[
#                 "-b:v", "1M",
#                 "-crf", "23",
#                 "-c:a", "copy",
#                 "-threads", str(MAX_WORKERS),
#                 "-preset", "fast",
#             ]
#         )

#         return output_path

#     except Exception as e:
#         logging.error(f"Fatal error in video creation: {e}")
#         return None
 
 
import os
import threading
import time
from datetime import datetime
from moviepy.editor import AudioFileClip, concatenate_audioclips
from gtts import gTTS
from tqdm import tqdm
import time

UPLOAD_FOLDER = "static"  # Directory where narrations are saved

# def generate_audio_segments(segments, output_dir=UPLOAD_FOLDER):
#     """Generates individual audio clips for each text-based segment and returns paths to each segment audio and timestamps."""
#     os.makedirs(output_dir, exist_ok=True)

#     segment_audio_paths = []
#     timestamps = []
#     total_duration = 0.0

#     for idx, segment in enumerate(segments):
#         text = segment.get("text", "").strip()
#         tag = segment.get("tag", None)

#         if not text and tag is None:
#             # Skip completely empty segments
#             continue

#         audio_path = os.path.join(output_dir, f"segment_{idx}.mp3")

#         if text:
#             # Generate audio narration
#             tts = gTTS(text, lang="en", slow=False)
#             tts.save(audio_path)

#             # Get audio duration
#             audio_clip = AudioFileClip(audio_path)
#             duration = audio_clip.duration
#             audio_clip.close()

#             # Calculate timestamps
#             segment_start = total_duration
#             segment_end = segment_start + duration
#             timestamps.append((segment_start, segment_end))

#             total_duration += duration
#         else:
#             # For segments with no text but a tag, set start and end times to 0.0
#             timestamps.append((0.0, 0.0))
#             audio_path = None  # No audio file for tag-only segments

#         segment_audio_paths.append(audio_path)

#     # Generate unique filename for the final narration
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     final_audio_filename = f"narration_{timestamp}.mp3"
#     final_audio_path = os.path.join(output_dir, final_audio_filename)

#     # Concatenate all audio clips into one final audio file
#     audio_clips = [AudioFileClip(path) for path in segment_audio_paths if path]
#     if audio_clips:
#         final_audio = concatenate_audioclips(audio_clips)
#         final_audio.write_audiofile(final_audio_path)
#         final_audio.close()

#     # Generate URL for the final audio file
#     final_audio_url = f"http://localhost:5000/static/{final_audio_filename}"

#     # Return the paths to each segment audio, the final concatenated audio path, the URL, and the list of timestamps
#     return segment_audio_paths, final_audio_path, final_audio_url, timestamps


import os
import time
from moviepy.editor import AudioFileClip, concatenate_audioclips
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = "voices" 

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_speech(text, output_path, voice):
    """Generates speech from text using OpenAI's latest TTS API."""
    speech_file_path = Path(output_path)

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    )

    response.stream_to_file(speech_file_path)
voice="shimmer"
text='You are using Alchemy Mix. The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.'
audio_path = os.path.join(UPLOAD_FOLDER, f"{voice}.mp3")

generate_speech(text,audio_path,voice)
