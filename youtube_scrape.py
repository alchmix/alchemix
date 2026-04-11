import yt_dlp
import requests
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips, AudioFileClip
from tempfile import TemporaryDirectory
from PIL import Image
import torch
import numpy as np
import cv2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DURATION = 5
MATCH_THRESHOLD = 0.2
MAX_WORKERS = 4
VIDEO_RESOLUTION = 720  # 720p

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Import YOLO and CLIP models
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# Initialize AI models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
yolo_model = YOLO("yolov8n.pt")

def time_to_seconds(timestamp):
    """Convert timestamp string to seconds."""
    try:
        h, m, s = map(float, timestamp.split(":"))
        return h * 3600 + m * 60 + s
    except ValueError as e:
        logger.error(f"Invalid timestamp format: {timestamp}")
        raise ValueError(f"Invalid timestamp format. Expected HH:MM:SS.ms, got {timestamp}") from e

def download_video_segment(url, output_path, start_time, end_time):
    """Download a specific segment of a video."""
    try:
        ydl_opts = {
            'format': f'bestvideo[height<={VIDEO_RESOLUTION}]+bestaudio/best[height<={VIDEO_RESOLUTION}]',
            'outtmpl': output_path,
            'merge_output_format': 'mp4',
            'quiet': True,
            'download_ranges': lambda _, __: [{'start_time': start_time, 'end_time': end_time}],
            'force_keyframes_at_cuts': True,
            'postprocessor_hooks': [lambda d: logger.info(f"Downloaded segment: {d.get('filename')}")],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        logger.error(f"Error downloading video segment: {e}")
        return False

def download_image(img_url, img_path):
    """Download an image with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            with open(img_path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to download image after {max_retries} attempts: {e}")
                return False
            time.sleep(1)  # Wait before retry

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def encode_text(text):
    """Encode text using CLIP model."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features[0].cpu().numpy()

def encode_image(image: Image.Image):
    """Encode image using CLIP model."""
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features[0].cpu().numpy()

def analyze_text_segments(narration_text):
    """Break down narration into semantic segments with timing estimates."""
    sentences = sent_tokenize(narration_text)
    
    # Initialize zero-shot classifier
    classifier = pipeline("zero-shot-classification", 
                        model="facebook/bart-large-mnli")
    
    segments = []
    current_segment = []
    current_words = 0
    WORDS_PER_SEGMENT = 50
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        current_words += len(words)
        current_segment.append(sentence)
        
        if current_words >= WORDS_PER_SEGMENT:
            segment_text = " ".join(current_segment)
            
            # Analyze themes
            themes = classifier(segment_text, 
                             candidate_labels=["people", "location", "action", 
                                             "emotion", "object"])
            
            # Extract key terms
            tokens = nltk.pos_tag(word_tokenize(segment_text))
            key_terms = [word for word, pos in tokens 
                        if pos.startswith(('NN', 'NNP'))]
            
            # Estimate duration (words * average seconds per word)
            estimated_duration = current_words * 0.4
            
            segments.append({
                'text': segment_text,
                'duration': estimated_duration,
                'themes': themes['labels'],
                'scores': themes['scores'],
                'key_terms': list(set(key_terms))
            })
            
            current_segment = []
            current_words = 0
    
    # Handle remaining text
    if current_segment:
        segment_text = " ".join(current_segment)
        themes = classifier(segment_text, 
                          candidate_labels=["people", "location", "action", 
                                          "emotion", "object"])
        tokens = nltk.pos_tag(word_tokenize(segment_text))
        key_terms = [word for word, pos in tokens 
                    if pos.startswith(('NN', 'NNP'))]
        
        segments.append({
            'text': segment_text,
            'duration': current_words * 0.4,
            'themes': themes['labels'],
            'scores': themes['scores'],
            'key_terms': list(set(key_terms))
        })
    
    return segments

def evaluate_media(pil_image, text):
    """Evaluate media against text using CLIP and YOLO."""
    # Get CLIP embeddings
    image_embedding = encode_image(pil_image)
    text_embedding = encode_text(text)
    clip_score = cosine_similarity(image_embedding, text_embedding)
    
    # Get YOLO scene description
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = yolo_model(cv_image)
    detected_objects = set()
    for result in results:
        for detection in result.boxes.cls.cpu().numpy():
            label = yolo_model.names[int(detection)]
            detected_objects.add(label)
    
    scene_description = "scene contains: " + ", ".join(detected_objects)
    scene_embedding = encode_text(scene_description)
    yolo_score = cosine_similarity(scene_embedding, text_embedding)
    
    # Combine scores
    return (clip_score + yolo_score) / 2

def download_all_media(clip_data_list, image_urls, temp_dir):
    """Download all videos and images concurrently."""
    successful_videos = []
    successful_images = []
    
    # Download video segments
    logger.info("Downloading video segments...")
    video_futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for clip_idx, clip_data in enumerate(clip_data_list):
            for vidclip_idx, vidclip in enumerate(clip_data['vidclips']):
                clip_path = os.path.join(temp_dir, f"clip_{clip_idx}_{vidclip_idx}.mp4")
                start = time_to_seconds(vidclip['startTime'])
                end = time_to_seconds(vidclip['endTime'])
                future = executor.submit(
                    download_video_segment,
                    clip_data['url'],
                    clip_path,
                    start,
                    end
                )
                video_futures.append((future, {
                    'path': clip_path,
                    'mute': vidclip.get('mute', False)
                }))
    
    for future, clip_info in video_futures:
        if future.result():
            successful_videos.append(clip_info)
    
    # Download images
    logger.info("Downloading images...")
    image_futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for img_idx, img_url in enumerate(image_urls):
            img_path = os.path.join(temp_dir, f"image_{img_idx}.jpg")
            future = executor.submit(download_image, img_url, img_path)
            image_futures.append((future, img_path))
    
    for future, img_path in image_futures:
        if future.result():
            successful_images.append(img_path)
    
    return successful_videos, successful_images

def create_smart_video(clip_data_list, image_urls, voice_over_path, narration_text, output_dir="outputs"):
    """Create a video with intelligent alignment between narration and visuals."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")
    
    with TemporaryDirectory() as temp_dir:
        try:
            # Analyze narration
            logger.info("Analyzing narration...")
            segments = analyze_text_segments(narration_text)
            
            # Download all media concurrently
            video_clips, image_paths = download_all_media(
                clip_data_list, 
                image_urls, 
                temp_dir
            )
            
            if not video_clips and not image_paths:
                raise ValueError("No media was successfully downloaded")
            
            # Create video sequence
            logger.info("Creating video sequence...")
            final_clips = []
            first_clip_size = None
            
            for segment in segments:
                # Find best matching media for segment
                best_score = -1
                best_media = None
                is_video = False
                
                # Try video clips first
                for clip in video_clips:
                    video = VideoFileClip(clip['path'])
                    frame = video.get_frame(video.duration / 2)
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    score = evaluate_media(pil_frame, segment['text'])
                    video.close()
                    
                    if score > best_score and score >= MATCH_THRESHOLD:
                        best_score = score
                        best_media = clip
                        is_video = True
                
                # Try images if no good video match
                if best_score < MATCH_THRESHOLD:
                    for img_path in image_paths:
                        pil_image = Image.open(img_path).convert("RGB")
                        score = evaluate_media(pil_image, segment['text'])
                        
                        if score > best_score and score >= MATCH_THRESHOLD:
                            best_score = score
                            best_media = img_path
                            is_video = False
                
                # Create clip from best matching media
                if best_media:
                    if is_video:
                        clip = VideoFileClip(best_media['path'])
                        if best_media['mute']:
                            clip = clip.without_audio()
                        # Loop if needed
                        if clip.duration < segment['duration']:
                            clip = clip.loop(duration=segment['duration'])
                        else:
                            clip = clip.subclip(0, segment['duration'])
                    else:
                        clip = ImageClip(best_media).with_duration(segment['duration'])
                    
                    # Set consistent size
                    if not first_clip_size:
                        first_clip_size = (clip.w, clip.h)
                    else:
                        clip = clip.resized(width=first_clip_size[0], height=first_clip_size[1])
                    
                    final_clips.append(clip)
            
            # Concatenate clips and add audio
            logger.info("Finalizing video...")
            final_video = concatenate_videoclips(final_clips, method="compose")
            
            if voice_over_path and os.path.exists(voice_over_path):
                logger.info("Adding voice-over...")
                audio = AudioFileClip(voice_over_path)
                final_video = final_video.with_audio(audio)
            
            # Export final video
            logger.info("Exporting video...")
            final_video.write_videofile(
                output_path,
                codec="libx264",
                fps=24,
                threads=4,
                preset="ultrafast",
                audio_codec="aac"
            )
            
            # Cleanup
            for clip in final_clips:
                clip.close()
            final_video.close()
            if 'audio' in locals():
                audio.close()
            
            logger.info(f"Video saved successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            raise

if __name__ == "__main__":
    # Example usage with the provided data
    
 clip_data = [
    {
        'url': 'https://www.youtube.com/watch?v=c3of5EUwwZo',
        'vidclips': [
            {'startTime': '00:00:07.93', 'endTime': '00:00:08.19', 'mute': True},
            {'startTime': '00:00:08.95', 'endTime': '00:00:09.15', 'mute': True}
        ]
    },
    {
        'url': 'https://www.youtube.com/watch?v=KkfiJpT7QI4',
        'vidclips': [
            {'startTime': '00:00:01.70', 'endTime': '00:00:01.88', 'mute': True},
            {'startTime': '00:00:02.52', 'endTime': '00:00:02.68', 'mute': True},
            {'startTime': '00:00:03.21', 'endTime': '00:00:03.38', 'mute': True},
            {'startTime': '00:00:03.78', 'endTime': '00:00:03.94', 'mute': True},
            {'startTime': '00:01:27.06', 'endTime': '00:01:27.24', 'mute': True}
        ]
    },
    {
        'url': 'https://www.youtube.com/watch?v=rHYKPFdDIyo',
        'vidclips': [
            {'startTime': '00:00:01.56', 'endTime': '00:00:08.23', 'mute': True},
            {'startTime': '00:00:28.86', 'endTime': '00:00:38.27', 'mute': True},
            {'startTime': '00:00:43.02', 'endTime': '00:00:43.78', 'mute': True}
        ]
    },
    {
        'url': 'https://www.youtube.com/watch?v=UMfCUvIK3V8',
        'vidclips': [
            {'startTime': '00:00:09.90', 'endTime': '00:00:17.73', 'mute': True},
            {'startTime': '00:00:21.84', 'endTime': '00:00:26.05', 'mute': True},
            {'startTime': '00:00:30.53', 'endTime': '00:00:34.00', 'mute': True},
            {'startTime': '00:00:37.12', 'endTime': '00:00:50.80', 'mute': True},
            {'startTime': '00:00:52.62', 'endTime': '00:00:55.77', 'mute': True}
        ]
    }
]

image_urls = [
    "https://live.staticflickr.com/2396/2071470089_c2ef9708a7_o.jpg",
    "https://live.staticflickr.com/86/233344084_497d8dcf49_o.jpg",
    "https://live.staticflickr.com/4009/4174971224_f99cb78787_o.jpg",
    "https://live.staticflickr.com/93/254826594_92ab57a73b_o.jpg",
    "https://live.staticflickr.com/83/254824777_118addd4f3_o.jpg",
    "https://live.staticflickr.com/4076/4922094906_2457775618_o.jpg",
    "https://live.staticflickr.com/4142/4923003130_851551ba3d_o.jpg",
    "https://live.staticflickr.com/8530/8675141209_cabf762d47_o.jpg",
    "https://live.staticflickr.com/5456/8864144937_abd6c42430_o.jpg",
    "https://farm9.staticflickr.com/8118/8602828910_651164e654_b.jpg",
    "https://live.staticflickr.com/65535/52164477528_87c236d4d8_o.jpg",
    "https://live.staticflickr.com/65535/52164477488_e7ef91b530_o.jpg",
    "https://live.staticflickr.com/2421/4093711193_b1b1be2490_o.jpg",
    "https://live.staticflickr.com/3254/2971063270_45e60361b1_o.jpg",
    "https://live.staticflickr.com/8013/7496767378_4883e7acff_o.jpg",
    "https://live.staticflickr.com/8430/7722871112_213d58924a_o.jpg",
    "https://live.staticflickr.com/8285/7722866286_dbf94e33a5_o.jpg",
    "https://live.staticflickr.com/8429/7722868136_def002be22_o.jpg",
    "https://live.staticflickr.com/7124/7722870126_b3a768f6da_o.jpg",
    "https://live.staticflickr.com/8289/7722869156_d58fe251e1_o.jpg"
]

# For this example, we assume you have the narration text already.
# In a real system, you might generate this by transcribing the voice-over with Whisper.
narration_text = """Oh, get ready for this spicy scoop that’s steaming hot off the press! Picture this: New York City, the glistening glamor of Radio City Music Hall, and none other than the legendary Meryl Streep making waves in a way you might not expect. Now, when you think Meryl, you think class, exemplary talent, maybe a dash of regal poise, right? But, hold onto your hats because Meryl just served up a slice of shocker humor that's got everyone talking!

So here’s the scene—it’s February 14th, a night sparkling with stars, thanks to SNL’s 50th Homecoming Concert, and the cameras are rolling live. There, amidst the glitterati, we spot Meryl Streep cozying up next to her rumored beau and "Only Murders in the Building" co-star, Martin Short. Love is in the air, or so it seems, right? But wait, there’s a twist coming that’s as juicy as a ripe peach in summer!

Cut to Ana Gasteyer and Will Ferrell, I mean, c’mon, these two alone could stir up a laugh riot with just a glance! They’re back as our favorite dysfunctional middle school music teachers, Marty and Bobbi Mohan-Culp. Oh, how we've missed them! Will Ferrell, in his impeccable style, complains about seeing quite a few middle fingers from the crowd, calling it "unbelievably rude." And what do you know, the comic arrow turns to point straight at our dear Meryl.

Ana reels it in with a saucy, "We're talking about you, Meryl. We see you." And boom! Meryl, the queen herself, without missing a beat, flips the bird right back at them. Yes, you heard that right—Meryl Streep gave the finger! And it wasn’t just a coy little gesture. No, no, no. It was bold, it was brassy, and oh, did it get people howling with laughter, including Martin Short, whose reaction was just priceless!

Now, don’t go thinking Meryl’s just thrown decorum out the window. Oh no, this was all in good fun, a cheeky part of the gag lined up by the SNL alums, and it clearly stole the show. But isn't it just fabulous to see such an icon letting loose, causing a stir, and just generally embracing the mirth of the moment?

So there you have it, folks! Meryl Streep, flipping the script (and the bird) in a hilariously unexpected twist at an SNL celebration. Who says award-winning actresses can’t have a little naughty fun? Let’s just say, the audience at Radio City Music Hall sure got an eyeful and a playful reminder that Meryl is not only an on-screen legend but a bit of a live-wire jokester too! Meryl, you cheeky thing, you’ve done it again!"""

voice_over_path = "narration_20250216_144312.mp3"

# ----------------- Run the Advanced Video Creation -----------------

try:
        output_file = create_smart_video(clip_data, image_urls, voice_over_path, narration_text)
        print(f"✅ Final Video Path: {output_file}")
except Exception as e:
        print(f"❌ Error creating video: {e}")