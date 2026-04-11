import yt_dlp
import requests
import os
import time
import threading
import shutil
from tqdm import tqdm
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip
from tempfile import TemporaryDirectory
import numpy

def time_to_seconds(timestamp):
    """ Converts a timestamp (HH:MM:SS) to seconds. """
    h, m, s = map(float, timestamp.split(":"))
    return h * 3600 + m * 60 + s

# Standard YouTube dimensions (1080p)
YOUTUBE_WIDTH = 1920
YOUTUBE_HEIGHT = 1080
DEFAULT_DURATION = 5

def progress_hook(d):
    """ Shows download progress. """
    if d['status'] == 'downloading':
        print(f"📥 Downloading: {d['_percent_str']} | Speed: {d['_speed_str']} | ETA: {d['_eta_str']}")

def download_video_segment(url, output_path, start_time, end_time):
    """ Downloads a specific segment of a YouTube video. """
    ydl_opts = {
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',  # Updated to 1080p
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'quiet': True,
        'progress_hooks': [progress_hook],
        'download_ranges': lambda _, __: [{'start_time': start_time, 'end_time': end_time}],
        'force_keyframes_at_cuts': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def download_image(img_url, img_path):
    """ Downloads an image and saves it. """
    img_data = requests.get(img_url).content
    with open(img_path, 'wb') as f:
        f.write(img_data)

def create_centered_image_clip(image_path, duration=DEFAULT_DURATION):
    """
    Creates an image clip that maintains aspect ratio and centers it on a YouTube-sized background.
    """
    # Create the base image clip
    img_clip = ImageClip(image_path)
    
    # Calculate scaling factor to fit within YouTube dimensions while maintaining aspect ratio
    width_ratio = YOUTUBE_WIDTH / img_clip.w
    height_ratio = YOUTUBE_HEIGHT / img_clip.h
    scale_factor = min(width_ratio, height_ratio)
    
    # Resize image maintaining aspect ratio
    resized_clip = img_clip.resized(scale_factor)
    
    # Create a black background clip
    bg_clip = ImageClip(
        numpy.zeros((YOUTUBE_HEIGHT, YOUTUBE_WIDTH, 3), dtype=numpy.uint8)
    ).with_duration(duration)
    
    # Calculate position to center the image
    x_center = (YOUTUBE_WIDTH - resized_clip.w) / 2
    y_center = (YOUTUBE_HEIGHT - resized_clip.h) / 2
    
    # Composite the image onto the background
    final_clip = CompositeVideoClip([
        bg_clip,
        resized_clip.with_position((x_center, y_center))
    ])
    
    return final_clip.with_duration(duration)

def create_video(clip_data_list, image_urls, voice_over_path, output_dir="outputs"):
    """ Generates a final video by combining YouTube clips, images, and voice-over. """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")

    with TemporaryDirectory() as temp_dir:
        clips = []
        first_clip_dimensions = None

        print("\n🎬 Processing YouTube Videos...")
        for clip_idx, clip_data in enumerate(tqdm(clip_data_list, desc="YouTube Clips", unit="clip")):
            yt_url = clip_data['url']
            for vidclip_idx, vidclip in enumerate(clip_data['vidclips']):
                start = time_to_seconds(vidclip['startTime'])
                end = time_to_seconds(vidclip['endTime'])
                clip_path = os.path.join(temp_dir, f"clip_{clip_idx}_{vidclip_idx}.mp4")

                # Download only the segment needed
                download_video_segment(yt_url, clip_path, start, end)

                # Load the clip and handle mute
                clip = VideoFileClip(clip_path)
                
                # Store dimensions of first video clip
                if not first_clip_dimensions:
                    first_clip_dimensions = (clip.w, clip.h)
                
                # Resize video if needed to match YouTube dimensions
                if clip.w != YOUTUBE_WIDTH or clip.h != YOUTUBE_HEIGHT:
                    clip = clip.resized(width=YOUTUBE_WIDTH, height=YOUTUBE_HEIGHT)
                
                if vidclip.get('mute', False):
                    clip = clip.without_audio()
                clips.append(clip)

        # Download images with progress bar
        print("\n🖼️ Downloading Images...")
        image_paths = []
        with tqdm(total=len(image_urls), desc="Images", unit="img") as pbar:
            threads = []
            for img_url in image_urls:
                img_name = os.path.basename(img_url)
                img_path = os.path.join(temp_dir, img_name)
                image_paths.append(img_path)
                thread = threading.Thread(target=lambda: (download_image(img_url, img_path), pbar.update(1)))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

        # Process images into clips
        print("\n🎞️ Adding Images...")
        for img_path in image_paths:
            img_clip = create_centered_image_clip(img_path)
            clips.append(img_clip)

        # Concatenate all clips
        if not clips:
            raise ValueError("No clips to process!")
        final_clip = concatenate_videoclips(clips, method="compose")

        # Add voice-over if available
        if voice_over_path and os.path.exists(voice_over_path):
            print("\n🎙️ Adding Voice-Over...")
            audio = AudioFileClip(voice_over_path)
            final_clip = final_clip.with_audio(audio)

        # Export final video
        print("\n📤 Exporting...")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            fps=24,
            threads=4,
            preset="ultrafast",
            audio_codec="aac"
        )

        # ✅ Cleanup: Close all clips properly
        for clip in clips:
            if hasattr(clip, 'reader'):
                clip.reader.close()
            if hasattr(clip, 'audio') and hasattr(clip.audio, 'reader'):
                clip.audio.reader.close()
            del clip

        if 'final_clip' in locals():
            if hasattr(final_clip, 'reader'):
                final_clip.reader.close()
            if hasattr(final_clip, 'audio') and hasattr(final_clip.audio, 'reader'):
                final_clip.audio.reader.close()
            del final_clip

        if 'audio' in locals():
            audio.close()
            del audio

        # ✅ Explicitly delete temp files before directory cleanup
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")

        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n✅ Video saved: {output_path}")
    return output_path