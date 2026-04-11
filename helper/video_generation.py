import yt_dlp
import requests
import os
import time
import threading
from tqdm import tqdm
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips, AudioFileClip
from tempfile import TemporaryDirectory

def time_to_seconds(timestamp):
    h, m, s = map(float, timestamp.split(":"))
    return h * 3600 + m * 60 + s

DEFAULT_DURATION = 5

def progress_hook(d):
    if d['status'] == 'downloading':
        print(f"📥 Downloading: {d['_percent_str']} | Speed: {d['_speed_str']} | ETA: {d['_eta_str']}")

def download_video_segment(url, output_path, start_time, end_time):
    """ Downloads a specific segment of a YouTube video. """
    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
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

def download_image(img_url, img_path, progress_bar):
    img_data = requests.get(img_url).content
    with open(img_path, 'wb') as f:
        f.write(img_data)
    progress_bar.update(1)

def create_video(clip_data_list, image_urls, voice_over_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")

    with TemporaryDirectory() as temp_dir:
        clips = []

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
                if vidclip.get('mute', False):
                    clip = clip.without_audio()
                clips.append(clip)

        # Download images with progress
        print("\n🖼️ Downloading Images...")
        image_paths = []
        with tqdm(total=len(image_urls), desc="Images", unit="img") as pbar:
            threads = []
            for img_url in image_urls:
                img_name = os.path.basename(img_url)
                img_path = os.path.join(temp_dir, img_name)
                image_paths.append(img_path)
                thread = threading.Thread(target=download_image, args=(img_url, img_path, pbar))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

        # Process images into clips
        print("\n🎞️ Adding Images...")
        for img_path in image_paths:
            img_clip = ImageClip(img_path).with_duration(DEFAULT_DURATION)
            if clips:
                img_clip = img_clip.resized(width=clips[0].w, height=clips[0].h)
            clips.append(img_clip)

        # Concatenate all clips
        if not clips:
            raise ValueError("No clips to process!")
        final_clip = concatenate_videoclips(clips, method="compose")

        # Add voice-over
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

        # Cleanup
        for clip in clips:
            clip.close()
        final_clip.close()
        if 'audio' in locals():
            audio.close()

    print(f"\n✅ Video saved: {output_path}")
    return output_path


# clip_data = [
#     {
#         'url': 'https://www.youtube.com/watch?v=rHYKPFdDIyo',
#         'vidclips': [
#             {'startTime': '00:00:18.11', 'endTime': '00:00:18.31', 'mute': True},
#             {'startTime': '00:00:30.53', 'endTime': '00:00:30.78', 'mute': True}
#         ]
#     }
# ]

# image_urls = [
#     "https://live.staticflickr.com/2396/2071470089_c2ef9708a7_o.jpg",
#     "https://live.staticflickr.com/4076/4922094906_2457775618_o.jpg"
# ]

# voice_over_path = "voiceover.mp3"

# output_file = create_video(clip_data, image_urls, voice_over_path)
# print(f"✅ Final Video Path: {output_file}")
