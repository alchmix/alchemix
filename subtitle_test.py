#!/usr/bin/env python3

import os
import subprocess
from datetime import timedelta
import whisperx
import argparse
import logging
import torch
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEMP_AUDIO = "temp_audio.mp3"
OUTPUT_SRT = "output.srt"
OUTPUT_VIDEO = "output_burned.mp4"


def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def extract_audio_ffmpeg(video_path):
    print("🎧 Extracting audio using FFmpeg...")
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "mp3", TEMP_AUDIO
    ]
    subprocess.run(command, check=True)
    print("✅ Audio saved to", TEMP_AUDIO)


def format_timestamp(seconds):
    td = timedelta(seconds=int(seconds))
    ms = int((seconds - int(seconds)) * 1000)
    return f"{str(td)},{ms:03}"


def generate_srt_whisperx(video_path, srt_path="output.srt", model_size="large-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    temp_audio = "temp_audio.wav"

    logging.info("🔧 Extracting and cleaning audio from video...")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        "-af", "highpass=f=200, lowpass=f=3000, dynaudnorm",
        "-c:a", "pcm_s16le", temp_audio
    ], check=True)

    logging.info("🧠 Transcribing audio with WhisperX...")
    model = whisperx.load_model(model_size, device)
    result = model.transcribe(temp_audio)

    logging.info("🧭 Performing word-level alignment...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, temp_audio, device
    )

    logging.info("📝 Writing SRT file...")
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, segment in enumerate(result_aligned["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    logging.info(f"✅ Subtitles saved to {srt_path}")
    return srt_path


def burn_subtitles_ffmpeg(input_video, output_video, srt_file):
    logging.info("🔥 Burning subtitles...")
    
    # Check if files exist
    if not os.path.exists(srt_file):
        raise FileNotFoundError(f"SRT file not found: {srt_file}")
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    # Convert paths to absolute paths and use forward slashes
    srt_abs_path = os.path.abspath(srt_file).replace("\\", "/")
    input_abs_path = os.path.abspath(input_video).replace("\\", "/")
    
    # Method 1: Try with ass filter (more reliable on Windows)
    try:
        logging.info("🎨 Trying ASS filter approach...")
        
        # First, convert SRT to ASS format for better compatibility
        ass_file = srt_file.replace('.srt', '.ass')
        subprocess.run([
            "ffmpeg", "-y", "-i", srt_file, ass_file
        ], check=True)
        
        ass_abs_path = os.path.abspath(ass_file).replace("\\", "/")
        
        subprocess.run([
            "ffmpeg", "-y", "-i", input_video,
            "-vf", f"ass={ass_abs_path}",
            "-c:a", "copy",
            output_video
        ], check=True)
        
        # Clean up temporary ASS file
        if os.path.exists(ass_file):
            os.remove(ass_file)
            
        logging.info(f"🎉 Final video with ASS subtitles: {output_video}")
        return
        
    except subprocess.CalledProcessError as e:
        logging.warning(f"ASS filter failed: {e}")
        # Clean up ASS file if it exists
        if 'ass_file' in locals() and os.path.exists(ass_file):
            os.remove(ass_file)
    
    # Method 2: Try with escaped subtitles filter
    try:
        logging.info("🔄 Trying escaped subtitles filter...")
        
        # Use double backslashes for Windows paths in the filter
        escaped_srt_path = srt_abs_path.replace("/", "\\\\")
        
        subprocess.run([
            "ffmpeg", "-y", "-i", input_video,
            "-vf", f"subtitles='{escaped_srt_path}'",
            "-c:a", "copy",
            output_video
        ], check=True)
        
        logging.info(f"🎉 Final video with escaped path subtitles: {output_video}")
        return
        
    except subprocess.CalledProcessError as e:
        logging.warning(f"Escaped subtitles filter failed: {e}")
    
    # Method 3: Copy SRT to same directory as video and use relative path
    try:
        logging.info("🔄 Trying relative path approach...")
        
        # Copy SRT file to same directory as input video
        video_dir = os.path.dirname(os.path.abspath(input_video))
        temp_srt = os.path.join(video_dir, "temp_subtitles.srt")
        shutil.copy2(srt_file, temp_srt)
        
        # Change to video directory and use relative paths
        original_cwd = os.getcwd()
        os.chdir(video_dir)
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", os.path.basename(input_video),
                "-vf", "subtitles=temp_subtitles.srt",
                "-c:a", "copy",
                os.path.basename(output_video)
            ], check=True)
            
            # Move output back to original location if needed
            if not os.path.samefile(video_dir, os.path.dirname(os.path.abspath(output_video))):
                shutil.move(os.path.basename(output_video), output_video)
                
        finally:
            os.chdir(original_cwd)
            if os.path.exists(temp_srt):
                os.remove(temp_srt)
        
        logging.info(f"🎉 Final video with relative path subtitles: {output_video}")
        return
        
    except (subprocess.CalledProcessError, OSError) as e:
        logging.warning(f"Relative path approach failed: {e}")
        # Ensure we're back in original directory
        try:
            os.chdir(original_cwd)
        except:
            pass
    
    # Method 4: Use drawtext filter to overlay text (fallback)
    try:
        logging.info("🔄 Trying drawtext overlay as final fallback...")
        
        # Read SRT content and create simple text overlay
        # This is a very basic implementation - shows first subtitle only
        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        
        # Extract first subtitle text (very basic parsing)
        lines = srt_content.strip().split('\n')
        subtitle_text = ""
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().isdigit() and '-->' not in line:
                subtitle_text = line.strip()
                break
        
        if subtitle_text:
            # Escape special characters for drawtext
            subtitle_text = subtitle_text.replace(":", "\\:").replace("'", "\\'")
            
            subprocess.run([
                "ffmpeg", "-y", "-i", input_video,
                "-vf", f"drawtext=text='{subtitle_text}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h-text_h-20",
                "-c:a", "copy",
                output_video
            ], check=True)
            
            logging.info(f"🎉 Final video with drawtext overlay: {output_video}")
            logging.warning("Note: Only showing first subtitle due to drawtext limitations")
            return
            
    except subprocess.CalledProcessError as e:
        logging.warning(f"Drawtext overlay failed: {e}")
    
    # If all methods fail
    logging.error("⚠️ All subtitle methods failed. Creating copy without subtitles...")
    try:
        shutil.copy2(input_video, output_video)
        logging.info(f"📁 Created copy without subtitles: {output_video}")
    except Exception as e:
        logging.error(f"❌ Failed to create copy: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Auto-caption a video using Whisper and burn subtitles with FFmpeg.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--model", type=str, default="large-v2", help="Whisper model size")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO, help="Output video file")
    args = parser.parse_args()

    video_path = args.video_path

    if not os.path.isfile(video_path):
        print("❌ Video file does not exist.")
        exit(1)

    if not check_ffmpeg():
        print("❌ FFmpeg is not installed or not available in PATH.")
        exit(1)

    try:
        # Extract audio for transcription
        extract_audio_ffmpeg(video_path)
        
        # Generate SRT file using the extracted audio
        generate_srt_whisperx(TEMP_AUDIO, OUTPUT_SRT, args.model)
        
        # Burn subtitles to video
        burn_subtitles_ffmpeg(video_path, args.output, OUTPUT_SRT)
        
        print(f"🎉 Process completed! Output: {args.output}")
        
    except Exception as e:
        logging.error(f"❌ Process failed: {e}")
        exit(1)
    finally:
        # Clean up temporary files
        for temp_file in [TEMP_AUDIO, "temp_audio.wav"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    main()