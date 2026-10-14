import shutil
import yt_dlp
import requests
import os
import base64
from contextlib import contextmanager
from threading import Semaphore
import time
import json
import uuid
from typing import Optional
import concurrent.futures
from multiprocessing import Manager
import gc
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import tempfile
from helper.telegram_notification_engine import send_telegram_notification
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips, AudioFileClip,CompositeVideoClip,ColorClip, CompositeAudioClip,vfx,concatenate_audioclips,clips_array
# import moviepy.video.fx.all as vfx
from tempfile import TemporaryDirectory
from PIL import Image, ImageFilter,ImageDraw,ImageFont
import numpy as np
import cv2
import openai
import random
import re
from urllib.parse import unquote, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
from functools import lru_cache
import os, uuid, glob, logging, subprocess
from math import ceil, floor
# from video_helper.segment_narration import segment_narration
import multiprocessing
import psutil
from functools import lru_cache, wraps
from datetime import datetime, timedelta
from threading import Lock


class _LazyTorchModule:
    """Defer `import torch` until first attribute access (avoids CUDA init in pool workers)."""

    _module = None

    def _load(self):
        if _LazyTorchModule._module is None:
            import torch as _torch_mod
            _LazyTorchModule._module = _torch_mod
        return _LazyTorchModule._module

    def __getattr__(self, name):
        return getattr(self._load(), name)


torch = _LazyTorchModule()


# ----------------- Configuration -----------------
_MEMORY_THRESHOLD = 85  # Percentage of RAM usage that triggers cleanup
_EMERGENCY_MEMORY_THRESHOLD = 90  # More aggressive threshold
_TEXT_CACHE_MAX_SIZE = 512
_TEXT_CACHE_TTL = timedelta(minutes=30)  # Shorter TTL for production
_GPU_MEMORY_THRESHOLD = 2  # GB

# ----------------- State Tracking -----------------
_TEXT_CACHE_LAST_CLEARED = datetime.now()
_active_operations = 0
_process_lock = Lock()
OUTPUT = "outputs"


# ---------------------------------------
# GLOBAL RUNTIME FLAGS (Same File)
# ---------------------------------------
CURRENT_WIDTH = 1920
CURRENT_HEIGHT = 1080
CURRENT_ORIENTATION = "landscape"

CURRENT_STORYBOARD_ENABLED = False
CURRENT_KENBURNS_ENABLED = True
_imagefx_auth_failed = False

# ----------------- Core Cache System -----------------
@lru_cache(maxsize=_TEXT_CACHE_MAX_SIZE)
def encode_text(text):
    """Cache-optimized text encoding with automatic cleanup"""
    _maybe_clear_text_cache()
    try:
        model, processor = get_clip_model()  # lazy singleton
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        return text_features[0].cpu().numpy()
    except Exception as e:
        logging.error(f"Error encoding text: {e}")
        return np.zeros((512,))


# def encode_text(text):
#     """Cache-optimized text encoding with automatic cleanup"""
#     _maybe_clear_text_cache()
#     try:
#         inputs = clip_processor(
#             text=[text], 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True, 
#             max_length=77
#         )
#         with torch.no_grad():
#             text_features = clip_model.get_text_features(**inputs)
#         return text_features[0].cpu().numpy()
#     except Exception as e:
#         logging.error(f"Error encoding text: {e}")
#         return np.zeros((512,))


def _maybe_clear_text_cache():
    """Conditionally clear the text cache based on TTL"""
    global _TEXT_CACHE_LAST_CLEARED
    if datetime.now() - _TEXT_CACHE_LAST_CLEARED > _TEXT_CACHE_TTL:
        encode_text.cache_clear()
        _TEXT_CACHE_LAST_CLEARED = datetime.now()
        logging.info("Cleared text encoding cache (TTL expired)")
        
def force_clear_text_cache():
    """Explicit cache clearing (call during memory pressure)"""
    encode_text.cache_clear()
    global _TEXT_CACHE_LAST_CLEARED
    _TEXT_CACHE_LAST_CLEARED = datetime.now()
    logging.warning("Forced text encoding cache clearance") 
    

# ----------------- Memory Management -----------------
def clear_gpu_memory():
    """Safely clear GPU cache without disrupting computations"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for current operations
        torch.cuda.empty_cache()  # Only clears unused memory
        logging.info("Safely cleared unused GPU memory")

def force_clear_text_cache():
    """Force-clear the text embedding cache"""
    global _TEXT_CACHE_LAST_CLEARED
    with _process_lock:
        encode_text.cache_clear()
        _TEXT_CACHE_LAST_CLEARED = datetime.now()
        logging.warning("Forced text encoding cache clearance")

def _maybe_clear_text_cache():
    """Conditionally clear caches based on TTL"""
    global _TEXT_CACHE_LAST_CLEARED
    if datetime.now() - _TEXT_CACHE_LAST_CLEARED > _TEXT_CACHE_TTL:
        with _process_lock:
            encode_text.cache_clear()
            _TEXT_CACHE_LAST_CLEARED = datetime.now()
            logging.info("Periodic text cache clearance (TTL expired)")

# ----------------- Decorator Implementation -----------------
def memory_safe_operation(func):
    """Decorator with complete process protection"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _active_operations
        
        # 1. Check memory before starting (only if no other active ops)
        if _active_operations == 0:
            current_mem = psutil.virtual_memory().percent
            if current_mem > _MEMORY_THRESHOLD:
                logging.warning(f"Pre-op memory high ({current_mem}%)")
                force_clear_text_cache()
                clear_gpu_memory()
            
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                if gpu_mem > _GPU_MEMORY_THRESHOLD:
                    clear_gpu_memory()
        
        # 2. Execute operation
        _active_operations += 1
        try:
            result = func(*args, **kwargs)
            
            # 3. Post-operation cleanup if we're the outermost call
            if _active_operations == 1:
                current_mem = psutil.virtual_memory().percent
                if current_mem > _EMERGENCY_MEMORY_THRESHOLD:
                    logging.critical(f"Post-op memory critical ({current_mem}%)")
                    force_clear_text_cache()
                    clear_gpu_memory()
            
            return result
        finally:
            _active_operations -= 1
    return wrapper

# ----------------- Monitoring Endpoint -----------------
def get_cache_status():
    """Returns current cache and memory status"""
    cache_info = encode_text.cache_info()
    mem = psutil.virtual_memory()
    
    status = {
        'memory': {
            'total_GB': round(mem.total / (1024**3), 1),
            'used_GB': round(mem.used / (1024**3), 1),
            'percent': mem.percent,
            'threshold': _MEMORY_THRESHOLD
        },
        'gpu': {
            'allocated_GB': round(torch.cuda.memory_allocated() / (1024**3), 2) 
                          if torch.cuda.is_available() else None,
            'threshold_GB': _GPU_MEMORY_THRESHOLD
        },
        'text_cache': {
            'current_size': cache_info.currsize,
            'max_size': _TEXT_CACHE_MAX_SIZE,
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'hit_rate': cache_info.hits / max(1, (cache_info.hits + cache_info.misses)),
            'last_cleared': _TEXT_CACHE_LAST_CLEARED.isoformat(),
            'next_clear': (_TEXT_CACHE_LAST_CLEARED + _TEXT_CACHE_TTL).isoformat()
        },
        'safety': {
            'active_operations': _active_operations,
            'process_lock_locked': _process_lock.locked()
        }
    }
    return status

#----------------------Story Board---------------------
from gradio_client import Client, handle_file
import threading

GRADIO_URL = os.getenv("GRADIO_URL", "http://127.0.0.1:7860")  # or your host

# Storyboard API back-pressure: shared across ALL segment worker processes via
# global_render_scheduler (Manager semaphore). Fallback threading sem for legacy paths.
STORYBOARD_MAX_CONCURRENT = max(1, int(os.getenv("STORYBOARD_MAX_CONCURRENT", "5")))
# Must stay BELOW SEGMENT_WALL_TIMEOUT_SEC (default 420). Otherwise the pool
# kills/requeues the segment before storyboard can timeout and fall back to Ken Burns.
STORYBOARD_TIMEOUT_SEC = float(os.getenv("STORYBOARD_TIMEOUT_SEC", "120"))
STORYBOARD_SLOT_TIMEOUT_SEC = float(
    os.getenv("STORYBOARD_SLOT_TIMEOUT_SEC", str(max(60, STORYBOARD_TIMEOUT_SEC + 30)))
)
MEDIA_LOCK_TIMEOUT_SEC = int(os.getenv("MEDIA_LOCK_TIMEOUT_SEC", "120"))
_storyboard_sem_fallback = threading.Semaphore(STORYBOARD_MAX_CONCURRENT)
_storyboard_sem_holder: dict = {"sem": None}


def _storyboard_tuning(segment_fps: int = 30) -> dict:
    """Env-driven storyboard speed/quality knobs (P1)."""
    fps_env = os.getenv("STORYBOARD_FPS")
    return {
        "split_len": int(os.getenv("STORYBOARD_SPLIT_LEN", "20")),
        "fps": int(fps_env) if fps_env else int(segment_fps),
        "obj_skip": int(os.getenv("STORYBOARD_OBJ_SKIP", "12")),
        "bg_skip": int(os.getenv("STORYBOARD_BG_SKIP", "20")),
        "hold_fraction": float(os.getenv("STORYBOARD_HOLD_FRACTION", "0.25")),
    }


def _register_storyboard_semaphore(sem) -> None:
    _storyboard_sem_holder["sem"] = sem


def _acquire_storyboard_slot(label: str = "") -> None:
    sem = _storyboard_sem_holder.get("sem") or _storyboard_sem_fallback
    if label:
        logging.info(f"⏳ Storyboard: waiting for slot ({label})...")
    if not sem.acquire(timeout=STORYBOARD_SLOT_TIMEOUT_SEC):
        raise TimeoutError(
            f"Storyboard slot unavailable after {STORYBOARD_SLOT_TIMEOUT_SEC:.0f}s ({label}) — "
            f"a worker may have died while holding the slot; restart wsgi.py"
        )
    if label:
        logging.info(f"🎨 Storyboard: slot acquired ({label})")


def _release_storyboard_slot(label: str = "") -> None:
    sem = _storyboard_sem_holder.get("sem") or _storyboard_sem_fallback
    sem.release()
    if label:
        logging.info(f"🎨 Storyboard: slot released ({label})")


@contextmanager
def media_lock_guard(media_lock, label: str = ""):
    """Timeout wrapper — avoids permanent deadlock if a worker dies while holding Manager lock."""
    if label:
        logging.debug(f"⏳ media_lock wait ({label})...")
    if not media_lock.acquire(timeout=MEDIA_LOCK_TIMEOUT_SEC):
        raise TimeoutError(
            f"media_lock timeout after {MEDIA_LOCK_TIMEOUT_SEC}s ({label}) — "
            f"a worker likely died while holding the lock; restart wsgi.py"
        )
    try:
        yield
    finally:
        media_lock.release()


# -------------------------------------------
# GLOBAL CLIENT POOL (thread-safe)
# -------------------------------------------

MAX_CLIENTS = STORYBOARD_MAX_CONCURRENT  # aligned with storyboard capacity
_executor = ThreadPoolExecutor(max_workers=MAX_CLIENTS)
_client_lock = threading.Lock()
_clients = []            # pool of persistent clients


def _get_client():
    """Return a reusable Gradio client (thread-safe)."""
    with _client_lock:
        if _clients:
            return _clients.pop()
        return Client(GRADIO_URL)


def _return_client(c):
    """Return a client to pool for reuse."""
    with _client_lock:
        _clients.append(c)


# -------------------------------------------
# MAIN PARALLEL CALLER
# -------------------------------------------

def storyboard_from_api_parallel(
    image_path: str,
    split_len="10",
    fps=30,
    object_skip_rate=8,
    bg_object_skip_rate=14,
    end_gray_img_duration_in_sec=0,
    target_dur_sec=None,
    hold_fraction=0.25,
):
    """
    Submit ONE storyboard job to the thread pool.
    Returns a Future that resolves to a video path string.
    Slot is held only until timeout or completion (no leak on future.cancel()).
    """

    def _task():
        return _storyboard_call_sync(
            image_path=image_path,
            split_len=split_len,
            fps=fps,
            object_skip_rate=object_skip_rate,
            bg_object_skip_rate=bg_object_skip_rate,
            end_gray_img_duration_in_sec=end_gray_img_duration_in_sec,
            target_dur_sec=target_dur_sec,
            hold_fraction=hold_fraction,
        )

    return _executor.submit(_task)


def _storyboard_call_sync(
    image_path: str,
    split_len="10",
    fps=30,
    object_skip_rate=8,
    bg_object_skip_rate=14,
    end_gray_img_duration_in_sec=0,
    target_dur_sec=None,
    hold_fraction=0.25,
    timeout_sec: float | None = None,
):
    """
    Call Gradio /main_func with caller-side slot + wall timeout.
    On timeout the slot is released immediately; orphan Gradio work may continue
    in a daemon thread (server-side concurrency still caps parallel renders).
    """
    label = os.path.basename(str(image_path)) if image_path else "storyboard"
    wait_sec = STORYBOARD_TIMEOUT_SEC if timeout_sec is None else float(timeout_sec)
    result_box: list = []
    error_box: list = []
    done = threading.Event()

    def _run():
        client = None
        try:
            client = _get_client()
            res = client.predict(
                image=handle_file(image_path),
                split_len=split_len,
                frame_rate=fps,
                object_skip_rate=object_skip_rate,
                bg_object_skip_rate=bg_object_skip_rate,
                end_gray_img_duration_in_sec=end_gray_img_duration_in_sec,
                target_dur_sec=target_dur_sec,
                hold_fraction=hold_fraction,
                api_name="/main_func",
            )
            if isinstance(res, dict) and "video" in res:
                result_box.append(res["video"])
            else:
                result_box.append(res)
        except Exception as exc:
            error_box.append(exc)
        finally:
            if client is not None:
                _return_client(client)
            done.set()

    _acquire_storyboard_slot(label)
    try:
        t = threading.Thread(target=_run, daemon=True, name=f"storyboard-{label}")
        t.start()
        if not done.wait(timeout=wait_sec):
            logging.warning(
                f"Storyboard timed out after {wait_sec:.0f}s ({label}) — "
                f"slot released; orphan Gradio call may still finish in background"
            )
            raise TimeoutError(
                f"Storyboard API timed out after {wait_sec:.0f}s "
                f"(max_concurrent={STORYBOARD_MAX_CONCURRENT})"
            )
        if error_box:
            raise error_box[0]
        if not result_box:
            raise RuntimeError(f"Storyboard returned no result ({label})")
        return result_box[0]
    finally:
        _release_storyboard_slot(label)



def create_text_segment_with_lines(
    text,
    duration,
    canvas_w,
    canvas_h,
    bg_color=(178, 34, 34),  # Red background
    text_color=(255, 255, 255),  # White text
    fontsize=None,
    fade_duration=0.8,
    font_path=None  # Optional: path to custom .ttf font
):
    """
    Creates a video segment with text on colored background using PIL (no ImageMagick needed).
    Text fades in smoothly with 2 horizontal + 2 vertical divider lines.
    """
    # Auto-calculate font size based on canvas size and text length
    if fontsize is None:
        text_length = len(text)
        if text_length < 30:
            fontsize = int(canvas_h * 0.08)
        elif text_length < 60:
            fontsize = int(canvas_h * 0.06)
        else:
            fontsize = int(canvas_h * 0.05)
    
    # Load font (try custom first, then fallback to default)
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, fontsize)
        else:
            # Try Impact font first (bold version)
            font_options = [
                "C:/Windows/Fonts/impact.ttf",    # Impact (bold by default)
                "C:/Windows/Fonts/IMPACT.TTF",    # Alternative capitalization
                "C:/Windows/Fonts/arialbd.ttf",   # Fallback: Arial Bold
                "C:/Windows/Fonts/calibrib.ttf",  # Fallback: Calibri Bold
            ]
            font = None
            for font_file in font_options:
                if os.path.exists(font_file):
                    font = ImageFont.truetype(font_file, fontsize)
                    logging.info(f"✅ Loaded font: {font_file}")
                    break
            
            if font is None:
                # Fallback to default PIL font
                font = ImageFont.load_default()
                logging.warning("⚠️ Using default PIL font - text may be small")
    except Exception as e:
        logging.warning(f"⚠️ Font loading failed: {e}, using default")
        font = ImageFont.load_default()
    
    # Create base image with background
    img = Image.new('RGB', (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Line settings
    line_thickness = max(3, int(canvas_h * 0.004))
    line_color = (255, 255, 255)
    
    # ========================================
    # HORIZONTAL LINES (2)
    # ========================================
    # Top line (30% from top)
    top_line_y = int(canvas_h * 0.30)
    draw.rectangle(
        [(0, top_line_y), (canvas_w, top_line_y + line_thickness)],
        fill=line_color
    )
    
    # Bottom line (70% from top)
    bottom_line_y = int(canvas_h * 0.70)
    draw.rectangle(
        [(0, bottom_line_y), (canvas_w, bottom_line_y + line_thickness)],
        fill=line_color
    )
    
    # ========================================
    # VERTICAL LINES (2)
    # ========================================
    # Left line (10% from left)
    left_line_x = int(canvas_w * 0.10)
    draw.rectangle(
        [(left_line_x, 0), (left_line_x + line_thickness, canvas_h)],
        fill=line_color
    )
    
    # Right line (90% from left)
    right_line_x = int(canvas_w * 0.90)
    draw.rectangle(
        [(right_line_x, 0), (right_line_x + line_thickness, canvas_h)],
        fill=line_color
    )
    
    # Wrap text to fit within 85% of canvas width
    max_width = int(canvas_w * 0.85)
    
    # Function to get text width
    def get_text_width(text_line):
        bbox = draw.textbbox((0, 0), text_line, font=font)
        return bbox[2] - bbox[0]
    
    # Wrap text manually
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if get_text_width(test_line) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Calculate total text height
    line_height = int(fontsize * 1.3)  # Line spacing
    total_text_height = len(lines) * line_height
    
    # Center text vertically
    start_y = (canvas_h - total_text_height) // 2
    
    # Draw each line of text (NO SHADOW)
    for i, line in enumerate(lines):
        # Get text width for centering
        text_width = get_text_width(line)
        x = (canvas_w - text_width) // 2
        y = start_y + (i * line_height)
        
        # Draw text directly without shadow
        draw.text((x, y), line, font=font, fill=text_color)
    
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Create ImageClip from numpy array
    image_clip = ImageClip(img_array).set_duration(duration)
    
    # Apply fade-in effect
    image_clip = image_clip.crossfadein(fade_duration)
    
    return image_clip


# def storyboard_from_api(
#     image_path: str,
#     split_len: str = "10",      # "5" | "10" | "20"
#     fps: int = 30,
#     object_skip_rate: int = 8,
#     bg_object_skip_rate: int = 14,
#     end_gray_img_duration_in_sec: int = 0,
#     target_dur_sec: float | None = None,   # ✅ NEW
# ) -> str:
#     """
#     Call your Gradio app and (optionally) pass a target duration so the server fits timing
#     without any client-side trimming.
#     """
#     client = Client(GRADIO_URL)
#     res = client.predict(
#         image=handle_file(image_path),
#         split_len=split_len,
#         frame_rate=fps,
#         object_skip_rate=object_skip_rate,
#         bg_object_skip_rate=bg_object_skip_rate,
#         end_gray_img_duration_in_sec=end_gray_img_duration_in_sec,
#         target_dur_sec=target_dur_sec,            # ✅ send it along
#         api_name="/main_func",                    # your existing endpoint name
#     )
#     return res["video"] if isinstance(res, dict) and "video" in res else res


def storyboard_from_api(
    image_path: str,
    split_len: str = "10",
    fps: int = 30,
    object_skip_rate: int = 8,
    bg_object_skip_rate: int = 14,
    end_gray_img_duration_in_sec: int = 0,
    target_dur_sec: float | None = None,
    hold_fraction: float = 0.25,
):
    """Blocking storyboard call with timeout-safe slot release."""
    return _storyboard_call_sync(
        image_path=image_path,
        split_len=split_len,
        fps=fps,
        object_skip_rate=object_skip_rate,
        bg_object_skip_rate=bg_object_skip_rate,
        end_gray_img_duration_in_sec=end_gray_img_duration_in_sec,
        target_dur_sec=target_dur_sec,
        hold_fraction=hold_fraction,
    )


#----------------------Story Boar d Close---------------------

# ----------------- Configuration -----------------
DEFAULT_DURATION = 5  # Default duration for image clips
MATCH_THRESHOLD = 0.2  # Threshold for media matching
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 20)) # Set via environment variable
TEMP_DIR ="temp_video_data_storage"

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO
)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ----------------- Model Initialization -----------------
# logging.info("Initializing models...")
# # Initialize CLIP
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# clip_model.eval()

# ─────────────────────────────────────────────
# Lazy CLIP singleton
# ─────────────────────────────────────────────
_clip_model = None
_clip_processor = None
_clip_lock = threading.Lock()

def get_clip_model():
    global _clip_model, _clip_processor

    if _clip_model is None:
        with _clip_lock:
            if _clip_model is None:
                logging.info("🔧 Initializing CLIP model (first use)...")
                from transformers import CLIPModel, CLIPProcessor
                _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                _clip_model.eval()
                logging.info("✅ CLIP model ready")

    return _clip_model, _clip_processor

_db = None


def _mongo_instance():
    """Defer mongodb import so Atlas blips during worker exec_module do not abort load."""
    import mongodb

    inst = getattr(mongodb, "mongo_instance", None)
    if inst is None:
        raise RuntimeError(
            "mongodb.mongo_instance is missing — MongoDB failed to initialize in mongodb.py "
            "(check Atlas primary / connection string)"
        )
    return inst


def get_db():
    """Lazy Mongo handle — avoid get_db() on import (segment pool exec_module)."""
    global _db
    if _db is None:
        _db = _mongo_instance().get_db()
    return _db


class _LazyDB:
    """Legacy `db` attribute access without connecting at import time."""

    def __getattr__(self, name):
        return getattr(get_db(), name)


db = _LazyDB()
# Ensure the directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

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

    elif d['status'] == 'finished':
        logging.info("✅ Download complete!")

    elif d['status'] == 'error':
        logging.error("❌ Download failed.")
        

STORYBOARD_TRANSITION_DUR = 0.25

def apply_storyboard_transitions(clips, transition_dur=STORYBOARD_TRANSITION_DUR, push_strength=0.06):
    """
    Adds storyboard-friendly transitions with proper crossfades.
    """
    if not clips:
        return None
    
    # 1) Add subtle camera push-in to each clip
    pushed = []
    for c in clips:
        dur = max(0.001, float(c.duration or 0.001))
        pushed.append(
            c.fx(vfx.resize, lambda t, d=dur: 1.0 + push_strength * (t / d))
        )
    
    # 2) Single clip - no transitions needed
    if len(pushed) == 1:
        return pushed[0]
    
    # 3) Apply crossfade transitions
    result = [pushed[0]]
    for c in pushed[1:]:
        fade_dur = min(transition_dur, c.duration / 3, pushed[0].duration / 3)
        result.append(c.crossfadein(fade_dur))
    
    # Use negative padding equal to fade duration to create overlap
    return concatenate_videoclips(result, method="compose", padding=-transition_dur)     


#I---------------------VIDEO DOWNLOAD FUNCTIONS START---------------------I

# def crop_with_audio(input_file, output_file, crop_spec):
#     """
#     Crops a video with the specified crop parameters while preserving audio,
#     auto-scaling the crop to the actual input resolution.
#     """
#     try:
#         # Get actual input resolution
#         probe = ffmpeg.probe(input_file)
#         stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
#         in_w, in_h = stream['width'], stream['height']

#         # Use expected reference resolution
#         expected = crop_spec.get("expectedInputResolution", {"width": 1920, "height": 1080})
#         scale_x = in_w / expected["width"]
#         scale_y = in_h / expected["height"]

#         # Apply scaling
#         x = int(crop_spec["x"] * scale_x)
#         y = int(crop_spec["y"] * scale_y)
#         w = int(crop_spec["width"] * scale_x)
#         h = int(crop_spec["height"] * scale_y)

#         # Validate crop bounds
#         w = min(w, in_w - x)
#         h = min(h, in_h - y)
#         if w <= 0 or h <= 0:
#             raise ValueError("Invalid crop bounds after scaling.")

#         video = ffmpeg.input(input_file)
#         cropped = video.video.crop(x, y, w, h)
#         ffmpeg.output(cropped, video.audio, output_file, vcodec='libx264', acodec='aac').run(overwrite_output=True)

#         logging.info(f"✂️ Cropping completed: {output_file}")
#         return True
#     except Exception as e:
#         logging.error(f"❌ Cropping failed: {e}")
#         return False

# import os
# import uuid
# import time
# import logging
# import subprocess

# import ffmpeg
# import yt_dlp


# def probe_video_size(input_file):
#     """Return (width, height) of the first video stream."""
#     probe = ffmpeg.probe(input_file)
#     stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
#     return int(stream["width"]), int(stream["height"])


# def scale_crop_to_input(crop_spec, input_w, input_h):
#     """
#     Scale crop coordinates from UI reference resolution to actual file pixels.
#     crop_spec must contain x, y, width, height.
#     expectedInputResolution is optional but strongly recommended from frontend.
#     """
#     expected = crop_spec.get("expectedInputResolution") or {}
#     ref_w = int(expected.get("width") or input_w)
#     ref_h = int(expected.get("height") or input_h)

#     if ref_w <= 0 or ref_h <= 0:
#         ref_w, ref_h = input_w, input_h

#     scale_x = input_w / ref_w
#     scale_y = input_h / ref_h

#     x = int(crop_spec["x"] * scale_x)
#     y = int(crop_spec["y"] * scale_y)
#     w = int(crop_spec["width"] * scale_x)
#     h = int(crop_spec["height"] * scale_y)

#     # Even dimensions for H.264
#     if w % 2 != 0:
#         w -= 1
#     if h % 2 != 0:
#         h -= 1

#     # Clamp to bounds
#     x = max(0, min(x, input_w - 2))
#     y = max(0, min(y, input_h - 2))
#     w = min(w, input_w - x)
#     h = min(h, input_h - y)

#     if w <= 0 or h <= 0:
#         raise ValueError(
#             f"Invalid crop after scaling: x={x}, y={y}, w={w}, h={h}, "
#             f"input={input_w}x{input_h}, ref={ref_w}x{ref_h}"
#         )

#     return x, y, w, h


# def crop_with_audio(input_file, output_file, crop_spec):
#     """
#     Crops a video with the specified crop parameters while preserving audio.
#     Auto-scales crop coords from expectedInputResolution to actual input size.
#     """
#     try:
#         in_w, in_h = probe_video_size(input_file)
#         x, y, w, h = scale_crop_to_input(crop_spec, in_w, in_h)

#         logging.info(
#             f"crop_with_audio: ref={crop_spec.get('expectedInputResolution')} "
#             f"actual={in_w}x{in_h} → crop={x},{y},{w}x{h}"
#         )

#         video = ffmpeg.input(input_file)
#         cropped = video.video.crop(x, y, w, h)

#         ffmpeg.output(
#             cropped,
#             video.audio,
#             output_file,
#             vcodec="libx264",
#             acodec="aac",
#         ).run(overwrite_output=True)

#         logging.info(f"✂️ Cropping completed: {output_file}")
#         return True

#     except Exception as e:
#         logging.error(f"❌ Cropping failed: {e}")
#         return False


# def download_video_segment(
#     user_id,
#     url,
#     output_path,
#     start_time,
#     end_time,
#     crop=None,
#     effect=None,
#     retries=3,
# ):
#     """
#     Downloads a video segment from a URL and applies FFmpeg manipulation
#     with optional effects.

#     Args:
#         user_id: User identifier for logo selection
#         url: Video URL to download
#         output_path: Final output file path
#         start_time: Start time in seconds
#         end_time: End time in seconds
#         crop: {
#             "x": int, "y": int, "width": int, "height": int,
#             "expectedInputResolution": {"width": int, "height": int}  # from frontend
#         } or None
#         effect: "zoom", "mirror", "zoom+mirror", or None
#         retries: Number of retry attempts

#     Returns:
#         output_path if successful, None if failed
#     """

#     if start_time is None or end_time is None or end_time <= start_time:
#         logging.warning(
#             f"Invalid time range: start_time={start_time}, end_time={end_time}. Skipping download."
#         )
#         return None

#     temp_raw_path = f"{output_path}.{uuid.uuid4()}.raw.mp4"
#     temp_crop_path = f"{output_path}.{uuid.uuid4()}.crop.mp4" if crop else None

#     for attempt in range(retries):
#         try:
#             # -------------------------
#             # YT-DLP DOWNLOAD
#             # -------------------------
#             ydl_opts = {
#                 "format": (
#                     "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
#                     "bestvideo[height<=1080]+bestaudio/best"
#                 ),
#                 "outtmpl": temp_raw_path,
#                 "merge_output_format": "mp4",
#                 "quiet": False,
#                 "progress_hooks": [progress_hook],
#                 "download_ranges": lambda _, __: [
#                     {"start_time": start_time, "end_time": end_time}
#                 ],
#                 "force_keyframes_at_cuts": True,
#                 "retries": retries,
#                 "extractor_args": {
#                     "youtube": {
#                         "player_client": ["default", "web", "android"]
#                     }
#                 },
#             }

#             logging.info(f"📥 Downloading segment from {start_time}s to {end_time}s...")

#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 ydl.download([url])

#             if not (os.path.exists(temp_raw_path) and os.path.getsize(temp_raw_path) > 0):
#                 raise Exception(f"Download failed: Empty or missing file at {temp_raw_path}")

#             input_path = temp_raw_path

#             # -------------------------
#             # OPTIONAL CROPPING (scaled to actual download resolution)
#             # -------------------------
#             if crop and isinstance(crop, dict) and all(
#                 k in crop for k in ["x", "y", "width", "height"]
#             ):
#                 logging.info("✂️ Applying custom crop...")

#                 in_w, in_h = probe_video_size(temp_raw_path)
#                 x, y, w, h = scale_crop_to_input(crop, in_w, in_h)

#                 logging.info(
#                     f"download_video_segment crop: ref={crop.get('expectedInputResolution')} "
#                     f"actual={in_w}x{in_h} → crop={x},{y},{w}x{h}"
#                 )

#                 crop_cmd = [
#                     "ffmpeg", "-y",
#                     "-i", temp_raw_path,
#                     "-filter:v", f"crop={w}:{h}:{x}:{y}",
#                     "-c:a", "copy",
#                     temp_crop_path,
#                 ]

#                 result = subprocess.run(crop_cmd, capture_output=True, text=True)
#                 if result.returncode != 0:
#                     logging.error(f"Crop FFmpeg error: {result.stderr}")
#                     raise Exception("Cropping failed")

#                 if not os.path.exists(temp_crop_path) or os.path.getsize(temp_crop_path) == 0:
#                     raise Exception("Cropping failed: output file missing or empty")

#                 input_path = temp_crop_path
#                 logging.info("✅ Crop applied successfully")

#             # -------------------------
#             # LOGO SELECTION
#             # -------------------------
#             logo_path = None
#             if user_id == "admin@example.com":
#                 logo_path = "branding/admin_logo.png"
#             elif user_id and user_id.endswith("@company.com"):
#                 logo_path = "branding/company_logo.png"

#             if logo_path and not os.path.exists(logo_path):
#                 logging.warning(f"Logo path {logo_path} not found, skipping logo overlay")
#                 logo_path = None

#             # -------------------------
#             # BUILD EFFECT FILTERS
#             # -------------------------
#             effect_filters = []

#             if effect == "zoom":
#                 effect_filters.append("scale=1.3*iw:1.3*ih")
#             elif effect == "mirror":
#                 effect_filters.append("hflip")
#             elif effect == "zoom+mirror":
#                 effect_filters.append("scale=1.3*iw:1.3*ih")
#                 effect_filters.append("hflip")

#             if effect_filters:
#                 effect_stage = f"[main]{','.join(effect_filters)}[processed];"
#             else:
#                 effect_stage = "[main]null[processed];"

#             # Only auto 9:16 crop when user did NOT provide custom crop
#             auto_crop_filter = "" if crop else "crop=h=iw*9/16,"

#             base_filter_graph = (
#                 "[0:v]split=2[main][blur];"
#                 "[blur]scale=1920:1080:force_original_aspect_ratio=increase,"
#                 "crop=1920:1080,"
#                 "boxblur=luma_radius=min(h\\,w)/20:luma_power=1:"
#                 "chroma_radius=min(cw\\,ch)/20:chroma_power=1[bg];"
#                 f"{effect_stage}"
#                 "[processed]scale=1920:1080:force_original_aspect_ratio=decrease[scaled];"
#                 "[bg][scaled]overlay=(W-w)/2:(H-h)/2,"
#                 f"{auto_crop_filter}"
#                 "eq=contrast=1.05:saturation=1.15:brightness=0.01[video]"
#             )

#             if logo_path:
#                 filter_graph = base_filter_graph + ";[video][1:v]overlay=W-w-10:H-h-10[final]"
#                 map_output = "[final]"
#             else:
#                 filter_graph = base_filter_graph
#                 map_output = "[video]"

#             command = ["ffmpeg", "-y", "-i", input_path]

#             if logo_path:
#                 command += ["-i", logo_path]

#             command += [
#                 "-filter_complex", filter_graph,
#                 "-map", map_output,
#                 "-map", "0:a?",
#                 "-c:v", "h264_nvenc",
#                 "-preset", "p5",
#                 "-tune", "hq",
#                 "-rc", "vbr",
#                 "-b:v", "8M",
#                 "-maxrate", "12M",
#                 "-bufsize", "24M",
#                 "-profile:v", "high",
#                 "-pix_fmt", "yuv420p",
#                 "-colorspace", "bt709",
#                 "-color_primaries", "bt709",
#                 "-color_trc", "bt709",
#                 "-c:a", "aac",
#                 "-b:a", "192k",
#                 "-movflags", "+faststart",
#                 "-shortest",
#                 output_path,
#             ]

#             logging.info("🎬 Processing video with FFmpeg...")

#             result = subprocess.run(command, capture_output=True, text=True)

#             if result.returncode != 0:
#                 logging.error(f"FFmpeg error: {result.stderr}")
#                 raise Exception(f"FFmpeg processing failed with code {result.returncode}")

#             for path in [temp_raw_path, temp_crop_path]:
#                 if path and os.path.exists(path):
#                     try:
#                         os.remove(path)
#                     except Exception as e:
#                         logging.warning(f"Failed to remove {path}: {e}")

#             if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#                 logging.info(f"✅ Video segment created successfully: {output_path}")
#                 return output_path

#             raise Exception(f"Final output file is empty or missing: {output_path}")

#         except Exception as e:
#             logging.error(f"❌ Attempt {attempt + 1}/{retries} failed: {e}")

#             for path in [temp_raw_path, temp_crop_path, output_path]:
#                 if path and os.path.exists(path):
#                     try:
#                         os.remove(path)
#                     except Exception:
#                         pass

#             if attempt < retries - 1:
#                 wait_time = 2 ** attempt
#                 logging.info(f"⏳ Retrying in {wait_time} seconds...")
#                 time.sleep(wait_time)
#             else:
#                 logging.error(f"❌ All {retries} attempts failed for {url}")

#     return None

# def download_video_segment(user_id, url, output_path, start_time, end_time,
#                            crop=None, effect=None, retries=3):
#     """
#     Downloads a video segment from a URL and applies FFmpeg manipulation with optional effects.
    
#     Args:
#         user_id: User identifier for logo selection
#         url: Video URL to download
#         output_path: Final output file path
#         start_time: Start time in seconds
#         end_time: End time in seconds
#         crop: Dictionary containing {"x": int, "y": int, "width": int, "height": int} or None
#         effect: Effect to apply - "zoom", "mirror", "zoom+mirror", or None
#         retries: Number of retry attempts
    
#     Returns:
#         output_path if successful, None if failed
#     """

#     if start_time is None or end_time is None or end_time <= start_time:
#         logging.warning(
#             f"Invalid time range: start_time={start_time}, end_time={end_time}. Skipping download.")
#         return None

#     temp_raw_path = f"{output_path}.{uuid.uuid4()}.raw.mp4"
#     temp_crop_path = f"{output_path}.{uuid.uuid4()}.crop.mp4" if crop else None

#     for attempt in range(retries):
#         try:
#             # -------------------------
#             # YT-DLP DOWNLOAD
#             # -------------------------
#             ydl_opts = {
#                 'format': 'bestvideo+bestaudio/best',
#                 'outtmpl': temp_raw_path,
#                 'merge_output_format': 'mp4',
#                 'quiet': False,
#                 'progress_hooks': [progress_hook],
#                 'download_ranges': lambda _, __: [
#                     {'start_time': start_time, 'end_time': end_time}
#                 ],
#                 'force_keyframes_at_cuts': True,
#                 'retries': retries,

#                 # Critical Fix for 2026 yt-dlp
#                 'extractor_args': {
#                     'youtube': {
#                         'player_client': ['default', 'web', 'android']
#                     }
#                 }
#             }

#             logging.info(f"📥 Downloading segment from {start_time}s to {end_time}s...")
            
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 ydl.download([url])

#             if not (os.path.exists(temp_raw_path) and os.path.getsize(temp_raw_path) > 0):
#                 raise Exception(
#                     f"Download failed: Empty or missing file at {temp_raw_path}")

#             input_path = temp_raw_path

#             # -------------------------
#             # OPTIONAL CROPPING
#             # -------------------------
#             if crop and isinstance(crop, dict) and all(k in crop for k in ["x", "y", "width", "height"]):
#                 logging.info("✂️  Applying custom crop...")

#                 crop_cmd = [
#                     "ffmpeg", "-y",
#                     "-i", temp_raw_path,
#                     "-filter:v",
#                     f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']}",
#                     "-c:a", "copy",
#                     temp_crop_path
#                 ]

#                 subprocess.run(crop_cmd, check=True, capture_output=True)

#                 if not os.path.exists(temp_crop_path) or os.path.getsize(temp_crop_path) == 0:
#                     raise Exception("Cropping failed: output file missing or empty")

#                 input_path = temp_crop_path
#                 logging.info("✅ Crop applied successfully")

#             # -------------------------
#             # LOGO SELECTION
#             # -------------------------
#             logo_path = None
#             if user_id == "admin@example.com":
#                 logo_path = "branding/admin_logo.png"
#             elif user_id and user_id.endswith("@company.com"):
#                 logo_path = "branding/company_logo.png"
            
#             # Verify logo exists if specified
#             if logo_path and not os.path.exists(logo_path):
#                 logging.warning(f"Logo path {logo_path} not found, skipping logo overlay")
#                 logo_path = None

#             # -------------------------
#             # BUILD EFFECT FILTERS
#             # -------------------------
#             effect_filters = []
            
#             if effect == "zoom":
#                 effect_filters.append("scale=1.3*iw:1.3*ih")
#             elif effect == "mirror":
#                 effect_filters.append("hflip")
#             elif effect == "zoom+mirror":
#                 effect_filters.append("scale=1.3*iw:1.3*ih")
#                 effect_filters.append("hflip")
            
#             # Build the effect stage with proper syntax
#             if effect_filters:
#                 effect_stage = f"[main]{','.join(effect_filters)}[processed];"
#             else:
#                 effect_stage = "[main]null[processed];"

#             # -------------------------
#             # BUILD AUTO-CROP FILTER
#             # -------------------------
#             # Only apply automatic 9:16 crop if user didn't provide custom crop
#             auto_crop_filter = "" if crop else "crop=h=iw*9/16,"

#             # -------------------------
#             # BUILD COMPLETE FILTER GRAPH
#             # Blur background always fills 1920x1080.
#             # Processed content is scaled to fit within canvas (no stretch).
#             # Overlaid centered on blur background for professional feel.
#             # Output is always exactly 1920x1080 regardless of crop or source.
#             # -------------------------
#             base_filter_graph = (
#                 "[0:v]split=2[main][blur];"
#                 "[blur]scale=1920:1080:force_original_aspect_ratio=increase,"
#                 "crop=1920:1080,"
#                 "boxblur=luma_radius=min(h\\,w)/20:luma_power=1:"
#                 "chroma_radius=min(cw\\,ch)/20:chroma_power=1[bg];"
#                 f"{effect_stage}"
#                 "[processed]scale=1920:1080:force_original_aspect_ratio=decrease[scaled];"
#                 "[bg][scaled]overlay=(W-w)/2:(H-h)/2,"
#                 f"{auto_crop_filter}"
#                 "eq=contrast=1.05:saturation=1.15:brightness=0.01[video]"
#             )

#             # Add logo overlay if present
#             if logo_path:
#                 filter_graph = base_filter_graph + ";[video][1:v]overlay=W-w-10:H-h-10[final]"
#                 map_output = "[final]"
#             else:
#                 filter_graph = base_filter_graph
#                 map_output = "[video]"

#             # -------------------------
#             # BUILD FFMPEG COMMAND
#             # -------------------------
#             command = ["ffmpeg", "-y", "-i", input_path]

#             if logo_path:
#                 command += ["-i", logo_path]

#             command += [
#                 "-filter_complex", filter_graph,
#                 "-map", map_output,
#                 "-map", "0:a?",
                
#                 # Video encoding settings
#                 "-c:v", "h264_nvenc",
#                 "-preset", "p5",
#                 "-tune", "hq",
#                 "-rc", "vbr",
#                 "-b:v", "8M",
#                 "-maxrate", "12M",
#                 "-bufsize", "24M",
#                 "-profile:v", "high",
#                 "-pix_fmt", "yuv420p",
                
#                 # Color settings
#                 "-colorspace", "bt709",
#                 "-color_primaries", "bt709",
#                 "-color_trc", "bt709",
                
#                 # Audio encoding settings
#                 "-c:a", "aac",
#                 "-b:a", "192k",
                
#                 # Output settings
#                 "-movflags", "+faststart",
#                 "-shortest",
#                 output_path
#             ]

#             logging.info("🎬 Processing video with FFmpeg...")
            
#             result = subprocess.run(command, capture_output=True, text=True)
            
#             if result.returncode != 0:
#                 logging.error(f"FFmpeg error: {result.stderr}")
#                 raise Exception(f"FFmpeg processing failed with code {result.returncode}")

#             # -------------------------
#             # CLEANUP TEMP FILES
#             # -------------------------
#             for path in [temp_raw_path, temp_crop_path]:
#                 if path and os.path.exists(path):
#                     try:
#                         os.remove(path)
#                         logging.debug(f"Cleaned up: {path}")
#                     except Exception as e:
#                         logging.warning(f"Failed to remove {path}: {e}")

#             # -------------------------
#             # VERIFY OUTPUT
#             # -------------------------
#             if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#                 logging.info(f"✅ Video segment created successfully: {output_path}")
#                 return output_path
#             else:
#                 raise Exception(
#                     f"Final output file is empty or missing: {output_path}")

#         except Exception as e:
#             logging.error(f"❌ Attempt {attempt + 1}/{retries} failed: {e}")

#             # Cleanup on failure
#             for path in [temp_raw_path, temp_crop_path, output_path]:
#                 if path and os.path.exists(path):
#                     try:
#                         os.remove(path)
#                     except:
#                         pass

#             if attempt < retries - 1:
#                 wait_time = 2 ** attempt
#                 logging.info(f"⏳ Retrying in {wait_time} seconds...")
#                 time.sleep(wait_time)
#             else:
#                 logging.error(f"❌ All {retries} attempts failed for {url}")

#     return None

# import os
# import time
# import logging
# import concurrent.futures

# def download_video_clips_in_parallel(clip_data_list, section_dir,user_id, max_workers=4, max_retries=5):
#     os.makedirs(section_dir, exist_ok=True)
#     downloaded_media = []

#     def task(character_name, yt_url, vidclip):
#         start = time_to_seconds(vidclip.get("startTime", "0:00:00"))
#         end = time_to_seconds(vidclip.get("endTime", "0:00:05"))
#         effect = vidclip.get("effect", None)
#         crop=vidclip.get('crop',None)

#         filename = f"{character_name}_{start}_{end}_{int(time.time())}.mp4"
#         clip_path = os.path.join(section_dir, filename)

#         for attempt in range(1, max_retries + 1):
#             try:
#                 downloaded_path = download_video_segment(
#                     user_id,
#                     yt_url,
#                     clip_path,
#                     start,
#                     end,
#                     crop,
#                     effect
                    
#                 )
#                 if downloaded_path and os.path.exists(downloaded_path):
#                     return {
#                         "character": character_name,
#                         "path": downloaded_path,
#                         "start": start,
#                         "end": end,
#                         "mute": vidclip.get("mute", False),
#                         "tag": vidclip.get("tag", None),
#                         "url": yt_url
#                     }
#                 else:
#                     raise RuntimeError("Downloaded path is invalid or file does not exist.")

#             except Exception as e:
#                 if attempt < max_retries:
#                     wait = 2 ** attempt
#                     logging.warning(f"🔁 Retry {attempt}/{max_retries} for {character_name} - {yt_url}: {e}. Retrying in {wait}s...")
#                     time.sleep(wait)
#                 else:
#                     logging.error(f"❌ Final retry failed for {character_name} - URL: {yt_url} | Error: {e}")
#                     return {
#                         "character": character_name,
#                         "url": yt_url,
#                         "error": str(e)
#                     }

#     futures = []
#     clip_context_map = {}  # for tracking what each future corresponds to

#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for item in clip_data_list:
#             character = item["characterName"]
#             for video in item["videos"]:
#                 yt_url = video["url"]
#                 for clip in video["vidclips"]:
#                     future = executor.submit(task, character, yt_url, clip)
#                     futures.append(future)
#                     clip_context_map[future] = (character, yt_url)

#         completed_count = 0
#         failed_clips = []

#         for future in concurrent.futures.as_completed(futures):
#             character, yt_url = clip_context_map[future]
#             try:
#                 result = future.result(timeout=180)
#                 if result and "path" in result:
#                     downloaded_media.append(result)
#                     completed_count += 1
#                 else:
#                     failed_clips.append(result)
#             except Exception as e:
#                 logging.error(f"⛔ Future task failed for {character} - URL: {yt_url} | Exception: {e}")
#                 failed_clips.append({"character": character, "url": yt_url, "error": str(e)})

#     logging.info(f"✅ Completed {completed_count}/{len(futures)} video clips successfully.")

#     if failed_clips:
#         logging.error(f"❌ {len(failed_clips)} clip(s) failed:")
#         for fail in failed_clips:
#             logging.error(f"   Character: {fail.get('character')} | URL: {fail.get('url')} | Error: {fail.get('error')}")

#     # if completed_count != len(futures):
#     #     raise RuntimeError(f"{len(futures) - completed_count} video segments failed to download after retries.")

#     return downloaded_media


from pathlib import Path
from collections import defaultdict
import hashlib, threading, concurrent.futures, shutil

SOURCE_CACHE_DIR = Path(os.environ.get("VIDEO_SOURCE_CACHE", "cache/source_videos"))
SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LONG_VIDEO_THRESHOLD_SEC = 600
_source_locks, _source_locks_guard = {}, threading.Lock()
# Grok/NVENC segment cuts share one GPU — parallel writes also raced on identical output names.
_grok_nvenc_cut_sem = threading.Semaphore(
    max(1, int(os.environ.get("GROK_CUT_NVENC_PARALLEL", "2")))
)

def _find_js_runtime(name: str, win_fallbacks: tuple[str, ...] = ()) -> str | None:
    """Locate deno/node on PATH or common Windows install dirs (no env vars)."""
    path = shutil.which(name)
    if path:
        return path
    for candidate in win_fallbacks:
        expanded = os.path.expandvars(os.path.expanduser(candidate))
        if os.path.isfile(expanded):
            return expanded
    return None


def _js_runtime_opts() -> dict:
    """
    YouTube now requires an external JS runtime + EJS scripts (yt-dlp 2025+).
    Auto-detect Deno (preferred) or Node 22+ — no env configuration.
    """
    runtimes: dict = {}
    deno = _find_js_runtime(
        "deno",
        (
            r"%USERPROFILE%\.deno\bin\deno.exe",
            r"C:\Program Files\deno\deno.exe",
            r"C:\Program Files (x86)\deno\deno.exe",
        ),
    )
    if deno:
        runtimes["deno"] = {"path": deno}
    node = _find_js_runtime(
        "node",
        (
            r"C:\Program Files\nodejs\node.exe",
            r"C:\Program Files (x86)\nodejs\node.exe",
        ),
    )
    if node:
        runtimes["node"] = {"path": node}
    if not runtimes:
        # yt-dlp default: try deno on PATH
        runtimes["deno"] = {}
        logging.warning(
            "⚠️ No Deno/Node JS runtime found — install Deno for YouTube downloads. "
            "See https://github.com/yt-dlp/yt-dlp/wiki/EJS"
        )
    return runtimes


def _ejs_component_opts() -> dict:
    """Enable EJS challenge solver fetch (GitHub) when yt-dlp-ejs pip package missing."""
    return {"remote_components": {"ejs:github"}}


def _ytdlp_tor_enabled() -> bool:
    return os.environ.get("YTDLP_TOR_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _ytdlp_fallback_proxy() -> str | None:
    """
    Tor/SOCKS proxy used ONLY after a geo-restriction error (not on every download).
    Set YTDLP_TOR_ENABLED=0 or YTDLP_PROXY=none to disable fallback entirely.
    """
    if not _ytdlp_tor_enabled():
        return None
    proxy = os.environ.get("YTDLP_PROXY", "socks5://127.0.0.1:9050").strip()
    if not proxy or proxy.lower() in ("0", "false", "no", "off", "none"):
        return None
    return proxy


def _is_ytdlp_geo_restricted(exc: BaseException) -> bool:
    geo_err = getattr(yt_dlp.utils, "GeoRestrictedError", None)
    if geo_err and isinstance(exc, geo_err):
        return True
    dl_err = getattr(yt_dlp.utils, "DownloadError", None)
    if dl_err and isinstance(exc, dl_err) and getattr(exc, "exc_info", None):
        inner = exc.exc_info[1]
        if geo_err and isinstance(inner, geo_err):
            return True
    msg = str(exc).lower()
    return (
        "not made this video available in your country" in msg
        or "georestricted" in msg
        or "geo restricted" in msg
    )


def _ytdlp_rotate_tor_circuit() -> None:
    """Optional NEWNYM when YTDLP_TOR_NEWNYM=1 and Tor control port is open."""
    if os.environ.get("YTDLP_TOR_NEWNYM", "").strip().lower() not in ("1", "true", "yes"):
        return
    try:
        from stem import Signal
        from stem.control import Controller

        control_port = int(os.environ.get("YTDLP_TOR_CONTROL_PORT", "9051"))
        with Controller.from_port(port=control_port) as controller:
            controller.authenticate()
            controller.signal(Signal.NEWNYM)
        logging.info("🔄 Tor circuit rotated (NEWNYM)")
        time.sleep(5)
    except Exception as e:
        logging.warning(f"⚠️ Tor NEWNYM skipped: {e}")


def _ytdlp_run(url: str, opts: dict, *, download: bool = True):
    """
    Run yt-dlp direct first. On GeoRestrictedError only, retry once via Tor proxy
    (YTDLP_PROXY, default socks5://127.0.0.1:9050). Normal clips never use Tor.
    """
    proxy = _ytdlp_fallback_proxy()

    def _attempt(use_proxy: bool):
        run_opts = dict(opts)
        if use_proxy:
            run_opts["proxy"] = proxy
        else:
            run_opts.pop("proxy", None)
        with yt_dlp.YoutubeDL(run_opts) as ydl:
            # extract_info(download=True) downloads exactly like ydl.download()
            # and also tells us which stream/resolution was picked.
            return ydl.extract_info(url, download=download)

    try:
        return _attempt(use_proxy=False)
    except Exception as e:
        if not _is_ytdlp_geo_restricted(e) or not proxy:
            raise
        logging.warning(
            f"⚠️ Geo-blocked on direct IP — retrying via Tor ({proxy}): {url[:80]}..."
        )
        try:
            return _attempt(use_proxy=True)
        except Exception as e2:
            _ytdlp_rotate_tor_circuit()
            raise e2

def _url_cache_key(url):
    return hashlib.sha256(url.encode()).hexdigest()[:20]

def _lock_for(key):
    with _source_locks_guard:
        return _source_locks.setdefault(key, threading.Lock())

def _base_ydl_opts(outtmpl, retries=3):
    """
    yt-dlp options for YouTube and other extractors.
    Cookies: live Firefox profile. JS: auto-detected Deno/Node + EJS scripts.
    """
    opts = {
        # Best stream up to 1080p (H.264 preferred: fast decode + NVENC friendly).
        # The 360p muxed file is only a last resort — see _ytdlp_download_hd.
        "format": "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
        "format_sort": ["res:1080", "vcodec:h264", "fps", "acodec:m4a"],
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "retries": retries,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 8,
        "noprogress": True,
        "cookiesfrombrowser": ("firefox",),
        "js_runtimes": _js_runtime_opts(),
    }
    opts.update(_ejs_component_opts())
    return opts


# ── HD guard ───────────────────────────────────────────────────────────
# YouTube sometimes withholds the real streams from a player client and only
# offers the 360p muxed file. Each attempt below is tried until one gives
# >= YTDLP_MIN_HEIGHT (default 720). (None = yt-dlp's own client choice.)
_YT_CLIENT_ATTEMPTS = (
    (None, True),
    (["web", "mweb", "default"], True),
    (["tv"], True),
    (["web_safari"], True),
    (["android_vr"], False),   # android_vr does not support cookies
)
_YT_URL_RE = re.compile(r"(?:youtube\.com|youtu\.be|youtube-nocookie\.com)", re.I)


def _ytdlp_min_height() -> int:
    try:
        return max(0, int(os.environ.get("YTDLP_MIN_HEIGHT", "720")))
    except ValueError:
        return 720


def _ytdlp_picked(info) -> dict:
    info = info or {}
    fmts = info.get("requested_formats") or [info]
    v = next((f for f in fmts if f.get("vcodec") not in (None, "none")), fmts[0] if fmts else {})
    return {
        "format_id": info.get("format_id"),
        "width": v.get("width"),
        "height": v.get("height") or 0,
        "vcodec": v.get("vcodec"),
    }


def _ytdlp_download_hd(url: str, opts: dict, out_path: str) -> dict:
    """
    Download with _ytdlp_run, log the picked resolution, and retry other
    YouTube player clients when the result is below YTDLP_MIN_HEIGHT.
    Keeps the best file if no attempt reaches the target. Returns picked info.
    """
    min_h = _ytdlp_min_height()
    attempts = _YT_CLIENT_ATTEMPTS if _YT_URL_RE.search(url or "") else ((None, True),)
    best = None  # (height, path, picked, label)
    last_err = None
    for i, (clients, use_cookies) in enumerate(attempts):
        run_opts = dict(opts)
        if clients:
            run_opts["extractor_args"] = {"youtube": {"player_client": clients}}
        else:
            run_opts.pop("extractor_args", None)
        if not use_cookies:
            run_opts.pop("cookiesfrombrowser", None)
            run_opts.pop("cookiefile", None)
        label = ("+".join(clients) if clients else "yt-dlp default") + ("" if use_cookies else " (no cookies)")
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        try:
            info = _ytdlp_run(url, run_opts, download=True)
        except Exception as e:
            last_err = e
            logging.warning(f"⚠️ yt-dlp [{label}] failed: {str(e).splitlines()[0][:160]}")
            continue
        if not os.path.exists(out_path):
            last_err = RuntimeError("yt-dlp produced no file")
            continue
        picked = _ytdlp_picked(info)
        h = int(picked.get("height") or 0)
        if not h:
            try:
                h = probe_video_size(out_path)[1]
                picked["height"] = h
            except Exception:
                pass
        logging.info(
            f"🎞️ yt-dlp picked format {picked.get('format_id')} "
            f"{picked.get('width') or '?'}x{h or '?'} {picked.get('vcodec') or ''} via {label}"
        )
        if best is None or h > best[0]:
            if best and os.path.exists(best[1]):
                try:
                    os.remove(best[1])
                except OSError:
                    pass
            keep = out_path if (h >= min_h or i == len(attempts) - 1) else f"{out_path}.try{i}"
            if keep != out_path:
                os.replace(out_path, keep)
            best = (h, keep, picked, label)
        if h >= min_h:
            break
        if i < len(attempts) - 1:
            logging.warning(f"⚠️ Only {h}p from {label} — trying another YouTube client for HD")
    if best is None:
        raise last_err or RuntimeError(f"yt-dlp download failed: {url}")
    if best[1] != out_path:
        os.replace(best[1], out_path)
    if best[0] and best[0] < min_h:
        logging.warning(
            f"⚠️ Best available was {best[0]}p ({best[3]}). "
            "Run `pip install -U yt-dlp` — old builds lose YouTube's HD streams."
        )
    return best[2]

def probe_video_size(input_file):
    probe = ffmpeg.probe(input_file)
    s = next(x for x in probe["streams"] if x["codec_type"] == "video")
    return int(s["width"]), int(s["height"])

def probe_video_duration(input_file):
    probe = ffmpeg.probe(input_file)
    s = next(x for x in probe["streams"] if x["codec_type"] == "video")
    return float(s.get("duration") or probe["format"]["duration"])

def scale_crop_to_input(crop_spec, input_w, input_h):
    expected = crop_spec.get("expectedInputResolution") or {}
    ref_w = int(expected.get("width") or input_w)
    ref_h = int(expected.get("height") or input_h)
    sx, sy = input_w / ref_w, input_h / ref_h
    x, y = int(crop_spec["x"] * sx), int(crop_spec["y"] * sy)
    w, h = int(crop_spec["width"] * sx), int(crop_spec["height"] * sy)
    w -= w % 2; h -= h % 2
    x = max(0, min(x, input_w - 2)); y = max(0, min(y, input_h - 2))
    w = min(w, input_w - x); h = min(h, input_h - y)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid crop: {x},{y},{w},{h}")
    return x, y, w, h

def get_video_metadata(url, retries=2):
    try:
        from helper.grok_imagine_extractor import (
            is_grok_imagine_share_link,
            get_grok_video_metadata,
        )

        if is_grok_imagine_share_link(url):
            return get_grok_video_metadata(url)
    except Exception as e:
        logging.error(f"❌ Grok metadata failed for {url}: {e}")
        raise

    for attempt in range(retries):
        try:
            opts = _base_ydl_opts("%(id)s")
            opts["skip_download"] = True
            opts["ignore_no_formats_error"] = True
            info = _ytdlp_run(url, opts, download=False)
            return {"duration": float(info.get("duration") or 0), "title": info.get("title") or ""}
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

def choose_download_strategy(clip_count, duration_sec):
    if clip_count >= 2:
        return "cache"
    return "partial"

def _source_cache_usable(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    try:
        if _probe_stream_duration(str(path), "video") <= 0:
            return False
        return _verify_video_decodable(str(path), timeout=90)
    except Exception:
        return False


def get_or_download_source(url, retries=3):
    key = _url_cache_key(url)
    cached = SOURCE_CACHE_DIR / f"{key}.mp4"
    if _source_cache_usable(cached):
        return str(cached)
    if cached.exists():
        logging.warning(
            "♻️ Removing bad Grok/source cache (undecodable): %s",
            cached.name,
        )
        try:
            cached.unlink()
        except OSError:
            pass
    lock = _lock_for(key)
    with lock:
        if _source_cache_usable(cached):
            return str(cached)
        if cached.exists():
            try:
                cached.unlink()
            except OSError:
                pass
        tmp = cached.with_suffix(".part.mp4")
        try:
            from helper.grok_imagine_extractor import (
                is_grok_imagine_share_link,
                download_grok_imagine_source,
            )

            if is_grok_imagine_share_link(url):
                path = download_grok_imagine_source(url, str(tmp))
                if not path:
                    raise RuntimeError(f"Grok Imagine download failed: {url}")
                path_obj = Path(path)
                if path_obj != cached:
                    if cached.exists():
                        cached.unlink()
                    path_obj.replace(cached)
                return str(cached)
        except RuntimeError:
            raise
        except Exception as e:
            logging.error(f"❌ Grok source download error: {e}")
            raise RuntimeError(f"Grok Imagine download failed: {url}") from e

        opts = _base_ydl_opts(str(tmp), retries)
        _ytdlp_download_hd(url, opts, str(tmp))
        if not tmp.exists() or tmp.stat().st_size < 10_000:
            raise RuntimeError(f"Source download failed: {url}")
        tmp.replace(cached)
        return str(cached)

def _nvenc_args():
    return [
        "-c:v", "h264_nvenc", "-preset", "p3", "-tune", "hq",
        "-rc", "vbr", "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    ]

def _build_filter_graph(crop, effect, logo_path, input_w, input_h):
    chain = ""
    if crop and all(k in crop for k in ("x", "y", "width", "height")):
        x, y, w, h = scale_crop_to_input(crop, input_w, input_h)
        chain += f"crop={w}:{h}:{x}:{y},"
    if effect == "zoom":
        chain += "scale=1.3*iw:1.3*ih,"
    elif effect == "mirror":
        chain += "hflip,"
    elif effect == "zoom+mirror":
        chain += "scale=1.3*iw:1.3*ih,hflip,"
    if chain.endswith(","):
        chain = chain[:-1]

    auto_crop = "" if crop else "crop=h=iw*9/16,"
    proc = f"[main]{chain}[processed];" if chain else "[main]null[processed];"
    graph = (
        "[0:v]split=2[main][blur];"
        "[blur]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
        "boxblur=luma_radius=min(h\\,w)/20:luma_power=1:"
        "chroma_radius=min(cw\\,ch)/20:chroma_power=1[bg];"
        f"{proc}"
        "[processed]scale=1920:1080:force_original_aspect_ratio=decrease[scaled];"
        "[bg][scaled]overlay=(W-w)/2:(H-h)/2,"
        f"{auto_crop}"
        "eq=contrast=1.05:saturation=1.15:brightness=0.01[video]"
    )
    if logo_path:
        return graph + ";[video][1:v]overlay=W-w-10:H-h-10[final]", "[final]"
    return graph, "[video]"

def _unique_clip_output_path(section_dir, character, start, end, clip) -> str:
    """Every vidclip gets its own file (parallel cuts must not share millis-based names)."""
    tag = (clip.get("tag") or "").strip()
    safe_tag = re.sub(r"[^\w.-]+", "_", tag)[:48] if tag else "clip"
    name = (
        f"{character}_{start:.2f}_{end:.2f}_{safe_tag}_{uuid.uuid4().hex[:12]}.mp4"
    )
    return os.path.join(section_dir, name)


def process_segment_single_pass(input_path, output_path, start_time, end_time,
                                user_id, crop=None, effect=None):
    """Accurate trim + crop + effects in ONE NVENC pass."""
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(f"Bad range {start_time}→{end_time}")

    in_w, in_h = probe_video_size(input_path)
    logo_path = None
    if user_id == "admin@example.com":
        logo_path = "branding/admin_logo.png"
    elif user_id and user_id.endswith("@company.com"):
        logo_path = "branding/company_logo.png"
    if logo_path and not os.path.exists(logo_path):
        logo_path = None

    graph, map_out = _build_filter_graph(crop, effect, logo_path, in_w, in_h)

    # Accurate trim: fast pre-seek + fine post-seek (frame-accurate)
    pre_seek = max(0.0, start_time - 3.0)
    fine_seek = start_time - pre_seek

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{pre_seek:.3f}", "-i", input_path,
        "-ss", f"{fine_seek:.3f}",
    ]
    if logo_path:
        cmd += ["-i", logo_path]
    cmd += [
        "-t", f"{duration:.3f}",
        "-filter_complex", graph,
        "-map", map_out, "-map", "0:a?",
        *_nvenc_args(),
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest",
        output_path,
    ]
    with _grok_nvenc_cut_sem:
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    if not _verify_video_decodable(output_path, timeout=60):
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Grok/NVENC cut produced undecodable file: {output_path}"
        )
    return output_path

def crop_with_audio(input_file, output_file, crop_spec):
    try:
        in_w, in_h = probe_video_size(input_file)
        x, y, w, h = scale_crop_to_input(crop_spec, in_w, in_h)
        cmd = [
            "ffmpeg", "-y", "-i", input_file,
            "-filter:v", f"crop={w}:{h}:{x}:{y}",
            "-map", "0:v:0", "-map", "0:a?",
            *_nvenc_args(),
            "-c:a", "aac", "-b:a", "192k",
            output_file,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        logging.info(f"✂️ Cropping completed: {output_file}")
        return True
    except Exception as e:
        logging.error(f"❌ Cropping failed: {e}")
        return False

def download_video_segment(user_id, url, output_path, start_time, end_time,
                           crop=None, effect=None, retries=3):
    if start_time is None or end_time is None or end_time <= start_time:
        logging.warning(f"Invalid time range: {start_time}→{end_time}")
        return None

    try:
        from helper.grok_imagine_extractor import is_grok_imagine_share_link

        if is_grok_imagine_share_link(url):
            source_path = get_or_download_source(url)
            if not source_path:
                return None
            process_segment_single_pass(
                source_path,
                output_path,
                start_time,
                end_time,
                user_id,
                crop=crop,
                effect=effect,
            )
            logging.info(f"✅ Grok clip segment created: {output_path}")
            return output_path
    except Exception as e:
        logging.error(f"❌ Grok segment download failed: {e}")
        return None

    temp_raw = f"{output_path}.{uuid.uuid4()}.raw.mp4"

    # Stream-copy a slightly wider range (no force_keyframes_at_cuts → no CPU
    # re-encode inside yt-dlp), then cut frame-exactly in the single NVENC pass.
    pad = 2.0
    range_start = max(0.0, float(start_time) - pad)
    range_end = float(end_time) + pad

    for attempt in range(retries):
        try:
            ydl_opts = _base_ydl_opts(temp_raw, retries)
            ydl_opts.update({
                "quiet": False,
                "progress_hooks": [progress_hook],
                "download_ranges": lambda _, __, a=range_start, b=range_end: [
                    {"start_time": a, "end_time": b}
                ],
            })

            _ytdlp_download_hd(url, ydl_opts, temp_raw)

            if not os.path.exists(temp_raw) or os.path.getsize(temp_raw) < 10_000:
                raise RuntimeError("Partial download empty")

            partial_dur = probe_video_duration(temp_raw)
            seg_start = float(start_time) - range_start
            seg_end = min(float(end_time) - range_start, partial_dur)
            if seg_end - seg_start < 0.05:
                raise RuntimeError(
                    f"Downloaded range too short ({partial_dur:.2f}s) for "
                    f"{start_time}→{end_time}"
                )
            process_segment_single_pass(
                temp_raw, output_path, seg_start, seg_end,
                user_id, crop=crop, effect=effect,
            )
            os.remove(temp_raw)
            logging.info(f"✅ Video segment created: {output_path}")
            return output_path

        except Exception as e:
            logging.error(f"❌ Attempt {attempt + 1}/{retries}: {e}")
            for p in (temp_raw, output_path):
                if p and os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None

def download_video_clips_in_parallel(clip_data_list, section_dir, user_id,
                                     max_workers=6, max_retries=5):
    os.makedirs(section_dir, exist_ok=True)
    by_url = defaultdict(list)

    for item in clip_data_list or []:
        char = item["characterName"]
        for video in item.get("videos") or []:
            url = video["url"]
            for clip in video.get("vidclips") or []:
                by_url[url].append({"character": char, "url": url, "clip": clip})

    if not by_url:
        return []

    # Metadata + strategy per URL
    url_meta, url_strategy = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        def _meta_job(url):
            try:
                meta = get_video_metadata(url)
                strategy = choose_download_strategy(len(by_url[url]), meta["duration"])
                return url, meta, strategy
            except Exception as e:
                logging.error(f"❌ YouTube metadata failed for {url}: {e}")
                return url, {"duration": 0, "title": ""}, "partial"

        for url, meta, strategy in ex.map(_meta_job, by_url.keys()):
            url_meta[url] = meta
            url_strategy[url] = strategy

    # Pre-cache sources where 2+ clips share a URL
    url_source = {}
    cache_urls = [u for u, s in url_strategy.items() if s == "cache"]
    if cache_urls:
        def _cache_job(url):
            try:
                return url, get_or_download_source(url)
            except Exception as e:
                logging.error(f"❌ YouTube source cache failed for {url}: {e}")
                return url, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            for url, path in ex.map(_cache_job, cache_urls):
                url_source[url] = path

    def _one(task):
        char, url, clip = task["character"], task["url"], task["clip"]
        start = time_to_seconds(clip.get("startTime", "0:00:00"))
        end = time_to_seconds(clip.get("endTime", "0:00:05"))
        crop, effect = clip.get("crop"), clip.get("effect")
        out = _unique_clip_output_path(section_dir, char, start, end, clip)

        for attempt in range(1, max_retries + 1):
            try:
                if url_strategy[url] == "cache":
                    source_path = url_source.get(url)
                    if not source_path:
                        raise RuntimeError(
                            f"Source unavailable for {url} (download or geo-block failed)"
                        )
                    process_segment_single_pass(
                        source_path, out, start, end, user_id, crop, effect
                    )
                else:
                    path = download_video_segment(
                        user_id, url, out, start, end, crop, effect, retries=2
                    )
                    if not path:
                        raise RuntimeError("Partial download failed")

                return {
                    "character": char, "path": out, "start": start, "end": end,
                    "mute": clip.get("mute", False), "tag": clip.get("tag"), "url": url,
                }
            except Exception as e:
                if attempt == max_retries:
                    return {"character": char, "url": url, "error": str(e)}
                time.sleep(2 ** attempt)

    jobs = []
    for url, tasks in by_url.items():
        for t in tasks:
            jobs.append(t)

    downloaded, failed = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(_one, jobs):
            if result.get("path"):
                downloaded.append(result)
            else:
                failed.append(result)

    logging.info(f"✅ Completed {len(downloaded)}/{len(jobs)} clips.")
    if failed:
        logging.error(
            f"❌ {len(failed)} clip(s) failed — continuing with {len(downloaded)} clip(s)"
        )
        for f in failed:
            logging.error(f"❌ {f.get('character')} | {f.get('url')} | {f.get('error')}")
    return downloaded



#I---------------------IMAGE DOWNLOAD FUNCTIONS START---------------------I
def create_session():
    """Create a requests session with standard headers"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    })
    return session


def setup_selenium_driver():
    """Configure Selenium with anti-detection measures"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    # Rotate user agents
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ]
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def unified_clean_filename(url_or_filename, session=None):
    """
    Sanitize and infer proper image filename.
    Accepts either a raw filename or a URL.
    Optionally uses a requests.Session to infer file extension via headers.
    """
    # If it's a URL
    if url_or_filename.startswith("http"):
        try:
            clean_url = unquote(url_or_filename.split('?')[0].split('#')[0])
            filename = os.path.basename(urlparse(clean_url).path)
        except:
            filename = f"image_{random.randint(1000,9999)}"

        filename = re.sub(r'[<>:"/\\|?*]', '', filename).strip()

        if not re.search(r'\.(jpg|jpeg|png|gif|webp)$', filename, re.I):
            if session:
                try:
                    content_type = session.head(url_or_filename, timeout=5).headers.get('Content-Type', '')
                    if 'jpeg' in content_type:
                        filename += '.jpg'
                    elif 'png' in content_type:
                        filename += '.png'
                    elif 'webp' in content_type:
                        filename += '.webp'
                    else:
                        filename += '.jpg'
                except:
                    filename += '.jpg'
            else:
                filename += '.jpg'

        return filename or f"image_{random.randint(1000,9999)}.jpg"

    else:
        # It's a given filename (e.g. from img_path)
        filename = re.sub(r'[<>:"/\\|?*]', '', url_or_filename).strip()
        return filename or f"image_{random.randint(1000,9999)}.jpg"

def direct_download(session, img_url, save_path):
    """Attempt direct download with streaming"""
    try:
        with session.get(img_url, stream=True, timeout=15) as response:
            response.raise_for_status()
            
            # Verify image content
            if 'image' not in response.headers.get('Content-Type', ''):
                return None
            
            # Handle CDN URLs with parameters
            if 'image' in response.url and img_url != response.url:
                return direct_download(session, response.url, save_path)
            
            # Save image
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            
            # Verify image integrity
            try:
                with Image.open(save_path) as img:
                    img.verify()
                return save_path
            except:
                os.remove(save_path)
                return None
    except:
        return None

def download_with_modified_headers(session, img_url, save_path):
    """Try different header combinations"""
    header_variations = [
        {"Referer": "https://www.google.com/"},
        {"Referer": img_url},
        {"Accept": "image/webp,*/*"},
        {}
    ]
    
    for headers in header_variations:
        try:
            with session.get(img_url, headers=headers, stream=True, timeout=10) as response:
                if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(8192):
                            f.write(chunk)
                    return save_path
        except:
            continue
    return None

def download_with_selenium(img_url, save_path):
    """Use Selenium to handle JavaScript-rendered pages"""
    driver = None
    try:
        driver = setup_selenium_driver()
        driver.get(img_url)
        
        # Wait for page to load
        WebDriverWait(driver, 3).until(
            lambda d: d.execute_script('return document.readyState') == 'complete'
        )
        
        # Find all potential image sources
        img_sources = []
        for tag in ['img', 'picture source', 'div[style*="background-image"]']:
            elements = driver.find_elements(By.CSS_SELECTOR, tag)
            for el in elements:
                for attr in ['src', 'data-src', 'data-url', 'data-original', 'style']:
                    src = el.get_attribute(attr) or ''
                    if 'http' in src:
                        img_sources.append(src.split('"')[1] if 'url(' in src else src)
                    elif 'base64' in src:
                        img_sources.append(src)
        
        # Try each found source
        session = create_session()
        for src in list(set(img_sources)):  # Remove duplicates
            try:
                if src.startswith('data:image'):
                    # Handle base64 images
                    header, data = src.split(',', 1)
                    image_data = base64.b64decode(data)
                    with open(save_path, 'wb') as f:
                        f.write(image_data)
                    return save_path
                else:
                    # Download found image URL
                    result = direct_download(session, src, save_path)
                    if result: return result
            except:
                continue
        
        # Fallback to screenshot if no images found
        driver.save_screenshot(save_path)
        return save_path if os.path.exists(save_path) else None
        
    except Exception as e:
        logging.error(f"Selenium failed: {str(e)}")
        return None
    finally:
        if driver: driver.quit()


def download_image(img_url, img_path):
    """Download an image using enhanced logic, fully compatible with the original function."""
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    session = create_session()

    # Smart clean: Use full URL if img_path has no clear name or extension
    input_filename = os.path.basename(img_path)
    if not re.search(r'\.(jpg|jpeg|png|gif|webp)$', input_filename, re.I):
        filename = unified_clean_filename(img_url, session)
    else:
        filename = unified_clean_filename(input_filename)

    # Final path to save image
    img_path = os.path.join(os.path.dirname(img_path), filename)

    # 1. Direct download
    result = direct_download(session, img_url, img_path)
    if result:
        return img_path

    # 2. Try modified headers
    result = download_with_modified_headers(session, img_url, img_path)
    if result:
        return img_path

    # 3. Fallback to Selenium has been removed for efficiency and compatibility
    return None  # Explicitly return None if all methods fail


@memory_safe_operation # Decorator to ensure memory safety
def download_images_in_parallel(image_data_list, section_dir, max_workers=10):
    os.makedirs(section_dir, exist_ok=True)
    results = {}

    def task(character_name, img_url, img_index):
        char_folder = os.path.join(section_dir, character_name.replace(" ", "_"))
        os.makedirs(char_folder, exist_ok=True)
        filename = f"{img_index}_{os.path.basename(unified_clean_filename(img_url))}"
        img_path = os.path.join(char_folder, filename)
        downloaded = download_image(img_url, img_path)
        return character_name, downloaded

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in image_data_list:
            character = item["characterName"]
            for idx, url in enumerate(item["images"]):  # 🚨 Fix: Process all images, not just the first
                futures.append(executor.submit(task, character, url, idx))  # Submit task for each image

        for future in as_completed(futures):
            try:
                character_name, downloaded_path = future.result()
                if downloaded_path:
                    results.setdefault(character_name, []).append(downloaded_path)
            except Exception as e:
                logging.error(f"Parallel image download error: {e}")

    return results

#---------------------IMAGE DOWNLOAD FUNCTIONS END---------------------


# ----------------- AI Processing Functions -----------------

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def encode_image_from_pil(image: Image.Image):
    """Encode an image using CLIP from a PIL image."""
    try:
        model, processor = get_clip_model()  # lazy singleton
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features[0].cpu().numpy()
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return np.zeros((512,))

def extract_representative_frame(video_clip):
    """Memory-optimized frame extraction with accuracy preservation"""
    try:
        # 1. Get frame at 1/4 duration (more stable than midpoint)
        frame = video_clip.get_frame(video_clip.duration / 4)
        
        # 2. Smart downscaling (preserve aspect ratio)
        height = 360  # Keeps sufficient detail
        width = int(frame.shape[1] * (height / frame.shape[0]))
        frame_resized = cv2.resize(frame, (width, height))
        
        # 3. Convert to RGB (ensure 3 channels even if source is grayscale)
        if len(frame_resized.shape) == 2:  # Grayscale
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 4. Normalize and convert to PIL
        frame_rgb = (frame_rgb * 255).astype(np.uint8)  # Ensure proper range
        return Image.fromarray(frame_rgb)
        
    except Exception as e:
        logging.error(f"Frame extraction error in extract_representative_frame: {e}")
        return None


def time_segments(audio_path, segments):
    """Estimate time boundaries for each segment while skipping empty text segments."""
    try:
        if not isinstance(audio_path, str):
            raise ValueError(f"audio_path must be a valid file path (str), but received: {type(audio_path)}")
        if not segments or not isinstance(segments, list):
            raise ValueError("Segments must be a non-empty list of dictionaries containing 'text'.")

        # Extract non-empty text segments
        valid_segments = [seg["text"] for seg in segments if isinstance(seg, dict) and "text" in seg and seg["text"].strip()]

        if not valid_segments:
            raise ValueError("All segments are empty; cannot allocate time.")

        # Get total duration of the audio file
        with AudioFileClip(audio_path) as audio:
            total_duration = audio.duration

        # Compute segment weights based on word count
        segment_word_counts = [len(seg.split()) for seg in valid_segments]
        total_words = sum(segment_word_counts)

        if total_words == 0:
            raise ValueError("All valid segments have zero words; cannot allocate time.")

        # Calculate proportional time allocation
        time_allocations = [(word_count / total_words) * total_duration for word_count in segment_word_counts]

        # Generate time boundaries
        time_boundaries = []
        start_time = 0.0

        for duration in time_allocations:
            end_time = start_time + duration
            time_boundaries.append((round(start_time, 2), round(end_time, 2)))
            start_time = end_time  # Move to next segment start time

        return time_boundaries

    except Exception as e:
        raise RuntimeError(f"Error in time_segments function: {str(e)}")


# def create_blurred_background_image(image_path, duration):
#     """
#     Creates a video clip from an image.
#     If the image doesn't fit 16:9 aspect ratio, it creates a blurred background
#     and places the image to always fill the full height of the frame.
#     Applies a random Ken Burns effect to the final video.
#     """
#     # Load image safely
#     with Image.open(image_path).convert("RGB") as opened_img:
#         img = opened_img.copy()  # Safely copy the image for later use

#     # Get image dimensions
#     img_width, img_height = img.size
#     aspect_ratio = img_width / img_height

#     # Define 16:9 dimensions
#     target_width, target_height = 1280, 720
#     target_aspect_ratio = target_width / target_height

#     # Determine if background is needed
#     needs_background = abs(aspect_ratio - target_aspect_ratio) > 0.01

#     if needs_background:
#         # Create blurred background
#         blurred_bg = img.resize((target_width, target_height)).filter(ImageFilter.GaussianBlur(40))

#         # Resize image to fill height
#         scale_factor = target_height / img_height
#         new_width = int(img_width * scale_factor)
#         resized_img = img.resize((new_width, target_height))

#         # Center image horizontally
#         x_offset = (target_width - new_width) // 2
#         composite = blurred_bg.copy()
#         composite.paste(resized_img, (x_offset, 0))

#         # Save temp composite image
#         composite_path = os.path.splitext(image_path)[0] + "_composite.jpg"
#         composite.save(composite_path)

#         # Create video clip
#         clip = ImageClip(composite_path).set_duration(duration)
#     else:
#         # Resize to 16:9 directly
#         resized_img = img.resize((target_width, target_height))
#         resized_path = os.path.splitext(image_path)[0] + "_resized.jpg"
#         resized_img.save(resized_path)

#         clip = ImageClip(resized_path).set_duration(duration)

#     # Apply slight zoom to support motion effects
#     clip = clip.resize(1.05)

#     # Modified Ken Burns effect (subtle panning/zooming)
#     effects = [
#     # Zoom in effect
#     lambda clip: CompositeVideoClip([clip.resize(lambda t: 1 + 0.05 * (t / clip.duration))]),

#     # Zoom out effect
#     lambda clip: CompositeVideoClip([clip.resize(lambda t: 1 - 0.05 * (t / clip.duration))]),

#     # Move up
#     lambda clip: CompositeVideoClip([clip.set_position(lambda t: ("center", -min(150, 11 * t)))]),

#     # Move down
#     lambda clip: CompositeVideoClip([clip.set_position(lambda t: ("center", min(150, 11 * t)))]),

#     # Move right
#     lambda clip: CompositeVideoClip([clip.set_position(lambda t: (min(150, 11 * t), "center"))]),

#     # Move left
#     lambda clip: CompositeVideoClip([clip.set_position(lambda t: (-min(150, 11 * t), "center"))]),
# ]

#     # Apply random Ken Burns effect
#     final_clip = random.choice(effects)(clip)
#     final_clip = final_clip.set_duration(duration)

#     # Wrap in final composite (avoids layering issues)
#     return CompositeVideoClip([final_clip])


logger = logging.getLogger("storyboard")

# --- TRIM-FREE LOADER --------------------------------------------------------
def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr[-500:]}")
    return float(result.stdout.strip())


def _validate_storyboard_video(path: str, duration: float) -> None:
    if not path:
        raise RuntimeError("Storyboard API returned empty path.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Storyboard path does not exist: {path}")
    if os.path.getsize(path) < 1000:
        raise RuntimeError(f"Storyboard file too small: {path}")

    clip_dur = _ffprobe_duration(path)
    if clip_dur <= 0.05:
        raise RuntimeError(f"Storyboard clip has invalid duration: {clip_dur}")
    if abs(clip_dur - duration) > 0.05:
        logger.warning(
            f"[Storyboard] Backend duration {clip_dur:.3f}s != target {duration:.3f}s. "
            "No trimming applied (by design)."
        )


def _copy_storyboard_to_output(src_path: str, output_path: str) -> str:
    """Stream-copy storyboard MP4 — skip redundant MoviePy re-encode (P1)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-c:v", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg copy failed: {result.stderr[-500:]}")
    return output_path


def _storyboard_video_from_api(
    composite_for_api: str,
    duration: float,
    split_len: int | None = None,
    fps: int | None = None,
    obj_skip: int | None = None,
    bg_skip: int | None = None,
    hold_fraction: float | None = None,
) -> str:
    tune = _storyboard_tuning(segment_fps=fps or 30)
    split_len = split_len if split_len is not None else tune["split_len"]
    fps = fps if fps is not None else tune["fps"]
    obj_skip = obj_skip if obj_skip is not None else tune["obj_skip"]
    bg_skip = bg_skip if bg_skip is not None else tune["bg_skip"]
    hold_fraction = hold_fraction if hold_fraction is not None else tune["hold_fraction"]

    logger.info(
        f"[Storyboard] Request -> img='{composite_for_api}', "
        f"target_dur={duration:.2f}s, split={split_len}, fps={fps}, "
        f"obj_skip={obj_skip}, bg_skip={bg_skip}, hold={hold_fraction}"
    )

    sb_video_path = storyboard_from_api(
        composite_for_api,
        split_len=int(split_len),
        fps=int(fps),
        object_skip_rate=int(obj_skip),
        bg_object_skip_rate=int(bg_skip),
        end_gray_img_duration_in_sec=0,
        target_dur_sec=duration,
        hold_fraction=float(hold_fraction),
    )

    _validate_storyboard_video(sb_video_path, duration)
    logger.info(f"[Storyboard] OK -> '{sb_video_path}'")
    return sb_video_path


def _load_storyboard_clip_or_raise(
    composite_for_api: str,
    duration: float,
    split_len: int,
    fps: int,
    obj_skip: int,
    bg_skip: int,
    hold_fraction: float | None = None,
) -> VideoFileClip:
    """
    Calls storyboard API and returns a validated MoviePy clip (legacy composite path).
    """
    sb_video_path = _storyboard_video_from_api(
        composite_for_api=composite_for_api,
        duration=duration,
        split_len=split_len,
        fps=fps,
        obj_skip=obj_skip,
        bg_skip=bg_skip,
        hold_fraction=hold_fraction,
    )
    clip = VideoFileClip(sb_video_path)
    logger.info(f"[Storyboard] clip loaded ({clip.duration:.2f}s)")
    return clip
import os
import math
import random
import numpy as np
from PIL import Image, ImageFilter
from moviepy.editor import ImageClip, CompositeVideoClip

def zoom_in_effect(clip, zoom_ratio=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        new_size = [
            math.ceil(base_size[0] * (1 + (zoom_ratio * t))),
            math.ceil(base_size[1] * (1 + (zoom_ratio * t)))
        ]

        new_size[0] += new_size[0] % 2
        new_size[1] += new_size[1] % 2

        img = img.resize(new_size, Image.LANCZOS)
        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)

        img = img.crop([
            x, y, new_size[0] - x, new_size[1] - y
        ]).resize(base_size, Image.LANCZOS)

        result = np.array(img)
        img.close()
        return result

    return clip.fl(effect)

def zoom_out_effect(clip, zoom_max_ratio=0.3, zoom_out_factor=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        scale_factor = zoom_max_ratio - (zoom_out_factor * t)
        scale_factor = max(scale_factor, 0)

        new_size = [
            math.ceil(base_size[0] * (1 + scale_factor)),
            math.ceil(base_size[1] * (1 + scale_factor))
        ]

        new_size[0] -= new_size[0] % 2
        new_size[1] -= new_size[1] % 2

        img = img.resize(new_size, Image.LANCZOS)
        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)

        img = img.crop([
            x, y, new_size[0] - x, new_size[1] - y
        ]).resize(base_size, Image.LANCZOS)

        result = np.array(img)
        img.close()
        return result

    return clip.fl(effect)

def pan_left_to_right_effect(clip, pan_ratio=0.1):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_width, base_height = img.size

        max_offset = int(base_width - (base_width / 1.2))
        pan_offset = int(max_offset * (t / clip.duration))

        crop_width = int(base_width / 1.2)
        crop_box = (
            pan_offset,
            0,
            pan_offset + crop_width,
            base_height
        )

        img = img.crop(crop_box).resize((base_width, base_height), Image.LANCZOS)
        return np.array(img)
    return clip.fl(effect)

def pan_right_to_left_effect(clip, pan_ratio=0.1):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_width, base_height = img.size

        max_offset = int(base_width - (base_width / 1.2))
        pan_offset = int(max_offset * (1 - t / clip.duration))

        crop_width = int(base_width / 1.2)
        crop_box = (
            pan_offset,
            0,
            pan_offset + crop_width,
            base_height
        )

        img = img.crop(crop_box).resize((base_width, base_height), Image.LANCZOS)
        return np.array(img)
    return clip.fl(effect)


# --- TRIM-FREE CREATOR -------------------------------------------------------
# def create_blurred_background_image(
#     image_path: str,
#     duration: float,
#     mode:str,
#     storyboard: bool =False,
#     storyboard_split_len: int = 10,
#     storyboard_fps: int = 30,
#     storyboard_obj_skip: int = 8,
#     storyboard_bg_skip: int = 14,
# ) -> CompositeVideoClip:
#     """
#     Builds a 1920x1080 still and, if enabled, requests a whiteboard animation
#     from the server with target_dur_sec. NO local trim/pad is performed.
#     """
#     target_w, target_h = 1920, 1080
#     with Image.open(image_path).convert("RGB") as im0:
#         img = im0.copy()

#     iw, ih = img.size
#     needs_bg = abs((iw/ih) - (target_w/target_h)) > 0.01

#     if needs_bg:
#         blurred = img.resize((target_w, target_h)).filter(ImageFilter.GaussianBlur(40))
#         scale = target_h / ih
#         new_w = int(iw * scale)
#         fore = img.resize((new_w, target_h))
#         x = (target_w - new_w) // 2
#         comp = blurred.copy()
#         comp.paste(fore, (x, 0))
#         comp_path = os.path.splitext(image_path)[0] + "_composite.jpg"
#         comp.save(comp_path, quality=95)
#         base_clip = ImageClip(comp_path).set_duration(duration)
#         composite_for_api = comp_path
#     else:
#         resized = img.resize((target_w, target_h))
#         resized_path = os.path.splitext(image_path)[0] + "_resized.jpg"
#         resized.save(resized_path, quality=95)
#         base_clip = ImageClip(resized_path).set_duration(duration)
#         composite_for_api = resized_path

#     if storyboard and mode == "original":
#         try:
#             sb_clip = _load_storyboard_clip_or_raise(
#                 composite_for_api=composite_for_api,
#                 duration=duration,
#                 split_len=storyboard_split_len,
#                 fps=storyboard_fps,
#                 obj_skip=storyboard_obj_skip,
#                 bg_skip=storyboard_bg_skip,
#             )
#             # No set_duration(duration) here either—respect backend timing.
#             return CompositeVideoClip([sb_clip.resize((target_w, target_h))])
#         except Exception as e:
#             logger.exception(f"[Storyboard] FAILED: {e}. Falling back to static image.")

#     logger.warning("[Storyboard] Using static image fallback.")
#     return CompositeVideoClip([base_clip])

# def create_blurred_background_image(
#     image_path: str,
#     duration: float,
#     mode: str,
#     storyboard: bool = False,
#     storyboard_split_len: int = 10,
#     storyboard_fps: int = 30,
#     storyboard_obj_skip: int = 8,
#     storyboard_bg_skip: int = 14,
# ) -> CompositeVideoClip:

#     target_w, target_h = 1920, 1080

#     with Image.open(image_path).convert("RGB") as im0:
#         img = im0.copy()

#     iw, ih = img.size
#     needs_bg = abs((iw / ih) - (target_w / target_h)) > 0.01

#     # ----------------------------------------------------
#     # 1. BUILD BASE STATIC IMAGE CLIP (NO MOTION YET)
#     # ----------------------------------------------------
#     if needs_bg:
#         blurred = img.resize((target_w, target_h)).filter(ImageFilter.GaussianBlur(40))
#         scale = target_h / ih
#         new_w = int(iw * scale)

#         fore = img.resize((new_w, target_h))
#         x = (target_w - new_w) // 2

#         comp = blurred.copy()
#         comp.paste(fore, (x, 0))
#         comp_path = os.path.splitext(image_path)[0] + "_composite.jpg"
#         comp.save(comp_path, quality=95)

#         base_clip = ImageClip(comp_path).set_duration(duration)
#         composite_for_api = comp_path

#     else:
#         resized_path = os.path.splitext(image_path)[0] + "_resized.jpg"
#         img.resize((target_w, target_h)).save(resized_path, quality=95)
#         base_clip = ImageClip(resized_path).set_duration(duration)
#         composite_for_api = resized_path

#     # ----------------------------------------------------
#     # 2. TRY STORYBOARD MODE FIRST (NO MOTION EFFECTS)
#     # ----------------------------------------------------
#     if storyboard and mode == "original":
#         try:
#             sb_clip = _load_storyboard_clip_or_raise(
#                 composite_for_api=composite_for_api,
#                 duration=duration,
#                 split_len=storyboard_split_len,
#                 fps=storyboard_fps,
#                 obj_skip=storyboard_obj_skip,
#                 bg_skip=storyboard_bg_skip,
#             )
#             return CompositeVideoClip([sb_clip.resize((target_w, target_h))])

#         except Exception as e:
#             logger.exception(f"[Storyboard] FAILED: {e}. Falling back to static + motion.")

#     # ----------------------------------------------------
#     # 3. APPLY KEN BURNS EFFECTS (from OLD FUNCTION)
#     # ----------------------------------------------------
#     logger.warning("[Storyboard] Using static image fallback — applying motion effect.")

#     # Slightly enlarge image for panning/zooming
#     motion_base = base_clip.resize(1.2)

#     # Define your effects (must be imported above)
#     effects = [
#         lambda c: zoom_in_effect(c, zoom_ratio=0.04),
#         lambda c: zoom_out_effect(c, zoom_max_ratio=0.3, zoom_out_factor=0.04),
#         # lambda c: pan_left_to_right_effect(c, pan_ratio=0.1),
#         # lambda c: pan_right_to_left_effect(c, pan_ratio=0.1),
#     ]

#     chosen_clip = random.choice(effects)(motion_base).set_duration(duration)

#     return CompositeVideoClip([chosen_clip])

# def create_image_video_with_effects(
#     image_path: str,
#     duration: float,
#     canvas_w: int,
#     canvas_h: int,
#     output_path: str,
#     mode: str,
#     index: int,
#     kenburns_enabled: bool = False,
#     storyboard_enabled: bool = False,
#     fps: int = 30,
# ) -> str:
#     """
#     Create video from image with optional Ken Burns and storyboard effects.
#     Uses GPU pipeline for maximum speed.
    
#     Args:
#         image_path: Path to source image
#         duration: Video duration in seconds
#         canvas_w: Target video width
#         canvas_h: Target video height
#         output_path: Where to save the video
#         mode: Video mode (e.g., "original", "commentary")
#         kenburns_enabled: Apply Ken Burns motion effects
#         storyboard_enabled: Apply storyboard effect
#         fps: Frames per second
    
#     Returns:
#         Path to created video file
#     """
    
#     logger = logging.getLogger(__name__)
    
#     # ============================================================
#     # STEP 1: Check for STORYBOARD mode (only exception to GPU pipeline)
#     # ============================================================
#     if storyboard_enabled and mode == "original":
#         logger.info(f"🎨 Storyboard mode enabled - using storyboard API")
#         try:
#             from PIL import Image, ImageFilter
            
#             # Create composite for storyboard API (it needs the composite path)
#             with Image.open(image_path).convert("RGB") as im0:
#                 img = im0.copy()
            
#             iw, ih = img.size
#             needs_bg = abs((iw / ih) - (canvas_w / canvas_h)) > 0.01
            
#             if needs_bg:
#                 blurred = img.resize((canvas_w, canvas_h)).filter(ImageFilter.GaussianBlur(40))
#                 scale = canvas_h / ih
#                 new_w = int(iw * scale)
#                 fore = img.resize((new_w, canvas_h))
#                 x = (canvas_w - new_w) // 2
                
#                 comp = blurred.copy()
#                 comp.paste(fore, (x, 0))
#                 composite_path = os.path.splitext(image_path)[0] + f"_composite_{uuid.uuid4()}.jpg"
#                 comp.save(composite_path, quality=95)
#             else:
#                 composite_path = os.path.splitext(image_path)[0] + f"_resized_{uuid.uuid4()}.jpg"
#                 img.resize((canvas_w, canvas_h)).save(composite_path, quality=95)
            
#             # Call storyboard function
#             sb_clip = _load_storyboard_clip_or_raise(
#                 composite_for_api=composite_path,
#                 duration=duration,
#                 split_len=10,
#                 fps=fps,
#                 obj_skip=8,
#                 bg_skip=14,
#             )
            
#             # Render storyboard clip to video file
#             temp_sb_path = os.path.join(os.path.dirname(output_path), f"temp_sb_{uuid.uuid4()}.mp4")
            
#             from moviepy.editor import CompositeVideoClip
#             final_sb = CompositeVideoClip([sb_clip.resize((canvas_w, canvas_h))])
            
#             use_nvenc = torch.cuda.is_available()
#             codec = "h264_nvenc" if use_nvenc else "libx264"
#             preset = "p1" if use_nvenc else "ultrafast"
            
#             final_sb.write_videofile(
#                 temp_sb_path,
#                 codec=codec,
#                 preset=preset,
#                 fps=fps,
#                 audio=False,
#                 threads=4
#             )
#             final_sb.close()
            
#             # Re-encode to final output
#             cmd = [
#                 'ffmpeg', '-y',
#                 '-i', temp_sb_path,
#                 '-c:v', 'copy',  # Just copy, no re-encoding
#                 '-movflags', 'faststart',
#                 output_path
#             ]
#             subprocess.run(cmd, capture_output=True, text=True, check=True)
            
#             # Cleanup
#             os.remove(temp_sb_path)
#             os.remove(composite_path)
            
#             logger.info(f"✅ Storyboard video created: {output_path}")
#             return output_path
            
#         except Exception as e:
#             logger.error(f"❌ Storyboard failed: {e} - falling back to GPU pipeline")
#             # Fall through to GPU pipeline
    
#     # ============================================================
#     # STEP 2: GPU PIPELINE (handles both Ken Burns and static)
#     # ============================================================
    
#     # Determine effect type
#     if kenburns_enabled:
#         effect_type = 'auto'  # GPU pipeline will choose random effect
#         logger.info(f"🎬 Ken Burns enabled - using GPU pipeline with random effect")
#     else:
#         effect_type = 'static'  # We'll handle this differently
#         logger.info(f"📸 Static mode - using GPU pipeline without motion")
    
#     # For static, we need to modify the approach
#     if effect_type == 'static':
#         # Use zoom_in with zoom_ratio=0 (no actual zoom, just static)
#         # OR create a simple static video using create_video_gpu_direct with very minimal zoom
#         logger.info(f"🚀 Creating static video with GPU pipeline")
        
#         try:
#             # Option 1: Use GPU with minimal/no zoom effect
#             # We'll use pan_left with 0 duration effectively making it static
#             # Actually, let's just use FFmpeg for true static since GPU is overkill
            
#             # Create composite on GPU first
#             from gpu_kenburns_simple import create_composite_gpu
#             import numpy as np
#             from PIL import Image
            
#             # Get GPU composite
#             composite_tensor = create_composite_gpu(image_path, canvas_w, canvas_h)
            
#             # Save composite to temp file
#             img_np = composite_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            
#             temp_composite = os.path.join(os.path.dirname(output_path), f"temp_comp_{uuid.uuid4()}.jpg")
#             Image.fromarray(img_np).save(temp_composite, quality=95)
            
#             # Create static video with FFmpeg
#             use_nvenc = torch.cuda.is_available()
#             codec = "h264_nvenc" if use_nvenc else "libx264"
#             preset = "p1" if use_nvenc else "ultrafast"
            
#             cmd = [
#                 'ffmpeg', '-y',
#                 '-loop', '1',
#                 '-i', temp_composite,
#                 '-t', str(duration),
#                 '-c:v', codec,
#                 '-preset', preset,
#                 '-pix_fmt', 'yuv420p',
#                 '-r', str(fps),
#                 '-movflags', 'faststart',
#                 output_path
#             ]
            
#             result = subprocess.run(cmd, capture_output=True, text=True)
#             if result.returncode != 0:
#                 logger.error(f"FFmpeg error: {result.stderr[-1000:]}")
#                 raise RuntimeError("FFmpeg failed")
            
#             # Cleanup
#             os.remove(temp_composite)
            
#             logger.info(f"✅ Static video created: {output_path}")
#             return output_path
            
#         except Exception as e:
#             logger.error(f"❌ GPU static mode failed: {e}")
#             raise
    
#     else:
#         # Ken Burns mode - use full GPU pipeline
#         logger.info(f"🚀 Using full GPU pipeline with Ken Burns effects")
        
#         try:
#             # This does EVERYTHING on GPU: composite creation + effects + encoding
#             result_path = create_video_gpu_direct(
#                 image_path=image_path,
#                 duration=duration,
#                 effect_type=effect_type,  # 'random'
#                 image_index=index,
#                 width=canvas_w,
#                 height=canvas_h,
#                 fps=fps,
#                 output_path=output_path
#             )
            
#             logger.info(f"✅ Ken Burns video created: {result_path}")
#             return result_path
            
#         except Exception as e:
#             logger.error(f"❌ GPU Ken Burns failed: {e}")
#             raise


def create_image_video_with_effects(
    image_path: str,
    duration: float,
    canvas_w: int,
    canvas_h: int,
    output_path: str,
    mode: str,
    index: int,
    kenburns_enabled: bool = False,
    storyboard_enabled: bool = False,
    fps: int = 30,
    effect_override: str = None,
    speed_override: float = None,
    segment_text: str = "",
    head_freeze_sec: float = 0.0,
    video_seed: str = "direct",
    nvenc_semaphore=None,
) -> str:

    logger = logging.getLogger(__name__)

    # ============================================================
    # STEP 1: STORYBOARD MODE
    # ============================================================
    if storyboard_enabled and mode == "original":
        logger.info(f"🎨 Storyboard mode enabled - using storyboard API")
        try:
            from PIL import Image, ImageFilter

            with Image.open(image_path).convert("RGB") as im0:
                img = im0.copy()

            iw, ih = img.size
            needs_bg = abs((iw / ih) - (canvas_w / canvas_h)) > 0.01

            if needs_bg:
                blurred = img.resize((canvas_w, canvas_h)).filter(ImageFilter.GaussianBlur(40))
                scale   = canvas_h / ih
                new_w   = int(iw * scale)
                fore    = img.resize((new_w, canvas_h))
                x       = (canvas_w - new_w) // 2
                comp    = blurred.copy()
                comp.paste(fore, (x, 0))
                composite_path = os.path.splitext(image_path)[0] + f"_composite_{uuid.uuid4()}.jpg"
                comp.save(composite_path, quality=95)
            else:
                composite_path = os.path.splitext(image_path)[0] + f"_resized_{uuid.uuid4()}.jpg"
                img.resize((canvas_w, canvas_h)).save(composite_path, quality=95)

            tune = _storyboard_tuning(segment_fps=fps)

            sb_video_path = _storyboard_video_from_api(
                composite_for_api=composite_path,
                duration=duration,
                split_len=tune["split_len"],
                fps=tune["fps"],
                obj_skip=tune["obj_skip"],
                bg_skip=tune["bg_skip"],
                hold_fraction=tune["hold_fraction"],
            )

            _copy_storyboard_to_output(sb_video_path, output_path)
            os.remove(composite_path)

            logger.info(f"✅ Storyboard video created: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"❌ Storyboard failed: {e} - falling back to GPU pipeline")

    # ============================================================
    # STEP 2: STATIC MODE
    # ============================================================
    if not kenburns_enabled:
        logger.info(f"📸 Static mode")
        try:
            from gpu_kenburns_simple import create_static_video_gpu

            return create_static_video_gpu(
                image_path=image_path,
                duration=duration,
                width=canvas_w,
                height=canvas_h,
                fps=fps,
                output_path=output_path,
                nvenc_semaphore=nvenc_semaphore,
            )
        except Exception as e:
            logger.error(f"❌ Static mode failed: {e}")
            raise

    # ============================================================
    # STEP 3: KEN BURNS — create_video_gpu_direct (ffmpeg default)
    # ============================================================
    from gpu_kenburns_simple import VALID_EFFECTS

    resolved_effect = effect_override if effect_override in VALID_EFFECTS else "zoom_in"
    resolved_speed  = speed_override

    logger.info(
        f"🚀 Ken Burns | effect={resolved_effect} "
        f"speed={resolved_speed if resolved_speed is not None else 'auto'} "
        f"dur={duration:.2f}s freeze={head_freeze_sec:.3f}s"
    )

    try:
        from gpu_kenburns_simple import create_video_gpu_direct

        result_path = create_video_gpu_direct(
            image_path=image_path,
            duration=duration,
            effect_type=resolved_effect,
            image_index=index,
            width=canvas_w,
            height=canvas_h,
            fps=fps,
            output_path=output_path,
            segment_text=segment_text,
            speed_multiplier=resolved_speed,
            head_freeze_sec=head_freeze_sec,
            video_seed=video_seed,
            nvenc_semaphore=nvenc_semaphore,
        )

        logger.info(f"✅ Ken Burns video created: {result_path}")
        return result_path

    except Exception as e:
        logger.error(f"❌ Ken Burns failed: {e}")
        raise


def create_blurred_background_image(
    image_path: str,
    duration: float,
    mode: str,
    storyboard: bool = False,          # compatibility only
    storyboard_split_len: int = 10,
    storyboard_fps: int = 30,
    storyboard_obj_skip: int = 8,
    storyboard_bg_skip: int = 14,
) -> CompositeVideoClip:
    """
    Builds a STILL clip with blurred background if needed.
    Uses global CURRENT_WIDTH / CURRENT_HEIGHT.
    Uses CURRENT_STORYBOARD_ENABLED for storyboard effect.
    Uses CURRENT_KENBURNS_ENABLED for Ken Burns motion.
    """

    target_w, target_h = CURRENT_WIDTH, CURRENT_HEIGHT

    with Image.open(image_path).convert("RGB") as im0:
        img = im0.copy()

    iw, ih = img.size
    needs_bg = abs((iw / ih) - (target_w / target_h)) > 0.01

    # ----------------------------------------------------
    # 1. STATIC BASE IMAGE
    # ----------------------------------------------------
    if needs_bg:
        blurred = img.resize((target_w, target_h)).filter(ImageFilter.GaussianBlur(40))
        scale = target_h / ih
        new_w = int(iw * scale)

        fore = img.resize((new_w, target_h))
        x = (target_w - new_w) // 2

        comp = blurred.copy()
        comp.paste(fore, (x, 0))
        comp_path = os.path.splitext(image_path)[0] + "_composite.jpg"
        comp.save(comp_path, quality=95)

        base_clip = ImageClip(comp_path).set_duration(duration)
        composite_for_api = comp_path

    else:
        resized_path = os.path.splitext(image_path)[0] + "_resized.jpg"
        img.resize((target_w, target_h)).save(resized_path, quality=95)
        base_clip = ImageClip(resized_path).set_duration(duration)
        composite_for_api = resized_path

    # ----------------------------------------------------
    # 2. STORYBOARD MODE
    # ----------------------------------------------------
    if CURRENT_STORYBOARD_ENABLED and mode == "original":
        try:
            sb_clip = _load_storyboard_clip_or_raise(
                composite_for_api=composite_for_api,
                duration=duration,
                split_len=storyboard_split_len,
                fps=storyboard_fps,
                obj_skip=storyboard_obj_skip,
                bg_skip=storyboard_bg_skip,
            )
            return CompositeVideoClip([sb_clip.resize((target_w, target_h))])
        except Exception as e:
            logger.exception(f"[Storyboard] FAILED: {e}. Falling back to static.")

    # ----------------------------------------------------
    # 3. KEN BURNS DISABLED
    # ----------------------------------------------------
    if not CURRENT_KENBURNS_ENABLED:
        logger.warning("[KenBurns] Disabled — returning static image.")
        return CompositeVideoClip([base_clip])

    # ----------------------------------------------------
    # 4. KEN BURNS EFFECT
    # ----------------------------------------------------
    logger.warning("[KenBurns] Applying motion effect.")

    motion_base = base_clip.resize(1.2)

    effects = [
        lambda c: zoom_in_effect(c, zoom_ratio=0.04),
        lambda c: zoom_out_effect(c, zoom_max_ratio=0.3, zoom_out_factor=0.04),
    ]

    chosen_clip = random.choice(effects)(motion_base).set_duration(duration)

    return CompositeVideoClip([chosen_clip])


@memory_safe_operation
def video_manipulation(input_video_path):
    """
    Apply a blur effect to the video using FFmpeg and save the processed video.
    
    Args:
        input_video_path (str): The path to the input video file.
        output_path (str): The path to save the output video file.
        
    Returns:
        str: "Done" when processing is complete.
    """
    # Create a temporary file to store the output video
    timestamp = int(time.time())
    random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    temp_filename = f"temp_video_{timestamp}_{random_str}.mp4"
    temp_output_path = os.path.join(tempfile.gettempdir(), temp_filename)

    # FFmpeg command to apply a blur effect to the video
    command = [
    "ffmpeg",
    "-i", input_video_path,
    "-lavfi",
    "[0:v]split=2[main][blur];"
    "[blur]scale=ih*16/9:-1,"
    "boxblur=luma_radius=min(h\\,w)/20:luma_power=1:"
    "chroma_radius=min(cw\\,ch)/20:chroma_power=1[bg];"
    "[bg][main]overlay=(W-w)/2:(H-h)/2,"
    "crop=h=iw*9/16",
    "-vb", "800K",
    temp_output_path
   ]


    # Run the FFmpeg command
    subprocess.run(command, check=True)

    # Load the resulting video into moviepy and apply an effect
    blurred_clip = VideoFileClip(temp_output_path).resize(height=720, width=1280)
    blurred_clip = blurred_clip.fx(vfx.mirror_x)
    # blurred_clip.fx(vfx.resize,2)
    

    # Delete the temporary file now that it's no longer in use
    # os.remove(temp_output.name)

    return blurred_clip


# ----------------- Media Selection -----------------

# @memory_safe_operation  # Decorator to ensure memory safety
# def find_best_media_combination(media_pool, text_segment, character_name, duration_needed, used_media_dict, media_lock):
#     DEFAULT_IMAGE_DURATION = 5
#     MIN_VIDEO_DURATION = 5
#     MIN_IMAGE_DURATION = 2

#     text_embedding = encode_text(text_segment)

#     # Filter relevant media by character (excluding tagged ones)
#     relevant_media = [m for m in media_pool if m['character'] == character_name and not m.get("tag")]

#     # Sort by similarity, prioritize unused media
#     scored = sorted(
#         [(m, cosine_similarity(text_embedding, m['embedding'])) 
#          for m in relevant_media if m["path"] not in used_media_dict],
#         key=lambda x: x[1], reverse=True
#     )

#     if not scored:
#         scored = sorted(
#             [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#             key=lambda x: x[1], reverse=True
#         )

#     selected = []
#     is_video_used = False

#     for media, score in scored:
#         if media['type'] == 'video' and media['duration'] >= MIN_VIDEO_DURATION:
#             clip_duration = min(media['duration'], duration_needed)
#             remaining = duration_needed - clip_duration

#             with media_lock:
#                 if media["path"] in used_media_dict:
#                     continue
#                 used_media_dict[media["path"]] = True


#             if remaining < 3 and remaining > 0:
#                 image = next((m for m, _ in scored if m['type'] == 'image' and m["path"] not in used_media_dict), None)
#                 if image:
#                     with media_lock:
#                         used_media_dict[image["path"]] = True
#                     selected.append({'media': image, 'duration': remaining, 'has_audio': False})

#             elif remaining >= 3:
#                 if remaining >= 5:
#                     if remaining % 5 == 0:
#                         num_images = remaining // 5
#                         for _ in range(num_images):
#                             image = next((m for m, _ in scored if m['type'] == 'image' and m["path"] not in used_media_dict), None)
#                             if image:
#                                 with media_lock:
#                                     used_media_dict[image["path"]] = True
#                                 selected.append({'media': image, 'duration': DEFAULT_IMAGE_DURATION, 'has_audio': False})
#                     else:
#                         num_images = 2
#                         each = remaining / num_images
#                         for _ in range(num_images):
#                             image = next((m for m, _ in scored if m['type'] == 'image' and m["path"] not in used_media_dict), None)
#                             if image:
#                                 with media_lock:
#                                     used_media_dict[image["path"]] = True
#                                 selected.append({'media': image, 'duration': each, 'has_audio': False})
#                 else:
#                     image = next((m for m, _ in scored if m['type'] == 'image' and m["path"] not in used_media_dict), None)
#                     if image:
#                         with media_lock:
#                             used_media_dict[image["path"]] = True
#                         selected.append({'media': image, 'duration': remaining, 'has_audio': False})

#             selected.insert(0, {
#                 'media': media,
#                 'duration': clip_duration,
#                 'has_audio': not media.get('mute', False)
#             })
#             is_video_used = True
#             break

#     # If no video was used, use only images
#     if not is_video_used:
#         image_list = [m for m, _ in scored if m['type'] == 'image' and m["path"] not in used_media_dict]
#         if not image_list:
#             image_list = [m for m, _ in scored if m['type'] == 'image']

#         if image_list:
#             if duration_needed % 5 == 0:
#                 num_images = duration_needed // 5
#                 for i in range(num_images):
#                     img = image_list[i % len(image_list)]
#                     with media_lock:
#                         used_media_dict[img["path"]] = True
#                     selected.append({
#                         'media': img,
#                         'duration': DEFAULT_IMAGE_DURATION,
#                         'has_audio': False
#                     })
#             else:
#                 num_images = 2 if duration_needed >= 6 else 1
#                 split_duration = duration_needed / num_images
#                 for i in range(num_images):
#                     img = image_list[i % len(image_list)]
#                     with media_lock:
#                         used_media_dict[img["path"]] = True
#                     selected.append({
#                         'media': img,
#                         'duration': split_duration,
#                         'has_audio': False
#                     })

#     return selected

# @memory_safe_operation  # Decorator to ensure memory safety
# def find_best_media_combination(media_pool, text_segment, character_name, duration_needed, used_media_dict, media_lock):
#     DEFAULT_IMAGE_DURATION = 5
#     MIN_VIDEO_DURATION = 5
#     MIN_IMAGE_DURATION = 2
#     MIN_MEDIA_SCREEN_TIME = 3

#     text_embedding = encode_text(text_segment)

#     # Filter relevant media for the character and exclude tagged ones
#     relevant_media = [m for m in media_pool if m['character'] == character_name and not m.get("tag")]

#     # Score and sort all media by similarity (prefer unused)
#     scored = sorted(
#         [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#         key=lambda x: x[1], reverse=True
#     )

#     selected = []
#     remaining_duration = duration_needed

#     # Try to select best combination of media that satisfies constraints
#     for media, score in scored:
#         path = media["path"]
#         if path in used_media_dict:
#             continue

#         # Determine the max safe duration for this media
#         if media['type'] == 'video':
#             if media['duration'] < MIN_VIDEO_DURATION:
#                 continue
#             media_duration = min(media['duration'], remaining_duration)
#         elif media['type'] == 'image':
#             media_duration = min(DEFAULT_IMAGE_DURATION, remaining_duration)
#             if media_duration < MIN_IMAGE_DURATION or media_duration > remaining_duration:
#                 continue
#         else:
#             continue

#         with media_lock:
#             used_media_dict[path] = True

#         selected.append({
#             'media': media,
#             'duration': media_duration,
#             'has_audio': media['type'] == 'video' and not media.get('mute', False)
#         })

#         remaining_duration -= media_duration

#         if remaining_duration <= 0:
#             break

#     # If not enough media were selected to fill the time, fill with more
#     if remaining_duration > 0:
#         available = [m for m, _ in scored if m["path"] not in used_media_dict]
#         max_images = int(remaining_duration // MIN_MEDIA_SCREEN_TIME)
#         num_fill = max_images if max_images > 0 else 1
#         split_duration = remaining_duration / num_fill

#         for i in range(num_fill):
#             media = available[i % len(available)]
#             path = media["path"]

#             if media['type'] == 'video' and media['duration'] < MIN_VIDEO_DURATION:
#                 continue
#             if media['type'] == 'image' and split_duration < MIN_IMAGE_DURATION:
#                 continue

#             with media_lock:
#                 used_media_dict[path] = True

#             selected.append({
#                 'media': media,
#                 'duration': min(split_duration, media.get('duration', split_duration)),
#                 'has_audio': media['type'] == 'video' and not media.get('mute', False)
#             })

#     # Enforce 3s minimum per media if using multiple
#     if len(selected) > 1:
#         total_selected_time = sum(s['duration'] for s in selected)
#         if total_selected_time < duration_needed:
#             scaling_factor = duration_needed / total_selected_time
#             for s in selected:
#                 s['duration'] *= scaling_factor

#         # Enforce 3s minimum screen time
#         selected = [s for s in selected if s['duration'] >= MIN_MEDIA_SCREEN_TIME]

#         # If some got filtered out due to short duration, re-balance again
#         if selected:
#             total = sum(s['duration'] for s in selected)
#             for s in selected:
#                 s['duration'] = (s['duration'] / total) * duration_needed

#     return selected

# @memory_safe_operation
# def find_best_media_combination(media_pool, text_segment, character_name, duration_needed, used_media_dict, media_lock):
#     DEFAULT_IMAGE_DURATION = 5
#     MIN_VIDEO_DURATION = 5
#     MIN_IMAGE_DURATION = 2
#     MIN_MEDIA_SCREEN_TIME = 3

#     text_embedding = encode_text(text_segment)

#     # Step 1: Filter relevant media (exclude tagged)
#     relevant_media = [m for m in media_pool if m['character'] == character_name and not m.get("tag")]
#     if not relevant_media:
#         logging.warning(f"⚠️ No media found for character: {character_name}")
#         return []

#     # Step 2: Score all media for similarity
#     scored = sorted(
#         [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#         key=lambda x: x[1],
#         reverse=True
#     )

#     selected = []
#     remaining_duration = duration_needed

#     # Step 3: Try unused media first
#     for media, score in scored:
#         path = media["path"]
#         if path in used_media_dict:
#             continue

#         media_duration = 0
#         if media['type'] == 'video':
#             if media['duration'] < MIN_VIDEO_DURATION:
#                 continue
#             media_duration = min(media['duration'], remaining_duration)
#         elif media['type'] == 'image':
#             media_duration = min(DEFAULT_IMAGE_DURATION, remaining_duration)
#             if media_duration < MIN_IMAGE_DURATION or media_duration > remaining_duration:
#                 continue
#         else:
#             continue

#         with media_lock:
#             used_media_dict[path] = True

#         selected.append({
#             'media': media,
#             'duration': media_duration,
#             'has_audio': media['type'] == 'video' and not media.get('mute', False)
#         })

#         remaining_duration -= media_duration
#         if remaining_duration <= 0:
#             break

#     # Step 4: Fallback — reuse previously used media
#     if remaining_duration > 0:
#         logging.info(f"♻️ Reusing media for character '{character_name}' to fill remaining {remaining_duration:.2f}s")
#         available = [m for m, _ in scored]  # reuse allowed now

#         max_images = int(remaining_duration // MIN_MEDIA_SCREEN_TIME)
#         num_fill = max(max_images, 1)

#         try:
#             split_duration = remaining_duration / num_fill
#         except ZeroDivisionError:
#             logging.error("❌ Division by zero in media fallback. Using 1 fallback image.")
#             split_duration = remaining_duration

#         for i in range(num_fill):
#             media = available[i % len(available)]
#             path = media["path"]

#             # Skip if too short
#             if media['type'] == 'video' and media['duration'] < MIN_VIDEO_DURATION:
#                 continue
#             if media['type'] == 'image' and split_duration < MIN_IMAGE_DURATION:
#                 continue

#             with media_lock:
#                 used_media_dict[path] = True

#             selected.append({
#                 'media': media,
#                 'duration': min(split_duration, media.get('duration', split_duration)),
#                 'has_audio': media['type'] == 'video' and not media.get('mute', False)
#             })

#     # Step 5: Normalize durations to exactly match duration_needed
#     total_selected_time = sum(s["duration"] for s in selected)

#     if total_selected_time > 0:
#         scaling_factor = duration_needed / total_selected_time
#         for s in selected:
#             s["duration"] *= scaling_factor

#     # Step 6: Filter clips too short after scaling
#     selected = [s for s in selected if s["duration"] >= MIN_IMAGE_DURATION]

#     # Step 7: Fill final gap by extending last media
#     final_time = sum(s["duration"] for s in selected)
#     gap = duration_needed - final_time

#     if selected and gap > 0:
#         selected[-1]["duration"] += gap
#         logging.info(f"🧩 Extended last media by {gap:.2f}s to match total duration.")

#     # Final check
#     if not selected:
#         logging.error(f"❌ No valid media could be selected for: \"{text_segment[:60]}...\"")

#     return selected

#Most important
# def find_best_media_combination(media_pool, text_segment, character_name, duration_needed, used_media_dict, media_lock):
#     DEFAULT_IMAGE_DURATION = 5
#     MIN_VIDEO_DURATION = 5
#     MIN_IMAGE_DURATION = 2
#     MIN_MEDIA_SCREEN_TIME = 3

#     if duration_needed < MIN_MEDIA_SCREEN_TIME:
#         return []

#     text_embedding = encode_text(text_segment)

#     # Step 1: Filter relevant media
#     relevant_media = [m for m in media_pool if m['character'] == character_name and not m.get("tag")]
#     if not relevant_media:
#         logging.warning(f"⚠️ No media found for character: {character_name}")
#         return []

#     # Step 2: Score and sort all relevant media
#     scored = sorted(
#         [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#         key=lambda x: x[1], reverse=True
#     )

#     available_media = []
#     fallback_media = []

#     for media, score in scored:
#         path = media["path"]
#         if path not in used_media_dict:
#             available_media.append(media)
#         else:
#             fallback_media.append(media)

#     # Step 3: Determine how many media items we need
#     media_count = max(1, int(duration_needed // DEFAULT_IMAGE_DURATION))

#     selected = []

#     # Step 4: Select from unused first
#     for media in available_media:
#         if len(selected) >= media_count:
#             break
#         with media_lock:
#             used_media_dict[media["path"]] = True
#         selected.append({
#             'media': media,
#             'duration': 0,
#             'has_audio': media["type"] == "video" and not media.get("mute", False)
#         })

#     # Step 5: If we didn’t get enough, reuse media from fallback
#     if len(selected) < media_count:
#         logging.info(f"♻️ Reusing media for character '{character_name}' to fill segment")
#         needed = media_count - len(selected)
#         reuse_pool = fallback_media[:needed]

#         for media in reuse_pool:
#             selected.append({
#                 'media': media,
#                 'duration': 0,
#                 'has_audio': media["type"] == "video" and not media.get("mute", False)
#             })

#     # Step 6: Assign equal durations across selected
#     if not selected:
#         logging.error(f"❌ No media could be selected for: \"{text_segment[:60]}...\"")
#         return []

#     duration_per_media = duration_needed / len(selected)
#     for s in selected:
#         s["duration"] = duration_per_media

#     return selected

from math import ceil
from collections import defaultdict

import logging
from math import ceil, floor
#last modified b4 tag based audio segments were included
# def find_best_media_combination(media_pool, text_segment, character_name, duration_needed, used_media_dict, media_lock):
#     # --- Configuration Constants ---
#     DEFAULT_IMAGE_DURATION = 5
#     MIN_IMAGE_DURATION = 2
#     MIN_VIDEO_DURATION = 3
#     PREFERRED_VIDEO_DURATION = 5
#     SPLIT_BASE_DURATION = 5
#     DURATION_TOLERANCE = 0.1

#     if duration_needed < MIN_IMAGE_DURATION + MIN_VIDEO_DURATION:
#         return []

#     # --- Step 1: Filter and score media ---
#     text_embedding = encode_text(text_segment)
#     relevant_media = [m for m in media_pool if m['character'] == character_name and not m.get("tag")]

#     if not relevant_media:
#         logging.warning(f"⚠️ No media found for character: {character_name}")
#         return []

#     scored = sorted(
#         [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#         key=lambda x: x[1], reverse=True
#     )

#     # --- OVERRIDE: Use top video if it can cover entire segment ---
#     top_media = scored[0][0]
#     if top_media['type'] == 'video' and top_media.get('duration', 0) >= duration_needed:
#         if top_media['path'] not in used_media_dict:
#             with media_lock:
#                 used_media_dict[top_media['path']] = True
#         return [{
#             'media': top_media,
#             'duration': duration_needed,
#             'has_audio': not top_media.get("mute", False)
#         }]

#     # --- Step 2: Categorize media (unused vs fallback) ---
#     available_media = []
#     fallback_media = []

#     for media, _ in scored:
#         if media['path'] not in used_media_dict:
#             available_media.append(media)
#         else:
#             fallback_media.append(media)

#     # --- Step 3: Decide how many media to use based on 5s blocks ---
#     media_count = max(1, ceil(duration_needed / 5))
#     selected = []

#     # --- Step 4: Pick from available media first ---
#     for media in available_media:
#         if len(selected) >= media_count:
#             break
#         with media_lock:
#             used_media_dict[media["path"]] = True
#         selected.append({
#             'media': media,
#             'duration': 0,
#             'has_audio': media["type"] == "video" and not media.get("mute", False)
#         })

#     # --- Step 5: Reuse fallback media if needed ---
#     if len(selected) < media_count:
#         needed = media_count - len(selected)
#         for media in fallback_media[:needed]:
#             selected.append({
#                 'media': media,
#                 'duration': 0,
#                 'has_audio': media["type"] == "video" and not media.get("mute", False)
#             })

#     # --- Step 6: Duration assignment ---
#     base_duration = duration_needed / len(selected)
#     remaining_duration = duration_needed

#     types_in_selection = {s['media']['type'] for s in selected}
#     mixed_types = len(types_in_selection) > 1

#     for s in selected:
#         media = s['media']
#         media_type = media['type']
#         media_max = media.get('duration', DEFAULT_IMAGE_DURATION if media_type == 'image' else 0)

#         if media_type == 'video':
#             chunks = floor(media_max / SPLIT_BASE_DURATION)
#             max_usable = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
#             s['duration'] = min(max(base_duration, MIN_VIDEO_DURATION), max_usable, remaining_duration)
#         else:
#             s['duration'] = max(base_duration, MIN_IMAGE_DURATION)

#         remaining_duration -= s['duration']

#     # --- Step 7: Redistribute leftover duration ---
#     if remaining_duration > DURATION_TOLERANCE:
#         flexible_media = [s for s in selected if (
#             (s['media']['type'] == 'image') or
#             (s['media']['type'] == 'video' and s['duration'] < s['media']['duration'])
#         )]
#         if flexible_media:
#             extra_per_item = remaining_duration / len(flexible_media)
#             for s in flexible_media:
#                 max_possible = s['media'].get('duration', float('inf')) if s['media']['type'] == 'video' else float('inf')
#                 add = min(extra_per_item, max_possible - s['duration'])
#                 s['duration'] += add
#                 remaining_duration -= add
#                 if remaining_duration <= DURATION_TOLERANCE:
#                     break

#     # --- Final Validation ---
#     total_duration = sum(s['duration'] for s in selected)
#     if abs(total_duration - duration_needed) > DURATION_TOLERANCE:
#         logging.warning(f"⚠️ Duration mismatch: Needed {duration_needed:.2f}s, got {total_duration:.2f}s")

#     return selected




# ============================================================================
# BLACK FRAME CREATION (ImageFX Style)
# ============================================================================

def create_black_frame_image(canvas_w, canvas_h, output_dir="temp"):
    """
    Creates a simple black PNG image that can be used as a fallback.
    Uses ImageFX-style approach - simple and efficient.
    
    Args:
        canvas_w: Width in pixels
        canvas_h: Height in pixels
        output_dir: Directory to save the image
    
    Returns:
        Path to the black frame PNG
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create nearly-black image (10,10,10 to avoid pure black encoding issues)
        black_img = Image.new('RGB', (canvas_w, canvas_h), color=(10, 10, 10))
        
        # Use consistent filename for caching
        black_frame_path = os.path.join(output_dir, f"black_frame_{canvas_w}x{canvas_h}.png")
        
        # Only create if doesn't exist (cache it)
        if not os.path.exists(black_frame_path):
            black_img.save(black_frame_path)
            logging.debug(f"📷 Created black frame: {black_frame_path}")
        else:
            logging.debug(f"📷 Using cached black frame: {black_frame_path}")
        
        return black_frame_path
        
    except Exception as e:
        logging.error(f"❌ Failed to create black frame image: {e}")
        return None


def create_full_segment_fallback(duration, canvas_w, canvas_h, segment_index, audio_path, chunk_folder, reason):
    """
    Creates a complete black video segment when all encoding attempts fail.
    Uses ImageFX approach: black frame image + MoviePy processing.
    
    Args:
        duration: Duration in seconds
        canvas_w: Canvas width
        canvas_h: Canvas height
        segment_index: Segment number
        audio_path: Path to audio file (if exists)
        chunk_folder: Output folder
        reason: Failure reason for logging
    
    Returns:
        Path to fallback video file
    """
    try:
        logging.info(f"🎬 Creating full segment fallback for segment {segment_index}")
        logging.info(f"   Duration: {duration:.2f}s, Resolution: {canvas_w}x{canvas_h}")
        logging.info(f"   Reason: {reason}")
        
        # Get or create black frame image
        black_frame_path = create_black_frame_image(canvas_w, canvas_h, output_dir=chunk_folder)
        
        if not black_frame_path or not os.path.exists(black_frame_path):
            raise RuntimeError("Cannot create black frame image")
        
        # Create video clip from black frame
        video_clip = ImageClip(black_frame_path, duration=duration)
        
        # Add audio if available
        if audio_path and os.path.exists(audio_path):
            try:
                audio_clip = AudioFileClip(audio_path, fps=44100)
                audio_clip = audio_clip.set_fps(44100)
                
                # Match durations
                if audio_clip.duration > duration:
                    audio_clip = audio_clip.subclip(0, duration)
                else:
                    audio_clip = audio_clip.set_duration(duration)
                
                audio_clip = audio_clip.volumex(2)
                video_clip = video_clip.set_audio(audio_clip)
                logging.info(f"   ✅ Audio attached")
                
            except Exception as audio_err:
                logging.warning(f"   ⚠️ Audio attachment failed: {audio_err}")
        else:
            logging.info(f"   ℹ️  No audio - creating silent fallback")
        
        # Generate output path
        fallback_filename = f"fallback_seg_{segment_index}_{uuid.uuid4().hex[:8]}.mp4"
        fallback_path = os.path.join(chunk_folder, fallback_filename)
        
        # Write video
        video_clip.write_videofile(
            fallback_path,
            codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
            audio_codec="aac" if (audio_path and os.path.exists(audio_path)) else None,
            audio_fps=44100 if (audio_path and os.path.exists(audio_path)) else None,
            threads=4,
            fps=30,
            preset="p3" if torch.cuda.is_available() else "medium",
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
            logger=None
        )
        
        # Cleanup
        video_clip.close()
        if audio_path and os.path.exists(audio_path):
            try:
                audio_clip.close()
            except:
                pass
        
        del video_clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Verify
        if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 10000:
            logging.info(f"✅ Fallback created: {fallback_filename}")
            return fallback_path
        else:
            logging.error(f"❌ Fallback file invalid")
            return None
            
    except Exception as e:
        logging.error(f"❌ Fallback creation failed: {e}")
        logging.error(traceback.format_exc())
        return None


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_video_file(path, max_retries=2):
    """
    Verify video file is valid and readable by MoviePy/FFmpeg.
    
    Args:
        path: Path to video file
        max_retries: Number of validation attempts
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(path):
        logging.error(f"❌ Video file does not exist: {path}")
        return False
    
    file_size = os.path.getsize(path)
    if file_size < 10000:  # Less than 10KB
        logging.error(f"❌ Video file too small: {file_size} bytes")
        return False
    
    for attempt in range(max_retries):
        try:
            clip = VideoFileClip(path)
            duration = clip.duration
            width = clip.w
            height = clip.h
            
            # Close immediately
            clip.close()
            if hasattr(clip, 'reader') and clip.reader:
                clip.reader.close()
            del clip
            
            # Validate properties
            if duration <= 0:
                logging.error(f"❌ Video has invalid duration: {duration}s")
                return False
            
            if width <= 0 or height <= 0:
                logging.error(f"❌ Video has invalid dimensions: {width}x{height}")
                return False
            
            logging.debug(f"✅ Video validated: {duration:.2f}s, {width}x{height}")
            return True
            
        except Exception as e:
            logging.warning(f"⚠️ Video validation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
    
    logging.error(f"❌ Video validation failed after {max_retries} attempts: {path}")
    return False


def validate_image_file(path):
    """
    Verify image file is valid and can be opened.
    
    Args:
        path: Path to image file
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(path):
        logging.error(f"❌ Image file does not exist: {path}")
        return False
    
    file_size = os.path.getsize(path)
    if file_size < 1000:  # Less than 1KB
        logging.error(f"❌ Image file too small: {file_size} bytes")
        return False
    
    try:
        img = Image.open(path)
        width, height = img.size
        img.close()
        
        if width <= 0 or height <= 0:
            logging.error(f"❌ Image has invalid dimensions: {width}x{height}")
            return False
        
        logging.debug(f"✅ Image validated: {width}x{height}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Image validation failed: {e}")
        return False



import os
import logging
import time
import requests
import itertools
from moviepy.editor import VideoFileClip, AudioFileClip

# --- Constants ---
PEXELS_API_KEYS = [
    "U68PkQZafYJ3NHqb0BppeQkTrHYNG7kGw5T1qoRoBIw9AAeTwApLMrya",
    "UY6ZwmiRj38FdL3l7rRbK99hx65DoKLcTQ0TEB3O16B5FOfrKydMQhMJ",
    "ekLBeXsr4e1Cpj5aaZ2fKcBQtcQpnmpxB7sV8CAHOObGMQhtUTm9TbQJ",
    "EK0PmgBMNFFN6eK4Q8zyFiqnDzxtR1gSC3mUjTH9vrhy06HkgmiIXeAB"
]

DEFAULT_IMAGE_DURATION = 5
MIN_IMAGE_DURATION = 2
MIN_VIDEO_DURATION = 3
PREFERRED_VIDEO_DURATION = 5
SPLIT_BASE_DURATION = 5
DURATION_TOLERANCE = 0.1

# Cycle between keys forever
_api_key_cycle = itertools.cycle(PEXELS_API_KEYS)

def get_next_api_key():
    return next(_api_key_cycle)


# ============================================================================
# DOWNLOAD FUNCTIONS WITH VALIDATION
# ============================================================================

def download_video(video_url, save_path, max_retries=3, timeout=60):
    """
    Download a video with integrity verification and retry logic.
    
    Args:
        video_url: URL of the video to download
        save_path: Path where video should be saved
        max_retries: Number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        bool: True if download successful and verified, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logging.info(f"📥 Downloading video (attempt {attempt + 1}/{max_retries}): {video_url}")
            
            response = requests.get(video_url, stream=True, timeout=timeout)
            
            if response.status_code != 200:
                logging.warning(f"⚠️ HTTP {response.status_code} for {video_url}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            
            # Get expected file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress tracking
            downloaded = 0
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                
                # Force flush to disk
                f.flush()
                os.fsync(f.fileno())
            
            # Verify download size
            actual_size = os.path.getsize(save_path)
            
            if total_size > 0:
                if actual_size < total_size * 0.95:  # Allow 5% tolerance
                    logging.warning(f"⚠️ Incomplete download: {actual_size}/{total_size} bytes")
                    os.remove(save_path)
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return False
            
            # Verify minimum file size (at least 10KB)
            if actual_size < 10000:
                logging.warning(f"⚠️ File too small: {actual_size} bytes")
                os.remove(save_path)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            
            # Wait for file system to stabilize
            time.sleep(0.5)
            
            # Final existence check
            if not os.path.exists(save_path):
                logging.error(f"❌ File disappeared after download: {save_path}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            
            # Verify video is readable by trying to open it
            if not validate_video_file(save_path):
                logging.warning(f"⚠️ Video file validation failed: {save_path}")
                os.remove(save_path)
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return False
            
            logging.info(f"✅ Video downloaded and verified: {os.path.basename(save_path)} ({actual_size} bytes)")
            return True
            
        except requests.exceptions.Timeout:
            logging.error(f"⏱️ Download timeout for {video_url}")
            if os.path.exists(save_path):
                os.remove(save_path)
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
                
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Download request error: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
                
        except Exception as e:
            logging.error(f"❌ Unexpected download error: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
    
    logging.error(f"💀 Failed to download video after {max_retries} attempts: {video_url}")
    return False


def download_pexels_image(image_url, save_path, max_retries=3, timeout=30):
    """
    Download an image with integrity verification and retry logic.
    
    Args:
        image_url: URL of the image to download
        save_path: Path where image should be saved
        max_retries: Number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        bool: True if download successful and verified, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logging.info(f"📥 Downloading image (attempt {attempt + 1}/{max_retries}): {image_url}")
            
            response = requests.get(image_url, stream=True, timeout=timeout)
            
            if response.status_code != 200:
                logging.warning(f"⚠️ HTTP {response.status_code} for {image_url}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            
            # Get expected file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress tracking
            downloaded = 0
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                
                # Force flush to disk
                f.flush()
                os.fsync(f.fileno())
            
            # Verify download size
            actual_size = os.path.getsize(save_path)
            
            if total_size > 0:
                if actual_size < total_size * 0.95:  # Allow 5% tolerance
                    logging.warning(f"⚠️ Incomplete image download: {actual_size}/{total_size} bytes")
                    os.remove(save_path)
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return False
            
            # Verify minimum file size (at least 1KB)
            if actual_size < 1000:
                logging.warning(f"⚠️ Image file too small: {actual_size} bytes")
                os.remove(save_path)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            
            # Wait for file system to stabilize
            time.sleep(0.3)
            
            # Verify image is valid
            if not validate_image_file(save_path):
                logging.warning(f"⚠️ Image file validation failed: {save_path}")
                os.remove(save_path)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            
            logging.info(f"✅ Image downloaded and verified: {os.path.basename(save_path)} ({actual_size} bytes)")
            return True
            
        except Exception as e:
            logging.error(f"❌ Image download error: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
    
    logging.error(f"💀 Failed to download image after {max_retries} attempts: {image_url}")
    return False



# def search_multiple_pexels_videos(query, max_results=5):
#     api_key = get_next_api_key()
#     headers = {"Authorization": api_key}
#     params = {"query": query, "per_page": max_results, "orientation": "landscape"}
#     url = "https://api.pexels.com/videos/search"
    
#     response = requests.get(url, headers=headers, params=params)

#     if response.status_code != 200:
#         return []

#     results = []
#     for video in response.json().get("videos", []):
#         best_quality = max(video["video_files"], key=lambda f: f["width"])
#         results.append({
#             "id": video["id"],  # keep for uniqueness
#             "url": best_quality["link"],
#             "duration": video["duration"]
#         })
#     return results
def generate_script_context(script: str):
    """
    Extracts a concise story context (4–6 sentences) from the first 600 words of a script.
    Designed for downstream use in AI image or video generation pipelines.
    """

    trimmed_script = " ".join(script.split()[:600])

    system = """
You are a professional script analyst.
Your task is to read the provided story and summarize its core context
in a clear, cinematic way — suitable for visual storytelling and scene generation.

Guidelines:
- Keep it short (4–6 sentences).
- Focus on who the main character is, what happens, and what the story means.
- No dialogue, no quotes, no commentary — just the essence.
- Maintain neutral, narrative tone (not overly dramatic).
- End with the central message or emotional theme.
"""

    user = f"""
Script (first 600 words):
{trimmed_script}

Now write the story context.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ],
        temperature=0.5,
    )

    context_summary = resp.choices[0].message.content.strip()
    return context_summary


# def search_multiple_pexels_media(
#     query,
#     media_type="video",
#     per_page=5,               # default 5; you can pass 20 when needed
#     page=1,
#     orientation="landscape",  # "landscape" | "portrait" | "square"
#     min_width=None,
#     timeout=(10, 30),
#     retries=3
# ):
#     """
#     Returns list of dicts:
#       {
#         "id": <int>,
#         "type": "video"|"image",
#         "url": <direct link>,
#         "duration": <seconds or None>,
#         "width": <int or None>,
#         "height": <int or None>
#       }
#     """
#     assert media_type in ("video", "image"), "media_type must be 'video' or 'image'"
#     base_url = "https://api.pexels.com/videos/search" if media_type == "video" else "https://api.pexels.com/v1/search"
#     params = {"query": query, "per_page": per_page, "page": page, "orientation": orientation}

#     last_err = None
#     for _ in range(retries):
#         headers = {"Authorization": get_next_api_key()}
#         try:
#             resp = requests.get(base_url, headers=headers, params=params, timeout=timeout)
#             if resp.status_code == 429:
#                 last_err = RuntimeError("Pexels rate limit (429)")
#                 continue
#             if resp.status_code != 200:
#                 last_err = RuntimeError(f"Pexels HTTP {resp.status_code}: {resp.text[:200]}")
#                 continue

#             data = resp.json()
#             results = []

#             if media_type == "video":
#                 for video in data.get("videos", []):
#                     files = [f for f in video.get("video_files", []) if str(f.get("file_type", "")).startswith("video")]
#                     if not files:
#                         continue
#                     best = max(files, key=lambda f: (f.get("width") or 0))
#                     w, h = best.get("width"), best.get("height")
#                     if min_width and (w or 0) < min_width:
#                         continue
#                     results.append({
#                         "id": video.get("id"),
#                         "type": "video",
#                         "url": best.get("link"),
#                         "duration": video.get("duration"),
#                         "width": w,
#                         "height": h,
#                     })
#             else:
#                 for photo in data.get("photos", []):
#                     src = photo.get("src", {})
#                     url = src.get("original") or src.get("large2x") or src.get("large") or src.get("medium")
#                     if not url:
#                         continue
#                     w, h = photo.get("width"), photo.get("height")
#                     if min_width and (w or 0) < min_width:
#                         continue
#                     results.append({
#                         "id": photo.get("id"),
#                         "type": "image",
#                         "url": url,
#                         "duration": None,
#                         "width": w,
#                         "height": h,
#                     })

#             return results

#         except Exception as e:
#             last_err = e
#             continue

#     logging.warning(f"⚠️ Pexels search failed after retries. Reason: {last_err}")
#     return []
def search_multiple_pexels_media(
    query,
    media_type="video",
    per_page=5,
    page=1,
    orientation=None,        # <- now optional, global takes priority
    min_width=None,
    timeout=(10, 30),
    retries=3
):
    """
    Pexels search adapted to global orientation:
      - If orientation is provided explicitly, use it.
      - If orientation=None → use CURRENT_ORIENTATION automatically.

    Returns list of dicts:
      {
        "id": <int>,
        "type": "video" | "image",
        "url": <direct link>,
        "duration": <seconds or None>,
        "width": <int or None>,
        "height": <int or None>
      }
    """
    # -----------------------------------------------
    # 🔥 Use GLOBAL ORIENTATION if not passed manually
    # -----------------------------------------------
    global CURRENT_ORIENTATION
    if orientation is None:
        # convert global orientation into Pexels orientation names
        if CURRENT_ORIENTATION == "portrait":
            orientation = "portrait"
        elif CURRENT_ORIENTATION == "landscape":
            orientation = "landscape"
        else:
            orientation = "square"

    assert media_type in ("video", "image"), "media_type must be 'video' or 'image'"

    base_url = (
        "https://api.pexels.com/videos/search"
        if media_type == "video"
        else "https://api.pexels.com/v1/search"
    )

    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "orientation": orientation,
    }

    last_err = None

    for _ in range(retries):
        headers = {"Authorization": get_next_api_key()}

        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=timeout)

            if resp.status_code == 429:
                last_err = RuntimeError("Pexels rate limit (429)")
                continue

            if resp.status_code != 200:
                last_err = RuntimeError(f"Pexels HTTP {resp.status_code}: {resp.text[:200]}")
                continue

            data = resp.json()
            results = []

            # -------------------------------
            # VIDEO SEARCH
            # -------------------------------
            if media_type == "video":
                for video in data.get("videos", []):
                    files = [
                        f for f in video.get("video_files", [])
                        if str(f.get("file_type", "")).startswith("video")
                    ]
                    if not files:
                        continue

                    best = max(files, key=lambda f: (f.get("width") or 0))
                    w, h = best.get("width"), best.get("height")

                    if min_width and (w or 0) < min_width:
                        continue

                    results.append({
                        "id": video.get("id"),
                        "type": "video",
                        "url": best.get("link"),
                        "duration": video.get("duration"),
                        "width": w,
                        "height": h,
                    })

            # -------------------------------
            # IMAGE SEARCH
            # -------------------------------
            else:
                for photo in data.get("photos", []):
                    src = photo.get("src", {})
                    url = (
                        src.get("original") or src.get("large2x")
                        or src.get("large") or src.get("medium")
                    )
                    if not url:
                        continue

                    w, h = photo.get("width"), photo.get("height")

                    if min_width and (w or 0) < min_width:
                        continue

                    results.append({
                        "id": photo.get("id"),
                        "type": "image",
                        "url": url,
                        "duration": None,
                        "width": w,
                        "height": h,
                    })

            return results

        except Exception as e:
            last_err = e
            continue

    logging.warning(f"⚠️ Pexels search failed after retries. Reason: {last_err}")
    return []


def generate_pexels_query(text_segment, segment_index, media_type="video"):
    mode = choose_query_mode(text_segment, segment_index)

    if media_type == "image":
        if mode == "normal":
            prompt = f"""
Given this narration or text segment, generate a short, clear search query to find images on pexels that best suit it.Respond with just the query.

Text: "{text_segment}"
"""
        else:
            prompt = f"""
Given this narration or text segment, generate a short, visually representative search query for Pexels Images.
Use realistic abstract or metaphorical imagery to capture the mood, tone, or meaning of the text, while still producing something that could realistically be filmed.
Keep it concise. Respond with just the query.

Text: "{text_segment}"
"""
    else:
        if mode == "normal":
            prompt = f"""
Given this narration or text segment, generate a short, clear search query to find the most visually relevant stock video on Pexels.
Focus on mood, setting, and visual keywords — avoid abstract ideas. Respond with just the query.

Text: "{text_segment}"
"""
        else:
            prompt = f"""
Given this narration or text segment, generate a short, visually representative search query for Pexels videos.
You may use abstract or metaphorical imagery to capture the mood, tone, or meaning of the text, while still producing something that could realistically be filmed.
Keep it concise. Respond with just the query.

Text: "{text_segment}"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

import hashlib

def _seed(s: str) -> int:
    return int(hashlib.sha1((s or "").encode("utf-8")).hexdigest(), 16) % (2**31)

import re

# openrouter_fx_prompts.py
# Self-contained: uses OpenRouter + DeepSeek V3.1 (free) to generate imageFX prompts.



def _trim_prompt(p: str, max_chars: int) -> str:
    p = " ".join((p or "").split())
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")) or (p.startswith("`") and p.endswith("`")):
        p = p[1:-1].strip()
    if len(p) > max_chars:
        cut = p[:max_chars]
        if "," in cut:
            cut = cut.rsplit(",", 1)[0]
        p = cut.rstrip()
    return p

def _append_style_spec(line: str, aspect_hint: str, style_spec: str | None, max_chars: int) -> str:
    # ensure cinematic + framing markers (helpful for consistency)
    low = line.lower()
    if "cinematic" not in low and "film still" not in low:
        line += ", cinematic film still"
    if aspect_hint and aspect_hint not in line:
        line += f", {aspect_hint} framing"
    # append unified style spec only when provided
    if style_spec:
        ss_low = style_spec.strip().lower()
        if ss_low not in low:
            line += ", " + style_spec.strip()
    # tidy
    line = re.sub(r"\s+,", ",", line)
    line = re.sub(r",\s*,", ", ", line).strip(" ,")
    return _trim_prompt(line, max_chars)

#     return prompts
def get_prev_two_segments(segments, idx):
    """
    segments: list of dicts or strings (e.g., [{'text': '...'}, ...] or ['...'])
    idx: current segment index
    returns: list of up to two previous segment texts, oldest -> most recent
    """
    if idx <= 0:
        return []
    start = max(0, idx - 4)
    prev = segments[start:idx]
    # normalize to text
    out = []
    for s in prev:
        out.append(s["text"] if isinstance(s, dict) else str(s))
    return out
def get_prev_two_context_text(segments, idx, sep=" "):
    """Convenience: join the last two previous segments into one context string."""
    return sep.join(get_prev_two_segments(segments, idx))




# def generate_image_fx_prompts(
#     text_segment: str,
#     prev_segments,

#     *,
#     style_spec: str | None = None,        # None => auto-pick best representation from the text
   
#     override_prompt: str | None = None,        # optional single override
#     override_prompts: list[str] | None = None, # optional per-shot overrides
# ):
#     """
#     Returns `count` explanatory prompts for the segment.
#     - Default (style_spec=None): GPT chooses the best-fitting visual treatment from the text.
#     - With style_spec: GPT must follow that style consistently across all prompts.
#     """

#     system = f"""
# You are a professional video editor crafting prompt for scene generation from provided text.
# Rules:
# - Understand the narration `{text_segment}` literally (who/what/where/mood).
# -Use Previous narration segment `{prev_segments}` to understand flow and create consistent and relatable prompts for the current segment.
# - Your output is NOT text for captions; it is a SCENE description an illustrator/model can render.
# - Make the prompt explanatory and readable: clear pose/gesture, simple prop(s), and foreground/midground staging.
# - Keep the prompt precisely detailed so the scene is unambiguous.
# - No text/logos/watermarks/UI in the image.
# - Return a SINGLE scene prompt.
# """

#     user = f"""
# Previous Segment Narration: "{prev_segments}"
# Narration: "{text_segment}"
# NOW FROM THE ABOVE NARRATION CRAFT AN ENGAGINGLY ACCURATE SCENE PROMPT TO REPRESENT IT USING THIS STYLE {style_spec} AND USING PREVIOUS NARRATION AS CONTEXT {prev_segments} TO ENSURE SMOOTH TRANSITION AND CONTINUITY
# Bans: no text, no logos, no watermarks
# Return exactly ONE scene prompt which which is designed according to {style_spec}.
# """

#     # Call your OpenAI client (assumes `client` is configured)
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "system", "content": system.strip()},
#                   {"role": "user", "content": user.strip()}],
#         temperature=0.5,
#     )

#     # Extract the single prompt from response
#     prompt = resp.choices[0].message.content.strip()
    
#     # Handle overrides if provided
#     if override_prompt:
#         return [override_prompt]
#     elif override_prompts and override_prompts:
#         return [override_prompts[0]]
    
#     # Return the single prompt AS A LIST
#     return [prompt]

# def generate_image_fx_prompts(
#     text_segment: str,
#     script_context: str,
#     prev_segments: str = "",
#     *,
#     style_spec: str | None = "explanatory",
#     override_prompt: str | None = None,
#     override_prompts: list[str] | None = None,
# ):
#     """
#     Generates one high-quality, context-aware image prompt for AI video generation.

#     Fixes:
#     - Maintains segment-level accuracy (no overgeneralization)
#     - Keeps global continuity via context and previous narration
#     - Auto-handles short narrations by blending with previous segment
#     """

#     # 🧩 STYLE LIBRARY
#     STYLE_LIBRARY = {
#         "explanatory": (
#             "hand-drawn pencil sketch; stick-figure characters; black graphite line art; "
#             "clean white background; medium line weight; light cross-hatching; no color; "
#             "simple, educational, storyboard-friendly layout."
#         ),
#         "explanatory_dark": (
#             "hand-drawn black and white pencil sketch; stick-figure characters or simple objects in a calm, subdued atmosphere; "
#             "lighting may vary—dim indoor, twilight, candlelight, or overcast outdoor scenes; "
#             "black graphite line art with medium, consistent line weight and balanced contrast; "
#             "light cross-hatching for shading and depth; no color fills; "
#             "clean contours and uncluttered composition with clear foreground and background separation; "
#             "cinematic yet minimal storybook mood, adaptable to different low-light or reflective scenes."
#         ),
# "ghibli": (
#     "Studio Ghibli–style hand-painted animation; soft watercolor brushstrokes and painterly textures; "
#     "pastel color palette with warm, nostalgic atmosphere; "
#     "natural sunlight and gentle ambient lighting creating subtle glow and depth; "
#     "characters are stylized, cartoon-like, and expressive — never realistic or photographic; "
#     "no real humans, no photorealism, no CGI or 3D rendering; "
#     "balanced cinematic composition with clear foreground, midground, and background layers; "
#     "scenes feel alive, whimsical, and emotionally warm, rooted in nature and everyday beauty; "
#     "emphasis on emotion, calmness, and storybook charm reminiscent of Studio Ghibli films such as Spirited Away, My Neighbor Totoro, and Howl’s Moving Castle; "
#     "no harsh outlines, no text or logos; "
#     "the entire image must clearly read as a 2D animated cartoon in Studio Ghibli’s painterly, timeless aesthetic."
# )
# ,
#         "war": (
#             "hyper realistic cinematic depiction inspired by early 20th-century warfare; "
#             "anonymous soldiers and civilians in historically inspired uniforms, weapons, and vehicles; "
#             "generic wartime environments such as trenches, ruined towns, fields, and camps; "
#             "neutral educational tone—no glorification, violence, or identifiable people; "
#             "soft overcast natural lighting; sepia or muted monochrome palette; "
#             "fine film grain texture and subtle smoke or dust for realism; "
#             "balanced composition emphasizing atmosphere and emotion over identity; "
#             "consistent visual continuity across scenes; "
#             "avoid text, logos, photorealistic faces, exaggerated cinematic effects, "
#             "national flags, or direct references to real historical figures, countries, or symbols. "
#             "If you must refer to a person, use descriptive language only — never real names."
#         ),
#         "cinematic": (
#             "cinematic composition; realistic lighting; expressive characters; depth of field; "
#             "emotional atmosphere; balanced contrast and color harmony."
#         ),
#         "dark_story": (
#             "noir-inspired mood; deep shadows; limited light sources; expressive lighting; "
#             "dramatic tension; emotional intensity."
#         ),
#         "luxury_motivation": (
#             "hyper-realistic detail; cinematic lighting; luxury environments; smooth reflections; "
#             "soft focus; gold/white tones; motivational aesthetic."
#         ),
#     }

#     # 🎨 Select style description
#     style_description = STYLE_LIBRARY.get(style_spec.lower(), style_spec or "")

#     # 🩹 Handle extremely short narration by blending with previous segment
#     if len(text_segment.split()) < 15 and prev_segments:
#         text_segment = f"{prev_segments.strip()} {text_segment.strip()}"

#     # 🧠 Build role messages
#     system = """
# You are a precise video-to-image prompt generator.
# Your goal is to visualize THIS segment accurately while maintaining visual continuity with prior scenes.

# Rules:
# - The CURRENT narration is the main focus.
# - Use the story context ONLY to understand the world, tone, and characters.
# - Use the previous narration ONLY to preserve lighting, perspective, or emotional continuity.
# - Describe a single vivid scene for AI image generation (no lists or camera directions).
# - Avoid summaries or generalizations — depict exactly what this narration implies.
# - No text, logos, UI elements, or artistic signatures.
# """

#     user = f"""
# STORY CONTEXT (background understanding only):
# {script_context}

# PREVIOUS NARRATION (for continuity):
# {prev_segments or "None — this is the opening scene."}

# CURRENT NARRATION (main focus to visualize):
# {text_segment}

# STYLE SPECIFICATION:
# {style_description}

# TASK:
# Describe one coherent, specific visual scene that literally illustrates the CURRENT narration.
# Keep it faithful to this segment while staying consistent with the story tone and prior visuals.
# Do not summarize or generalize — show what is happening in this moment.
# """

#     # ⚙️ Call the model
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system.strip()},
#             {"role": "user", "content": user.strip()},
#         ],
#         temperature=0.35,  # Lower for accuracy and consistency
#     )

#     prompt = resp.choices[0].message.content.strip()

#     # 🔄 Override options (manual)
#     if override_prompt:
#         return [override_prompt]
#     if override_prompts:
#         return override_prompts[:1]

#     return [prompt]

# def generate_image_fx_prompts(
#     text_segment: str,
#     script_context: str,
#     prev_segments: str = "",
#     *,
#     style_spec: str | None = "explanatory",
#     override_prompt: str | None = None,
#     override_prompts: list[str] | None = None,
# ):
#     """
#     Generates one high-quality, context-aware image prompt for AI video generation.

#     Fixes:
#     - Maintains segment-level accuracy (no overgeneralization)
#     - Keeps global continuity via context and previous narration
#     - Auto-handles short narrations by blending with previous segment
#     """

#     # 🧩 STYLE LIBRARY (original preserved, only extended)
#     STYLE_LIBRARY = {
#         "explanatory": (
#     "hand-drawn pencil sketch with selective splashes of color; mostly monochrome graphite line art enhanced with soft watercolor accents where necessary (for emphasis or emotion); "
#     "clean white background; medium line weight; light cross-hatching for depth; minimal shading; simple, educational, storyboard-friendly composition arranged in clear panels or frames; "
#     "maintain consistent perspective and spacing for sequential storytelling; characters appear as expressive stick-figures when humans are needed, while objects, buildings, or animals are drawn as simplified cartoon sketches; "
#     "expressive poses and gestures prioritized over detail; artistic balance between clarity, warmth, and narrative flow."
# ),

#         "explanatory_dark": (
#             "hand-drawn black and white pencil sketch; stick-figure characters or simple objects in a calm, subdued atmosphere; "
#             "lighting may vary—dim indoor, twilight, candlelight, or overcast outdoor scenes; "
#             "black graphite line art with medium, consistent line weight and balanced contrast; "
#             "light cross-hatching for shading and depth; no color fills; "
#             "clean contours and uncluttered composition with clear foreground and background separation; "
#             "cinematic yet minimal storybook mood, adaptable to different low-light or reflective scenes."
#         ),
#         "ghibli": (
#             "Studio Ghibli–style hand-painted animation; soft watercolor brushstrokes and painterly textures; "
#             "pastel color palette with warm, nostalgic atmosphere; "
#             "natural sunlight and gentle ambient lighting creating subtle glow and depth; "
#             "characters are stylized, cartoon-like, and expressive — never realistic or photographic; "
#             "no real humans, no photorealism, no CGI or 3D rendering; "
#             "balanced cinematic composition with clear foreground, midground, and background layers; "
#             "scenes feel alive, whimsical, and emotionally warm, rooted in nature and everyday beauty; "
#             "emphasis on emotion, calmness, and storybook charm reminiscent of Studio Ghibli films such as Spirited Away, My Neighbor Totoro, and Howl’s Moving Castle; "
#             "no harsh outlines, no text or logos; "
#             "the entire image must clearly read as a 2D animated cartoon in Studio Ghibli’s painterly, timeless aesthetic."
#         ),
#         # "war": (
#         #     """""hyper-realistic cinematic depiction inspired by early 20th-century warfare; anonymous soldiers and civilians shown in historically accurate but non-specific uniforms, weapons, and vehicles; environments include muddy trenches, ruined European towns, open countryside battlefields, coastal fortifications, and temporary military camps; soft overcast daylight with natural shadows and muted atmospheric haze; sepia-tinted or monochrome color palette evoking 1930–1945 film photography; subtle smoke, drifting dust, and fine film-grain texture for realism; balanced wide 16:9 composition with realistic depth of field and gentle motion blur; foreground elements like mud, barbed wire, and scattered equipment add depth; camera perspective resembles a war-era documentary still captured with a Leica camera — grounded, archival, and emotionally restrained; tone is neutral, historical, and educational, focusing on atmosphere and human endurance rather than heroism or violence; emphasize mood, weather, and environment over identity; maintain consistent visual continuity across all generated scenes so that each feels part of the same historical world; avoid any text, logos, watermarks, photorealistic celebrity faces, exaggerated cinematic lighting, blood, gore, nationalism, glorification, fantasy effects, or real historical figures, countries, or symbols; if a person must be referred to, describe only their role or action, never a real name."""""
#         # ),
#         "war":(

#     "hyper-realistic cinematic depiction inspired by early 20th-century warfare; "
#     "anonymous soldiers and civilians shown in historically accurate but non-specific uniforms, weapons, and vehicles; "
#     "environments include muddy trenches, ruined towns, open countryside battlefields, coastal fortifications, and temporary military camps; "
#     "soft overcast daylight with diffused natural shadows and atmospheric haze; "
#     "render entirely in black-and-white analog film photography style with soft silver-gray tonal range, "
#     "lifted blacks, and moderate mid-tone contrast; "
#     "simulate authentic aged film texture using fine grain, faint scratches, and light vignette; "
#     "balanced 16:9 composition with realistic depth of field and gentle motion blur; "
#     "foreground elements such as mud, barbed wire, and scattered gear add depth and realism; "
#     "camera perspective resembles a 1940s Leica or Contax war documentary still—grounded, archival, and emotionally restrained; "
#     "tone remains neutral, historical, and educational, emphasizing atmosphere and human endurance over heroism or violence; "
#     "focus on mood, weather, and environment rather than identity; "
#     "maintain consistent visual continuity across all scenes for a cohesive historical world; "
#     "avoid any text, logos, watermarks, photorealistic celebrity faces, exaggerated lighting, blood, gore, nationalism, glorification, fantasy effects, "
#     "or direct references to real historical figures, countries, or symbols; "
#     "if people appear, describe only their roles or actions—never real names."
#         ),
#         "cinematic": (
#             "cinematic composition; realistic lighting; expressive characters; depth of field; "
#             "emotional atmosphere; balanced contrast and color harmony."
#         ),
#         "dark_story": (
#             "noir-inspired mood; deep shadows; limited light sources; expressive lighting; "
#             "dramatic tension; emotional intensity."
#         ),
#         "luxury_motivation": (
#             "hyper-realistic detail; cinematic lighting; luxury environments; smooth reflections; "
#             "soft focus; gold/white tones; motivational aesthetic."
#         ),
#         # 🆕 Additional styles added for flexibility (do not overwrite existing)
#         "chalk": (
#             "chalk pastel illustration style; visible strokes and textures; soft color blending; "
#             "lightly glowing pastel hues; artistic yet readable composition on dark or textured background; "
#             "no text, logos, or gradients."
#         ),
#         "sketch": (
#             "simple pencil sketch style; medium line weight; minimal shading; clear outlines; "
#             "focus on gesture and shape clarity; clean white background; storyboard-friendly layout."
#         ),
#         "comic": (
#             "2D comic book art style; bold outlines; flat color panels; dynamic poses; "
#             "dramatic expressions; cinematic framing; no text bubbles or typography."
#         ),
#     }

#     # 🎨 Select style description
#     style_description = STYLE_LIBRARY.get(style_spec.lower(), style_spec or "")

#     # 🩹 Handle short narration by blending with previous segment
#     # if len(text_segment.split()) < 15 and prev_segments:
#     #     text_segment = f"{prev_segments.strip()} {text_segment.strip()}"

#     # 🔁 Override support (NEW)
#     if override_prompt:
#         style_description=override_prompt
#     if override_prompts:
#         return override_prompts[:1]

#     # 🧠 Build prompt messages (original preserved)
# #     system = """
# # You are a precise video-to-image prompt generator.
# # Your goal is to visualize THIS segment accurately while maintaining visual continuity with prior scenes.

# # Rules:
# # - The CURRENT narration is the main focus.
# # - Use the story context ONLY to understand the world, tone, and characters.
# # - Use the previous narration ONLY to preserve lighting, perspective, or emotional continuity.
# # - Describe a single vivid scene for AI image generation (no lists or camera directions).
# # - Avoid summaries or generalizations — depict exactly what this narration implies.
# # - No text, logos, UI elements, or artistic signatures.
# # """

# #     user = f"""
# # STORY CONTEXT (background understanding only):
# # {script_context}

# # PREVIOUS NARRATION (for continuity):
# # {prev_segments or "None — this is the opening scene."}

# # CURRENT NARRATION (main focus to visualize):
# # {text_segment}

# # STYLE SPECIFICATION:
# # {style_description}

# # TASK:
# # Describe one coherent, specific visual scene that literally illustrates the CURRENT narration.
# # Keep it faithful to this segment while staying consistent with the story tone and prior visuals.
# # Do not summarize or generalize — show what is happening in this moment.
# # """
#     system = f"""
# You are a professional video-to-image scene designer creating clear, visually coherent prompts.
# Rules:
# - Interpret the narration literally to visualize characters, actions, and settings.
# - Preserve visual and emotional continuity between previous and current narration.
# - Focus on clarity, composition, and storytelling logic.
# - No text, logos, watermarks, or UI elements.
# - Return ONE scene prompt — no lists or captions.
# """

#     user = f"""
# Previous Narration:"{prev_segments}"
# Current Narration: "{text_segment}"

# Style Guide: {style_description}

# OUR NARRATION IS PART OF A STORY. USE PREVIOUS NARRATION TO UNDERSTAND WHAT CAME BEFORE CURRENT NARRATION AND TAKING CONTEXT FROM IT UNDERSTAND WHAT CURRENT NARRATION IS ABOUT

# Now create one strong, descriptive scene prompt that accurately represents the narration
# and transitions naturally from the previous scene, while following the style guide.

# """
#     # ⚙️ Model call (unchanged)
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system.strip()},
#             {"role": "user", "content": user.strip()},
#         ],
#         temperature=0.35,  # Lower for accuracy and consistency
#     )

#     prompt = resp.choices[0].message.content.strip()

#     return [prompt]
# def generate_image_fx_prompts(
#     text_segment: str,
#     script_context: str,
#     prev_segments: list[str] = None,
#     *,
#     style_spec: str | None = "explanatory",
#     override_prompt: str | None = None,
#     override_prompts: list[str] | None = None,
# ):
#     """
#     Generates ONE high-quality, context-aware image prompt
#     that visually continues the story from previous segments.
#     """

#     # ensure prev_segments is a list
#     if prev_segments is None:
#         prev_segments = []

#     # combine last 3 narration segments for stronger continuity
#     continuity_text = "\n".join(prev_segments[-3:]) if prev_segments else "None — this is the opening scene."

#     # ------------------------------
#     # 🔹 STYLE LIBRARY
#     # ------------------------------
#     STYLE_LIBRARY = {
#         "explanatory": (
#             "hand-drawn pencil sketch with selective splashes of color; mostly monochrome graphite line art enhanced "
#             "with soft watercolor accents; clean white background; medium line weight; light cross-hatching; clear, "
#             "educational storyboard composition; expressive stick-figures; simplified cartoon objects; clarity-first."
#         ),

#         "explanatory_dark": (
#             "hand-drawn black-and-white pencil sketch; dim, subdued mood; medium graphite line weight; light shading; "
#             "clean contours; atmospheric but minimal; cinematic low-light storybook tone; no color."
#         ),

#         "ghibli": (
#             "Studio Ghibli hand-painted watercolor look; soft pastel tones; gentle ambient lighting; expressive stylized "
#             "characters; natural warm atmosphere; layered cinematic composition; no realism, no 3D."
#         ),

#         "war": (
#             "hyper-realistic black-and-white analog war photography style; soft gray tonal range; moderate contrast; "
#             "fine grain; mild scratches; 1940s Leica reportage look; trenches or ruined towns; atmospheric haze; "
#             "realistic documentary tone without glorification."
#         ),

#         "cinematic": (
#             "cinematic composition; expressive characters; dramatic lighting; shallow depth of field; emotional tone."
#         ),

#         "dark_story": (
#             "noir style; deep shadows; small light pools; expressive contrast; tense dramatic composition."
#         ),

#         "luxury_motivation": (
#             "hyper-realistic luxury aesthetic; gold/white palette; polished reflections; cinematic lighting; elegant mood."
#         ),

#         "chalk": (
#             "chalk pastel drawing; textured strokes; soft glow; blended color patches; artful, minimal composition."
#         ),

#         "sketch": (
#             "simple pencil sketch; medium lines; minimal shading; clean white background; storyboard-friendly."
#         ),

#         "comic": (
#             "2D comic art style; bold outlines; flat colors; dramatic poses; cinematic framing; no text bubbles."
#         ),
#     }

#     # select style description
#     style_description = STYLE_LIBRARY.get(style_spec.lower(), style_spec or "")

#     # override prompt direct return
#     if override_prompt:
#         return [override_prompt]

#     if override_prompts:
#         return override_prompts[:1]

#     # ------------------------------
#     # 🔥 SYSTEM PROMPT (continuity-enforced)
#     # ------------------------------
#     system = """
# You are a professional storyboard image designer for video generation.

# Your TOP PRIORITY is VISUAL CONTINUITY between scenes.

# Rules:
# 1. Every new scene MUST visually continue from the previous scenes.
#    - reuse the same characters, environment, lighting, objects, and emotional tone
#    - unless narration explicitly changes them.

# 2. Interpret the CURRENT narration literally,
#    but anchor it inside the SAME WORLD created by previous narration.

# 3. If narration is ambiguous:
#    - keep same setting
#    - keep same perspective
#    - keep same characters
#    - keep same mood

# 4. Absolutely NO:
#    - TEXT
#    - speech bubbles
#    - logos
#    - watermarks
#    - UI elements
#    - multiple scene options

# 5. Output ONE coherent scene description.

# 6. PRIORITY order:
#    CONTINUITY → CURRENT NARRATION → STYLE → CREATIVITY.
# """.strip()

#     # ------------------------------
#     # 🔥 USER PROMPT (context-aware)
#     # ------------------------------
#     user = f"""
# STORY CONTEXT (background world understanding):
# {script_context}

# PREVIOUS SCENES (last 1–3 narrations, defines where the story visually is):
# {continuity_text}

# CURRENT NARRATION (must be illustrated literally):
# {text_segment}

# STYLE GUIDE:
# {style_description}

# TASK:
# Create ONE detailed visual scene that:
# - depicts the CURRENT narration,
# - continues naturally from the previous scenes,
# - preserves the same characters, environment, lighting, and emotional continuity,
# - and follows the style guide.

# Only describe the image. No extra commentary.
# """.strip()

#     # ------------------------------
#     # 🔥 MODEL CALL
#     # ------------------------------
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user",  "content": user},
#         ],
#         temperature=0.25,  # tuned for stability & continuity
#     )

#     prompt = resp.choices[0].message.content.strip()

#     return [prompt]

def generate_image_fx_prompts(
    text_segment: str,
    script_context: str,  # unused but preserved for compatibility
    prev_segments: list[str] = None,
    *,
    style_spec: str | None = "cinematic",
    override_prompt: str | None = None,
    override_prompts: list[str] | None = None,
):
    """
    SMART CONTEXTUAL PROMPT GENERATOR
    Dynamically reads previous narration and current narration together.
    If the two are semantically related (same character/scene), the prompt continues naturally.
    If not, it resets and generates a fresh scene.

    Space style only: uses layered visual-bible prompt format (see helper.imagefx_styles).
    """
    from helper.imagefx_styles import generate_space_image_fx_prompts, is_space_style

    if is_space_style(style_spec):
        return generate_space_image_fx_prompts(
            text_segment,
            script_context,
            prev_segments,
            client=client,
            append_style_fn=_append_style_spec,
            override_prompt=override_prompt,
        )

    if prev_segments is None:
        prev_segments = []

    continuity_text = "\n".join(prev_segments[-2:]) if prev_segments else "None — this is the opening scene."

    STYLE_LIBRARY = {
        "explanatory": (
    "hand-drawn stick figure illustration with selective splashes of vibrant color; predominantly monochrome graphite "
    "stick-man line art enhanced with deliberate soft watercolor washes in key areas; pristine clean white background "
    "with ample negative space; simple stick figures with circular heads, straight-line bodies and limbs; consistent "
    "medium line weight (0.5-0.7mm equivalent) for stick figure outlines; expressive stick-man poses with dynamic "
    "gestures and clear body language; hands shown as simple circles or mitten shapes; feet as small ovals or dots; "
    "facial expressions using minimal dots and curves (two dots for eyes, simple curved line for mouth); light, "
    "directional cross-hatching only for background elements and depth; well-organized educational storyboard "
    "composition with clear visual hierarchy and logical flow between stick figure scenes; simplified geometric shapes "
    "for objects (circles, squares, triangles); smooth tonal gradients where minimal shading appears; crisp stick-figure "
    "outlines with deliberate placement of color accents (blues, warm oranges, soft greens, gentle yellows) on clothing "
    "or key objects; minimal atmospheric effects focused on clarity; high contrast between ink-black stick-figure "
    "linework and white background; strategic use of empty space to guide eye movement; motion lines and simple arrows "
    "to show movement and direction; teaching illustration quality with stick-man characters as protagonists; bright, "
    "optimistic lighting implied through minimal shadows; clean pencil strokes with no smudging; watercolor blooms "
    "contained and purposeful on stick figure clothing or objects; infographic-style visual storytelling with stick "
    "people; accessibility-first design; NO TEXT, NO LABELS, NO WRITTEN WORDS IN IMAGE; purely visual narrative using "
    "expressive stick figures; clarity-first aesthetic with every stick-man element serving educational purpose; "
    "kindergarten-friendly stick figure art style elevated with professional composition and selective color."
),

        "explanatory_dark": (
            "hand-drawn pencil sketch with selective splashes of dark color; mostly monochrome charcoal and graphite with "
"heavy shadows; murky grey or aged parchment background; varying line weights with emphasis on thick, jagged strokes; "
"dense cross-hatching and heavy shading in corners; unsettling storyboard composition; angular, distorted figures "
"with exaggerated proportions; warped, unsettling objects; atmospheric fog or mist effects; high contrast between "
"deep blacks and pale grays; occasional crimson, deep purple, or sickly green accents; scratchy, nervous linework; "
"vignette darkening at edges; ominous negative space; tension-first aesthetic. KEEP IT AI IMAGE GENERATOR SAFE (NO PUBLIC UNSAFE PROMPT) SO IT DOES NOT CENSOR IT."
"DO NOT METION PEOPLE NAMES TOO AS IT IS NOT AI GENERATION SAFE ONLY GIVE PROMPT THAT DESCRIBES THE CHARACTERS AND SCENE"
        ),

        "ghibli": (
            "Studio Ghibli hand-painted watercolor look; soft pastel tones; gentle ambient lighting; expressive stylized "
            "characters; natural warm atmosphere; layered cinematic composition; no realism, no 3D."
        ),

        "war": (
            "hyper-realistic black-and-white analog war photography style; soft gray tonal range; moderate contrast; "
            "fine grain; mild scratches; 1940s Leica reportage look; trenches or ruined towns; atmospheric haze; "
            "realistic documentary tone without glorification."
        ),

        "cinematic": (
            "cinematic composition; expressive characters; dramatic lighting; shallow depth of field; emotional tone."
        ),

        "dark_story": (
            "noir style; deep shadows; small light pools; expressive contrast; tense dramatic composition."
        ),
"explanatory_illustrated": (
    "hand-drawn cartoon illustration style with bold ink outlines and selective color application; clean black "
    "ink linework defining all forms; simplified cartoon anatomy with rounded, expressive features; solid black "
    "fill areas for hair, shadows, and clothing elements creating strong graphic contrast; white or cream faces with "
    "minimal facial details - simple dots for eyes, basic curves for mouths and expressions; selective muted color "
    "applied only to key clothing or objects - deep burgundy, maroon, plum, forest green, or navy tones; "
    "visible sketch lines and directional hatching on colored areas showing texture, folds, and dimension; "
    "background on textured paper with visible grain, fibers, and subtle imperfections; aged paper aesthetic with "
    "warm cream, beige, or off-white base tone; soft watercolor or ink wash stains creating atmospheric cloudy effects "
    "in warm gray, subtle purple, dusty pink, or pale blue; organic wash bleeding and gradient transitions on paper surface; "
    "paper texture maintained throughout - never pure digital smoothness; slight paper warping or texture bumps visible; "
    "gentle vignette with darker wash concentrating around edges; figures positioned with clear silhouettes against "
    "textured paper backgrounds; stark contrast between solid black fill areas and white character elements; "
    "sketchbook or art journal page aesthetic; hand-drawn authenticity with traditional media feel; minimal shading "
    "achieved through linework rather than gradients; warm, tactile overall quality; scan-like paper grain and subtle "
    "noise throughout; clean compositions with thoughtful negative space; consistent medium line weight; "
    "NO TEXT IN IMAGE; versatile style adaptable to any character type or scene with authentic paper texture foundation"
),

"fitness_video": (
    "Explanatory scene representation for the text for a fitness video in hand-drawn pencil style featuring cartoonish "
"gym figures. Depict in predominantly black graphite line art on a clean white background with selective splashes of "
"vibrant watercolor; consistent medium line weight (0.5-0.7mm equivalent) for character outlines; cartoonish gym "
"figures with exaggerated muscular definition and dynamic athletic proportions; expressive faces showing effort, "
"determination, and motivation using bold eyebrows and clear mouth expressions; crisp black pencil contours with "
"deliberate placement of soft watercolor washes in key areas (energetic reds and warm oranges on abdominal muscles "
"to highlight core engagement and six-pack definition, cool blues and teals on gym equipment like dumbbells, benches, "
"and resistance bands, vibrant greens for healthy food elements and supplements, bright yellows for energy bursts "
"and motivational emphasis); minimal shading using light directional cross-hatching for depth and muscle definition; "
"watercolor blooms contained and purposeful on specific muscle groups being activated, workout gear, or nutritional "
"concepts; clean uncluttered layout with ample negative space; storyboard-friendly composition showing exercise "
"progressions with clear visual hierarchy and logical flow between workout scenes; well-organized foreground/midground "
"separation with gym figures prominent; smooth tonal gradients where color accents appear on core areas during "
"contraction; high contrast between ink-black linework and pristine white background; strategic use of empty space "
"to guide eye through exercise sequences; motion lines and directional arrows in simple black strokes showing proper "
"form and movement flow; 16:9 framing always; neutral paper-like white background with subtle texture optional; "
"camera at eye-level unless low/high angle better explains the exercise concept; teaching illustration quality with "
"cartoonish gym athletes demonstrating abdominal exercises (crunches, planks, leg raises, bicycle kicks, mountain "
"climbers, hanging knee raises); hands gripping equipment or supporting bodyweight shown with clear anatomy; feet "
"positioned to show proper stance and balance; bright optimistic lighting implied through minimal shadows; clean "
"pencil strokes with no smudging; color washes enhance without overwhelming the black line art foundation; "
"infographic-style fitness storytelling; high-energy motivational aesthetic; gym setting elements simplified to "
"essential forms (weight plates as circles, benches as rectangles, mats as simple planes); NO photorealism, NO 3D, "
"NO gradients beyond watercolor blooms, NO digital painting, NO complex rendering, NO logos, NO watermarks, "
"NO text, NO labels, NO typography in image; purely visual narrative using expressive cartoonish gym figures with "
"athletic builds; clarity-first exercise instruction with every element serving educational purpose for abs training "
"and six-pack development; hand-drawn sketch aesthetic elevated with professional fitness instruction composition "
"and selective strategic color highlighting on activated muscle groups, equipment, and motivational focal points."
),
        "luxury_motivation": (
        "cinematic documentary-style photography with dramatic natural lighting; high-contrast realism with deep shadows "
    "and golden hour warmth; gritty urban or minimalist environments; sharp focus on human determination and effort; "
    "desaturated color palette with selective warm amber, burnt orange, or steel blue accents; strong directional light "
    "creating powerful silhouettes; textured surfaces—concrete, weathered wood, morning mist; dynamic composition with "
    "leading lines and rule of thirds; shallow depth of field emphasizing singular focus; raw, unpolished aesthetic; "
    "atmospheric dust particles or morning fog; strong geometric shapes and clean lines; moody pre-dawn or late evening "
    "ambiance; emphasis on solitary figures in motion—running, climbing, writing, building; visible sweat, effort, "
    "concentration; inspirational negative space suggesting possibility; subtle lens flare or backlit subjects; "
    "professional editorial photography style; authentic human emotion without staged smiles; environmental storytelling "
    "through minimal props—journal, weights, laptop, blueprint; tension between struggle and breakthrough; grounded, "
    "aspirational, and viscerally real. SAFE FOR AI IMAGE GENERATION, NO RESTRICTED CONTENT"),

        "chalk": (
           "digital oil painting style with classical Renaissance technique enforced throughout; rich impasto brushwork texture "
"with visible paint buildup and thick layered appearance on all surfaces; dramatic chiaroscuro lighting mandatory - "
"single warm light source (amber, golden ochre, candlelight quality) creating strong contrast against deep shadows; "
"deep burnt umber, sepia, and raw umber shadows with no pure black backgrounds; painterly directional brushstrokes "
"visible in every area - faces, clothing, backgrounds, objects; smooth blended transitions between light and shadow "
"using traditional glazing and scumbling techniques; strict muted earthy color palette: burnt sienna for skin midtones, "
"raw umber for darks, golden yellow ochre for highlights, deep warm blacks, venetian red accents only; Rembrandt triangle "
"lighting on faces where applicable; textured canvas weave visible throughout composition - never smooth digital surface; "
"varnished finish appearance with subtle sheen and aged masterwork quality; atmospheric sfumato in backgrounds with "
"softer edges and reduced detail; foreground figures painted with sharper definition and heavier impasto; strategic "
"golden highlights from single light source on faces, hands, and focal points; warm ambient glow with cool neutral shadows; "
"classical portrait composition rules - clear focal hierarchy, balanced negative space in darker areas, emotional depth "
"through light modeling; contemporary modern subjects in present-day clothing and settings painted with timeless technique; "
"realistic human anatomy and proportions with expressive faces showing psychological nuance; gestural loose brushwork in "
"backgrounds, tighter controlled strokes on faces; visible paint texture with directional flow suggesting form and volume; "
"layered paint appearance with underpainting showing through in shadow areas; traditional oil painting color mixing - "
"no digital gradients or airbrush effects; warm color temperature dominant with selective cool accents; cinematic staging "
"merged with fine art composition; emotional storytelling through light, shadow, and brushwork; 16:9 aspect ratio maintained; "
"canvas texture grain and subtle imperfections visible; paint buildup on highlights creating dimensional quality; "
"NO photorealism, NO smooth digital painting, NO flat illustration, NO 3D rendering, NO graphic design elements, "
"NO vector art, NO anime style, NO pure black backgrounds, NO text, NO logos, NO watermarks, NO typography; "
"consistent oil-on-canvas aesthetic across all subjects and scenes with authentic traditional painting media feel"
        ),
        
        
        "female_chalk": (
    "digital oil painting style with classical Renaissance technique enforced throughout; rich impasto brushwork texture "
    "with visible paint buildup and thick layered appearance on all surfaces; dramatic chiaroscuro lighting mandatory - "
    "single warm light source (amber, golden ochre, candlelight quality) creating strong contrast against deep shadows; "
    "deep burnt umber, sepia, and raw umber shadows with no pure black backgrounds; painterly directional brushstrokes "
    "visible in every area - faces, clothing, backgrounds, objects; smooth blended transitions between light and shadow "
    "using traditional glazing and scumbling techniques; strict muted earthy color palette: burnt sienna for skin midtones, "
    "raw umber for darks, golden yellow ochre for highlights, deep warm blacks, venetian red accents only; Rembrandt triangle "
    "lighting on faces where applicable; textured canvas weave visible throughout composition - never smooth digital surface; "
    "varnished finish appearance with subtle sheen and aged masterwork quality; atmospheric sfumato in backgrounds with "
    "softer edges and reduced detail; foreground figures painted with sharper definition and heavier impasto; strategic "
    "golden highlights from single light source on faces, hands, and focal points; warm ambient glow with cool neutral shadows; "
    "classical portrait composition rules - clear focal hierarchy, balanced negative space in darker areas, emotional depth "
    "through light modeling; contemporary modern subjects in present-day clothing and settings painted with timeless technique; "
    "DEFAULT TO FEMALE SUBJECTS: when gender is not explicitly specified, depict women and female figures as primary subjects; "
    "strong, empowered female presence with confident posture and commanding composition; women as central protagonists and "
    "focal points of narrative; include male figures only when explicitly requested or when supporting female-centered stories; "
    "realistic human anatomy and proportions with expressive faces showing psychological nuance; gestural loose brushwork in "
    "backgrounds, tighter controlled strokes on faces; visible paint texture with directional flow suggesting form and volume; "
    "layered paint appearance with underpainting showing through in shadow areas; traditional oil painting color mixing - "
    "no digital gradients or airbrush effects; warm color temperature dominant with selective cool accents; cinematic staging "
    "merged with fine art composition; emotional storytelling through light, shadow, and brushwork; 16:9 aspect ratio maintained; "
    "canvas texture grain and subtle imperfections visible; paint buildup on highlights creating dimensional quality; "
    "NO photorealism, NO smooth digital painting, NO flat illustration, NO 3D rendering, NO graphic design elements, "
    "NO vector art, NO anime style, NO pure black backgrounds, NO text, NO logos, NO watermarks, NO typography; "
    "consistent oil-on-canvas aesthetic across all subjects and scenes with authentic traditional painting media feel"
),
        "sketch": (
            "simple pencil sketch; medium lines; minimal shading; clean white background; storyboard-friendly."
        ),

        "comic": (
            "2D comic art style; bold outlines; flat colors; dramatic poses; cinematic framing; no text bubbles."
        ),
    }

    style_description = STYLE_LIBRARY.get(style_spec.lower(), style_spec or "")

    if override_prompt:
       style_description= override_prompt
    if override_prompts:
       style_description= override_prompts[:1]

    system = """
You are a professional cinematic visual designer and story continuity analyst.

Your task is to generate ONE detailed VISUAL image description for the current narration.

Before generating the image, you must decide:
- Does the current narration naturally continue the same subject, scene, or perspective as the previous narration?
- Or does it introduce a new focus (new character, new place, or new time)?

If it continues the same thread → preserve the same main character(s), environment, lighting, and tone.
If it shifts focus → start a fresh visual composition that fits only the current narration.

Rules:
1. Treat previous narration as visual memory, not as mandatory context.
2. Infer semantic connection like a human reader — if the sentences flow naturally, they are related.
3. Prioritize clarity of subject: do not mix characters or merge unrelated scenes.
4. Avoid text, captions, or symbolic metaphors. Keep visuals grounded and cinematic.
5. DO NOT METION PEOPLE NAMES AS IT IS NOT AI GENERATION SAFE ONLY GIVE PROMPT THAT DESCRIBES THE CHARACTERS AND SCENE
6. KEEP IT AI IMAGE GENERATOR SAFE (NO PUBLIC UNSAFE PROMPT) SO IMAGE GENERATOR DOES NOT FLAG IT DOES NOT CENSOR IT BREAKING OUR PIPELINE.
""".strip()

    user = f"""
PREVIOUS NARRATION:
{continuity_text}

CURRENT NARRATION:
{text_segment.strip()}

STYLE GUIDE:
{style_description}

Key principles:
1. Focus on WHAT IS HAPPENING in the text - the verbs, actions, and movements
2. Identify WHO or WHAT is performing the action
3. Capture the SETTING and ATMOSPHERE
4. Visualize the EMOTIONAL TONE
5. Keep it literal and grounded - represent what the text actually describes
6. DO NOT add narrative elements not present in the text
7. DO NOT mention people's names - describe characters by appearance, role, or action instead
8. KEEP IT SAFE for AI image generators - no content that would be flagged or censored

TASK:
Create a single, detailed visual scene description that:
1. Identifies the main action(s) or event(s) in the text
2. Describes who/what is involved (by appearance/role, NOT by name)
3. Captures the setting and environment
4. Reflects the emotional tone and atmosphere
5. Applies the specified visual style consistently
6. Results in a clear, specific image that an AI generator can create

Focus on concrete, visual elements. Avoid abstract concepts.

Return ONLY the final scene description — no commentary or reasoning.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.65,
    )

    prompt = resp.choices[0].message.content.strip()
    return [prompt]

# def generate_image_fx_prompts(
#     text_segment: str,
#     script_context: str = None,  # unused but preserved for compatibility
#     prev_segments: list[str] = None,  # unused but preserved for compatibility
#     *,
#     style_spec: str | None = "cinematic",
#     override_prompt: str | None = None,
#     override_prompts: list[str] | None = None,
# ):
#     """
#     SIMPLIFIED ACTION-FOCUSED PROMPT GENERATOR
#     Takes a text segment and generates a visual description that accurately
#     represents the actions, subjects, and scene described in that segment only.
#     No continuity tracking - each segment is independent.
#     """

#     # ------------------------------
#     # 🎨 STYLE LIBRARY
#     # ------------------------------
#     STYLE_LIBRARY = {
#         "explanatory": (
#             "hand-drawn stick figure illustration with selective splashes of vibrant color; predominantly monochrome graphite "
#             "stick-man line art enhanced with deliberate soft watercolor washes in key areas; pristine clean white background "
#             "with ample negative space; simple stick figures with circular heads, straight-line bodies and limbs; consistent "
#             "medium line weight (0.5-0.7mm equivalent) for stick figure outlines; expressive stick-man poses with dynamic "
#             "gestures and clear body language; hands shown as simple circles or mitten shapes; feet as small ovals or dots; "
#             "facial expressions using minimal dots and curves (two dots for eyes, simple curved line for mouth); light, "
#             "directional cross-hatching only for background elements and depth; well-organized educational storyboard "
#             "composition with clear visual hierarchy and logical flow between stick figure scenes; simplified geometric shapes "
#             "for objects (circles, squares, triangles); smooth tonal gradients where minimal shading appears; crisp stick-figure "
#             "outlines with deliberate placement of color accents (blues, warm oranges, soft greens, gentle yellows) on clothing "
#             "or key objects; minimal atmospheric effects focused on clarity; high contrast between ink-black stick-figure "
#             "linework and white background; strategic use of empty space to guide eye movement; motion lines and simple arrows "
#             "to show movement and direction; teaching illustration quality with stick-man characters as protagonists; bright, "
#             "optimistic lighting implied through minimal shadows; clean pencil strokes with no smudging; watercolor blooms "
#             "contained and purposeful on stick figure clothing or objects; infographic-style visual storytelling with stick "
#             "people; accessibility-first design; NO TEXT, NO LABELS, NO WRITTEN WORDS IN IMAGE; purely visual narrative using "
#             "expressive stick figures; clarity-first aesthetic with every stick-man element serving educational purpose; "
#             "kindergarten-friendly stick figure art style elevated with professional composition and selective color."
#         ),

#         "explanatory_dark": (
#             "hand-drawn pencil sketch with selective splashes of dark color; mostly monochrome charcoal and graphite with "
#             "heavy shadows; murky grey or aged parchment background; varying line weights with emphasis on thick, jagged strokes; "
#             "dense cross-hatching and heavy shading in corners; unsettling storyboard composition; angular, distorted figures "
#             "with exaggerated proportions; warped, unsettling objects; atmospheric fog or mist effects; high contrast between "
#             "deep blacks and pale grays; occasional crimson, deep purple, or sickly green accents; scratchy, nervous linework; "
#             "vignette darkening at edges; ominous negative space; tension-first aesthetic. KEEP IT AI IMAGE GENERATOR SAFE (NO PUBLIC UNSAFE PROMPT) SO IT DOES NOT CENSOR IT."
#             "DO NOT MENTION PEOPLE NAMES TOO AS IT IS NOT AI GENERATION SAFE ONLY GIVE PROMPT THAT DESCRIBES THE CHARACTERS AND SCENE"
#         ),

#         "ghibli": (
#             "Studio Ghibli hand-painted watercolor look; soft pastel tones; gentle ambient lighting; expressive stylized "
#             "characters; natural warm atmosphere; layered cinematic composition; no realism, no 3D."
#         ),

#         "war": (
#             "hyper-realistic black-and-white analog war photography style; soft gray tonal range; moderate contrast; "
#             "fine grain; mild scratches; 1940s Leica reportage look; trenches or ruined towns; atmospheric haze; "
#             "realistic documentary tone without glorification."
#         ),

#         "cinematic": (
#             "cinematic composition; expressive characters; dramatic lighting; shallow depth of field; emotional tone."
#         ),

#         "dark_story": (
#             "noir style; deep shadows; small light pools; expressive contrast; tense dramatic composition."
#         ),

#         "explanatory_illustrated": (
#             "hand-drawn cartoon illustration style with bold ink outlines and selective color application; clean black "
#             "ink linework defining all forms; simplified cartoon anatomy with rounded, expressive features; solid black "
#             "fill areas for hair, shadows, and clothing elements creating strong graphic contrast; white or cream faces with "
#             "minimal facial details - simple dots for eyes, basic curves for mouths and expressions; selective muted color "
#             "applied only to key clothing or objects - deep burgundy, maroon, plum, forest green, or navy tones; "
#             "visible sketch lines and directional hatching on colored areas showing texture, folds, and dimension; "
#             "background on textured paper with visible grain, fibers, and subtle imperfections; aged paper aesthetic with "
#             "warm cream, beige, or off-white base tone; soft watercolor or ink wash stains creating atmospheric cloudy effects "
#             "in warm gray, subtle purple, dusty pink, or pale blue; organic wash bleeding and gradient transitions on paper surface; "
#             "paper texture maintained throughout - never pure digital smoothness; slight paper warping or texture bumps visible; "
#             "gentle vignette with darker wash concentrating around edges; figures positioned with clear silhouettes against "
#             "textured paper backgrounds; stark contrast between solid black fill areas and white character elements; "
#             "sketchbook or art journal page aesthetic; hand-drawn authenticity with traditional media feel; minimal shading "
#             "achieved through linework rather than gradients; warm, tactile overall quality; scan-like paper grain and subtle "
#             "noise throughout; clean compositions with thoughtful negative space; consistent medium line weight; "
#             "NO TEXT IN IMAGE; versatile style adaptable to any character type or scene with authentic paper texture foundation"
#         ),

#         "fitness_video": (
#             "energetic hand-drawn illustration for fitness and exercise content; dynamic stick figures or simplified anatomical "
#             "figures in active poses - running, jumping, lifting, stretching, planking; clear body positioning with emphasis on "
#             "correct form and movement; bold, confident linework in charcoal or dark gray; vibrant color splashes highlighting "
#             "muscle groups in action - bright orange for arms, electric blue for legs, energetic yellow for core; warm off-white "
#             "or light cream background; motion lines and directional arrows showing movement flow and exercise progression; "
#             "multiple figure positions showing exercise sequences or rep counts; anatomical accuracy in simplified form with "
#             "visible joint positions; sweat drops or energy bursts illustrated as simple graphic elements; equipment drawn as "
#             "basic geometric shapes - dumbbells as rectangles with circles, yoga mats as simple rectangles; encouraging body "
#             "language with figures showing effort and determination; clean, motivational aesthetic; strategic use of red or "
#             "orange accents for emphasis on target areas; minimal background detail to keep focus on movement; infographic-style "
#             "layout with visual flow between exercises; breathing indicators shown with simple curved lines; proper spacing "
#             "between figure demonstrations; professional fitness instruction quality; energetic yet clear composition; "
#             "selective shading on muscles to show engagement; before-and-after position comparisons side-by-side; "
#             "accessible visual guidance for all fitness levels; NO TEXT IN IMAGE; purely visual movement instruction; "
#             "high-energy, motivational atmosphere with clean execution"
#         ),

#         "luxury_motivation": (
#             "cinematic documentary-style photography with dramatic natural lighting; high-contrast realism with deep shadows "
#             "and golden hour warmth; gritty urban or minimalist environments; sharp focus on human determination and effort; "
#             "desaturated color palette with selective warm amber, burnt orange, or steel blue accents; strong directional light "
#             "creating powerful silhouettes; textured surfaces—concrete, weathered wood, morning mist; dynamic composition with "
#             "leading lines and rule of thirds; shallow depth of field emphasizing singular focus; raw, unpolished aesthetic; "
#             "atmospheric dust particles or morning fog; strong geometric shapes and clean lines; moody pre-dawn or late evening "
#             "ambiance; emphasis on solitary figures in motion—running, climbing, writing, building; visible sweat, effort, "
#             "concentration; inspirational negative space suggesting possibility; subtle lens flare or backlit subjects; "
#             "professional editorial photography style; authentic human emotion without staged smiles; environmental storytelling "
#             "through minimal props—journal, weights, laptop, blueprint; tension between struggle and breakthrough; grounded, "
#             "aspirational, and viscerally real. SAFE FOR AI IMAGE GENERATION, NO RESTRICTED CONTENT"
#         ),

#         "chalk": (
#             "hand-drawn 2D character illustration with simple, clean linework; cartoon figure with rounded features and "
#             "expressive face; black ink outlines with consistent medium line weight; character has simplified anatomy with "
#             "circular head, defined hair with flowing strokes, large expressive eyes, and basic body proportions; "
#             "selective color application with muted, desaturated palette - deep burgundy or maroon for clothing, black for "
#             "pants and sleeves; character shown in casual standing pose with natural gesture; soft watercolor or digital "
#             "wash background in warm gray, beige, and subtle purple-pink tones; atmospheric cloudy or misty effect behind "
#             "figure created with blended gradients; textured, slightly grainy background with organic paint-like quality; "
#             "gentle vignette darkening at edges; figure positioned off-center with thoughtful negative space; minimal shading "
#             "on character using simple cell-shading technique; white highlights on face for dimension; character design suitable "
#             "for educational or narrative content; approachable, relatable protagonist style; clean digital illustration "
#             "quality with hand-drawn authenticity; warm cream or soft tan overall tone; balanced contrast between dark linework "
#             "and gentle background; webcomic or animated explainer aesthetic; figure maintains clear silhouette; professional "
#             "character design with personality and emotional expression; suitable for storytelling and philosophical content; "
#             "NO TEXT IN IMAGE; purely visual character-focused composition"
#         ),

#         "sketch": (
#             "simple pencil sketch; medium lines; minimal shading; clean white background; storyboard-friendly."
#         ),

#         "comic": (
#             "2D comic art style; bold outlines; flat colors; dramatic poses; cinematic framing; no text bubbles."
#         ),
#     }

#     style_description = STYLE_LIBRARY.get(style_spec.lower(), style_spec or "")

#     if override_prompt:
#         style_description = override_prompt
#     if override_prompts:
#         style_description = override_prompts[0]

#     # ------------------------------
#     # 🧠 SYSTEM PROMPT — Action Extraction Focus
#     # ------------------------------
#     system = """
# You are a professional visual scene designer specializing in extracting and visualizing key actions from text.

# Your task: Generate ONE detailed visual image description that accurately represents the ACTIONS, SUBJECTS, and SCENE described in the given text segment.

# Key principles:
# 1. Focus on WHAT IS HAPPENING in the text - the verbs, actions, and movements
# 2. Identify WHO or WHAT is performing the action
# 3. Capture the SETTING and ATMOSPHERE
# 4. Visualize the EMOTIONAL TONE
# 5. Keep it literal and grounded - represent what the text actually describes
# 6. DO NOT add narrative elements not present in the text
# 7. DO NOT mention people's names - describe characters by appearance, role, or action instead
# 8. KEEP IT SAFE for AI image generators - no content that would be flagged or censored

# Think like a storyboard artist: your job is to create a single frame that captures the essence of this moment.
# """.strip()

#     # ------------------------------
#     # 🎬 USER PROMPT — Direct Action Visualization
#     # ------------------------------
#     user = f"""
# TEXT SEGMENT TO VISUALIZE:
# {text_segment.strip()}

# VISUAL STYLE TO APPLY:
# {style_description}

# TASK:
# Create a single, detailed visual scene description that:
# 1. Identifies the main action(s) or event(s) in the text
# 2. Describes who/what is involved (by appearance/role, NOT by name)
# 3. Captures the setting and environment
# 4. Reflects the emotional tone and atmosphere
# 5. Applies the specified visual style consistently
# 6. Results in a clear, specific image that an AI generator can create

# Focus on concrete, visual elements. Avoid abstract concepts unless they can be represented visually.

# Return ONLY the final scene description - no explanation or commentary.
# """.strip()

#     # ------------------------------
#     # 🔥 MODEL CALL
#     # ------------------------------
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user", "content": user},
#         ],
#         temperature=0.7,  # balanced creativity and consistency
#     )

#     prompt = resp.choices[0].message.content.strip()
#     return [prompt]

# import os
# import re
# from openai import OpenAI
# from typing import Optional, List

# # Configure OpenRouter client
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
# )

# def generate_image_fx_prompts(
#     text_segment: str,
#     prev_segments: str,
#     count: int,
#     *,
#     style_spec: Optional[str] = None,
#     aspect_hint: str = "16:9",
#     style_hint: str = "cinematic, film still",
#     detail_level: str = "high detail",
#     camera_hint: str = "35mm lens, shallow depth of field",
#     lighting_hint: str = "cinematic lighting, soft key light, gentle rim light",
#     max_chars: int = 260,
#     override_prompt: Optional[str] = None,
#     override_prompts: Optional[List[str]] = None,
# ) -> List[str]:
#     """
#     Returns a single explanatory prompt for the segment using DeepSeek V3.1 (free).
#     Matches the behavior of your original paid version.
#     """
    
#     context = (
#         "All narrations focus on personal growth, discipline, and self-mastery — delivered in a motivational, "
#         "cinematic tone that feels raw, real, and transformative. Every prompt you generate must visually represent "
#         "the emotions and ideas described in the narration — such as overthinking, energy drain, self-discipline, "
#         "focus, or cutting toxic habits — through symbolic or lifestyle visuals."
#     )
    
#     system = f"""
# You are a professional video editor crafting prompt for scene generation from provided text.
# Rules:
# - Understand the narration `{text_segment}` literally (who/what/where/mood).
# - Use Previous narration segment `{prev_segments}` to understand flow and create consistent and relatable prompts for the current segment.
# - Your output is NOT text for captions; it is a SCENE description an illustrator/model can render.
# - Make the prompt explanatory and readable: clear pose/gesture, simple prop(s), and foreground/midground staging.
# - Keep the prompt precisely detailed so the scene is unambiguous.
# - No text/logos/watermarks/UI in the image.
# - Return a SINGLE scene prompt.
# """.strip()

#     user = f"""
# Previous Segment Narration: "{prev_segments}"
# Narration: "{text_segment}"
# NOW FROM THE ABOVE NARRATION CRAFT AN ENGAGINGLY ACCURATE SCENE PROMPT TO REPRESENT IT USING THIS STYLE {style_spec} AND FOLLOWING THIS CONTEXT:{context} AND USING PREVIOUS NARRATION {prev_segments} TO ENSURE SMOOTH TRANSITION AND CONTINUITY
# Bans: no text, no logos, no watermarks
# Return exactly ONE scene prompt which is designed according to {style_spec}.
# """.strip()

#     # Handle overrides first (same as original)
#     if override_prompt:
#         return [override_prompt]
#     elif override_prompts and override_prompts:
#         return [override_prompts[0]]
    
#     # Call DeepSeek V3.1 (free) via OpenRouter
#     resp = client.chat.completions.create(
#         model="openai/gpt-oss-20b:free",
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user", "content": user},
#         ],
#         temperature=0.5,
#     )

#     # Extract the single prompt from response
#     prompt = resp.choices[0].message.content.strip()
    
#     # Return the single prompt AS A LIST (matching original behavior)
#     return [prompt]

def choose_query_mode(text_segment: str, segment_index: int) -> str:
    """
    Returns 'normal' most of the time, but every 3rd segment (1-based: 3, 6, 9…)
    we classify the text and let GPT decide between 'normal' and 'abstract'.
    """
    # Most of the time: normal
    if (segment_index + 1) % 3 != 0:
        return "normal"

    # Only every 3rd segment: lightweight classifier
    prompt = f"""
Classify the best visual query mode for stock footage.
If the text is concrete (places/actions/objects), output: normal
If it is metaphorical/introspective/emotion-heavy, output: abstract

Text:
\"\"\"{text_segment.strip()}\"\"\"

Respond with one word: normal or abstract.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    label = resp.choices[0].message.content.strip().lower()
    return "abstract" if label.startswith("abstract") else "normal"

import os, time, uuid, json, subprocess, requests
from contextlib import contextmanager

PEXELS_DIR = "pexels_videos"
os.makedirs(PEXELS_DIR, exist_ok=True)

@contextmanager
def file_lock(lock_path, timeout=120):
    # Very simple cross-process lock: create-exclusive file, else wait
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            try:
                yield
            finally:
                try: os.remove(lock_path)
                except: pass
            return
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Lock timeout: {lock_path}")
            time.sleep(0.3)

def _remux_or_reencode(src, dst):
    # Try fast remux to fix index/duration
    cmd = ["ffmpeg","-y","-loglevel","error","-i",src,"-movflags","+faststart","-c","copy","-fflags","+genpts","-map","0",dst]
    if subprocess.run(cmd).returncode == 0 and os.path.getsize(dst) > 0:
        return True
    # Fallback: full re-encode to CFR 30
    cmd = ["ffmpeg","-y","-loglevel","error","-i",src,"-r","30","-pix_fmt","yuv420p","-c:v","libx264","-c:a","aac",dst]
    return subprocess.run(cmd).returncode == 0 and os.path.getsize(dst) > 0

def fetch_pexels_asset(video_id: str, url: str):
    """
    Process-safe: multiple workers can call this; only one actually downloads.
    Returns absolute validated path or None.
    """
    base = os.path.join(PEXELS_DIR, f"pexels_{video_id}.mp4")
    okflag = base + ".ok.json"
    lock = base + ".lock"

    # If someone already finished, use it
    if os.path.exists(base) and os.path.exists(okflag):
        return os.path.abspath(base)

    with file_lock(lock):
        # Double-check after taking lock
        if os.path.exists(base) and os.path.exists(okflag):
            return os.path.abspath(base)

        tmp = base + ".part"
        # Download
        with requests.get(url, stream=True, timeout=(10, 120)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024*1024):
                    if chunk: f.write(chunk)

        # Validate size
        if os.path.getsize(tmp) < 256_000:
            try: os.remove(tmp)
            except: pass
            return None

        # Fix container/index
        fixed = base + ".fixed"
        if not _remux_or_reencode(tmp, fixed):
            try: os.remove(tmp)
            except: pass
            try: os.remove(fixed)
            except: pass
            return None

        os.replace(fixed, base)
        try: os.remove(tmp)
        except: pass

        # Mark good with metadata
        with open(okflag, "w") as f:
            json.dump({"video_id": video_id, "url": url, "ts": time.time()}, f)

        return os.path.abspath(base)
    
def probe_duration(path):
    # Light-weight safety net; or trust your API duration after remux
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", path],
            stderr=subprocess.STDOUT
        )
        return float(out.decode().strip())
    except:
        return 0.0



# def find_best_media_combination(media_pool, segment, character_name, duration_needed, used_media_dict, media_lock, mode, segment_index):
#     def loop_tagged_media(tagged_media, duration_needed):
#         looped, total, i = [], 0.0, 0
#         while total < duration_needed and tagged_media:
#             media = tagged_media[i % len(tagged_media)]
#             media_duration = media.get("duration", DEFAULT_IMAGE_DURATION if media["type"] == "image" else 0)
#             duration_to_use = min(media_duration, duration_needed - total)
#             looped.append({
#                 "media": media,
#                 "duration": duration_to_use,
#                 "has_audio": False  # tagged media always muted
#             })
#             with media_lock:
#                 used_media_dict[media["path"]] = True
#             total += duration_to_use
#             i += 1
#         return looped

#     # 1) Tag override (always wins)
#     tag = segment.get("tag")
#     if tag:
#         tagged_media = [m for m in media_pool if m.get('tag') == tag]
#         if tagged_media:
#             return loop_tagged_media(tagged_media, duration_needed)
#         else:
#             logging.info(f"🔁 No tagged media found for tag '{tag}' — falling back to selection")

#     # 2) Pexels path
#     if segment.get("pexels_required", False):
#         query = generate_pexels_query(segment.get("text", ""), segment_index)
#         logging.info(f"🔍 Pexels search query: {query}")

#         pexels_videos = search_multiple_pexels_videos(query, max_results=5)
#         if not pexels_videos:
#             logging.warning("⚠️ No videos returned from Pexels.")
#             return []

#         temp_media_pool = []
#         for i, video in enumerate(pexels_videos):
#             unique_name = f"pexels_{uuid.uuid4().hex}_{i}.mp4"
#             path = os.path.join("pexels_videos", unique_name)
#             os.makedirs(os.path.dirname(path), exist_ok=True)
#             try:
#                 download_video(video["url"], path)
#             except Exception as e:
#                 logging.warning(f"⚠️ Failed to download Pexels video: {e}")
#                 continue

#             temp_media_pool.append({
#                 "path": path,
#                 "type": "video",
#                 "duration": video.get("duration", 0),
#                 "character": character_name,
#                 "tag": None,
#                 "embedding": None,
#                 "source": "pexels",
#                 "rank": i
#             })

#         if not temp_media_pool:
#             return []

#         scored_media = sorted(temp_media_pool, key=lambda m: m["rank"])
#         available, fallback = [], []
#         for media in scored_media:
#             (available if media["path"] not in used_media_dict else fallback).append(media)

#         media_count = max(1, ceil(duration_needed / SPLIT_BASE_DURATION))
#         selected = []

#         for media in available:
#             if len(selected) >= media_count:
#                 break
#             with media_lock:
#                 used_media_dict[media["path"]] = True
#             selected.append({"media": media, "duration": 0, "has_audio": False})  # keep Pexels muted

#         if len(selected) < media_count:
#             for media in fallback[:media_count - len(selected)]:
#                 selected.append({"media": media, "duration": 0, "has_audio": False})

#         base_duration = duration_needed / len(selected)
#         remaining_duration = duration_needed

#         for s in selected:
#             media = s["media"]
#             media_max = media.get("duration", 0)
#             chunks = floor(media_max / SPLIT_BASE_DURATION)
#             max_usable = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
#             s["duration"] = min(max(base_duration, MIN_VIDEO_DURATION), max_usable, remaining_duration)
#             remaining_duration -= s["duration"]

#         # ⬇️ EXACT leftover distribution from old version
#         if remaining_duration > DURATION_TOLERANCE:
#             flexible_media = [s for s in selected if (
#                 (s['media']['type'] == 'image') or
#                 (s['media']['type'] == 'video' and s['duration'] < s['media']['duration'])
#             )]
#             if flexible_media:
#                 extra_per_item = remaining_duration / len(flexible_media)
#                 for s in flexible_media:
#                     max_possible = s['media'].get('duration', float('inf')) if s['media']['type'] == 'video' else float('inf')
#                     add = min(extra_per_item, max_possible - s['duration'])
#                     s['duration'] += add
#                     remaining_duration -= add
#                     if remaining_duration <= DURATION_TOLERANCE:
#                         break

#         total_duration = sum(s["duration"] for s in selected)
#         if abs(total_duration - duration_needed) > DURATION_TOLERANCE:
#             logging.warning(f"⚠️ Duration mismatch (Pexels): Needed {duration_needed:.2f}s, got {total_duration:.2f}s")

#         return selected

#     # 3) Default path (score BOTH images and videos)
#     if duration_needed < 2:
#         return []

#     relevant_media = [
#         m for m in media_pool
#         if m.get('character') == character_name
#         and not m.get("tag")
#         # no type filter → includes videos and images
#     ]

#     if not relevant_media:
#         logging.warning(f"⚠️ No media found for character: {character_name}")
#         return []

#     text_embedding = encode_text(segment.get("text", ""))
#     scored = sorted(
#         [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#         key=lambda x: x[1],
#         reverse=True
#     )

#     available_media, fallback_media = [], []
#     for media, _ in scored:
#         (available_media if media['path'] not in used_media_dict else fallback_media).append(media)

#     media_count = max(1, ceil(duration_needed / SPLIT_BASE_DURATION))
#     selected = []

#     for media in available_media:
#         if len(selected) >= media_count:
#             break
#         with media_lock:
#             used_media_dict[media["path"]] = True
#         selected.append({
#             "media": media,
#             "duration": 0,
#             "has_audio": (media.get("type") == "video" and not media.get("mute", False))
#         })

#     if len(selected) < media_count:
#         for media in fallback_media[:media_count - len(selected)]:
#             selected.append({
#                 "media": media,
#                 "duration": 0,
#                 "has_audio": (media.get("type") == "video" and not media.get("mute", False))
#             })

#     base_duration = duration_needed / len(selected)
#     remaining_duration = duration_needed

#     for s in selected:
#         media = s["media"]
#         mtype = media.get("type")

#         if mtype == "video":
#             media_max = media.get("duration", 0)
#             chunks = floor(media_max / SPLIT_BASE_DURATION)
#             max_usable = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
#             s["duration"] = min(max(base_duration, MIN_VIDEO_DURATION), max_usable, remaining_duration)
#         else:
#             media_max = media.get("duration", DEFAULT_IMAGE_DURATION)
#             s["duration"] = min(max(base_duration, MIN_IMAGE_DURATION), media_max, remaining_duration)

#         remaining_duration -= s["duration"]

#     # ⬇️ EXACT leftover distribution from old version
#     if remaining_duration > DURATION_TOLERANCE:
#         flexible_media = [s for s in selected if (
#             (s['media']['type'] == 'image') or
#             (s['media']['type'] == 'video' and s['duration'] < s['media']['duration'])
#         )]
#         if flexible_media:
#             extra_per_item = remaining_duration / len(flexible_media)
#             for s in flexible_media:
#                 max_possible = s['media'].get('duration', float('inf')) if s['media']['type'] == 'video' else float('inf')
#                 add = min(extra_per_item, max_possible - s['duration'])
#                 s['duration'] += add
#                 remaining_duration -= add
#                 if remaining_duration <= DURATION_TOLERANCE:
#                     break

#     total_duration = sum(s['duration'] for s in selected)
#     if abs(total_duration - duration_needed) > DURATION_TOLERANCE:
#         logging.warning(f"⚠️ Duration mismatch: Needed {duration_needed:.2f}s, got {total_duration:.2f}s")

#     return selected

import os, subprocess, uuid, glob, json, logging
from math import ceil

IMAGEFX_SIZE_FOR_16x9 = "LANDSCAPE"  # check the repo; WIDE/landscape usually maps closest to 16:9
IMAGEFX_MODEL = "IMAGEN_3_5"

#MULTIPLE GENERARATION PER SEGMENT
# def imagefx_generate_images(prompt: str, count: int, out_dir: str, cookie: str, seed: int | None = None):
#     os.makedirs(out_dir, exist_ok=True)
#     cmd = [
#         "imagefx", "generate",
#         "--prompt", prompt,
#         "--model", IMAGEFX_MODEL,
#         "--size", IMAGEFX_SIZE_FOR_16x9,
#         "--count", str(count),
#         "--dir", out_dir,
#         "--cookie", cookie
#     ]
#     if seed is not None:
#         cmd += ["--seed", str(seed)]

#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#     except subprocess.CalledProcessError as e:
#         logging.warning(f"ImageFX generation failed: {e.stdout[:500]}")
#         return []

#     # The CLI saves files to out_dir; grab newest images
#     # (the repo commonly outputs WEBP; convert later if needed)
#     files = sorted(glob.glob(os.path.join(out_dir, "*")), key=os.path.getmtime, reverse=True)
#     return files[:count]

#SINGLE GENERATION PER SEGMENT
# --- helpers ---

# import hashlib, os, glob, subprocess, logging, shutil, uuid
# from math import ceil, floor

# def _wrap_if_ps1(path: str):
#     if path.lower().endswith(".ps1"):
#         return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path]
#     return [path]

# def _resolve_imagefx_cmd():
#     override = os.getenv("IMAGEFX_BIN")
#     if override and os.path.isfile(override):
#         return _wrap_if_ps1(override)
#     found = shutil.which("imagefx") or shutil.which("ImageFX")
#     if found:
#         return _wrap_if_ps1(found)
#     for c in (
#         os.path.expandvars(r"%APPDATA%\npm\imagefx.ps1"),
#         os.path.expanduser(r"~\AppData\Roaming\npm\imagefx.ps1"),
#         os.path.expandvars(r"%APPDATA%\npm\imagefx.cmd"),
#         os.path.expanduser(r"~\AppData\Roaming\npm\imagefx.cmd"),
#     ):
#         if os.path.isfile(c):
#             return _wrap_if_ps1(c)
#     return ["npx", "-y", "@rohitaryal/imagefx-api"]

# def _normalize_size(aspect_hint: str | None) -> str:
#     if not aspect_hint: 
#         return "UNSPECIFIED"
#     a = aspect_hint.strip().lower()
#     if a in ("16:9", "landscape", "wide", "widescreen", "horizontal"): return "LANDSCAPE"
#     if a in ("9:16", "portrait", "vertical", "phone"):                 return "PORTRAIT"
#     if a in ("1:1", "square"):                                         return "SQUARE"
#     return "UNSPECIFIED"

# def _stable_seed_from_prompt(p: str) -> int:
#     return int(hashlib.sha1(p.encode("utf-8")).hexdigest(), 16) % (2**31)

# def _imagefx_generate(prompt: str, count: int, out_dir: str, cookie: str, seed: int | None = None,
#                       aspect_hint: str = "16:9") -> list[str]:
#     os.makedirs(out_dir, exist_ok=True)
#     base_cmd = _resolve_imagefx_cmd()
#     size = _normalize_size(aspect_hint)

#     # The CLI allows up to 8 per call; split into batches if needed
#     remaining, all_paths = count, []
#     batch_index = 0
#     while remaining > 0:
#         this_batch = min(remaining, 8)
#         # For reproducible variety, offset the seed per batch (and per image inside batch)
#         base_seed = seed if seed is not None else _stable_seed_from_prompt(prompt)
#         # The CLI only accepts one --seed; to ensure unique images, we’ll shift per batch.
#         # (Inside the service, a single seed often yields a set of variations; offset helps cross-batch.)
#         seed_for_batch = base_seed + (batch_index * 97)

#         cmd = base_cmd + [
#             "generate",
#             "--prompt", prompt,
#             "--model", "IMAGEN_3_5",
#             "--size", size,
#             "--count", str(this_batch),
#             "--dir", out_dir,
#             "--cookie", cookie,
#             "--seed", str(seed_for_batch),
#         ]
#         logging.info(f"Running ImageFX: {' '.join(cmd)}")
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#         except subprocess.CalledProcessError as e:
#             logging.error(f"❌ ImageFX generation failed.\nCommand: {' '.join(cmd)}\nOutput:\n{e.stdout[:1000]}")
#             break

#         # collect newest files from out_dir after each batch
#         files = sorted(glob.glob(os.path.join(out_dir, "*")), key=os.path.getmtime, reverse=True)
#         # take the most recent `this_batch` not already in all_paths
#         new_files = [f for f in files if f not in all_paths][:this_batch]
#         all_paths.extend(new_files)
#         remaining -= this_batch
#         batch_index += 1

#     return all_paths[:count]

import hashlib, os, glob, subprocess, logging, shutil, uuid, time, re, shlex
from math import ceil, floor
from pathlib import Path
from typing import Optional

# ----------------------------
# Helpers (kept; minimal add-ons)
# ----------------------------
def _wrap_if_ps1(path: str):
    if path.lower().endswith(".ps1"):
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path]
    return [path]

def _resolve_imagefx_cmd():
    override = os.getenv("IMAGEFX_BIN")
    if override and os.path.isfile(override):
        return _wrap_if_ps1(override)
    found = shutil.which("imagefx") or shutil.which("ImageFX")
    if found:
        return _wrap_if_ps1(found)
    for c in (
        os.path.expandvars(r"%APPDATA%\npm\imagefx.ps1"),
        os.path.expanduser(r"~\AppData\Roaming\npm\imagefx.ps1"),
        os.path.expandvars(r"%APPDATA%\npm\imagefx.cmd"),
        os.path.expanduser(r"~\AppData\Roaming\npm\imagefx.cmd"),
    ):
        if os.path.isfile(c):
            return _wrap_if_ps1(c)
    # final fallback via npx
    return ["npx", "-y", "@rohitaryal/imagefx-api"]

def _normalize_size(aspect_hint: str | None) -> str:
    if not aspect_hint: 
        return "UNSPECIFIED"
    a = aspect_hint.strip().lower()
    if a in ("16:9", "landscape", "wide", "widescreen", "horizontal"): return "LANDSCAPE"
    if a in ("9:16", "portrait", "vertical", "phone"):                 return "PORTRAIT"
    if a in ("1:1", "square"):                                         return "SQUARE"
    return "UNSPECIFIED"

def _stable_seed_from_prompt(p: str) -> int:
    return int(hashlib.sha1(p.encode("utf-8")).hexdigest(), 16) % (2**31)

# ----------------------------
# NEW small, surgical hardening helpers
# ----------------------------
def _flatten_prompt(p: str, max_len: int = 1500) -> str:
    """
    Make the prompt safe for Windows/Node CLIs:
    - remove CR/LF, collapse whitespace
    - strip common markdown markers
    - cap length to avoid argv issues
    """
    if not isinstance(p, str):
        p = str(p)
    p = p.replace("\r", " ").replace("\n", " ")
    p = p.replace("**", "").replace("__", "")
    
    # ADD THESE TWO LINES:
    p = p.replace(" & ", " and ")  # ← FIX LINE 1
    p = p.replace("&", " and ")     # ← FIX LINE 2
    
    p = re.sub(r"\s+", " ", p).strip()
    if len(p) > max_len:
        p = p[: max_len - 1] + "…"
    return p

def _safe_cookie_env(cookie: str) -> dict:
    """
    Set GOOGLE_COOKIE in env so we don't rely on a fragile --cookie arg.
    """
    env = os.environ.copy()
    env["GOOGLE_COOKIE"] = cookie
    return env

def _list_new_images(out_dir: str) -> list[str]:
    """
    Collect generated images only; sort newest first.
    """
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(out_dir, pat)))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files

# ----------------------------
# PATCHED: ImageFX generator with minimal changes
# ----------------------------

def refine_prompt_for_safety(unsafe_prompt: str, style_spec: str) -> str:
    """
    Takes a prompt that was rejected as unsafe and refines it to be safe
    while maintaining the artistic style and visual intent.

    This is a DEDICATED instance of ChatGPT with ONE JOB: make it safe.
    """

    system = """
You are a prompt safety specialist for AI image generation.

Your ONLY job is to take image prompts that were rejected as "unsafe" and rewrite them to be completely safe while preserving:
- The artistic style and technique
- The compositional intent
- The lighting and mood
- The general visual aesthetic

You must REMOVE or SOFTEN:
- Violence, weapons, injuries, blood
- Threatening poses or expressions
- Dark, disturbing, or intense subject matter
- Anything that could be interpreted as harmful, dangerous, or inappropriate

REPLACEMENT STRATEGIES:
- Replace conflict with neutral interaction or solitude
- Replace weapons with neutral objects
- Replace aggressive poses with contemplative or neutral stances
- Replace disturbing imagery with atmospheric or symbolic alternatives
- Keep the same art style, lighting, and composition techniques

Be aggressive in your safety adjustments. It's better to be too safe than to get rejected again.
""".strip()

    user = f"""
ORIGINAL PROMPT (REJECTED AS UNSAFE):
{unsafe_prompt}

ARTISTIC STYLE TO PRESERVE:
{style_spec}

TASK:
Rewrite this prompt to be COMPLETELY SAFE for AI image generation while maintaining the artistic style, lighting technique, and compositional approach.

Return ONLY the rewritten safe prompt - no explanation, no commentary.
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,  # Very low for consistent safety
            max_tokens=500,
        )
        
        safe_prompt = resp.choices[0].message.content.strip()
        logging.info(f"🛡️ Safety refinement complete:\nORIGINAL: {unsafe_prompt[:80]}...\nSAFE: {safe_prompt[:80]}...")
        return safe_prompt
        
    except Exception as e:
        logging.error(f"❌ Safety refinement failed: {e}")
        # Fallback: very generic safe prompt
        return f"A serene, peaceful scene in {style_spec} style with soft lighting and calm atmosphere"

# def _imagefx_generate(
#     prompt: str,
#     count: int,
#     out_dir: str,
#     cookie: str,
#     seed: int | None = None,
#     aspect_hint: str = None,        # <-- NOW optional
#     model: str = "IMAGEN_3_5",
#     retries: int = 3,
#     timeout_sec: int = 120,
# ) -> list[str]:
#     """
#     Generate images via the ImageFX CLI, now orientation-aware:
#     - If aspect_hint is provided manually, use it.
#     - If aspect_hint=None → use CURRENT_ORIENTATION automatically.
#     """

#     # -------------------------------------------
#     # 🔥 DYNAMIC ORIENTATION FROM GLOBALS
#     # -------------------------------------------
#     global CURRENT_ORIENTATION

#     if aspect_hint is None:
#         if CURRENT_ORIENTATION == "portrait":
#             aspect_hint = "9:16"
#         elif CURRENT_ORIENTATION == "landscape":
#             aspect_hint = "16:9"
#         else:
#             aspect_hint = "1:1"  # fallback square

#     # normalize to ImageFX size tag
#     size = _normalize_size(aspect_hint)

#     # -------------------------------------------
#     # ORIGINAL LOGIC BELOW (UNCHANGED)
#     # -------------------------------------------

#     os.makedirs(out_dir, exist_ok=True)
#     base_cmd = _resolve_imagefx_cmd()

#     if not cookie or not cookie.strip():
#         logging.error("❌ ImageFX: cookie is empty or missing.")
#         return []

#     # sanitize prompt for CLI
#     prompt_flat = _flatten_prompt(prompt)

#     # allow up to 8 per call; original batching preserved
#     remaining, all_paths = count, []
#     batch_index = 0

#     # prepare env with cookie
#     env = _safe_cookie_env(cookie)

#     while remaining > 0:
#         this_batch = min(remaining, 8)
#         base_seed = seed if seed is not None else _stable_seed_from_prompt(prompt_flat)
#         seed_for_batch = base_seed + (batch_index * 97)

#         cmd = base_cmd + [
#             "generate",
#             "--prompt", prompt_flat,
#             "--model", model,
#             "--size", size,              # <--- NOW USING GLOBAL ORIENTATION
#             "--count", str(this_batch),
#             "--dir", out_dir,
#             "--seed", str(seed_for_batch),
#             "--retry", str(max(1, retries)),
#             "--cookie", cookie,
#         ]

#         invocation_attempts = 2
#         success = False
#         last_out = ""

#         for attempt in range(1, invocation_attempts + 1):
#             logging.info(
#                 "🚀 Running ImageFX (batch %d, attempt %d/%d): %s",
#                 batch_index + 1, attempt, invocation_attempts,
#                 " ".join(shlex.quote(c) for c in cmd)
#             )
#             try:
#                 result = subprocess.run(
#                     cmd,
#                     check=False,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.STDOUT,
#                     text=True,
#                     env=env,
#                     timeout=timeout_sec,
#                 )
#                 last_out = result.stdout or ""
#                 if result.returncode != 0:
#                     logging.error("❌ ImageFX generation failed (rc=%s).\n%s",
#                                   result.returncode, last_out[:2000])
#                 else:
#                     # allow filesystem to settle on Windows
#                     time.sleep(0.3)
#                     files = _list_new_images(out_dir)
#                     new_files = [f for f in files if f not in all_paths][:this_batch]
#                     if new_files:
#                         all_paths.extend(new_files)
#                         success = True
#                         break
#                     else:
#                         logging.warning("⚠️ ImageFX reported success but no image files found in %s", out_dir)

#             except subprocess.TimeoutExpired:
#                 logging.error("⏳ ImageFX CLI timed out after %ss.", timeout_sec)
#             except Exception as e:
#                 logging.exception("💥 Failed to invoke ImageFX CLI: %s", e)

#             time.sleep(0.5)  # small retry backoff

#         if not success:
#             logging.error("❌ ImageFX: no images generated for this batch after retries.")
#             break

#         remaining -= this_batch
#         batch_index += 1

#     return all_paths[:count]

# IMAGEFX_COOKIE_FILE = "cookies/imagefxcookies.txt"

# def read_imagefx_cookie_file():
#     if not os.path.exists(IMAGEFX_COOKIE_FILE):
#         return None

#     try:
#         with open(IMAGEFX_COOKIE_FILE, "r", encoding="utf-8") as f:
#             content = f.read().strip()
#             return content if content else None
#     except Exception as e:
#         logging.error(f"❌ Failed to read ImageFX cookie file: {e}")
#         return None


# ─────────────────────────────────────────────
# Cookie file path
# ─────────────────────────────────────────────
IMAGEFX_COOKIE_FILE = "cookies/imagefx_cookies.json"


def read_imagefx_cookie_file() -> str | None:
    from cookie_timers import cookie_timer  # import the global daemon timer
    """Load the JSON cookie file and return the cookie the timer is currently pointing at."""
    if not os.path.exists(IMAGEFX_COOKIE_FILE):
        logging.error("❌ ImageFX cookie file not found: %s", IMAGEFX_COOKIE_FILE)
        return None

    try:
        with open(IMAGEFX_COOKIE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        cookies = data.get("cookies", [])
        if not cookies:
            logging.error("❌ No cookies found in JSON file.")
            return None

        cookie = cookie_timer.get_cookie(cookies)
        logging.info(
            "🍪 Using cookie index %d of %d",
            cookie_timer.current_index(), len(cookies)
        )
        return cookie

    except Exception as e:
        logging.error("❌ Failed to read ImageFX cookie file: %s", e)
        return None


def _handle_401(last_output: str) -> None:
    """
    Cloudflare worker returned 401. Cookie rotation is legacy — auth is API-key only.
    """
    global _imagefx_auth_failed
    _imagefx_auth_failed = True
    logging.error(
        "🔐 Cloudflare image worker: 401 Unauthorized — check CLOUDFLARE_IMAGE_API_KEY "
        f"({last_output})"
    )

# ─────────────────────────────────────────────
# Core ImageFX generation function
# ─────────────────────────────────────────────
# def _imagefx_generate(
#     prompt: str,
#     count: int,
#     out_dir: str,
#     cookie: str,
#     seed: int | None = None,
#     aspect_hint: str = None,
#     model: str = "IMAGEN_3_5",
#     retries: int = 3,
#     timeout_sec: int = 120,
# ) -> list[str]:
#     """
#     Generate images via the ImageFX CLI.
#     - Auto-detects orientation from CURRENT_ORIENTATION global if aspect_hint is None.
#     - On 401, immediately rotates cookie and fires a Telegram alert.
#     """

#     global CURRENT_ORIENTATION

#     # ── Orientation ──────────────────────────────────────────────────────────
#     if aspect_hint is None:
#         if CURRENT_ORIENTATION == "portrait":
#             aspect_hint = "9:16"
#         elif CURRENT_ORIENTATION == "landscape":
#             aspect_hint = "16:9"
#         else:
#             aspect_hint = "1:1"

#     size = _normalize_size(aspect_hint)

#     # ── Setup ─────────────────────────────────────────────────────────────────
#     os.makedirs(out_dir, exist_ok=True)
#     base_cmd = _resolve_imagefx_cmd()

#     if not cookie or not cookie.strip():
#         logging.error("❌ ImageFX: cookie is empty or missing.")
#         return []

#     prompt_flat = _flatten_prompt(prompt)
#     remaining, all_paths = count, []
#     batch_index = 0
#     env = _safe_cookie_env(cookie)

#     while remaining > 0:
#         this_batch = min(remaining, 8)
#         base_seed = seed if seed is not None else _stable_seed_from_prompt(prompt_flat)
#         seed_for_batch = base_seed + (batch_index * 97)

#         cmd = base_cmd + [
#             "generate",
#             "--prompt", prompt_flat,
#             "--model", model,
#             "--size", size,
#             "--count", str(this_batch),
#             "--dir", out_dir,
#             "--seed", str(seed_for_batch),
#             "--retry", str(max(1, retries)),
#             "--cookie", cookie,
#         ]

#         invocation_attempts = 2
#         success = False
#         last_out = ""

#         for attempt in range(1, invocation_attempts + 1):
#             logging.info(
#                 "🚀 Running ImageFX (batch %d, attempt %d/%d): %s",
#                 batch_index + 1, attempt, invocation_attempts,
#                 " ".join(shlex.quote(c) for c in cmd)
#             )
#             try:
#                 result = subprocess.run(
#                     cmd,
#                     check=False,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.STDOUT,
#                     text=True,
#                     env=env,
#                     timeout=timeout_sec,
#                 )
#                 last_out = result.stdout or ""

#                 if result.returncode != 0:
#                     logging.error(
#                         "❌ ImageFX generation failed (rc=%s).\n%s",
#                         result.returncode, last_out[:2000]
#                     )

#                     # ── 401 handler ──────────────────────────────────────────
#                     if "401" in last_out or "UNAUTHENTICATED" in last_out:
#                         _handle_401(last_out)
#                         # Return empty — caller will retry with fresh cookie
#                         # on the next generate_image_fx_prompts loop iteration
#                         return []
#                     # ─────────────────────────────────────────────────────────

#                 else:
#                     time.sleep(0.3)
#                     files = _list_new_images(out_dir)
#                     new_files = [f for f in files if f not in all_paths][:this_batch]
#                     if new_files:
#                         all_paths.extend(new_files)
#                         success = True
#                         break
#                     else:
#                         logging.warning(
#                             "⚠️ ImageFX reported success but no image files found in %s", out_dir
#                         )

#             except subprocess.TimeoutExpired:
#                 logging.error("⏳ ImageFX CLI timed out after %ss.", timeout_sec)
#             except Exception as e:
#                 logging.exception("💥 Failed to invoke ImageFX CLI: %s", e)

#             time.sleep(0.5)

#         if not success:
#             logging.error("❌ ImageFX: no images generated for this batch after retries.")
#             break

#         remaining -= this_batch
#         batch_index += 1

#     return all_paths[:count]


def _imagefx_generate(
    prompt: str,
    count: int,
    out_dir: str,
    cookie: str | None = None,
    seed: int | None = None,
    aspect_hint: str = None,
    model: str = "IMAGEN_3_5",
    retries: int = 3,
    timeout_sec: int = 120,
) -> list[str]:
    """
    Generate images via the Cloudflare Workers AI endpoint (Bearer API key).
    Legacy ``cookie`` param is ignored — kept for call-site compatibility only.
    """

    global CURRENT_ORIENTATION

    # ── Orientation ──────────────────────────────────────────────────────────
    if aspect_hint is None:
        if CURRENT_ORIENTATION == "portrait":
            aspect_hint = "9:16"
        elif CURRENT_ORIENTATION == "landscape":
            aspect_hint = "16:9"
        else:
            aspect_hint = "1:1"

    # ── Map aspect ratio to dimensions ────────────────────────────────────────
    size_map = {
        "9:16":  (768, 1344),
        "16:9":  (1344, 768),
        "1:1":   (1024, 1024),
    }
    width, height = size_map.get(aspect_hint, (1024, 1024))

    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    WORKER_URL = os.getenv(
        "CLOUDFLARE_IMAGE_WORKER_URL",
        "https://alchemy-mix.techwebdev3.workers.dev",
    )
    API_KEY = os.getenv("CLOUDFLARE_IMAGE_API_KEY", "alchemy-mix#44@245")

    prompt_flat = _flatten_prompt(prompt)
    remaining, all_paths = count, []
    batch_index = 0

    while remaining > 0:
        this_batch = min(remaining, 8)
        success = False

        for attempt in range(1, retries + 1):
            logging.info(
                "🚀 Running ImageFX (batch %d, attempt %d/%d): POST %s",
                batch_index + 1, attempt, retries, WORKER_URL
            )
            try:
                response = requests.post(
                    WORKER_URL,
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "prompt": prompt_flat,
                        "width":  width,
                        "height": height,
                    },
                    timeout=timeout_sec,
                )

                if response.status_code == 401:
                    logging.error("❌ ImageFX: 401 Unauthorized.")
                    _handle_401("401 UNAUTHENTICATED")
                    return []

                if response.status_code != 200:
                    logging.error(
                        "❌ ImageFX generation failed (rc=%s).\n%s",
                        response.status_code, response.text[:2000]
                    )

                else:
                    # ── Save image to disk ────────────────────────────────────
                    timestamp = int(time.time() * 1000)
                    filename  = os.path.join(out_dir, f"image_{timestamp}_{batch_index}_{attempt}.jpg")
                    with open(filename, "wb") as f:
                        f.write(response.content)

                    time.sleep(0.3)
                    new_files = [filename] if os.path.exists(filename) else []

                    if new_files:
                        all_paths.extend(new_files)
                        success = True
                        break
                    else:
                        logging.warning(
                            "⚠️ ImageFX reported success but no image files found in %s", out_dir
                        )

            except requests.Timeout:
                logging.error("⏳ ImageFX CLI timed out after %ss.", timeout_sec)
            except Exception as e:
                logging.exception("💥 Failed to invoke ImageFX CLI: %s", e)

            time.sleep(0.5)

        if not success:
            logging.error("❌ ImageFX: no images generated for this batch after retries.")
            break

        remaining -= this_batch
        batch_index += 1

    return all_paths[:count]



















# def find_best_media_combination(media_pool, segment, character_name, duration_needed, used_media_dict, media_lock, mode, segment_index,script_context,prev_segments=None):
#     def loop_tagged_media(tagged_media, duration_needed):
#         looped, total, i = [], 0.0, 0
#         while total < duration_needed and tagged_media:
#             media = tagged_media[i % len(tagged_media)]
#             media_duration = media.get("duration", DEFAULT_IMAGE_DURATION if media["type"] == "image" else 0)
#             duration_to_use = min(media_duration, duration_needed - total)
#             looped.append({
#                 "media": media,
#                 "duration": duration_to_use,
#                 "has_audio": False
#             })
#             with media_lock:
#                 used_media_dict[media["path"]] = True
#             total += duration_to_use
#             i += 1
#         return looped

#     # per-video Pexels-ID lock
#     with media_lock:
#      if "_pexels_downloaded_ids" not in used_media_dict:
#          used_media_dict["_pexels_downloaded_ids"] = set()
#     downloaded_pexels_ids = used_media_dict["_pexels_downloaded_ids"]

#     # 1) Tag override
#     tag = segment.get("tag")
#     if tag:
#         tagged_media = [m for m in media_pool if m.get('tag') == tag]
#         if tagged_media:
#             return loop_tagged_media(tagged_media, duration_needed)
#         logging.info(f"🔁 No tagged media found for tag '{tag}' — falling back to selection")

#     # 2) Pexels videos
# # 2) Pexels videos
#     if segment.get("pexels_required", False):
#         query = generate_pexels_query(segment.get("text", ""), segment_index, media_type='video')
#         logging.info(f"🔍 Pexels video query: {query}")

#         media_count = max(1, ceil(duration_needed / SPLIT_BASE_DURATION))

#         # First try with 10
#         batch = search_multiple_pexels_media(query, media_type="video", per_page=10, page=1)
#         fresh = [v for v in batch if v.get("id") not in downloaded_pexels_ids]

#         # If not enough unique, fetch 50
#         if len(fresh) < media_count:
#             batch2 = search_multiple_pexels_media(query, media_type="video", per_page=50, page=1)
#             seen = {v.get("id") for v in fresh}
#             fresh.extend([v for v in batch2 if v.get("id") not in downloaded_pexels_ids and v.get("id") not in seen])

#         if not fresh:
#             logging.warning("⚠️ No fresh videos returned from Pexels.")
#             return []

#         temp_media_pool = []
#         successful_downloads = 0
#         failed_downloads = 0
        
#         for i, video in enumerate(fresh[:media_count * 2]):  # Try to download 2x needed in case of failures
#             if successful_downloads >= media_count:
#                 break
                
#             path = os.path.join("pexels_videos", f"pexels_{uuid.uuid4().hex}_{i}.mp4")
            
#             # Ensure directory exists
#             os.makedirs(os.path.dirname(path), exist_ok=True)
            
#             try:
#                 logging.info(f"🎥 Downloading Pexels video {successful_downloads + 1}/{media_count}...")
                
#                 # Download with validation
#                 if not download_video(video["url"], path):
#                     failed_downloads += 1
#                     logging.warning(f"⚠️ Failed to download/validate video {i+1}")
#                     continue
                
#                 # Double-check file exists and is valid
#                 if not os.path.exists(path):
#                     failed_downloads += 1
#                     logging.error(f"❌ Downloaded file doesn't exist: {path}")
#                     continue
                
#                 file_size = os.path.getsize(path)
#                 if file_size < 10000:
#                     failed_downloads += 1
#                     logging.error(f"❌ Downloaded file too small: {file_size} bytes")
#                     os.remove(path)
#                     continue
                
#                 # Mark as downloaded
#                 with media_lock:
#                     downloaded_pexels_ids.add(video.get("id"))
                
#                 # Add to pool
#                 temp_media_pool.append({
#                     "path": path,
#                     "type": "video",
#                     "duration": video.get("duration", 0),
#                     "character": character_name,
#                     "tag": None,
#                     "embedding": None,
#                     "source": "pexels",
#                     "rank": successful_downloads
#                 })
                
#                 successful_downloads += 1
#                 logging.info(f"✅ Successfully downloaded video {successful_downloads}/{media_count}")
                
#             except Exception as e:
#                 failed_downloads += 1
#                 logging.error(f"❌ Exception downloading Pexels video {i+1}: {e}")
#                 if os.path.exists(path):
#                     try:
#                         os.remove(path)
#                     except:
#                         pass
#                 continue

#         logging.info(f"📊 Pexels download results: {successful_downloads} successful, {failed_downloads} failed")

#         if not temp_media_pool:
#             logging.error("❌ No Pexels videos successfully downloaded")
#             return []

#         # Choose & duration-allocate
#         scored_media = sorted(temp_media_pool, key=lambda m: m["rank"])
#         available, fallback = [], []
#         for m in scored_media:
#             (available if m["path"] not in used_media_dict else fallback).append(m)

#         selected = []
#         for m in available:
#             if len(selected) >= media_count:
#                 break
#             with media_lock:
#                 used_media_dict[m["path"]] = True
#             selected.append({"media": m, "duration": 0, "has_audio": False})

#         if len(selected) < media_count:
#             for m in fallback[:media_count - len(selected)]:
#                 selected.append({"media": m, "duration": 0, "has_audio": False})

#         if not selected:
#             logging.error("❌ No videos selected from downloaded pool")
#             return []

#         base = duration_needed / len(selected)
#         remaining = duration_needed
#         for s in selected:
#             media = s["media"]
#             media_max = media.get("duration", 0)
#             chunks = floor(media_max / SPLIT_BASE_DURATION)
#             max_usable = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
#             s["duration"] = min(max(base, MIN_VIDEO_DURATION), max_usable, remaining)
#             remaining -= s["duration"]

#         if remaining > DURATION_TOLERANCE:
#             flex = [s for s in selected if (s['media']['type'] == 'image') or (s['media']['type'] == 'video' and s['duration'] < s['media']['duration'])]
#             if flex:
#                 extra = remaining / len(flex)
#                 for s in flex:
#                     max_possible = s['media'].get('duration', float('inf')) if s['media']['type'] == 'video' else float('inf')
#                     add = min(extra, max_possible - s['duration'])
#                     s['duration'] += add
#                     remaining -= add
#                     if remaining <= DURATION_TOLERANCE:
#                         break

#         total = sum(s["duration"] for s in selected)
#         if abs(total - duration_needed) > DURATION_TOLERANCE:
#             logging.warning(f"⚠️ Duration mismatch (Pexels videos): Need {duration_needed:.2f}s, got {total:.2f}s")
        
#         logging.info(f"✅ Returning {len(selected)} Pexels videos with total duration {total:.2f}s")
#         return selected

#     # 2b) Pexels images
#     if segment.get("pexels_image_required", False):
#         query = generate_pexels_query(segment.get("text", ""), segment_index, media_type="image")
#         logging.info(f"🖼️ Pexels image query: {query}")

#         media_count = max(1, ceil(duration_needed / DEFAULT_IMAGE_DURATION))

#         # First try with 5
#         batch = search_multiple_pexels_media(query, media_type="image", per_page=5, page=1)
#         fresh = [img for img in batch if img.get("id") not in downloaded_pexels_ids]

#         # If not enough unique, fetch 20
#         if len(fresh) < media_count:
#             batch2 = search_multiple_pexels_media(query, media_type="image", per_page=20, page=1)
#             seen = {img.get("id") for img in fresh}
#             fresh.extend([img for img in batch2 if img.get("id") not in downloaded_pexels_ids and img.get("id") not in seen])

#         if not fresh:
#             logging.warning("⚠️ No fresh images returned from Pexels.")
#             return []

#         temp_media_pool = []
#         successful_downloads = 0
#         failed_downloads = 0
        
#         for i, image in enumerate(fresh[:media_count * 2]):  # Try 2x in case of failures
#             if successful_downloads >= media_count:
#                 break
                
#             path = os.path.join("pexels_images", f"pexels_{uuid.uuid4().hex}_{i}.jpg")
            
#             # Ensure directory exists
#             os.makedirs(os.path.dirname(path), exist_ok=True)
            
#             try:
#                 logging.info(f"🖼️ Downloading Pexels image {successful_downloads + 1}/{media_count}...")
                
#                 if not download_pexels_image(image["url"], path):
#                     failed_downloads += 1
#                     logging.warning(f"⚠️ Failed to download/validate image {i+1}")
#                     continue
                
#                 # Double-check file
#                 if not os.path.exists(path):
#                     failed_downloads += 1
#                     logging.error(f"❌ Downloaded image doesn't exist: {path}")
#                     continue
                
#                 file_size = os.path.getsize(path)
#                 if file_size < 1000:
#                     failed_downloads += 1
#                     logging.error(f"❌ Image file too small: {file_size} bytes")
#                     os.remove(path)
#                     continue

#                 with media_lock:
#                     downloaded_pexels_ids.add(image.get("id"))

#                 temp_media_pool.append({
#                     "path": path,
#                     "type": "image",
#                     "duration": DEFAULT_IMAGE_DURATION,
#                     "character": character_name,
#                     "tag": None,
#                     "embedding": None,
#                     "source": "pexels_image",
#                     "rank": successful_downloads
#                 })
                
#                 successful_downloads += 1
#                 logging.info(f"✅ Successfully downloaded image {successful_downloads}/{media_count}")
                
#             except Exception as e:
#                 failed_downloads += 1
#                 logging.error(f"❌ Exception downloading Pexels image {i+1}: {e}")
#                 if os.path.exists(path):
#                     try:
#                         os.remove(path)
#                     except:
#                         pass
#                 continue

#         logging.info(f"📊 Pexels image results: {successful_downloads} successful, {failed_downloads} failed")

#         if not temp_media_pool:
#             logging.error("❌ No Pexels images successfully downloaded")
#             return []

#         scored_media = sorted(temp_media_pool, key=lambda m: m["rank"])
#         available, fallback = [], []
#         for m in scored_media:
#             (available if m["path"] not in used_media_dict else fallback).append(m)

#         selected = []
#         for m in available:
#             if len(selected) >= media_count:
#                 break
#             with media_lock:
#                 used_media_dict[m["path"]] = True
#             selected.append({"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False})

#         if len(selected) < media_count:
#             for m in fallback[:media_count - len(selected)]:
#                 selected.append({"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False})

#         if not selected:
#             logging.error("❌ No images selected from downloaded pool")
#             return []

#         # Rebalance durations
#         base = max(MIN_IMAGE_DURATION, duration_needed / len(selected))
#         remaining = duration_needed
#         for s in selected:
#             s["duration"] = min(DEFAULT_IMAGE_DURATION, base, remaining)
#             remaining -= s["duration"]

#         if remaining > DURATION_TOLERANCE:
#             idx, n = 0, len(selected)
#             while remaining > DURATION_TOLERANCE and idx < n * 2:
#                 s = selected[idx % n]
#                 can_add = DEFAULT_IMAGE_DURATION - s["duration"]
#                 if can_add > 0:
#                     add = min(can_add, remaining)
#                     s["duration"] += add
#                     remaining -= add
#                 idx += 1

#         total = sum(s["duration"] for s in selected)
#         if abs(total - duration_needed) > DURATION_TOLERANCE:
#             logging.warning(f"⚠️ Duration mismatch (Pexels images): Need {duration_needed:.2f}s, got {total:.2f}s")
        
#         logging.info(f"✅ Returning {len(selected)} Pexels images with total duration {total:.2f}s")
#         return selected
#     # 2c) ImageFX (Google ImageFX via community CLI) mULTIPLE GENERATION
#       # 2c) ImageFX (Google ImageFX via community CLI) MULTIPLE GENERATION
#     # if segment.get("imagefx_required", True) and mode == "original":
#     #     cookie = os.getenv("GOOGLE_COOKIE")
#     #     if not cookie:
#     #         logging.warning("⚠️ GOOGLE_COOKIE not set; skipping ImageFX branch.")
#     #         return []

#     #     # how many images are needed for this segment
#     #     media_count = max(1, ceil(duration_needed / DEFAULT_IMAGE_DURATION))

#     #     # build N distinct, cinematic prompts (first → last)
#     #     # style_spec ="" "Explanatory scene in hand-drawn pencil style; STICK-FIGURE character or characters(More than one base on scene) (or objects only) chosen for the clearest explanation; black graphite line art on clean white background; consistent medium line weight; minimal shading via light cross-hatching only; NO color fills; clean contours; uncluttered layout; storyboard-friendly  staging with clear foreground/midground separation; neutral paper-like background (subtle, optional); static camera at eye-level unless a low/high angle BETTER explains the idea; BAN photorealism, 3D, gradients, complex rendering, UI, logos, and ANY text/typography."""
#     #     style_spec="ghibli"
        
#     #     # prompts = generate_image_fx_prompts(segment["text"], prev_segments, count=media_count, style_spec=style_spec)
#     #     prompts = generate_image_fx_prompts(segment["text"],script_context, prev_segments, style_spec=style_spec)

#     #     # per-run cache: map each UNIQUE prompt -> list of generated file paths
#     #     with media_lock:
#     #         cache = used_media_dict.setdefault("_imagefx_cache_multi", {})  # {prompt: [paths]}

#     #     selected_paths: list[str] = []

#     #     # stable base seed for this segment (but still varies per prompt via +i)
#     #     base_seed = _seed(segment.get("imagefx_prompt") or segment.get("text", ""))

#     #     for i, p in enumerate(prompts):
#     #         # 1) reuse from cache if available & unused
#     #         path_from_cache = None
#     #         with media_lock:
#     #             cached_list = cache.get(p, [])
#     #             for cand in cached_list:
#     #                 if cand not in used_media_dict:
#     #                     used_media_dict[cand] = True
#     #                     path_from_cache = cand
#     #                     break
#     #         if path_from_cache:
#     #             selected_paths.append(path_from_cache)
#     #             continue

#     #         # 2) generate exactly ONE image for this prompt with a deterministic seed
#     #         seed_i = (base_seed + i) % (2**31)
#     #         out_dir = os.path.join("imagefx_images", f"{uuid.uuid4().hex}_{i:02d}")

#     #         new_paths = _imagefx_generate(
#     #             prompt=p,
#     #             count=1,
#     #             out_dir=out_dir,
#     #             cookie=cookie,
#     #             seed=seed_i,
#     #             aspect_hint="16:9",  # maps to LANDSCAPE inside _imagefx_generate
#     #         )

#     #         if not new_paths:
#     #             logging.warning(f"⚠️ ImageFX produced no image for prompt #{i+1}.")
#     #             continue

#     #         path = new_paths[0]
#     #         with media_lock:
#     #             cache.setdefault(p, []).append(path)
#     #             used_media_dict[path] = True
#     #         selected_paths.append(path)

#     #     if not selected_paths:
#     #         logging.warning("⚠️ ImageFX: no images generated/selected.")
#     #         return []

#     #     # build media objects
#     #     temp_media_pool = []
#     #     for i, path in enumerate(selected_paths[:media_count]):
#     #         temp_media_pool.append({
#     #             "path": path,
#     #             "type": "image",
#     #             "duration": DEFAULT_IMAGE_DURATION,
#     #             "character": character_name,
#     #             "tag": None,
#     #             "embedding": None,
#     #             "source": "imagefx_multi",
#     #             "rank": i
#     #         })

#     #     # allocate durations list
#     #     selected = [{"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False} for m in temp_media_pool]

#     #     # >>> Single-Image ImageFX: hold for entire segment <<<
#     #     if len(selected) == 1 and selected[0]["media"]["type"] == "image" and selected[0]["media"].get("source", "").startswith("imagefx"):
#     #         selected[0]["duration"] = duration_needed
#     #         total = selected[0]["duration"]
#     #         if abs(total - duration_needed) > DURATION_TOLERANCE:
#     #             logging.warning(f"⚠️ Duration mismatch (ImageFX single): Need {duration_needed:.2f}s, got {total:.2f}s")
#     #         return selected
#     #     # >>> END special-case <<<

#     #     # multi-image ImageFX: rebalance durations to match target
#     #     base = max(MIN_IMAGE_DURATION, duration_needed / len(selected))
#     #     remaining = duration_needed
#     #     for s in selected:
#     #         s["duration"] = min(DEFAULT_IMAGE_DURATION, base, remaining)
#     #         remaining -= s["duration"]

#     #     if remaining > DURATION_TOLERANCE:
#     #         idx, n = 0, len(selected)
#     #         while remaining > DURATION_TOLERANCE and idx < n * 2:
#     #             s = selected[idx % n]
#     #             can_add = DEFAULT_IMAGE_DURATION - s["duration"]
#     #             if can_add > 0:
#     #                 add = min(can_add, remaining)
#     #                 s["duration"] += add
#     #                 remaining -= add
#     #             idx += 1

#     #     total = sum(s["duration"] for s in selected)
#     #     if abs(total - duration_needed) > DURATION_TOLERANCE:
#     #         logging.warning(f"⚠️ Duration mismatch (ImageFX multi): Need {duration_needed:.2f}s, got {total:.2f}s")

#     #     return selected
    
#     # if segment.get("imagefx_required", True) and mode == "original":
#     #    cookie = os.getenv("GOOGLE_COOKIE")
#     #    if not cookie:
#     #        logging.warning("⚠️ GOOGLE_COOKIE not set; skipping ImageFX branch.")
#     #        return []

#     #    # how many images are needed for this segment
#     #    media_count = max(1, ceil(duration_needed / DEFAULT_IMAGE_DURATION))

#     #    # 🎨 Determine style or custom override
#     #    style_spec = "ghibli"  # fallback
#     #    override_prompt = None

#     #    # Handle user-provided ai_prompt (from assign_timeline_phases)
#     #    if segment.get("ai_prompt"):
#     #        prompt_text = segment["ai_prompt"].strip()
#     #        known_styles = {
#     #         "ghibli", "explanatory", "explanatory_dark",
#     #         "war", "cinematic", "dark_story", "luxury_motivation"
#     #     }

#     #        if prompt_text.lower() in known_styles:
#     #         style_spec = prompt_text.lower()
#     #        else:
#     #         override_prompt = prompt_text  # treat as literal custom style

#     #    # 🧠 Generate prompts dynamically
#     #    prompts = generate_image_fx_prompts(
#     #     text_segment=segment["text"],
#     #     script_context=script_context,
#     #     prev_segments=prev_segments,
#     #     style_spec=style_spec,
#     #     override_prompt=override_prompt
#     # )

#     #    # 🔁 Caching system
#     #    with media_lock:
#     #        cache = used_media_dict.setdefault("_imagefx_cache_multi", {})

#     #    selected_paths: list[str] = []
#     #    base_seed = _seed(segment.get("ai_prompt") or segment.get("text", ""))

#     #    for i, p in enumerate(prompts):
#     #        # 1️⃣ Reuse from cache if possible
#     #        path_from_cache = None
#     #        with media_lock:
#     #            cached_list = cache.get(p, [])
#     #            for cand in cached_list:
#     #                if cand not in used_media_dict:
#     #                    used_media_dict[cand] = True
#     #                    path_from_cache = cand
#     #                    break
#     #        if path_from_cache:
#     #            selected_paths.append(path_from_cache)
#     #            continue

#     #        # 2️⃣ Generate one image for this prompt
#     #        seed_i = (base_seed + i) % (2**31)
#     #        out_dir = os.path.join("imagefx_images", f"{uuid.uuid4().hex}_{i:02d}")

#     #        new_paths = _imagefx_generate(
#     #         prompt=p,
#     #         count=1,
#     #         out_dir=out_dir,
#     #         cookie=cookie,
#     #         seed=seed_i,
            
#     #     )

#     #        if not new_paths:
#     #         logging.warning(f"⚠️ ImageFX produced no image for prompt #{i+1}.")
#     #         continue

#     #        path = new_paths[0]
#     #        with media_lock:
#     #            cache.setdefault(p, []).append(path)
#     #            used_media_dict[path] = True
#     #        selected_paths.append(path)

#     #    if not selected_paths:
#     #        logging.warning("⚠️ ImageFX: no images generated/selected.")
#     #        return []

#     #    # Build media pool
#     #    temp_media_pool = []
#     #    for i, path in enumerate(selected_paths[:media_count]):
#     #        temp_media_pool.append({
#     #         "path": path,
#     #         "type": "image",
#     #         "duration": DEFAULT_IMAGE_DURATION,
#     #         "character": character_name,
#     #         "tag": None,
#     #         "embedding": None,
#     #         "source": "imagefx_multi",
#     #         "rank": i
#     #     })

#     #    # Allocate durations
#     #    selected = [{"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False} for m in temp_media_pool]

#     #    # Single-image segment (entire narration)
#     #    if len(selected) == 1:
#     #        selected[0]["duration"] = duration_needed
#     #        return selected

#     #    # Multi-image balancing
#     #    base = max(MIN_IMAGE_DURATION, duration_needed / len(selected))
#     #    remaining = duration_needed
#     #    for s in selected:
#     #        s["duration"] = min(DEFAULT_IMAGE_DURATION, base, remaining)
#     #        remaining -= s["duration"]

#     #    total = sum(s["duration"] for s in selected)
#     #    if abs(total - duration_needed) > DURATION_TOLERANCE:
#     #        logging.warning(f"⚠️ Duration mismatch (ImageFX multi): Need {duration_needed:.2f}s, got {total:.2f}s")

#     #    return selected
#     if segment.get("imagefx_required", True) and mode == "original":
#         # cookie = os.getenv("GOOGLE_COOKIE")
#         from cookie_reader import read_active_cookie
#         cookie = read_active_cookie()
#         if not cookie:
#             logging.warning("⚠️ No active cookie available; skipping ImageFX branch.")
#             return []

#         media_count = max(1, ceil(duration_needed / DEFAULT_IMAGE_DURATION))

#         # 🎨 Determine style or custom override
#         style_spec = "ghibli"
#         override_prompt = None

#         if segment.get("ai_prompt"):
#             prompt_text = segment["ai_prompt"].strip()
#             known_styles = {
#                 "ghibli", "explanatory", "explanatory_dark","chalk", "explanatory_illustrated","fitness_video",
#                 "war", "cinematic", "dark_story", "luxury_motivation","female_chalk"
#             }
#             if prompt_text.lower() in known_styles:
#                 style_spec = prompt_text.lower()
#             else:
#                 override_prompt = prompt_text

#         # 🧠 Generate initial prompts
#         prompts = generate_image_fx_prompts(
#             text_segment=segment["text"],
#             script_context=script_context,
#             prev_segments=prev_segments,
#             style_spec=style_spec,
#             override_prompt=override_prompt
#         )

#         with media_lock:
#             cache = used_media_dict.setdefault("_imagefx_cache_multi", {})

#         selected_paths: list[str] = []
#         base_seed = _seed(segment.get("ai_prompt") or segment.get("text", ""))

#         for i, original_prompt in enumerate(prompts):
#             # 1️⃣ Check cache first
#             path_from_cache = None
#             with media_lock:
#                 cached_list = cache.get(original_prompt, [])
#                 for cand in cached_list:
#                     if cand not in used_media_dict:
#                         used_media_dict[cand] = True
#                         path_from_cache = cand
#                         break
            
#             if path_from_cache:
#                 selected_paths.append(path_from_cache)
#                 continue

#             # 2️⃣ Try generating with progressive safety refinement
#             seed_i = (base_seed + i) % (2**31)
#             out_dir = os.path.join("imagefx_images", f"{uuid.uuid4().hex}_{i:02d}")

#             current_prompt = original_prompt
#             new_paths = None
#             attempt = 0
#             max_attempts = 4  # Original + 3 safety refinements

#             while attempt < max_attempts and not new_paths:
#                 attempt += 1
                
#                 logging.info(f"🎬 Attempt {attempt}/{max_attempts} for segment {segment_index}")
                
#                 new_paths = _imagefx_generate(
#                     prompt=current_prompt,
#                     count=1,
#                     out_dir=out_dir,
#                     cookie=cookie,
#                     seed=seed_i,
#                 )

#                 # If failed, refine and try again
#                 if not new_paths and attempt < max_attempts:
#                     logging.warning(f"⚠️ Generation failed on attempt {attempt}. Refining prompt...")
#                     logging.info(f"🔍 Current prompt: {current_prompt[:100]}...")
                    
#                     # Refine the current prompt to make it safer
#                     current_prompt = refine_prompt_for_safety(current_prompt, style_spec)
#                     logging.info(f"🛡️ Refined prompt: {current_prompt[:100]}...")

#             # 3️⃣ Final fallback: dark frame
#             if not new_paths:
#                 logging.warning(f"⚠️ All {max_attempts} attempts failed for prompt #{i+1}. Skipping.")
#                 continue

#             # 4️⃣ Success - cache and add to selection
#             if new_paths:
#                 path = new_paths[0]
#                 with media_lock:
#                     cache.setdefault(current_prompt, []).append(path)
#                     used_media_dict[path] = True
#                 selected_paths.append(path)
#                 logging.info(f"✅ Successfully generated image for segment {segment_index} (attempt {attempt})")

#         if not selected_paths:
#             logging.warning("⚠️ ImageFX: no images generated/selected.")
#             return []

#         # Build media pool
#         temp_media_pool = []
#         for i, path in enumerate(selected_paths[:media_count]):
#             temp_media_pool.append({
#                 "path": path,
#                 "type": "image",
#                 "duration": DEFAULT_IMAGE_DURATION,
#                 "character": character_name,
#                 "tag": None,
#                 "embedding": None,
#                 "source": "imagefx_multi",
#                 "rank": i
#             })

#         selected = [{"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False} for m in temp_media_pool]

#         # Single-image segment
#         if len(selected) == 1:
#             selected[0]["duration"] = duration_needed
#             return selected

#         # Multi-image balancing
#         base = max(MIN_IMAGE_DURATION, duration_needed / len(selected))
#         remaining = duration_needed
#         for s in selected:
#             s["duration"] = min(DEFAULT_IMAGE_DURATION, base, remaining)
#             remaining -= s["duration"]

#         total = sum(s["duration"] for s in selected)
#         if abs(total - duration_needed) > DURATION_TOLERANCE:
#             logging.warning(f"⚠️ Duration mismatch (ImageFX multi): Need {duration_needed:.2f}s, got {total:.2f}s")

#         return selected


#     # 3) Default: score local images & videos together
#     if duration_needed < 2:
#         return []

#     relevant_media = [
#         m for m in media_pool
#         if m.get('character') == character_name and not m.get("tag")
#     ]

#     if not relevant_media:
#         logging.warning(f"⚠️ No media found for character: {character_name}")
#         return []

#     # You must have encode_text & cosine_similarity defined elsewhere
#     text_embedding = encode_text(segment.get("text", ""))
#     scored = sorted(
#         [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#         key=lambda x: x[1],
#         reverse=True
#     )

#     available_media, fallback_media = [], []
#     for media, _ in scored:
#         (available_media if media['path'] not in used_media_dict else fallback_media).append(media)

#     media_count = max(1, ceil(duration_needed / SPLIT_BASE_DURATION))
#     selected = []

#     for media in available_media:
#         if len(selected) >= media_count:
#             break
#         with media_lock:
#             used_media_dict[media["path"]] = True
#         selected.append({
#             "media": media,
#             "duration": 0,
#             "has_audio": (media.get("type") == "video" and not media.get("mute", False))
#         })

#     if len(selected) < media_count:
#         for media in fallback_media[:media_count - len(selected)]:
#             selected.append({
#                 "media": media,
#                 "duration": 0,
#                 "has_audio": (media.get("type") == "video" and not media.get("mute", False))
#             })

#     base = duration_needed / len(selected)
#     remaining = duration_needed

#     for s in selected:
#         media = s["media"]
#         if media.get("type") == "video":
#             media_max = media.get("duration", 0)
#             chunks = floor(media_max / SPLIT_BASE_DURATION)
#             max_usable = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
#             s["duration"] = min(max(base, MIN_VIDEO_DURATION), max_usable, remaining)
#         else:
#             media_max = media.get("duration", DEFAULT_IMAGE_DURATION)
#             s["duration"] = min(max(base, MIN_IMAGE_DURATION), media_max, remaining)
#         remaining -= s["duration"]

#     if remaining > DURATION_TOLERANCE:
#         flex = [s for s in selected if (s['media']['type'] == 'image') or (s['media']['type'] == 'video' and s['duration'] < s['media']['duration'])]
#         if flex:
#             extra = remaining / len(flex)
#             for s in flex:
#                 max_possible = s['media'].get('duration', float('inf')) if s['media']['type'] == 'video' else float('inf')
#                 add = min(extra, max_possible - s['duration'])
#                 s['duration'] += add
#                 remaining -= add
#                 if remaining <= DURATION_TOLERANCE:
#                     break

#     total = sum(s['duration'] for s in selected)
#     if abs(total - duration_needed) > DURATION_TOLERANCE:
#         logging.warning(f"⚠️ Duration mismatch: Need {duration_needed:.2f}s, got {total:.2f}s")

#     return selected

from whisk_character_handler import whisk_generate_for_segment


# def find_best_media_combination(
#     media_pool,
#     segment,
#     character_name,
#     duration_needed,
#     used_media_dict,
#     media_lock,
#     mode,
#     segment_index,
#     script_context,
#     prev_segments=None,
#     whisk_shared_state=None,  # ✅ NEW parameter
# ):
#     """
#     Find best media for segment.
#     Now supports Whisk character refinement (highest priority).
#     """

#     def loop_tagged_media(tagged_media, duration_needed):
#         looped, total, i = [], 0.0, 0
#         while total < duration_needed and tagged_media:
#             media = tagged_media[i % len(tagged_media)]
#             media_duration = media.get("duration", DEFAULT_IMAGE_DURATION if media["type"] == "image" else 0)
#             duration_to_use = min(media_duration, duration_needed - total)
#             looped.append({
#                 "media": media,
#                 "duration": duration_to_use,
#                 "has_audio": False
#             })
#             with media_lock:
#                 used_media_dict[media["path"]] = True
#             total += duration_to_use
#             i += 1
#         return looped

#     # per-video Pexels-ID lock
#     with media_lock:
#         if "_pexels_downloaded_ids" not in used_media_dict:
#             used_media_dict["_pexels_downloaded_ids"] = set()
#     downloaded_pexels_ids = used_media_dict["_pexels_downloaded_ids"]

#     # -------------------------------------------------
#     # ✅ 0) WHISK CHARACTER HANDLING (HIGHEST PRIORITY)
#     # -------------------------------------------------
#     if segment.get("whisk_required") and mode == "original":
#         if not whisk_shared_state or not whisk_shared_state.get("media_id"):
#             logging.error("❌ whisk_required but no character context available")
#             return []

#         logging.info(f"🎭 Segment {segment_index} using Whisk character")

#         return whisk_generate_for_segment(
#             segment=segment,
#             whisk_shared_state=whisk_shared_state,
#             media_lock=media_lock,
#             duration_needed=duration_needed,
#             segment_index=segment_index
#         )

#     # -------------------------------------------------
#     # 1) Tag override
#     # -------------------------------------------------
#     tag = segment.get("tag")
#     if tag:
#         tagged_media = [m for m in media_pool if m.get('tag') == tag]
#         if tagged_media:
#             return loop_tagged_media(tagged_media, duration_needed)
#         logging.info(f"🔁 No tagged media found for tag '{tag}' — falling back to selection")

#     # -------------------------------------------------
#     # 2) Pexels videos
#     # -------------------------------------------------
#     if segment.get("pexels_required", False):
#         query = generate_pexels_query(segment.get("text", ""), segment_index, media_type='video')
#         logging.info(f"🔍 Pexels video query: {query}")

#         media_count = max(1, ceil(duration_needed / SPLIT_BASE_DURATION))

#         # First try with 10
#         batch = search_multiple_pexels_media(query, media_type="video", per_page=10, page=1)
#         fresh = [v for v in batch if v.get("id") not in downloaded_pexels_ids]

#         # If not enough unique, fetch 50
#         if len(fresh) < media_count:
#             batch2 = search_multiple_pexels_media(query, media_type="video", per_page=50, page=1)
#             seen = {v.get("id") for v in fresh}
#             fresh.extend([v for v in batch2 if v.get("id") not in downloaded_pexels_ids and v.get("id") not in seen])

#         if not fresh:
#             logging.warning("⚠️ No fresh videos returned from Pexels.")
#             return []

#         temp_media_pool = []
#         successful_downloads = 0
#         failed_downloads = 0

#         for i, video in enumerate(fresh[:media_count * 2]):  # Try 2x needed in case of failures
#             if successful_downloads >= media_count:
#                 break

#             path = os.path.join("pexels_videos", f"pexels_{uuid.uuid4().hex}_{i}.mp4")
#             os.makedirs(os.path.dirname(path), exist_ok=True)

#             try:
#                 logging.info(f"🎥 Downloading Pexels video {successful_downloads + 1}/{media_count}...")

#                 if not download_video(video["url"], path):
#                     failed_downloads += 1
#                     logging.warning(f"⚠️ Failed to download/validate video {i+1}")
#                     continue

#                 if not os.path.exists(path):
#                     failed_downloads += 1
#                     logging.error(f"❌ Downloaded file doesn't exist: {path}")
#                     continue

#                 file_size = os.path.getsize(path)
#                 if file_size < 10000:
#                     failed_downloads += 1
#                     logging.error(f"❌ Downloaded file too small: {file_size} bytes")
#                     os.remove(path)
#                     continue

#                 with media_lock:
#                     downloaded_pexels_ids.add(video.get("id"))

#                 temp_media_pool.append({
#                     "path": path,
#                     "type": "video",
#                     "duration": video.get("duration", 0),
#                     "character": character_name,
#                     "tag": None,
#                     "embedding": None,
#                     "source": "pexels",
#                     "rank": successful_downloads
#                 })

#                 successful_downloads += 1
#                 logging.info(f"✅ Successfully downloaded video {successful_downloads}/{media_count}")

#             except Exception as e:
#                 failed_downloads += 1
#                 logging.error(f"❌ Exception downloading Pexels video {i+1}: {e}")
#                 if os.path.exists(path):
#                     try:
#                         os.remove(path)
#                     except:
#                         pass
#                 continue

#         logging.info(f"📊 Pexels download results: {successful_downloads} successful, {failed_downloads} failed")

#         if not temp_media_pool:
#             logging.error("❌ No Pexels videos successfully downloaded")
#             return []

#         scored_media = sorted(temp_media_pool, key=lambda m: m["rank"])
#         available, fallback = [], []
#         for m in scored_media:
#             (available if m["path"] not in used_media_dict else fallback).append(m)

#         selected = []
#         for m in available:
#             if len(selected) >= media_count:
#                 break
#             with media_lock:
#                 used_media_dict[m["path"]] = True
#             selected.append({"media": m, "duration": 0, "has_audio": False})

#         if len(selected) < media_count:
#             for m in fallback[:media_count - len(selected)]:
#                 selected.append({"media": m, "duration": 0, "has_audio": False})

#         if not selected:
#             logging.error("❌ No videos selected from downloaded pool")
#             return []

#         base = duration_needed / len(selected)
#         remaining = duration_needed
#         for s in selected:
#             media = s["media"]
#             media_max = media.get("duration", 0)
#             chunks = floor(media_max / SPLIT_BASE_DURATION)
#             max_usable = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
#             s["duration"] = min(max(base, MIN_VIDEO_DURATION), max_usable, remaining)
#             remaining -= s["duration"]

#         if remaining > DURATION_TOLERANCE:
#             flex = [s for s in selected if (s['media']['type'] == 'image') or (s['media']['type'] == 'video' and s['duration'] < s['media']['duration'])]
#             if flex:
#                 extra = remaining / len(flex)
#                 for s in flex:
#                     max_possible = s['media'].get('duration', float('inf')) if s['media']['type'] == 'video' else float('inf')
#                     add = min(extra, max_possible - s['duration'])
#                     s['duration'] += add
#                     remaining -= add
#                     if remaining <= DURATION_TOLERANCE:
#                         break

#         total = sum(s["duration"] for s in selected)
#         if abs(total - duration_needed) > DURATION_TOLERANCE:
#             logging.warning(f"⚠️ Duration mismatch (Pexels videos): Need {duration_needed:.2f}s, got {total:.2f}s")

#         logging.info(f"✅ Returning {len(selected)} Pexels videos with total duration {total:.2f}s")
#         return selected

#     # -------------------------------------------------
#     # 2b) Pexels images
#     # -------------------------------------------------
#     if segment.get("pexels_image_required", False):
#         query = generate_pexels_query(segment.get("text", ""), segment_index, media_type="image")
#         logging.info(f"🖼️ Pexels image query: {query}")

#         media_count = max(1, ceil(duration_needed / DEFAULT_IMAGE_DURATION))

#         batch = search_multiple_pexels_media(query, media_type="image", per_page=5, page=1)
#         fresh = [img for img in batch if img.get("id") not in downloaded_pexels_ids]

#         if len(fresh) < media_count:
#             batch2 = search_multiple_pexels_media(query, media_type="image", per_page=20, page=1)
#             seen = {img.get("id") for img in fresh}
#             fresh.extend([img for img in batch2 if img.get("id") not in downloaded_pexels_ids and img.get("id") not in seen])

#         if not fresh:
#             logging.warning("⚠️ No fresh images returned from Pexels.")
#             return []

#         temp_media_pool = []
#         successful_downloads = 0
#         failed_downloads = 0

#         for i, image in enumerate(fresh[:media_count * 2]):
#             if successful_downloads >= media_count:
#                 break

#             path = os.path.join("pexels_images", f"pexels_{uuid.uuid4().hex}_{i}.jpg")
#             os.makedirs(os.path.dirname(path), exist_ok=True)

#             try:
#                 logging.info(f"🖼️ Downloading Pexels image {successful_downloads + 1}/{media_count}...")

#                 if not download_pexels_image(image["url"], path):
#                     failed_downloads += 1
#                     logging.warning(f"⚠️ Failed to download/validate image {i+1}")
#                     continue

#                 if not os.path.exists(path):
#                     failed_downloads += 1
#                     logging.error(f"❌ Downloaded image doesn't exist: {path}")
#                     continue

#                 file_size = os.path.getsize(path)
#                 if file_size < 1000:
#                     failed_downloads += 1
#                     logging.error(f"❌ Image file too small: {file_size} bytes")
#                     os.remove(path)
#                     continue

#                 with media_lock:
#                     downloaded_pexels_ids.add(image.get("id"))

#                 temp_media_pool.append({
#                     "path": path,
#                     "type": "image",
#                     "duration": DEFAULT_IMAGE_DURATION,
#                     "character": character_name,
#                     "tag": None,
#                     "embedding": None,
#                     "source": "pexels_image",
#                     "rank": successful_downloads
#                 })

#                 successful_downloads += 1
#                 logging.info(f"✅ Successfully downloaded image {successful_downloads}/{media_count}")

#             except Exception as e:
#                 failed_downloads += 1
#                 logging.error(f"❌ Exception downloading Pexels image {i+1}: {e}")
#                 if os.path.exists(path):
#                     try:
#                         os.remove(path)
#                     except:
#                         pass
#                 continue

#         logging.info(f"📊 Pexels image results: {successful_downloads} successful, {failed_downloads} failed")

#         if not temp_media_pool:
#             logging.error("❌ No Pexels images successfully downloaded")
#             return []

#         scored_media = sorted(temp_media_pool, key=lambda m: m["rank"])
#         available, fallback = [], []
#         for m in scored_media:
#             (available if m["path"] not in used_media_dict else fallback).append(m)

#         selected = []
#         for m in available:
#             if len(selected) >= media_count:
#                 break
#             with media_lock:
#                 used_media_dict[m["path"]] = True
#             selected.append({"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False})

#         if len(selected) < media_count:
#             for m in fallback[:media_count - len(selected)]:
#                 selected.append({"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False})

#         if not selected:
#             logging.error("❌ No images selected from downloaded pool")
#             return []

#         base = max(MIN_IMAGE_DURATION, duration_needed / len(selected))
#         remaining = duration_needed
#         for s in selected:
#             s["duration"] = min(DEFAULT_IMAGE_DURATION, base, remaining)
#             remaining -= s["duration"]

#         if remaining > DURATION_TOLERANCE:
#             idx, n = 0, len(selected)
#             while remaining > DURATION_TOLERANCE and idx < n * 2:
#                 s = selected[idx % n]
#                 can_add = DEFAULT_IMAGE_DURATION - s["duration"]
#                 if can_add > 0:
#                     add = min(can_add, remaining)
#                     s["duration"] += add
#                     remaining -= add
#                 idx += 1

#         total = sum(s["duration"] for s in selected)
#         if abs(total - duration_needed) > DURATION_TOLERANCE:
#             logging.warning(f"⚠️ Duration mismatch (Pexels images): Need {duration_needed:.2f}s, got {total:.2f}s")

#         logging.info(f"✅ Returning {len(selected)} Pexels images with total duration {total:.2f}s")
#         return selected

#     # -------------------------------------------------
#     # 2c) ImageFX
#     # -------------------------------------------------
#     if segment.get("imagefx_required", True) and mode == "original":
#         from cookie_reader import read_active_cookie
#         cookie = read_active_cookie()
#         if not cookie:
#             logging.warning("⚠️ No active cookie available; skipping ImageFX branch.")
#             return []

#         media_count = max(1, ceil(duration_needed / DEFAULT_IMAGE_DURATION))

#         style_spec = "ghibli"
#         override_prompt = None

#         if segment.get("ai_prompt"):
#             prompt_text = segment["ai_prompt"].strip()
#             known_styles = {
#                 "ghibli", "explanatory", "explanatory_dark", "chalk", "explanatory_illustrated",
#                 "fitness_video", "war", "cinematic", "dark_story", "luxury_motivation", "female_chalk"
#             }
#             if prompt_text.lower() in known_styles:
#                 style_spec = prompt_text.lower()
#             else:
#                 override_prompt = prompt_text

#         prompts = generate_image_fx_prompts(
#             text_segment=segment["text"],
#             script_context=script_context,
#             prev_segments=prev_segments,
#             style_spec=style_spec,
#             override_prompt=override_prompt
#         )

#         with media_lock:
#             cache = used_media_dict.setdefault("_imagefx_cache_multi", {})

#         selected_paths: list[str] = []
#         base_seed = _seed(segment.get("ai_prompt") or segment.get("text", ""))

#         for i, original_prompt in enumerate(prompts):
#             # 1️⃣ Check cache first
#             path_from_cache = None
#             with media_lock:
#                 cached_list = cache.get(original_prompt, [])
#                 for cand in cached_list:
#                     if cand not in used_media_dict:
#                         used_media_dict[cand] = True
#                         path_from_cache = cand
#                         break

#             if path_from_cache:
#                 selected_paths.append(path_from_cache)
#                 continue

#             # 2️⃣ Try generating with progressive safety refinement
#             seed_i = (base_seed + i) % (2**31)
#             out_dir = os.path.join("imagefx_images", f"{uuid.uuid4().hex}_{i:02d}")

#             current_prompt = original_prompt
#             new_paths = None
#             attempt = 0
#             max_attempts = 4  # Original + 3 safety refinements

#             while attempt < max_attempts and not new_paths:
#                 attempt += 1

#                 logging.info(f"🎬 Attempt {attempt}/{max_attempts} for segment {segment_index}")

#                 new_paths = _imagefx_generate(
#                     prompt=current_prompt,
#                     count=1,
#                     out_dir=out_dir,
#                     cookie=cookie,
#                     seed=seed_i,
#                 )

#                 if not new_paths and attempt < max_attempts:
#                     logging.warning(f"⚠️ Generation failed on attempt {attempt}. Refining prompt...")
#                     logging.info(f"🔍 Current prompt: {current_prompt[:100]}...")
#                     current_prompt = refine_prompt_for_safety(current_prompt, style_spec)
#                     logging.info(f"🛡️ Refined prompt: {current_prompt[:100]}...")

#             if not new_paths:
#                 logging.warning(f"⚠️ All {max_attempts} attempts failed for prompt #{i+1}. Skipping.")
#                 continue

#             path = new_paths[0]
#             with media_lock:
#                 cache.setdefault(current_prompt, []).append(path)
#                 used_media_dict[path] = True
#             selected_paths.append(path)
#             logging.info(f"✅ Successfully generated image for segment {segment_index} (attempt {attempt})")

#         if not selected_paths:
#             logging.warning("⚠️ ImageFX: no images generated/selected.")
#             return []

#         temp_media_pool = []
#         for i, path in enumerate(selected_paths[:media_count]):
#             temp_media_pool.append({
#                 "path": path,
#                 "type": "image",
#                 "duration": DEFAULT_IMAGE_DURATION,
#                 "character": character_name,
#                 "tag": None,
#                 "embedding": None,
#                 "source": "imagefx_multi",
#                 "rank": i
#             })

#         selected = [{"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False} for m in temp_media_pool]

#         if len(selected) == 1:
#             selected[0]["duration"] = duration_needed
#             return selected

#         base = max(MIN_IMAGE_DURATION, duration_needed / len(selected))
#         remaining = duration_needed
#         for s in selected:
#             s["duration"] = min(DEFAULT_IMAGE_DURATION, base, remaining)
#             remaining -= s["duration"]

#         total = sum(s["duration"] for s in selected)
#         if abs(total - duration_needed) > DURATION_TOLERANCE:
#             logging.warning(f"⚠️ Duration mismatch (ImageFX multi): Need {duration_needed:.2f}s, got {total:.2f}s")

#         return selected

#     # -------------------------------------------------
#     # 3) Default: score local images & videos together
#     # -------------------------------------------------
#     if duration_needed < 2:
#         return []

#     relevant_media = [
#         m for m in media_pool
#         if m.get('character') == character_name and not m.get("tag")
#     ]

#     if not relevant_media:
#         logging.warning(f"⚠️ No media found for character: {character_name}")
#         return []

#     text_embedding = encode_text(segment.get("text", ""))
#     scored = sorted(
#         [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#         key=lambda x: x[1],
#         reverse=True
#     )

#     available_media, fallback_media = [], []
#     for media, _ in scored:
#         (available_media if media['path'] not in used_media_dict else fallback_media).append(media)

#     media_count = max(1, ceil(duration_needed / SPLIT_BASE_DURATION))
#     selected = []

#     for media in available_media:
#         if len(selected) >= media_count:
#             break
#         with media_lock:
#             used_media_dict[media["path"]] = True
#         selected.append({
#             "media": media,
#             "duration": 0,
#             "has_audio": (media.get("type") == "video" and not media.get("mute", False))
#         })

#     if len(selected) < media_count:
#         for media in fallback_media[:media_count - len(selected)]:
#             selected.append({
#                 "media": media,
#                 "duration": 0,
#                 "has_audio": (media.get("type") == "video" and not media.get("mute", False))
#             })

#     base = duration_needed / len(selected)
#     remaining = duration_needed

#     for s in selected:
#         media = s["media"]
#         if media.get("type") == "video":
#             media_max = media.get("duration", 0)
#             chunks = floor(media_max / SPLIT_BASE_DURATION)
#             max_usable = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
#             s["duration"] = min(max(base, MIN_VIDEO_DURATION), max_usable, remaining)
#         else:
#             media_max = media.get("duration", DEFAULT_IMAGE_DURATION)
#             s["duration"] = min(max(base, MIN_IMAGE_DURATION), media_max, remaining)
#         remaining -= s["duration"]

#     if remaining > DURATION_TOLERANCE:
#         flex = [s for s in selected if (s['media']['type'] == 'image') or (s['media']['type'] == 'video' and s['duration'] < s['media']['duration'])]
#         if flex:
#             extra = remaining / len(flex)
#             for s in flex:
#                 max_possible = s['media'].get('duration', float('inf')) if s['media']['type'] == 'video' else float('inf')
#                 add = min(extra, max_possible - s['duration'])
#                 s['duration'] += add
#                 remaining -= add
#                 if remaining <= DURATION_TOLERANCE:
#                     break

#     total = sum(s['duration'] for s in selected)
#     if abs(total - duration_needed) > DURATION_TOLERANCE:
#         logging.warning(f"⚠️ Duration mismatch: Need {duration_needed:.2f}s, got {total:.2f}s")

#     return selected



# -----------------------------------------------------------------------------
# 2. find_best_media_combination
# Full drop-in replacement.
# Changes are isolated to section 2c (ImageFX) only — all other sections
# (Whisk, tag override, Pexels video, Pexels image, local media) are unchanged.
# -----------------------------------------------------------------------------


# =============================================================================
# DROP-IN REPLACEMENTS
# =============================================================================
# Place at module level (top of tests.py, outside any function)
# =============================================================================
_imagefx_auth_failed = False


# =============================================================================
# 1. _handle_401
# =============================================================================
def _handle_401(last_output: str) -> None:
    """
    Cloudflare worker returned 401. Cookie rotation is legacy — auth is API-key only.
    """
    global _imagefx_auth_failed
    _imagefx_auth_failed = True
    logging.error(
        "🔐 Cloudflare image worker: 401 Unauthorized — check CLOUDFLARE_IMAGE_API_KEY "
        f"({last_output})"
    )


# =============================================================================
# 2. find_best_media_combination
# =============================================================================
def find_best_media_combination(
    media_pool,
    segment,
    character_name,
    duration_needed,
    used_media_dict,
    media_lock,
    mode,
    segment_index,
    script_context,
    prev_segments=None,
    whisk_shared_state=None,
):
    """
    Find best media for a segment.

    ImageFX section behaviour:
    ─────────────────────────────────────────────────────────────────────────
    PROMPT CACHING
      OpenAI is called exactly once per segment. The prompt is written to the
      shared manager dict before the first _imagefx_generate attempt so every
      retry finds it and skips the API call entirely.

    PROMPT REFINEMENT
      On non-auth failures (safety blocks, graphic content) the prompt is
      progressively refined. Each refined version is persisted immediately to
      a separate refined_cache_key so retries resume from the latest version
      rather than starting over.

    401 / UNAUTHENTICATED HANDLING
      On a 401 the current prompt state (original or already refined) is
      cached, the global _imagefx_auth_failed flag is set in the shared dict
      (visible to all processes), and the worker enters a blocking poll loop.

      While waiting:
        - Telegram notifications fire at IMAGEFX_401_NOTIFY_INTERVAL_SEC
          intervals with occurrence count and elapsed downtime.
        - Every POLL_INTERVAL_SEC the worker reads a fresh cookie and probes
          it with a real generation attempt using the CACHED prompt — zero
          new OpenAI calls.
        - The first worker to recover clears the shared flag and sends a
          recovery notification — all other waiting workers see the flag
          cleared on their next poll and also resume.
        - Workers processing non-ImageFX segments are completely unaffected.

      If no valid cookie appears within WAIT_TIMEOUT_SEC the segment gives up
      and returns [] as an absolute last resort — only then does the fallback
      chain produce a black frame.
    ─────────────────────────────────────────────────────────────────────────
    """

    # ── Tunables ─────────────────────────────────────────────────────────────
    IMAGEFX_401_NOTIFY_INTERVAL_SEC = 300   # Telegram at most once per 5 min
    POLL_INTERVAL_SEC               = 15    # check for fresh cookie every 15s
    WAIT_TIMEOUT_SEC                = 18000  # give up after 30 min

    # ── Helpers ───────────────────────────────────────────────────────────────
    def loop_tagged_media(tagged_media, duration_needed):
        looped, total, i = [], 0.0, 0
        while total < duration_needed and tagged_media:
            media      = tagged_media[i % len(tagged_media)]
            media_dur  = media.get("duration", DEFAULT_IMAGE_DURATION if media["type"] == "image" else 0)
            dur_to_use = min(media_dur, duration_needed - total)
            looped.append({"media": media, "duration": dur_to_use, "has_audio": False})
            with media_lock_guard(media_lock, f"seg{segment_index}:loop_tag"):
                used_media_dict[media["path"]] = True
            total += dur_to_use
            i     += 1
        return looped

    def _send_telegram_safe(message, title, url, button):
        try:
            send_telegram_notification(message, title, url, button)
        except Exception as e:
            logging.error(f"❌ Telegram notification failed: {e}")

    def _update_401_counters_and_check_notify(now):
        """
        Must be called inside media_lock.
        Returns (should_notify, failure_count, first_seen_at).
        """
        last_notified = used_media_dict.get("_imagefx_401_last_notified_at", 0)
        should_notify = (now - last_notified) >= IMAGEFX_401_NOTIFY_INTERVAL_SEC
        if should_notify:
            used_media_dict["_imagefx_401_last_notified_at"] = now
        if "_imagefx_401_first_seen_at" not in used_media_dict:
            used_media_dict["_imagefx_401_first_seen_at"] = now
        used_media_dict["_imagefx_401_count"] = (
            used_media_dict.get("_imagefx_401_count", 0) + 1
        )
        return (
            should_notify,
            used_media_dict["_imagefx_401_count"],
            used_media_dict["_imagefx_401_first_seen_at"],
        )

    def _clear_auth_failure_state():
        """Must be called inside media_lock."""
        used_media_dict["_imagefx_auth_failed"]      = False
        used_media_dict["_imagefx_401_count"]         = 0
        used_media_dict["_imagefx_401_first_seen_at"] = 0

    # ── Pexels ID set initialisation ─────────────────────────────────────────
    with media_lock_guard(media_lock, f"seg{segment_index}:pexels_init"):
        if "_pexels_downloaded_ids" not in used_media_dict:
            used_media_dict["_pexels_downloaded_ids"] = set()
    downloaded_pexels_ids = used_media_dict["_pexels_downloaded_ids"]

    # =========================================================================
    # 0) WHISK (highest priority)
    # =========================================================================
    if segment.get("whisk_required") and mode == "original":
        if not whisk_shared_state or not whisk_shared_state.get("media_id"):
            logging.error("❌ whisk_required but no character context available")
            return []
        logging.info(f"🎭 Segment {segment_index}: using Whisk character")
        return whisk_generate_for_segment(
            segment=segment,
            whisk_shared_state=whisk_shared_state,
            media_lock=media_lock,
            duration_needed=duration_needed,
            segment_index=segment_index,
        )

    # =========================================================================
    # 1) Tag override
    # =========================================================================
    tag = segment.get("tag")
    if tag:
        tagged_media = [m for m in media_pool if m.get("tag") == tag]
        if tagged_media:
            return loop_tagged_media(tagged_media, duration_needed)
        logging.info(f"🔁 No tagged media for tag '{tag}' — falling back")

    # =========================================================================
    # 2) Pexels videos
    # =========================================================================
    if segment.get("pexels_required", False):
        query = generate_pexels_query(segment.get("text", ""), segment_index, media_type="video")
        logging.info(f"🔍 Pexels video query: {query}")

        media_count = max(1, ceil(duration_needed / SPLIT_BASE_DURATION))
        batch       = search_multiple_pexels_media(query, media_type="video", per_page=10, page=1)
        fresh       = [v for v in batch if v.get("id") not in downloaded_pexels_ids]

        if len(fresh) < media_count:
            batch2 = search_multiple_pexels_media(query, media_type="video", per_page=50, page=1)
            seen   = {v.get("id") for v in fresh}
            fresh.extend([v for v in batch2 if v.get("id") not in downloaded_pexels_ids and v.get("id") not in seen])

        if not fresh:
            logging.warning("⚠️ No fresh Pexels videos returned.")
            return []

        temp_media_pool       = []
        successful_downloads  = 0
        failed_downloads      = 0

        for i, video in enumerate(fresh[: media_count * 2]):
            if successful_downloads >= media_count:
                break
            path = os.path.join("pexels_videos", f"pexels_{uuid.uuid4().hex}_{i}.mp4")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                logging.info(f"🎥 Downloading Pexels video {successful_downloads + 1}/{media_count}...")
                if not download_video(video["url"], path):
                    failed_downloads += 1
                    continue
                if not os.path.exists(path) or os.path.getsize(path) < 10000:
                    failed_downloads += 1
                    if os.path.exists(path):
                        os.remove(path)
                    continue
                with media_lock_guard(media_lock, f"seg{segment_index}:pexels_vid_id"):
                    downloaded_pexels_ids.add(video.get("id"))
                temp_media_pool.append({
                    "path": path, "type": "video",
                    "duration": video.get("duration", 0),
                    "character": character_name, "tag": None,
                    "embedding": None, "source": "pexels",
                    "rank": successful_downloads,
                })
                successful_downloads += 1
            except Exception as e:
                failed_downloads += 1
                logging.error(f"❌ Pexels video download error: {e}")
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass

        logging.info(f"📊 Pexels video: {successful_downloads} ok / {failed_downloads} failed")
        if not temp_media_pool:
            logging.error("❌ No Pexels videos downloaded")
            return []

        scored_media   = sorted(temp_media_pool, key=lambda m: m["rank"])
        available, fallback = [], []
        for m in scored_media:
            (available if m["path"] not in used_media_dict else fallback).append(m)

        selected = []
        for m in available:
            if len(selected) >= media_count:
                break
            with media_lock_guard(media_lock, f"seg{segment_index}:pexels_vid_mark"):
                used_media_dict[m["path"]] = True
            selected.append({"media": m, "duration": 0, "has_audio": False})
        if len(selected) < media_count:
            for m in fallback[: media_count - len(selected)]:
                selected.append({"media": m, "duration": 0, "has_audio": False})
        if not selected:
            return []

        base      = duration_needed / len(selected)
        remaining = duration_needed
        for s in selected:
            media_max = s["media"].get("duration", 0)
            chunks    = floor(media_max / SPLIT_BASE_DURATION)
            max_use   = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
            s["duration"] = min(max(base, MIN_VIDEO_DURATION), max_use, remaining)
            remaining    -= s["duration"]

        if remaining > DURATION_TOLERANCE:
            flex = [s for s in selected if s["media"]["type"] == "image" or s["duration"] < s["media"]["duration"]]
            if flex:
                extra = remaining / len(flex)
                for s in flex:
                    cap = s["media"].get("duration", float("inf")) if s["media"]["type"] == "video" else float("inf")
                    add = min(extra, cap - s["duration"])
                    s["duration"] += add
                    remaining     -= add
                    if remaining <= DURATION_TOLERANCE:
                        break

        total = sum(s["duration"] for s in selected)
        if abs(total - duration_needed) > DURATION_TOLERANCE:
            logging.warning(f"⚠️ Pexels video duration mismatch: need {duration_needed:.2f}s got {total:.2f}s")
        logging.info(f"✅ Returning {len(selected)} Pexels videos ({total:.2f}s)")
        return selected

    # =========================================================================
    # 2b) Pexels images
    # =========================================================================
    if segment.get("pexels_image_required", False):
        query = generate_pexels_query(segment.get("text", ""), segment_index, media_type="image")
        logging.info(f"🖼️ Pexels image query: {query}")

        media_count = max(1, ceil(duration_needed / DEFAULT_IMAGE_DURATION))
        batch       = search_multiple_pexels_media(query, media_type="image", per_page=5, page=1)
        fresh       = [img for img in batch if img.get("id") not in downloaded_pexels_ids]

        if len(fresh) < media_count:
            batch2 = search_multiple_pexels_media(query, media_type="image", per_page=20, page=1)
            seen   = {img.get("id") for img in fresh}
            fresh.extend([img for img in batch2 if img.get("id") not in downloaded_pexels_ids and img.get("id") not in seen])

        if not fresh:
            logging.warning("⚠️ No fresh Pexels images returned.")
            return []

        temp_media_pool      = []
        successful_downloads = 0
        failed_downloads     = 0

        for i, image in enumerate(fresh[: media_count * 2]):
            if successful_downloads >= media_count:
                break
            path = os.path.join("pexels_images", f"pexels_{uuid.uuid4().hex}_{i}.jpg")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                logging.info(f"🖼️ Downloading Pexels image {successful_downloads + 1}/{media_count}...")
                if not download_pexels_image(image["url"], path):
                    failed_downloads += 1
                    continue
                if not os.path.exists(path) or os.path.getsize(path) < 1000:
                    failed_downloads += 1
                    if os.path.exists(path):
                        os.remove(path)
                    continue
                with media_lock_guard(media_lock, f"seg{segment_index}:pexels_img_id"):
                    downloaded_pexels_ids.add(image.get("id"))
                temp_media_pool.append({
                    "path": path, "type": "image",
                    "duration": DEFAULT_IMAGE_DURATION,
                    "character": character_name, "tag": None,
                    "embedding": None, "source": "pexels_image",
                    "rank": successful_downloads,
                })
                successful_downloads += 1
            except Exception as e:
                failed_downloads += 1
                logging.error(f"❌ Pexels image download error: {e}")
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass

        logging.info(f"📊 Pexels image: {successful_downloads} ok / {failed_downloads} failed")
        if not temp_media_pool:
            logging.error("❌ No Pexels images downloaded")
            return []

        scored_media = sorted(temp_media_pool, key=lambda m: m["rank"])
        available, fallback = [], []
        for m in scored_media:
            (available if m["path"] not in used_media_dict else fallback).append(m)

        selected = []
        for m in available:
            if len(selected) >= media_count:
                break
            with media_lock_guard(media_lock, f"seg{segment_index}:pexels_vid_mark"):
                used_media_dict[m["path"]] = True
            selected.append({"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False})
        if len(selected) < media_count:
            for m in fallback[: media_count - len(selected)]:
                selected.append({"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False})
        if not selected:
            return []

        base      = max(MIN_IMAGE_DURATION, duration_needed / len(selected))
        remaining = duration_needed
        for s in selected:
            s["duration"] = min(DEFAULT_IMAGE_DURATION, base, remaining)
            remaining    -= s["duration"]

        if remaining > DURATION_TOLERANCE:
            idx, n = 0, len(selected)
            while remaining > DURATION_TOLERANCE and idx < n * 2:
                s       = selected[idx % n]
                can_add = DEFAULT_IMAGE_DURATION - s["duration"]
                if can_add > 0:
                    add           = min(can_add, remaining)
                    s["duration"] += add
                    remaining     -= add
                idx += 1

        total = sum(s["duration"] for s in selected)
        if abs(total - duration_needed) > DURATION_TOLERANCE:
            logging.warning(f"⚠️ Pexels image duration mismatch: need {duration_needed:.2f}s got {total:.2f}s")
        logging.info(f"✅ Returning {len(selected)} Pexels images ({total:.2f}s)")
        return selected

    # =========================================================================
    # 2c) ImageFX
    # =========================================================================
    if segment.get("imagefx_required", True) and mode == "original":

        # Cache keys — all stored in shared manager dict
        prompt_cache_key  = f"_imagefx_prompt_{segment_index}"
        refined_cache_key = f"_imagefx_refined_prompt_{segment_index}"

        media_count = max(1, ceil(duration_needed / DEFAULT_IMAGE_DURATION))
        style_spec  = "ghibli"
        override_prompt = None

        if segment.get("ai_prompt"):
            prompt_text  = segment["ai_prompt"].strip()
            known_styles = {
                "ghibli", "explanatory", "explanatory_dark", "chalk",
                "explanatory_illustrated", "fitness_video", "war", "cinematic",
                "dark_story", "luxury_motivation", "female_chalk", "space",
            }
            style_spec      = prompt_text.lower() if prompt_text.lower() in known_styles else style_spec
            override_prompt = prompt_text          if prompt_text.lower() not in known_styles else None

        # ── PROMPT CACHE ─────────────────────────────────────────────────────
        # OpenAI is called exactly once per segment regardless of how many
        # retries occur. Prompt is written BEFORE any generation attempt.
        with media_lock_guard(media_lock, f"seg{segment_index}:prompt_cache_read"):
            cached_prompts = used_media_dict.get(prompt_cache_key)

        if cached_prompts:
            logging.info(f"♻️  Segment {segment_index}: reusing cached prompt — no OpenAI call.")
            prompts = cached_prompts
        else:
            logging.info(f"🤖 Segment {segment_index}: generating OpenAI prompt...")
            prompts = generate_image_fx_prompts(
                text_segment=segment["text"],
                script_context=script_context,
                prev_segments=prev_segments,
                style_spec=style_spec,
                override_prompt=override_prompt,
            )
            with media_lock_guard(media_lock, f"seg{segment_index}:prompt_cache_write"):
                used_media_dict[prompt_cache_key] = prompts
            logging.info(f"✅ Segment {segment_index}: prompt cached.")

        with media_lock_guard(media_lock, f"seg{segment_index}:image_cache"):
            image_cache = used_media_dict.setdefault("_imagefx_cache_multi", {})

        selected_paths: list[str] = []
        base_seed                 = _seed(segment.get("ai_prompt") or segment.get("text", ""))

        for prompt_idx, original_prompt in enumerate(prompts):

            # Check image cache first
            path_from_cache = None
            with media_lock_guard(media_lock, f"seg{segment_index}:cache_lookup"):
                for cand in image_cache.get(original_prompt, []):
                    if cand not in used_media_dict:
                        used_media_dict[cand] = True
                        path_from_cache       = cand
                        break

            if path_from_cache:
                selected_paths.append(path_from_cache)
                continue

            seed_i  = (base_seed + prompt_idx) % (2 ** 31)
            out_dir = os.path.join("imagefx_images", f"{uuid.uuid4().hex}_{prompt_idx:02d}")

            # Resume from last refined prompt if one exists, else use original
            with media_lock_guard(media_lock, f"seg{segment_index}:refined_prompt"):
                current_prompt = used_media_dict.get(refined_cache_key, original_prompt)

            new_paths = None
            attempt   = 0
            max_attempts = 4

            while attempt < max_attempts and not new_paths:
                attempt += 1
                logging.info(
                    f"🎬 Segment {segment_index} — Cloudflare image attempt {attempt}/{max_attempts} "
                    f"(prompt: {current_prompt[:60]}...)"
                )

                global _imagefx_auth_failed
                _imagefx_auth_failed = False

                new_paths = _imagefx_generate(
                    prompt=current_prompt,
                    count=1,
                    out_dir=out_dir,
                    seed=seed_i,
                )

                if new_paths:
                    with media_lock_guard(media_lock, f"seg{segment_index}:prompt_done"):
                        used_media_dict[prompt_cache_key] = [current_prompt]
                        if refined_cache_key in used_media_dict:
                            del used_media_dict[refined_cache_key]
                    break

                if _imagefx_auth_failed:
                    logging.error(
                        f"❌ Segment {segment_index}: Cloudflare worker returned 401 — "
                        f"check CLOUDFLARE_IMAGE_API_KEY (no cookie wait; failing fast)."
                    )
                    break

                if attempt < max_attempts:
                    logging.warning(
                        f"⚠️  Segment {segment_index}: generation failed (non-auth, "
                        f"likely safety block) on attempt {attempt}. Refining prompt..."
                    )
                    logging.info(f"🔍 Current: {current_prompt[:100]}...")
                    current_prompt = refine_prompt_for_safety(current_prompt, style_spec)
                    logging.info(f"🛡️  Refined: {current_prompt[:100]}...")
                    with media_lock_guard(media_lock, f"seg{segment_index}:refined_write"):
                        used_media_dict[refined_cache_key] = current_prompt

            # End of attempt loop
            if not new_paths:
                logging.warning(
                    f"⚠️  Segment {segment_index}: all attempts exhausted for prompt #{prompt_idx + 1}."
                )
                continue

            path = new_paths[0]
            with media_lock_guard(media_lock, f"seg{segment_index}:image_store"):
                image_cache.setdefault(current_prompt, []).append(path)
                used_media_dict[path] = True
            selected_paths.append(path)
            logging.info(f"✅ Segment {segment_index}: image obtained (prompt #{prompt_idx + 1}).")

        if not selected_paths:
            logging.warning(f"⚠️  Segment {segment_index}: ImageFX produced no images.")
            return []

        # Build output list
        temp_media_pool = [
            {
                "path": path, "type": "image",
                "duration": DEFAULT_IMAGE_DURATION,
                "character": character_name, "tag": None,
                "embedding": None, "source": "imagefx_multi",
                "rank": i,
            }
            for i, path in enumerate(selected_paths[:media_count])
        ]

        selected = [{"media": m, "duration": DEFAULT_IMAGE_DURATION, "has_audio": False} for m in temp_media_pool]

        if len(selected) == 1:
            selected[0]["duration"] = duration_needed
            return selected

        base      = max(MIN_IMAGE_DURATION, duration_needed / len(selected))
        remaining = duration_needed
        for s in selected:
            s["duration"] = min(DEFAULT_IMAGE_DURATION, base, remaining)
            remaining    -= s["duration"]

        total = sum(s["duration"] for s in selected)
        if abs(total - duration_needed) > DURATION_TOLERANCE:
            logging.warning(
                f"⚠️  ImageFX duration mismatch: need {duration_needed:.2f}s got {total:.2f}s"
            )

        return selected

    # =========================================================================
    # 3) Default: score local images & videos by embedding similarity
    # =========================================================================
    if duration_needed < 2:
        return []

    relevant_media = [
        m for m in media_pool
        if m.get("character") == character_name and not m.get("tag")
    ]

    if not relevant_media:
        logging.warning(f"⚠️ No local media for character: {character_name}")
        return []

    text_embedding = encode_text(segment.get("text", ""))
    scored = sorted(
        [(m, cosine_similarity(text_embedding, m["embedding"])) for m in relevant_media],
        key=lambda x: x[1],
        reverse=True,
    )

    available_media, fallback_media = [], []
    for media, _ in scored:
        (available_media if media["path"] not in used_media_dict else fallback_media).append(media)

    media_count = max(1, ceil(duration_needed / SPLIT_BASE_DURATION))
    selected    = []

    for media in available_media:
        if len(selected) >= media_count:
            break
        with media_lock_guard(media_lock, f"seg{segment_index}:media_select"):
            used_media_dict[media["path"]] = True
        selected.append({
            "media": media, "duration": 0,
            "has_audio": media.get("type") == "video" and not media.get("mute", False),
        })

    if len(selected) < media_count:
        for media in fallback_media[: media_count - len(selected)]:
            selected.append({
                "media": media, "duration": 0,
                "has_audio": media.get("type") == "video" and not media.get("mute", False),
            })

    base      = duration_needed / len(selected)
    remaining = duration_needed

    for s in selected:
        media = s["media"]
        if media.get("type") == "video":
            media_max = media.get("duration", 0)
            chunks    = floor(media_max / SPLIT_BASE_DURATION)
            max_use   = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
            s["duration"] = min(max(base, MIN_VIDEO_DURATION), max_use, remaining)
        else:
            s["duration"] = min(max(base, MIN_IMAGE_DURATION), media.get("duration", DEFAULT_IMAGE_DURATION), remaining)
        remaining -= s["duration"]

    if remaining > DURATION_TOLERANCE:
        flex = [
            s for s in selected
            if s["media"]["type"] == "image"
            or s["duration"] < s["media"].get("duration", 0)
        ]
        if flex:
            extra = remaining / len(flex)
            for s in flex:
                cap = s["media"].get("duration", float("inf")) if s["media"]["type"] == "video" else float("inf")
                add           = min(extra, cap - s["duration"])
                s["duration"] += add
                remaining     -= add
                if remaining <= DURATION_TOLERANCE:
                    break

    total = sum(s["duration"] for s in selected)
    if abs(total - duration_needed) > DURATION_TOLERANCE:
        logging.warning(f"⚠️ Local media duration mismatch: need {duration_needed:.2f}s got {total:.2f}s")

    return selected

# def find_best_media_combination(media_pool, segment, character_name, duration_needed, used_media_dict, media_lock):
#     # --- Configuration Constants ---
#     DEFAULT_IMAGE_DURATION = 5
#     MIN_IMAGE_DURATION = 2
#     MIN_VIDEO_DURATION = 3
#     PREFERRED_VIDEO_DURATION = 5
#     SPLIT_BASE_DURATION = 5
#     DURATION_TOLERANCE = 0.1

#     # ✅ Helper: Repeat tagged media in loop until duration is filled
#     def loop_tagged_media(tagged_media, duration_needed):
#         looped = []
#         total = 0
#         i = 0
#         while total < duration_needed:
#             media = tagged_media[i % len(tagged_media)]
#             media_duration = media.get("duration", DEFAULT_IMAGE_DURATION if media["type"] == "image" else 0)
#             duration_to_use = min(media_duration, duration_needed - total)
#             looped.append({
#                 "media": media,
#                 "duration": duration_to_use,
#                 "has_audio": False  # Always mute for tagged media
#             })
#             with media_lock:
#                 used_media_dict[media["path"]] = True
#             total += duration_to_use
#             i += 1
#         return looped

#     # ✅ TAG OVERRIDE
#     tag = segment.get("tag")
#     if tag:
#         tagged_media = [m for m in media_pool if m.get('tag') == tag]
#         if tagged_media:
#             return loop_tagged_media(tagged_media, duration_needed)
#         else:
#             logging.info(f"🔁 No tagged media found for tag '{tag}' — falling back to scoring")

#     # --- DEFAULT MATCHING LOGIC (unchanged) ---
#     if duration_needed < 2:
#         return []

#     text_embedding = encode_text(segment["text"])
#     relevant_media = [m for m in media_pool if m['character'] == character_name and not m.get("tag")]

#     if not relevant_media:
#         logging.warning(f"⚠️ No media found for character: {character_name}")
#         return []

#     scored = sorted(
#         [(m, cosine_similarity(text_embedding, m['embedding'])) for m in relevant_media],
#         key=lambda x: x[1], reverse=True
#     )

#     top_media = scored[0][0]
#     if top_media['type'] == 'video' and top_media.get('duration', 0) >= duration_needed:
#         if top_media['path'] not in used_media_dict:
#             with media_lock:
#                 used_media_dict[top_media['path']] = True
#         return [{
#             'media': top_media,
#             'duration': duration_needed,
#             'has_audio': not top_media.get("mute", False)
#         }]

#     available_media = []
#     fallback_media = []

#     for media, _ in scored:
#         if media['path'] not in used_media_dict:
#             available_media.append(media)
#         else:
#             fallback_media.append(media)

#     media_count = max(1, ceil(duration_needed / 5))
#     selected = []

#     for media in available_media:
#         if len(selected) >= media_count:
#             break
#         with media_lock:
#             used_media_dict[media["path"]] = True
#         selected.append({
#             'media': media,
#             'duration': 0,
#             'has_audio': media["type"] == "video" and not media.get("mute", False)
#         })

#     if len(selected) < media_count:
#         needed = media_count - len(selected)
#         for media in fallback_media[:needed]:
#             selected.append({
#                 'media': media,
#                 'duration': 0,
#                 'has_audio': media["type"] == "video" and not media.get("mute", False)
#             })

#     base_duration = duration_needed / len(selected)
#     remaining_duration = duration_needed

#     for s in selected:
#         media = s['media']
#         media_type = media['type']
#         media_max = media.get('duration', DEFAULT_IMAGE_DURATION if media_type == 'image' else 0)

#         if media_type == 'video':
#             chunks = floor(media_max / SPLIT_BASE_DURATION)
#             max_usable = chunks * SPLIT_BASE_DURATION if chunks > 0 else media_max
#             s['duration'] = min(max(base_duration, MIN_VIDEO_DURATION), max_usable, remaining_duration)
#         else:
#             s['duration'] = max(base_duration, MIN_IMAGE_DURATION)

#         remaining_duration -= s['duration']

#     if remaining_duration > DURATION_TOLERANCE:
#         flexible_media = [s for s in selected if (
#             (s['media']['type'] == 'image') or
#             (s['media']['type'] == 'video' and s['duration'] < s['media']['duration'])
#         )]
#         if flexible_media:
#             extra_per_item = remaining_duration / len(flexible_media)
#             for s in flexible_media:
#                 max_possible = s['media'].get('duration', float('inf')) if s['media']['type'] == 'video' else float('inf')
#                 add = min(extra_per_item, max_possible - s['duration'])
#                 s['duration'] += add
#                 remaining_duration -= add
#                 if remaining_duration <= DURATION_TOLERANCE:
#                     break

#     total_duration = sum(s['duration'] for s in selected)
#     if abs(total_duration - duration_needed) > DURATION_TOLERANCE:
#         logging.warning(f"⚠️ Duration mismatch: Needed {duration_needed:.2f}s, got {total_duration:.2f}s")

#     return selected


# ----------------- Video Creation -----------------

def save_metadata(media_pool, section_dir):
    """Save media metadata to a JSON file inside the appropriate section folder (intro/body)."""
    metadata_file = os.path.join(section_dir, "media_metadata.json")
    metadata = []
    
    for media in media_pool:
        metadata.append({
            "path": media.get("path", None) or (media["clip"].filename if "clip" in media else None),
            "character": media["character"],
            "type": media["type"],
            "embedding": media['embedding'].tolist() if isinstance(media['embedding'], np.ndarray) else media['embedding'],
            "duration": media["duration"],
            "tag": media.get("tag", None),
            "mute": media.get("mute", False)
        })
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"📄 Metadata saved to {metadata_file}")

def load_metadata(section_dir):
    """Load media metadata from the existing section folder (intro/body)."""
    metadata_file = os.path.join(section_dir, "media_metadata.json")
    
    if not os.path.exists(metadata_file):
        return []

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    logging.info(f"📂 Loaded metadata from {metadata_file}")
    return metadata


import multiprocessing
from functools import partial
import os
import uuid
import logging
import gc
from collections import deque
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
#very very most important b4 overlays
# def encode_video_segment(segment_info):
   
#     segment, media_pool, segment_audio_path, section_dir, chunk_folder, i, used_media_dict, media_lock = segment_info
#     chunk_filename = f"chunk_{i}_{uuid.uuid4()}.mp4"
#     chunk_path = os.path.join(chunk_folder, chunk_filename)
#     segment, media_pool, segment_audio_path, section_dir, chunk_folder, i, used_media_dict, media_lock = segment_info
#     print(f"[{i}] Audio path: {segment_audio_path}")

    
#     logging.info(f"🔹 Processing segment {i}: {segment['text'][:50]}...")

#     try:
#         character_name = segment["character"]
#         segment_tag = segment.get("tag")

#         # Handle tagged assets
#         if segment_audio_path is None and segment_tag:
#             tagged_media = [m for m in media_pool if m["character"] == character_name and m.get("tag") == segment_tag]
#             if tagged_media:
#                 selected = tagged_media[0]
#                 clip = (
#                     VideoFileClip(selected["path"]).subclip(0, selected["duration"])
#                     if selected["type"] == "video"
#                     else create_blurred_background_image(selected["path"], selected["duration"])
#                 )
#                 clip.write_videofile(
#                     chunk_path,
#                     codec="libx264",
#                     audio_codec="aac",
#                     threads=4,
#                     fps=30,
#                     preset="medium",
#                     ffmpeg_params=["-pix_fmt", "yuv420p"]
#                 )
#                 clip.close()
#                 return chunk_path

#         # Main processing
#         segment_audio = AudioFileClip(segment_audio_path)
#         duration_needed = segment_audio.duration
#         selected_media = find_best_media_combination(
#             media_pool, segment["text"], character_name, duration_needed, used_media_dict, media_lock
#         )

#         if not selected_media:
#             logging.error(f"❌ No media selected for segment {i}")
#             return None

#         segment_clips = []
#         for item in selected_media:
#             media = item["media"]
#             assigned_duration = item["duration"]
            
#             try:
#                 if media["type"] == "video":
#                     # MODIFIED: Only trim the end
#                     start_time = 0
#                     end_time = min(assigned_duration, media["duration"])
                    
#                     if media["duration"] < assigned_duration:
#                         logging.warning(f"⚠️ Video too short: {media['path']} ({media['duration']}s < {assigned_duration}s needed)")
                    
#                     clip = VideoFileClip(media["path"]).subclip(start_time, end_time)
#                     clip = clip.set_duration(assigned_duration)
                    
#                     if media.get("mute", False):
#                         clip = clip.without_audio()
                    
#                     logging.info(f"🎥 Video (end-trimmed): {os.path.basename(media['path'])} [0-{end_time:.2f}s]")
#                 else:
#                     clip = create_blurred_background_image(media["path"], assigned_duration)
#                     logging.info(f"🖼️ Image: {os.path.basename(media['path'])} ({assigned_duration:.2f}s)")
                
#                 segment_clips.append(clip)
#             except Exception as e:
#                 logging.error(f"Failed to process {media['path']}: {str(e)}")
#                 continue

#         # Concatenate and export
#         final_clip = concatenate_videoclips(segment_clips, method="compose")
#         final_clip = final_clip.set_audio(segment_audio).set_duration(duration_needed)
        
#         final_clip.write_videofile(
#             chunk_path,
#             codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#             audio_codec="aac",
#             threads=4,
#             fps=30,
#             preset="fast",
#             ffmpeg_params=["-pix_fmt", "yuv420p"]
#         )

#         return chunk_path

#     except Exception as e:
#         logging.error(f"❌ Segment {i} failed: {str(e)}", exc_info=True)
#         return None

#     finally:
#         # Cleanup
#         for var in ["segment_audio", "final_clip", *[f"clip_{i}" for i in range(10)]]:
#             if var in locals():
#                 try:
#                     locals()[var].close()
#                     del locals()[var]
#                 except:
#                     pass

#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         gc.collect()

used_overlay_paths_in_current_clip = set()  # global usage tracking

# Global image queue
overlay_image_queue = None

def reset_overlay_usage_for_new_clip():
    """
    Resets the overlay image tracker so each new clip starts with a clean slate.
    Must be called once per new clip.
    """
    global used_overlay_paths_in_current_clip
    used_overlay_paths_in_current_clip = set()

def get_next_overlay_image(media_pool, character_name=None):
    """
    Selects a random overlay image without repeating within the current clip.
    Falls back to character-specific images if no 'overlay' images exist.
    """
    global used_overlay_paths_in_current_clip

    # Try overlay character images first
    overlay_images = [m for m in media_pool if m["character"] == "overlay" and m["type"] == "image"]

    # If no overlay images, fallback to character-specific images
    if not overlay_images and character_name:
        overlay_images = [m for m in media_pool if m["character"] == character_name and m["type"] == "image"]
        if not overlay_images:
            raise RuntimeError(f"❌ No images available for character: {character_name}")
        logging.warning(f"⚠️ No overlay images found, using fallback from: {character_name}")
    elif not overlay_images:
        raise RuntimeError("❌ No overlay or fallback images available.")

    # Filter out used images within the same clip
    unused_images = [img for img in overlay_images if img["path"] not in used_overlay_paths_in_current_clip]

    if not unused_images:
        # If all have been used, reset the set (within the same clip)
        used_overlay_paths_in_current_clip = set()
        unused_images = overlay_images

    selected_image = random.choice(unused_images)
    used_overlay_paths_in_current_clip.add(selected_image["path"])
    return selected_image

# def split_segments(duration: float):
#     segments = []

#     min_video = 3
#     fixed_image = 5
#     max_video = 10

#     if 10 < duration < 15:
#         # Handle 10s < duration < 15s → [video, image]
#         steal = 15 - duration
#         video_time = max(min_video, max_video - steal)
#         image_time = duration - video_time
#         segments = [("video", video_time), ("image", image_time)]

#     elif 15 <= duration < 25:
#         # Handle 15s ≤ duration < 25s → [video, image, video]
#         steal = 25 - duration
#         steal_left = steal // 2
#         steal_right = steal - steal_left

#         v1 = max(min_video, max_video - steal_left)
#         v2 = max(min_video, max_video - steal_right)
#         image_time = duration - (v1 + v2)

#         # Ensure image has exactly 5s
#         if image_time < fixed_image:
#             extra_needed = fixed_image - image_time
#             if v1 - extra_needed // 2 >= min_video and v2 - (extra_needed - extra_needed // 2) >= min_video:
#                 v1 -= extra_needed // 2
#                 v2 -= (extra_needed - extra_needed // 2)
#                 image_time = fixed_image
#             else:
#                 raise ValueError("Not enough time to maintain minimums in 3-part segment")

#         segments = [("video", v1), ("image", image_time), ("video", v2)]

#     else:
#         # Default multi-segment breakdown: [video, image, video, image...]
#         t = 0
#         while True:
#             if t + max_video + fixed_image + min_video > duration:
#                 break
#             segments.append(("video", max_video))
#             t += max_video

#             if t + fixed_image > duration:
#                 break
#             segments.append(("image", fixed_image))
#             t += fixed_image

#         remaining = duration - t

#         if remaining >= min_video:
#             segments.append(("video", remaining))
#         elif remaining > 0:
#             # Try to steal from previous video segment
#             for i in range(len(segments) - 1, -1, -1):
#                 if segments[i][0] == "video":
#                     prev_dur = segments[i][1]
#                     eat_amount = min_video - remaining
#                     if prev_dur - eat_amount >= min_video:
#                         segments[i] = ("video", prev_dur - eat_amount)
#                         segments.append(("video", min_video))
#                         break
#             else:
#                 raise ValueError("Unable to rebalance final video segment")

#     return segments
def split_segments(
    duration: float,
    v_min: float = 3.0,
    v_max: float = 20.0,
    i_len: float = 10.0,
    enforce_exact_image: bool = True,
):
    """
    Returns a list of (kind, seconds) where kind ∈ {"video","image"}.
    Tuning knobs:
      - v_min: minimum video segment length
      - v_max: max continuous video length before forcing an image
      - i_len: fixed image segment length
      - enforce_exact_image: in the 3-part layout, force image to be exactly i_len (raise if not possible)

    Rules (generalized from your original):
      1) If duration <= v_max: one video segment of 'duration'.
      2) If v_max < duration < v_max + i_len: [video, image] with image at the end.
      3) If v_max + i_len <= duration <= 2*v_max + i_len: [video, image, video],
         trying to keep the image exactly i_len and videos as balanced as possible.
      4) Else: alternate [video=v_max, image=i_len, ...] and finish with a final video
         (>= v_min) by stealing from the previous video if needed.
    """
    if duration <= 0:
        return []

    segments = []

    # ---- Case 1: short, all video
    if duration <= v_max:
        return [("video", duration)]

    # ---- Case 2: one image at end
    if duration < v_max + i_len:
        v1 = max(v_min, min(v_max, duration - i_len))
        i1 = duration - v1
        return [("video", v1), ("image", i1)]

    # ---- Case 3: video + image + video (aim for symmetry, image ≈ i_len)
    if duration <= 2 * v_max + i_len:
        # start with ideal split
        extra = (2 * v_max + i_len) - duration  # how much we must remove from v1+v2
        steal_left = extra // 2
        steal_right = extra - steal_left
        v1 = max(v_min, v_max - steal_left)
        v2 = max(v_min, v_max - steal_right)
        img = duration - (v1 + v2)

        if enforce_exact_image:
            # force image segment to be exactly i_len if possible by shaving video
            if img < i_len:
                need = i_len - img
                # take half from v1, half from v2 while respecting v_min
                take_left = min(need // 2, max(0.0, v1 - v_min))
                take_right = min(need - take_left, max(0.0, v2 - v_min))
                v1 -= take_left
                v2 -= take_right
                img = duration - (v1 + v2)
                if img < i_len - 1e-6:  # still short
                    raise ValueError("Not enough time to maintain minimums in 3-part segment")
            elif img > i_len:
                # if image is longer than desired, push excess into videos (up to v_max)
                give = img - i_len
                add_left = min(give // 2, max(0.0, v_max - v1))
                add_right = min(give - add_left, max(0.0, v_max - v2))
                v1 += add_left
                v2 += add_right
                img = duration - (v1 + v2)
                # if still > i_len, we accept a slightly longer image (can be tightened if needed)

        return [("video", v1), ("image", img), ("video", v2)]

    # ---- Case 4: alternating blocks
    t = 0.0
    while True:
        if t + v_max + i_len + v_min > duration:
            break
        segments.append(("video", v_max))
        t += v_max

        if t + i_len > duration:
            break
        segments.append(("image", i_len))
        t += i_len

    remaining = duration - t

    if remaining >= v_min:
        segments.append(("video", remaining))
    elif remaining > 0:
        # steal from the last video to ensure final video >= v_min
        for idx in range(len(segments) - 1, -1, -1):
            if segments[idx][0] == "video":
                prev = segments[idx][1]
                need = v_min - remaining
                if prev - need >= v_min:
                    segments[idx] = ("video", prev - need)
                    segments.append(("video", v_min))
                    break
        else:
            # If we cannot rebalance, fall back to one final video chunk
            # (Alternatively, raise to mimic original strictness)
            segments.append(("video", remaining))

    return segments

def create_zoomed_overlay_with_blur_background(image_path, duration, target_size=(1280, 720), zoom_ratio=0.04):
    """
    Creates an ImageClip with a blurred background and overlay image,
    then applies a zoom-in effect to the entire clip.
    """
    with Image.open(image_path).convert("RGB") as opened_img:
        img = opened_img.copy()

    img_width, img_height = img.size
    target_width, target_height = target_size
    aspect_ratio = img_width / img_height
    target_aspect_ratio = target_width / target_height

    needs_background = abs(aspect_ratio - target_aspect_ratio) > 0.01

    if needs_background:
        # Create blurred background
        blurred_bg = img.resize((target_width, target_height)).filter(ImageFilter.GaussianBlur(40))

        # Resize main image to fit height
        scale_factor = target_height / img_height
        new_width = int(img_width * scale_factor)
        resized_img = img.resize((new_width, target_height))

        # Paste resized image onto blurred background
        x_offset = (target_width - new_width) // 2
        composite = blurred_bg.copy()
        composite.paste(resized_img, (x_offset, 0))

        # Save composite
        composite_path = os.path.splitext(image_path)[0] + "_blur_zoom.jpg"
        composite.save(composite_path)
        base_clip = ImageClip(composite_path).set_duration(duration)
    else:
        resized_img = img.resize((target_width, target_height))
        resized_path = os.path.splitext(image_path)[0] + "_resized_zoom.jpg"
        resized_img.save(resized_path)
        base_clip = ImageClip(resized_path).set_duration(duration)

    # 🔍 Apply zoom-in effect
    return zoom_in_effect(base_clip, zoom_ratio=zoom_ratio).set_duration(duration)




from moviepy.editor import *
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
import os
import uuid
import logging
import gc
import ffmpeg

# def overlay_avatar_no_audio(resized_clip, avatar_path, background_path, output_path, tag,
#                             media_pool=None, character_name=None):
#     """
#     Keeps your bordered panel video over a looped 1920x1080 background.
#     For each 'image' window from split_segments(duration), it does EXACTLY what your
#     encode path does: it builds a zoomed overlay clip via
#     create_zoomed_overlay_with_blur_background(...), pre-renders that short segment,
#     then overlays it fullscreen at the correct time window.

#     Notes:
#       - We keep the original audio from `resized_clip` (this function is *no_audio*).
#       - The overlay image clips are pre-rendered (video-only) and composited with ffmpeg.
#     """
#     try:
#         import os
#         import logging
#         import shutil
#         import tempfile
#         import ffmpeg

#         # 👉 You must have this helper available in your module:
#         # from your_module import create_zoomed_overlay_with_blur_background

#         # 1) Validate background
#         if not background_path or not os.path.exists(background_path):
#             raise FileNotFoundError(f"❌ Background video not found: {background_path}")

#         # 2) Probe base clip info and duration
#         probe = ffmpeg.probe(resized_clip)
#         vstream = next(s for s in probe['streams'] if s.get('codec_type') == 'video')
#         duration = float(
#             vstream.get('duration')
#             or probe.get('format', {}).get('duration')
#             or 0.0
#         )
#         if duration <= 0:
#             raise ValueError("❌ Could not determine input video duration.")

#         # 3) Inputs
#         main_input = ffmpeg.input(resized_clip)
#         main_video = main_input.video
#         main_audio = main_input.audio  # we keep original audio

#         # 4) Build looped 1080p background
#         bg = (
#             ffmpeg.input(background_path, stream_loop=-1, t=duration)
#             .video
#             .filter('scale', 1920, 1080, force_original_aspect_ratio='increase')
#             .filter('crop', 1920, 1080)
#             .filter('format', 'yuv420p')
#         )

#         # 5) Compute avatar panel size and border
#         orig_w = int(vstream['width'])
#         orig_h = int(vstream['height'])
#         aspect = orig_w / orig_h

#         max_w, max_h = 1600, 900
#         if aspect > (max_w / max_h):
#             panel_w = max_w
#             panel_h = int(max_w / aspect)
#         else:
#             panel_h = max_h
#             panel_w = int(max_h * aspect)

#         if panel_w % 2:  panel_w -= 1
#         if panel_h % 2:  panel_h -= 1

#         panel = (
#             main_video
#             .filter('scale', panel_w, panel_h)
#             .filter('format', 'yuv420p')
#         )

#         border = 16
#         bordered = panel.filter(
#             'pad',
#             w=panel_w + border * 2,
#             h=panel_h + border * 2,
#             x=border, y=border,
#             color='white'
#         )

#         pos_x = (1920 - (panel_w + border * 2)) // 2
#         pos_y = (1080 - (panel_h + border * 2)) // 2
#         total_h = panel_h + border * 2

#         # 6) Base composite (BG + bordered panel for full duration)
#         composed = ffmpeg.overlay(bg, bordered, x=pos_x, y=pos_y)

#         # 7) Optional tag label
#         if tag == 'intro':
#             composed = composed.filter(
#                 'drawtext',
#                 text='COMING UP...',
#                 fontsize=90,
#                 fontcolor='red',
#                 x=pos_x + border + 20,
#                 y=pos_y + total_h - border - 40,
#                 fontfile='C:/Windows/Fonts/impact.ttf',
#                 enable='between(t,0,3)'
#             )

#         # 8) Image windows: use the SAME helper you use in encode_segment
#         #    create_zoomed_overlay_with_blur_background(image_path, seg_duration, target_size=(1920,1080))
#         #    Then pre-render that short clip and overlay it fullscreen at the right time.
#         rendered_paths = []
#         tmp_dir = None

#         if media_pool is not None and character_name is not None:
#             try:
#                 reset_overlay_usage_for_new_clip()
#                 plan = split_segments(duration)  # e.g., [("video", 10), ("image", 5), ...]
#             except Exception as e:
#                 logging.warning(f"⚠️ split_segments failed; skipping image overlays: {e}")
#                 plan = []

#             # temp dir for pre-rendered overlay segments
#             tmp_dir = tempfile.mkdtemp(prefix="ovsegs_")

#             # We'll align in time by shifting PTS; still guard with enable window.
#             t_cursor = 0.0
#             seg_idx = 0

#             # We need the target canvas size for your helper
#             target_size = (1920, 1080)

#             # If you want the overlay image segment to carry audio from the base,
#             # your encode path did: .set_audio(audio.subclip(t, t+seg_dur).set_fps(44100).volumex(1.5))
#             # BUT: ffmpeg's overlay uses only the VIDEO from the overlay stream,
#             # and we keep the base audio. So we skip overlay audio here (no change in final audio).

#             for seg_type, seg_dur in plan:
#                 if seg_type == "image" and seg_dur > 0:
#                     try:
#                         overlay = get_next_overlay_image(media_pool, character_name)  # {'path': ...}
#                         img_path = overlay["path"]

#                         # >>> EXACTLY THE SAME BUILDER as in encode_segment <<<
#                         ov_clip = create_zoomed_overlay_with_blur_background(
#                             img_path, seg_dur, target_size=target_size
#                         )
#                         # (We do NOT attach audio here because this function preserves the main audio.)

#                         # Pre-render this short overlay segment to MP4 (video-only)
#                         seg_path = os.path.join(tmp_dir, f"overlay_{seg_idx:04d}.mp4")
#                         ov_clip.write_videofile(
#                             seg_path,
#                             codec="libx264", audio=False, fps=30,
#                             preset="medium", threads=4,
#                             verbose=False, logger=None
#                         )
#                         try:
#                             ov_clip.close()
#                         except Exception:
#                             pass

#                         rendered_paths.append(seg_path)

#                         # Bring the rendered overlay video into ffmpeg and time-shift it
#                         seg_stream = (
#                             ffmpeg
#                             .input(seg_path)
#                             .video
#                             .filter('setpts', f'PTS+{t_cursor}/TB')  # align start time
#                             .filter('format', 'yuv420p')
#                         )

#                         # Composite fullscreen over base during this window
#                         composed = ffmpeg.overlay(
#                             composed, seg_stream,
#                             x=0, y=0,
#                             enable=f'between(t,{t_cursor:.3f},{(t_cursor + seg_dur):.3f})'
#                         )

#                         seg_idx += 1

#                     except Exception as e:
#                         logging.warning(f"⚠️ Overlay image render/overlay failed: {e}")

#                 t_cursor += seg_dur

#         # 9) Keep the original audio from the main clip
#         audio = (
#             main_audio
#             .filter('aresample', 48000)
#             .filter('aformat', channel_layouts='stereo')
#         )

#         # 10) Output encode
#         ffmpeg.output(
#             composed, audio, output_path,
#             vcodec='h264_nvenc', acodec='aac',
#             audio_bitrate='192k', ar=48000, ac=2,
#             preset='p3', crf=18, r=30,
#             pix_fmt='yuv420p', movflags='+faststart'
#         ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

#         # 11) Cleanup temp overlay renders
#         try:
#             for p in rendered_paths:
#                 try:
#                     if p and os.path.exists(p):
#                         os.remove(p)
#                 except Exception:
#                     pass
#             if tmp_dir:
#                 shutil.rmtree(tmp_dir, ignore_errors=True)
#         except Exception:
#             pass

#         return output_path

#     except ffmpeg.Error as e:
#         print("🔴 FFmpeg STDERR:\n", (e.stderr.decode() if getattr(e, "stderr", None) else ""))
#         print("🔴 FFmpeg STDOUT:\n", (e.stdout.decode() if getattr(e, "stdout", None) else ""))
#         raise

#     except Exception as e:
#         print(f"❌ Error in overlay_avatar_no_audio: {e}")
#         raise

# def overlay_avatar_no_audio(resized_clip, avatar_path, background_path, output_path, tag,
#                             media_pool=None, character_name=None):
#     """
#     Portrait + landscape aware version.
#     Breaking news / TV broadcast style design.
#     """
#     try:
#         import os
#         import logging
#         import shutil
#         import tempfile
#         import ffmpeg

#         global CURRENT_WIDTH, CURRENT_HEIGHT, CURRENT_ORIENTATION

#         canvas_w = CURRENT_WIDTH
#         canvas_h = CURRENT_HEIGHT
#         is_portrait = (CURRENT_ORIENTATION == "portrait")

#         rendered_paths = []
#         tmp_dir = None

#         # ------------------------------------------
#         # 1) Validate background
#         # ------------------------------------------
#         if not background_path or not os.path.exists(background_path):
#             raise FileNotFoundError(f"❌ Background video not found: {background_path}")

#         # ------------------------------------------
#         # 2) Base clip probe
#         # ------------------------------------------
#         probe = ffmpeg.probe(resized_clip)
#         vstream = next(s for s in probe['streams'] if s.get('codec_type') == 'video')
#         duration = float(
#             vstream.get('duration')
#             or probe.get('format', {}).get('duration')
#             or 0.0
#         )
#         if duration <= 0:
#             raise ValueError("❌ Could not determine input video duration.")

#         main_input = ffmpeg.input(resized_clip)
#         main_video = main_input.video
#         main_audio = main_input.audio

#         # ------------------------------------------
#         # 3) Background — looped + darkened
#         # ------------------------------------------
#         bg = (
#             ffmpeg.input(background_path, stream_loop=-1, t=duration)
#             .video
#             .filter('scale', canvas_w, canvas_h, force_original_aspect_ratio='increase')
#             .filter('crop', canvas_w, canvas_h)
#             .filter('colorchannelmixer', rr=0.45, gg=0.45, bb=0.45)  # darker = more contrast
#             .filter('format', 'yuv420p')
#         )

#         # ------------------------------------------
#         # 4) Compute panel size
#         # ------------------------------------------
#         orig_w = int(vstream['width'])
#         orig_h = int(vstream['height'])
#         aspect = orig_w / orig_h

#         if is_portrait:
#             max_w, max_h = int(canvas_w * 0.92), int(canvas_h * 0.55)
#         else:
#             max_w, max_h = int(canvas_w * 0.83), int(canvas_h * 0.83)

#         if aspect > (max_w / max_h):
#             panel_w = max_w
#             panel_h = int(max_w / aspect)
#         else:
#             panel_h = max_h
#             panel_w = int(max_h * aspect)

#         if panel_w % 2: panel_w -= 1
#         if panel_h % 2: panel_h -= 1

#         panel = (
#             main_video
#             .filter('scale', panel_w, panel_h)
#             .filter('format', 'yuv420p')
#         )

#         # ------------------------------------------
#         # 5) FRAME DESIGN
#         #
#         #    Layer structure (inside → outside):
#         #    [content panel]
#         #    [white inner border - 4px]       ← clean inner edge
#         #    [black separator - 3px]          ← depth gap
#         #    [white outer border - 10px]      ← the bold white frame
#         #    [dark shadow ring - 6px]         ← lifts frame off background
#         # ------------------------------------------

#         inner_white   = 4
#         black_sep     = 3
#         outer_white   = 10
#         shadow        = 6

#         framed = (
#             panel
#             # inner white line
#             .filter('pad',
#                     w=panel_w + inner_white * 2,
#                     h=panel_h + inner_white * 2,
#                     x=inner_white, y=inner_white,
#                     color='white')
#             # black separator gap
#             .filter('pad',
#                     w=panel_w + inner_white * 2 + black_sep * 2,
#                     h=panel_h + inner_white * 2 + black_sep * 2,
#                     x=black_sep, y=black_sep,
#                     color='0x111111')
#             # bold outer white border — the main frame
#             .filter('pad',
#                     w=panel_w + inner_white * 2 + black_sep * 2 + outer_white * 2,
#                     h=panel_h + inner_white * 2 + black_sep * 2 + outer_white * 2,
#                     x=outer_white, y=outer_white,
#                     color='white')
#             # dark shadow ring
#             .filter('pad',
#                     w=panel_w + inner_white * 2 + black_sep * 2 + outer_white * 2 + shadow * 2,
#                     h=panel_h + inner_white * 2 + black_sep * 2 + outer_white * 2 + shadow * 2,
#                     x=shadow, y=shadow,
#                     color='0x0A0A0A')
#         )

#         total_border = (inner_white + black_sep + outer_white + shadow) * 2
#         framed_w = panel_w + total_border
#         framed_h = panel_h + total_border

#         # ------------------------------------------
#         # 6) Position panel — centered on canvas
#         # ------------------------------------------
#         pos_x = (canvas_w - framed_w) // 2
#         pos_y = (canvas_h - framed_h) // 2

#         composed = ffmpeg.overlay(bg, framed, x=pos_x, y=pos_y)

#         # ------------------------------------------
#         # 7) COMING UP CHYRON — big, bold, unmissable
#         #
#         #    Layout:
#         #
#         #    ╔═══════════════════════════════════════╗
#         #    ║  [panel content video plays here]     ║
#         #    ╠═══════════════════════════════════════╣  ← top accent line (red, 8px)
#         #    ║                                       ║
#         #    ║   ▌  COMING UP...                     ║  ← dark strip + red flag + big text
#         #    ║                                       ║
#         #    ╚═══════════════════════════════════════╝
#         #
#         # ------------------------------------------
#         if tag == 'intro':

#             # Sizes scale with orientation
#             chyron_h     = 110 if is_portrait else 90   # tall strip = more presence
#             accent_bar_h = 8                             # red line above strip
#             text_size    = 72 if is_portrait else 62     # large, unmissable
#             flag_w       = 14                            # left red tab width
#             text_pad_x   = 28                            # text left padding after flag

#             # Y positions
#             strip_y  = pos_y + framed_h - chyron_h - accent_bar_h
#             accent_y = strip_y
#             # Vertically center text inside the strip
#             text_y   = strip_y + accent_bar_h + ((chyron_h - text_size) // 2) - 4

#             # 1) Full-width dark strip — deep black, strong contrast
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x,
#                 y=strip_y + accent_bar_h,
#                 w=framed_w,
#                 h=chyron_h,
#                 color='0x000000EE',   # near-opaque black
#                 t='fill'
#             )

#             # 2) Red accent bar — sits on top of dark strip
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x,
#                 y=accent_y,
#                 w=framed_w,
#                 h=accent_bar_h,
#                 color='0xFF1A1AFF',   # vivid broadcast red
#                 t='fill'
#             )

#             # 3) Red left flag tab — broadcast identity marker
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x,
#                 y=strip_y + accent_bar_h,
#                 w=flag_w,
#                 h=chyron_h,
#                 color='0xFF1A1AFF',
#                 t='fill'
#             )

#             # 4) White separator line between flag and text
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x + flag_w,
#                 y=strip_y + accent_bar_h,
#                 w=3,
#                 h=chyron_h,
#                 color='0xFFFFFFCC',
#                 t='fill'
#             )

#             # 5) "COMING UP..." — big white bold text with strong shadow
#             composed = composed.filter(
#                 'drawtext',
#                 text='COMING UP...',
#                 fontsize=text_size,
#                 fontcolor='white',
#                 x=pos_x + flag_w + text_pad_x,
#                 y=text_y,
#                 fontfile='C:/Windows/Fonts/arialbd.ttf',
#                 shadowcolor='0x000000CC',
#                 shadowx=3,
#                 shadowy=3,
#                 enable='gte(t,0)'
#             )

#             # 6) Thin white bottom line — closes the chyron cleanly
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x,
#                 y=strip_y + accent_bar_h + chyron_h - 3,
#                 w=framed_w,
#                 h=3,
#                 color='0xFFFFFFAA',
#                 t='fill'
#             )

#         # ------------------------------------------
#         # 8) Audio
#         # ------------------------------------------
#         audio = (
#             main_audio
#             .filter('aresample', 48000)
#             .filter('aformat', channel_layouts='stereo')
#         )

#         # ------------------------------------------
#         # 9) Encode final
#         # ------------------------------------------
#         ffmpeg.output(
#             composed, audio, output_path,
#             vcodec='h264_nvenc', acodec='aac',
#             audio_bitrate='192k', ar=48000, ac=2,
#             preset='p3', crf=18, r=30,
#             pix_fmt='yuv420p', movflags='+faststart'
#         ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

#         # ------------------------------------------
#         # 10) Cleanup
#         # ------------------------------------------
#         try:
#             for p in rendered_paths:
#                 if p and os.path.exists(p):
#                     os.remove(p)
#             if tmp_dir:
#                 shutil.rmtree(tmp_dir, ignore_errors=True)
#         except Exception:
#             pass

#         return output_path

#     except ffmpeg.Error as e:
#         print("🔴 FFmpeg STDERR:\n", e.stderr.decode() if e.stderr else "")
#         raise

#     except Exception as e:
#         print(f"❌ Error in overlay_avatar_no_audio: {e}")
#         raise

# def overlay_avatar_no_audio(resized_clip, avatar_path, background_path, output_path, tag,
#                             media_pool=None, character_name=None):
#     """
#     Portrait + landscape aware version.
#     Breaking news / TV broadcast style design.
#     Falls back to procedural news gradient if no background video is provided.
#     """
#     try:
#         import os
#         import logging
#         import shutil
#         import tempfile
#         import ffmpeg

#         global CURRENT_WIDTH, CURRENT_HEIGHT, CURRENT_ORIENTATION

#         canvas_w = CURRENT_WIDTH
#         canvas_h = CURRENT_HEIGHT
#         is_portrait = (CURRENT_ORIENTATION == "portrait")

#         rendered_paths = []
#         tmp_dir = None

#         # ------------------------------------------
#         # 1) Base clip probe
#         # ------------------------------------------
#         probe = ffmpeg.probe(resized_clip)
#         vstream = next(s for s in probe['streams'] if s.get('codec_type') == 'video')
#         duration = float(
#             vstream.get('duration')
#             or probe.get('format', {}).get('duration')
#             or 0.0
#         )
#         if duration <= 0:
#             raise ValueError("❌ Could not determine input video duration.")

#         main_input = ffmpeg.input(resized_clip)
#         main_video = main_input.video
#         main_audio = main_input.audio

#         # ------------------------------------------
#         # 2) Background — real video or news gradient fallback
#         # ------------------------------------------
#         if background_path and os.path.exists(background_path):
#             bg = (
#                 ffmpeg.input(background_path, stream_loop=-1, t=duration)
#                 .video
#                 .filter('scale', canvas_w, canvas_h, force_original_aspect_ratio='increase')
#                 .filter('crop', canvas_w, canvas_h)
#                 .filter('colorchannelmixer', rr=0.45, gg=0.45, bb=0.45)
#                 .filter('format', 'yuv420p')
#             )
#         else:
#             logging.warning(f"⚠️ Background video not found: {background_path} — using news gradient fallback")
#             # Procedural broadcast news background via lavfi
#             # Deep navy base with diagonal geq gradient + vignette
#             bg = (
#                 ffmpeg.input(
#                     f'color=c=0x0A0F1E:size={canvas_w}x{canvas_h}:rate=30:duration=999',
#                     f='lavfi'
#                 )
#                 .video
#                 .filter('geq',
#                         r='clip(50-(X/W)*30+(Y/H)*10,10,80)',
#                         g='clip(70-(X/W)*20+(Y/H)*15,20,90)',
#                         b='clip(120-(X/W)*30-(Y/H)*20,60,140)')
#                 .filter('vignette', angle=0.5)
#                 .filter('format', 'yuv420p')
#             )

#         # ------------------------------------------
#         # 3) Compute panel size
#         # ------------------------------------------
#         orig_w = int(vstream['width'])
#         orig_h = int(vstream['height'])
#         aspect = orig_w / orig_h

#         if is_portrait:
#             max_w, max_h = int(canvas_w * 0.92), int(canvas_h * 0.55)
#         else:
#             max_w, max_h = int(canvas_w * 0.83), int(canvas_h * 0.83)

#         if aspect > (max_w / max_h):
#             panel_w = max_w
#             panel_h = int(max_w / aspect)
#         else:
#             panel_h = max_h
#             panel_w = int(max_h * aspect)

#         if panel_w % 2: panel_w -= 1
#         if panel_h % 2: panel_h -= 1

#         panel = (
#             main_video
#             .filter('scale', panel_w, panel_h)
#             .filter('format', 'yuv420p')
#         )

#         # ------------------------------------------
#         # 4) FRAME DESIGN
#         #
#         #    Layer structure (inside → outside):
#         #    [content panel]
#         #    [white inner border - 4px]       ← clean inner edge
#         #    [black separator - 3px]          ← depth gap
#         #    [white outer border - 10px]      ← the bold white frame
#         #    [dark shadow ring - 6px]         ← lifts frame off background
#         # ------------------------------------------

#         inner_white = 4
#         black_sep   = 3
#         outer_white = 10
#         shadow      = 6

#         framed = (
#             panel
#             .filter('pad',
#                     w=panel_w + inner_white * 2,
#                     h=panel_h + inner_white * 2,
#                     x=inner_white, y=inner_white,
#                     color='white')
#             .filter('pad',
#                     w=panel_w + inner_white * 2 + black_sep * 2,
#                     h=panel_h + inner_white * 2 + black_sep * 2,
#                     x=black_sep, y=black_sep,
#                     color='0x111111')
#             .filter('pad',
#                     w=panel_w + inner_white * 2 + black_sep * 2 + outer_white * 2,
#                     h=panel_h + inner_white * 2 + black_sep * 2 + outer_white * 2,
#                     x=outer_white, y=outer_white,
#                     color='white')
#             .filter('pad',
#                     w=panel_w + inner_white * 2 + black_sep * 2 + outer_white * 2 + shadow * 2,
#                     h=panel_h + inner_white * 2 + black_sep * 2 + outer_white * 2 + shadow * 2,
#                     x=shadow, y=shadow,
#                     color='0x0A0A0A')
#         )

#         total_border = (inner_white + black_sep + outer_white + shadow) * 2
#         framed_w = panel_w + total_border
#         framed_h = panel_h + total_border

#         # ------------------------------------------
#         # 5) Position panel — centered on canvas
#         # ------------------------------------------
#         pos_x = (canvas_w - framed_w) // 2
#         pos_y = (canvas_h - framed_h) // 2

#         composed = ffmpeg.overlay(bg, framed, x=pos_x, y=pos_y)

#         # ------------------------------------------
#         # 6) COMING UP CHYRON — big, bold, unmissable
#         #
#         #    Layout:
#         #
#         #    ╔═══════════════════════════════════════╗
#         #    ║  [panel content video plays here]     ║
#         #    ╠═══════════════════════════════════════╣  ← top accent line (red, 8px)
#         #    ║                                       ║
#         #    ║   ▌  COMING UP...                     ║  ← dark strip + red flag + big text
#         #    ║                                       ║
#         #    ╚═══════════════════════════════════════╝
#         #
#         # ------------------------------------------
#         if tag == 'intro':

#             chyron_h     = 110 if is_portrait else 90
#             accent_bar_h = 8
#             text_size    = 72 if is_portrait else 62
#             flag_w       = 14
#             text_pad_x   = 28

#             strip_y  = pos_y + framed_h - chyron_h - accent_bar_h
#             accent_y = strip_y
#             text_y   = strip_y + accent_bar_h + ((chyron_h - text_size) // 2) - 4

#             # 1) Full-width dark strip
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x,
#                 y=strip_y + accent_bar_h,
#                 w=framed_w,
#                 h=chyron_h,
#                 color='0x000000EE',
#                 t='fill'
#             )

#             # 2) Red accent bar
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x,
#                 y=accent_y,
#                 w=framed_w,
#                 h=accent_bar_h,
#                 color='0xFF1A1AFF',
#                 t='fill'
#             )

#             # 3) Red left flag tab
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x,
#                 y=strip_y + accent_bar_h,
#                 w=flag_w,
#                 h=chyron_h,
#                 color='0xFF1A1AFF',
#                 t='fill'
#             )

#             # 4) White separator line between flag and text
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x + flag_w,
#                 y=strip_y + accent_bar_h,
#                 w=3,
#                 h=chyron_h,
#                 color='0xFFFFFFCC',
#                 t='fill'
#             )

#             # 5) "COMING UP..." text
#             composed = composed.filter(
#                 'drawtext',
#                 text='COMING UP...',
#                 fontsize=text_size,
#                 fontcolor='white',
#                 x=pos_x + flag_w + text_pad_x,
#                 y=text_y,
#                 fontfile='C:/Windows/Fonts/arialbd.ttf',
#                 shadowcolor='0x000000CC',
#                 shadowx=3,
#                 shadowy=3,
#                 enable='gte(t,0)'
#             )

#             # 6) Thin white bottom line
#             composed = composed.filter(
#                 'drawbox',
#                 x=pos_x,
#                 y=strip_y + accent_bar_h + chyron_h - 3,
#                 w=framed_w,
#                 h=3,
#                 color='0xFFFFFFAA',
#                 t='fill'
#             )

#         # ------------------------------------------
#         # 7) Audio
#         # ------------------------------------------
#         audio = (
#             main_audio
#             .filter('aresample', 48000)
#             .filter('aformat', channel_layouts='stereo')
#         )

#         # ------------------------------------------
#         # 8) Encode final
#         # ------------------------------------------
#         ffmpeg.output(
#             composed, audio, output_path,
#             vcodec='h264_nvenc', acodec='aac',
#             audio_bitrate='192k', ar=48000, ac=2,
#             preset='p3', crf=18, r=30,
#             pix_fmt='yuv420p', movflags='+faststart'
#         ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

#         # ------------------------------------------
#         # 9) Cleanup
#         # ------------------------------------------
#         try:
#             for p in rendered_paths:
#                 if p and os.path.exists(p):
#                     os.remove(p)
#             if tmp_dir:
#                 shutil.rmtree(tmp_dir, ignore_errors=True)
#         except Exception:
#             pass

#         return output_path

#     except ffmpeg.Error as e:
#         print("🔴 FFmpeg STDERR:\n", e.stderr.decode() if e.stderr else "")
#         raise

#     except Exception as e:
#         print(f"❌ Error in overlay_avatar_no_audio: {e}")
#         raise

def overlay_avatar_no_audio(
    resized_clip,
    avatar_path,
    background_path,
    output_path,
    tag,
    media_pool=None,
    character_name=None,
    talking_heads_mode=False,
    teaser_config=None,
):
    try:
        import os
        import logging
        import ffmpeg
        from helper.teaser_backend import normalize_teaser_config, is_intro_tag, hex_to_ffmpeg_color

        global CURRENT_WIDTH, CURRENT_HEIGHT, CURRENT_ORIENTATION

        canvas_w = CURRENT_WIDTH
        canvas_h = CURRENT_HEIGHT
        is_portrait = (CURRENT_ORIENTATION == "portrait")

        cfg = normalize_teaser_config(teaser_config) if teaser_config else None
        is_intro = is_intro_tag(tag)

        probe = ffmpeg.probe(resized_clip)
        vstream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
        duration = float(
            vstream.get("duration")
            or probe.get("format", {}).get("duration")
            or 0.0
        )
        if duration <= 0:
            raise ValueError("❌ Could not determine input video duration.")

        main_input = ffmpeg.input(resized_clip)
        main_video = main_input.video
        main_audio = main_input.audio

        def _apply_content_look(video_stream):
            if not is_intro or not cfg:
                return video_stream

            effect = cfg.get("contentEffect", "none")
            color_mode = cfg.get("colorMode", "color")

            if effect == "grayscale" or color_mode == "grayscale":
                video_stream = video_stream.filter("hue", s=0)
            elif color_mode == "sepia":
                video_stream = video_stream.filter(
                    "colorchannelmixer",
                    rr=0.393, rg=0.769, rb=0.189,
                    gr=0.349, gg=0.686, gb=0.168,
                    br=0.272, bg=0.534, bb=0.131,
                )
            elif effect == "desaturate":
                desat = float(cfg.get("contentDesaturate", 0.65))
                sat = max(0.0, 1.0 - desat)
                video_stream = video_stream.filter("hue", s=sat)
            elif effect == "contrast":
                video_stream = video_stream.filter(
                    "eq", contrast=float(cfg.get("contrast", 1.15))
                )

            if cfg.get("vignette") or effect == "vignette":
                video_stream = video_stream.filter("vignette")

            return video_stream

        def _apply_intro_chyron(composed_stream, pos_x=0, framed_w=None, pos_y=0, framed_h=None):
            if not is_intro:
                return composed_stream

            from helper.teaser_backend import apply_chyron_to_stream

            return apply_chyron_to_stream(
                composed_stream,
                cfg,
                canvas_w,
                canvas_h,
                pos_x=pos_x,
                pos_y=pos_y,
                framed_w=framed_w,
                framed_h=framed_h,
            )

        # ── TALKING HEADS / FULLSCREEN ──────────────────────────
        if talking_heads_mode:
            composed = (
                main_video
                .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="increase")
                .filter("crop", canvas_w, canvas_h)
            )
            composed = _apply_content_look(composed)
            composed = composed.filter("format", "yuv420p")
            composed = _apply_intro_chyron(composed)

            audio = (
                main_audio
                .filter("aresample", 48000)
                .filter("aformat", channel_layouts="stereo")
            )

            ffmpeg.output(
                composed, audio, output_path,
                vcodec="h264_nvenc", acodec="aac",
                audio_bitrate="192k", ar=48000, ac=2,
                preset="p3", crf=18, r=30,
                pix_fmt="yuv420p", movflags="+faststart",
            ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

            logging.info(f"✅ Intro teaser encoded (fullscreen): {output_path}")
            return output_path

        # ── STANDARD PANEL MODE ─────────────────────────────────
        if background_path and os.path.exists(background_path):
            bg = (
                ffmpeg.input(background_path, stream_loop=-1, t=duration)
                .video
                .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="increase")
                .filter("crop", canvas_w, canvas_h)
                .filter("colorchannelmixer", rr=0.45, gg=0.45, bb=0.45)
                .filter("format", "yuv420p")
            )
        else:
            logging.warning(f"⚠️ Background video not found: {background_path} — using navy fallback")
            bg = (
                ffmpeg.input(
                    f"color=c=0x0A0F1E:size={canvas_w}x{canvas_h}:rate=30:duration={round(duration, 4)}",
                    f="lavfi",
                )
                .video
                .filter("format", "yuv420p")
            )

        orig_w = int(vstream["width"])
        orig_h = int(vstream["height"])
        aspect = orig_w / orig_h

        if is_portrait:
            max_w, max_h = int(canvas_w * 0.92), int(canvas_h * 0.55)
        else:
            max_w, max_h = int(canvas_w * 0.83), int(canvas_h * 0.83)

        if aspect > (max_w / max_h):
            panel_w = max_w
            panel_h = int(max_w / aspect)
        else:
            panel_h = max_h
            panel_w = int(max_h * aspect)

        if panel_w % 2:
            panel_w -= 1
        if panel_h % 2:
            panel_h -= 1

        panel = main_video.filter("scale", panel_w, panel_h)
        panel = _apply_content_look(panel)
        panel = panel.filter("format", "yuv420p")

        inner_white, black_sep, outer_white, shadow = 4, 3, 10, 6
        framed = (
            panel
            .filter("pad", w=panel_w + inner_white * 2, h=panel_h + inner_white * 2, x=inner_white, y=inner_white, color="white")
            .filter("pad", w=panel_w + inner_white * 2 + black_sep * 2, h=panel_h + inner_white * 2 + black_sep * 2, x=black_sep, y=black_sep, color="0x111111")
            .filter("pad", w=panel_w + inner_white * 2 + black_sep * 2 + outer_white * 2, h=panel_h + inner_white * 2 + black_sep * 2 + outer_white * 2, x=outer_white, y=outer_white, color="white")
            .filter("pad", w=panel_w + inner_white * 2 + black_sep * 2 + outer_white * 2 + shadow * 2, h=panel_h + inner_white * 2 + black_sep * 2 + outer_white * 2 + shadow * 2, x=shadow, y=shadow, color="0x0A0A0A")
        )

        total_border = (inner_white + black_sep + outer_white + shadow) * 2
        framed_w = panel_w + total_border
        framed_h = panel_h + total_border
        pos_x = (canvas_w - framed_w) // 2
        pos_y = (canvas_h - framed_h) // 2

        composed = ffmpeg.overlay(bg, framed, x=pos_x, y=pos_y)
        composed = _apply_intro_chyron(composed, pos_x=pos_x, framed_w=framed_w, pos_y=pos_y, framed_h=framed_h)

        audio = (
            main_audio
            .filter("aresample", 48000)
            .filter("aformat", channel_layouts="stereo")
        )

        ffmpeg.output(
            composed, audio, output_path,
            vcodec="h264_nvenc", acodec="aac",
            audio_bitrate="192k", ar=48000, ac=2,
            preset="p3", crf=18, r=30,
            pix_fmt="yuv420p", movflags="+faststart",
        ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

        return output_path

    except ffmpeg.Error as e:
        print("🔴 FFmpeg STDERR:\n", e.stderr.decode() if e.stderr else "")
        raise
    except Exception as e:
        print(f"❌ Error in overlay_avatar_no_audio: {e}")
        raise

import os
import ffmpeg


def overlay_avatar_with_audio(
    resized_clip,
    avatar_path,
    audio_path,
    output_path,
    avatar_active,
    background_path=None,
    **_
):
    try:
        # ------------------------------------------
        # GLOBAL ORIENTATION + RESOLUTION
        # ------------------------------------------
        global CURRENT_WIDTH, CURRENT_HEIGHT, CURRENT_ORIENTATION

        canvas_w = CURRENT_WIDTH        # 1920 or 1080
        canvas_h = CURRENT_HEIGHT       # 1080 or 1920
        is_portrait = (CURRENT_ORIENTATION == "portrait")

        # ------------------------------------------
        # 0) Validate inputs
        # ------------------------------------------
        if not resized_clip or not os.path.exists(resized_clip):
            raise FileNotFoundError(f"❌ Main video not found: {resized_clip}")
        if avatar_active and (not avatar_path or not os.path.exists(avatar_path)):
            raise FileNotFoundError(f"❌ Avatar image not found: {avatar_path}")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ Audio not found: {audio_path}")

        # ------------------------------------------
        # 1) Probe audio duration
        # ------------------------------------------
        audio_probe = ffmpeg.probe(audio_path)
        astreams = [s for s in audio_probe['streams'] if s.get('codec_type') == 'audio']

        if not astreams:
            raise ValueError("No audio stream found")

        audio_duration = float(astreams[0].get('duration', 0.0))
        if audio_duration <= 0:
            raise ValueError("Invalid audio duration")

        # ------------------------------------------
        # 2) MAIN VIDEO — SCALE/CROP TO CANVAS SIZE
        # ------------------------------------------
        main = (
            ffmpeg
            .input(resized_clip, stream_loop=-1)
            .video
            .filter('trim', duration=audio_duration)
            .filter('setpts', 'PTS-STARTPTS')
            .filter('scale', canvas_w, canvas_h, force_original_aspect_ratio='increase')
            .filter('crop', canvas_w, canvas_h)
            .filter('format', 'yuv420p')
        )

        # ------------------------------------------
        # 3) Audio -> stereo 48k
        # ------------------------------------------
        narration_audio = (
            ffmpeg.input(audio_path)
            .audio
            .filter('aresample', 48000)
            .filter(
                'aformat',
                sample_fmts='fltp',
                sample_rates='48000',
                channel_layouts='stereo'
            )
        )

        # ------------------------------------------
        # 4) CONDITIONAL AVATAR OVERLAY
        # ------------------------------------------
        if avatar_active:
            # ------------------------------------------
            # AVATAR PANEL SIZE (different for portrait)
            # ------------------------------------------
            if is_portrait:
                avatar_w, avatar_h = int(canvas_w * 0.42), int(canvas_h * 0.18)
            else:
                avatar_w, avatar_h = 420, 470

            border_size = 6
            margin = 24

            # ------------------------------------------
            # 5) Avatar image scaling
            # ------------------------------------------
            avatar_scaled = (
                ffmpeg.input(avatar_path)
                .video
                .filter('scale', avatar_w, avatar_h, force_original_aspect_ratio='decrease')
                .filter('pad', avatar_w, avatar_h, '(ow-iw)/2', '(oh-ih)/2', color='black')
                .filter('format', 'yuv420p')
            )

            # ------------------------------------------
            # 6) Waveform (matches avatar size)
            # ------------------------------------------
            waveform = (
                ffmpeg.input(audio_path)
                .filter('aformat', channel_layouts='mono')
                .filter('showwaves', s=f'{avatar_w}x{avatar_h}', mode='line', n='2', colors='ffffff')
                .filter('format', 'yuva420p')
            )

            avatar_with_waveform = ffmpeg.overlay(avatar_scaled, waveform, format='auto')

            # ------------------------------------------
            # 7) Add border + separator
            # ------------------------------------------
            pad_w = avatar_w + 2 * border_size
            pad_h = avatar_h + 2 * border_size
            sep_x = avatar_w + border_size - 1

            avatar_bordered = (
                avatar_with_waveform
                .filter('pad', w=pad_w, h=pad_h, x=border_size, y=border_size, color='white')
                .filter('drawbox', x=sep_x, y=border_size, w=2, h=avatar_h, color='white', t='fill')
            )

            # ------------------------------------------
            # 8) AVATAR POSITIONING
            # ------------------------------------------
            if is_portrait:
                # Portrait → place avatar centered at bottom
                pos_x = f'(W-w)/2'
                pos_y = f'H-h-{margin}'
            else:
                # Landscape → classic bottom-left
                pos_x = margin
                pos_y = f'H-h-{margin}'

            # ------------------------------------------
            # 9) Compose final video with avatar
            # ------------------------------------------
            composed = ffmpeg.overlay(
                main,
                avatar_bordered.filter('format', 'yuv420p'),
                x=pos_x,
                y=pos_y,
                eof_action='pass'
            )
        else:
            # No avatar - use main video as-is
            composed = main

        # ------------------------------------------
        # 10) Encode final
        # ------------------------------------------
        (
            ffmpeg
            .output(
                composed,
                narration_audio,
                output_path,
                vcodec='h264_nvenc',
                acodec='aac',
                audio_bitrate='192k',
                ar=48000,
                ac=2,
                rc='vbr',
                cq=19,
                r=30,
                pix_fmt='yuv420p',
                movflags='+faststart'
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        return output_path

    except ffmpeg.Error as e:
        try:
            err = e.stderr.decode()
        except Exception:
            err = str(e)
        print("🔴 FFmpeg STDERR:\n", err)
        raise

    except Exception as e:
        print(f"❌ Error in overlay_avatar_with_audio: {e}")
        raise


# MUSE_PY = r"C:\Miniconda3\envs\musetalk\python.exe"
# MUSE_ROOT = r"D:\MuseTalk"

# CODEFORMER_PY = r"C:\Miniconda3\envs\codeformer\python.exe"
# CODEFORMER_ROOT = r"D:\CodeFormer"

# import os, time, uuid, shutil, subprocess, traceback, gc, logging
# import yaml
# import ffmpeg
# import torch

# def overlay_avatar_with_audio(
#     resized_clip,
#     avatar_path,
#     audio_path,
#     output_path,
#     avatar_active,
#     bbox_shift=0
# ):
#     """
#     MuseTalk avatar pipeline with CodeFormer enhancement.
#     """

#     # -------------------------------
#     # CONFIG: YOUR KNOWN PATHS
#     # -------------------------------
#     MUSE_PY = r"C:\Miniconda3\envs\musetalk\python.exe"
#     MUSE_ROOT = r"D:\MuseTalk"
#     MUSE_UNET = r"D:\MuseTalk\models\musetalkV15\unet.pth"
#     MUSE_JSON = r"D:\MuseTalk\models\musetalkV15\musetalk.json"
#     FFMPEG_BIN = r"C:\ffmpeg\bin"

#     CODEFORMER_PY = r"C:\Miniconda3\envs\codeformer\python.exe"
#     CODEFORMER_ROOT = r"D:\CodeFormer"
#     CODEFORMER_RESULTS = os.path.join(CODEFORMER_ROOT, "results")

#     try:
#         # -------------------------------
#         # VALIDATION
#         # -------------------------------
#         if not avatar_active:
#             raise RuntimeError("overlay_avatar_with_audio called while avatar_active=False")

#         if not audio_path or not os.path.exists(audio_path):
#             raise RuntimeError(f"Audio not found: {audio_path}")

#         if not avatar_path or not os.path.exists(avatar_path):
#             raise RuntimeError(f"Avatar video not found: {avatar_path}")

#         # Absolute paths
#         avatar_path = os.path.abspath(avatar_path)
#         audio_path = os.path.abspath(audio_path)
#         output_path = os.path.abspath(output_path)

#         work_dir = os.path.dirname(output_path)
#         os.makedirs(work_dir, exist_ok=True)

#         uid = str(uuid.uuid4())[:10]
#         yaml_path = os.path.join(work_dir, f"musetalk_{uid}.yaml")
#         musetalk_out_dir = os.path.join(work_dir, f"musetalk_out_{uid}")
#         os.makedirs(musetalk_out_dir, exist_ok=True)

#         logging.info("🎭 Avatar pipeline starting")
#         logging.info(f"   Avatar: {os.path.basename(avatar_path)}")
#         logging.info(f"   Audio: {os.path.basename(audio_path)}")
#         logging.info(f"   BBox shift: {bbox_shift}")

#         # Validate input files are not corrupted
#         avatar_size = os.path.getsize(avatar_path)
#         audio_size = os.path.getsize(audio_path)
        
#         logging.info(f"   Avatar size: {avatar_size / 1024:.1f} KB")
#         logging.info(f"   Audio size: {audio_size / 1024:.1f} KB")
        
#         if avatar_size < 1000:
#             raise RuntimeError(f"Avatar file too small: {avatar_size} bytes")
#         if audio_size < 1000:
#             raise RuntimeError(f"Audio file too small: {audio_size} bytes")

#         # -------------------------------
#         # 1) WRITE MUSETALK YAML
#         # -------------------------------
#         musetalk_yaml = {
#             "task_0": {
#                 "video_path": avatar_path,
#                 "audio_path": audio_path,
#                 "bbox_shift": int(bbox_shift)
#             }
#         }

#         with open(yaml_path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(musetalk_yaml, f, sort_keys=False)

#         if not os.path.exists(yaml_path):
#             raise RuntimeError(f"MuseTalk YAML not created: {yaml_path}")

#         logging.info(f"✅ Config created: {yaml_path}")
        
#         # Log the config content
#         with open(yaml_path, "r") as f:
#             yaml_content = f.read()
#             logging.info(f"   YAML content:\n{yaml_content}")

#         # -------------------------------
#         # 2) RUN MUSETALK WITH FULL OUTPUT CAPTURE
#         # -------------------------------
#         logging.info("🎬 Running MuseTalk...")
#         logging.info(f"   Command: {MUSE_PY} -m scripts.inference")
#         logging.info(f"   Working dir: {MUSE_ROOT}")
#         logging.info(f"   Output dir: {musetalk_out_dir}")

#         muse_cmd = [
#             MUSE_PY,
#             "-m", "scripts.inference",
#             "--inference_config", yaml_path,
#             "--result_dir", musetalk_out_dir,
#             "--unet_model_path", MUSE_UNET,
#             "--unet_config", MUSE_JSON,
#             "--version", "v15",
#             "--ffmpeg_path", FFMPEG_BIN
#         ]

#         start_time = time.time()
        
#         # Run WITHOUT check=True first to capture output even on failure
#         muse_result = subprocess.run(
#             muse_cmd,
#             cwd=MUSE_ROOT,
#             capture_output=True,
#             text=True,
#             timeout=600
#         )
        
#         elapsed_time = time.time() - start_time
        
#         logging.info(f"✅ MuseTalk process completed in {elapsed_time:.1f}s")
#         logging.info(f"   Return code: {muse_result.returncode}")
        
#         # 🔥 LOG THE FULL OUTPUT
#         if muse_result.stdout:
#             logging.info("=" * 60)
#             logging.info("MUSETALK STDOUT:")
#             logging.info("=" * 60)
#             logging.info(muse_result.stdout)
#             logging.info("=" * 60)
        
#         if muse_result.stderr:
#             logging.warning("=" * 60)
#             logging.warning("MUSETALK STDERR:")
#             logging.warning("=" * 60)
#             logging.warning(muse_result.stderr)
#             logging.warning("=" * 60)
        
#         # Check if MuseTalk actually failed
#         if muse_result.returncode != 0:
#             raise RuntimeError(f"MuseTalk failed with return code {muse_result.returncode}")
        
#         # Warn if completed too quickly (likely failed silently)
#         if elapsed_time < 30:
#             logging.warning(f"⚠️ MuseTalk completed suspiciously fast ({elapsed_time:.1f}s)")
#             logging.warning("   This usually means it failed silently. Check the logs above.")

#         # -------------------------------
#         # 3) FIND MUSETALK OUTPUT
#         # -------------------------------
#         logging.info("🔍 Locating MuseTalk output...")
#         logging.info(f"   Searching in: {musetalk_out_dir}")
        
#         # Give filesystem a moment to sync
#         time.sleep(1)
        
#         all_videos = []
#         all_files = []
        
#         for root, dirs, files in os.walk(musetalk_out_dir):
#             for file in files:
#                 full_path = os.path.join(root, file)
#                 file_size = os.path.getsize(full_path)
#                 rel_path = os.path.relpath(full_path, musetalk_out_dir)
                
#                 all_files.append({
#                     'path': full_path,
#                     'name': file,
#                     'size': file_size,
#                     'rel_path': rel_path
#                 })
                
#                 if file.lower().endswith((".mp4", ".avi", ".mov")):
#                     if file_size > 10_000:
#                         all_videos.append({
#                             'path': full_path,
#                             'size': file_size,
#                             'mtime': os.path.getmtime(full_path),
#                             'rel_path': rel_path
#                         })
#                         logging.info(f"   ✅ Found video: {rel_path} ({file_size / 1024:.1f} KB)")
#                     else:
#                         logging.warning(f"   ⚠️ Found video but too small: {rel_path} ({file_size} bytes)")

#         # Debug: Show ALL files created
#         if all_files:
#             logging.info(f"   Total files created: {len(all_files)}")
#             for f in all_files:
#                 logging.info(f"      📄 {f['rel_path']} ({f['size']} bytes)")
#         else:
#             logging.error(f"   ❌ No files created at all in output directory!")

#         if not all_videos:
#             logging.error(f"❌ No valid .mp4 files found")
#             logging.error(f"   MuseTalk likely failed to generate output")
#             logging.error(f"   Check the STDOUT/STDERR logs above for errors")
#             logging.error(f"")
#             logging.error(f"   Common issues:")
#             logging.error(f"   1. Avatar video format not supported")
#             logging.error(f"   2. Audio format not supported (need WAV PCM 16kHz)")
#             logging.error(f"   3. FFmpeg not found or not working")
#             logging.error(f"   4. CUDA/GPU memory issue")
#             logging.error(f"   5. Model files not loaded correctly")
            
#             raise RuntimeError(f"MuseTalk produced no valid mp4 output")

#         # Sort by modification time (newest first)
#         all_videos.sort(key=lambda x: x['mtime'], reverse=True)
        
#         musetalk_video = all_videos[0]['path']
#         muse_size = all_videos[0]['size']
        
#         logging.info(f"✅ Selected MuseTalk output: {all_videos[0]['rel_path']} ({muse_size / 1024:.1f} KB)")
        
#         if len(all_videos) > 1:
#             logging.info(f"   (Found {len(all_videos)} videos total, using newest)")

#         # -------------------------------
#         # 4) RUN CODEFORMER
#         # -------------------------------
#         logging.info("✨ Running CodeFormer enhancement...")

#         cf_start_time = time.time()

#         cf_cmd = [
#             CODEFORMER_PY,
#             "inference_codeformer.py",
#             "--bg_upsampler", "realesrgan",
#             "--face_upsample",
#             "-w", "1.0",
#             "--input_path", musetalk_video
#         ]

#         cf_result = subprocess.run(
#             cf_cmd,
#             cwd=CODEFORMER_ROOT,
#             capture_output=True,
#             text=True,
#             timeout=600
#         )
        
#         cf_elapsed = time.time() - cf_start_time
        
#         logging.info(f"✅ CodeFormer completed in {cf_elapsed:.1f}s")
#         logging.info(f"   Return code: {cf_result.returncode}")
        
#         if cf_result.returncode != 0:
#             logging.warning(f"⚠️ CodeFormer failed, using unenhanced video")
#             logging.warning(f"   STDERR: {cf_result.stderr[:500]}")
#             codeformer_out = musetalk_video
#         else:
#             # ... [rest of CodeFormer output finding logic] ...
            
#             muse_basename = os.path.basename(musetalk_video)
#             muse_name_no_ext = os.path.splitext(muse_basename)[0]

#             expected_folder = os.path.join(CODEFORMER_RESULTS, muse_name_no_ext)
#             expected_final = os.path.join(expected_folder, "final_results")
#             expected_video = os.path.join(expected_final, muse_basename)
            
#             if os.path.exists(expected_video) and os.path.getsize(expected_video) > 10_000:
#                 codeformer_out = expected_video
#                 logging.info(f"✅ CodeFormer output: {os.path.relpath(codeformer_out, CODEFORMER_RESULTS)}")
#             else:
#                 logging.warning(f"⚠️ CodeFormer output not found at: {expected_video}")
#                 logging.warning(f"   Using unenhanced video")
#                 codeformer_out = musetalk_video

#         # -------------------------------
#         # 6) FINAL MUX
#         # -------------------------------
#         logging.info("🎵 Final mux with audio...")

#         try:
#             audio_probe = ffmpeg.probe(audio_path)
#             audio_streams = [s for s in audio_probe["streams"] if s.get("codec_type") == "audio"]
#             audio_duration = float(audio_streams[0].get("duration", 0) or 0) if audio_streams else None
#         except:
#             audio_duration = None

#         ffmpeg_cmd = [
#             "ffmpeg", "-y",
#             "-i", codeformer_out,
#             "-i", audio_path,
#             "-map", "0:v:0",
#             "-map", "1:a:0",
#             "-c:v", "h264_nvenc" if torch.cuda.is_available() else "libx264",
#             "-preset", "p5" if torch.cuda.is_available() else "medium",
#             "-pix_fmt", "yuv420p",
#             "-c:a", "aac",
#             "-b:a", "192k",
#             "-ar", "48000",
#             "-movflags", "+faststart",
#             "-shortest"
#         ]

#         if audio_duration and audio_duration > 0:
#             ffmpeg_cmd.extend(["-t", str(audio_duration)])

#         ffmpeg_cmd.append(output_path)

#         subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=120)

#         final_size = os.path.getsize(output_path)
#         logging.info(f"✅ Final output: {os.path.basename(output_path)} ({final_size / (1024*1024):.2f} MB)")

#         # -------------------------------
#         # 7) CLEANUP
#         # -------------------------------
#         try:
#             if os.path.exists(yaml_path):
#                 os.remove(yaml_path)
#             if os.path.exists(musetalk_out_dir):
#                 shutil.rmtree(musetalk_out_dir, ignore_errors=True)
#         except:
#             pass

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         return output_path

#     except Exception as e:
#         logging.error(f"❌ Avatar pipeline error: {e}")
#         logging.error(traceback.format_exc())
#         raise


# def encode_video_segment(segment_info):
#     segment, media_pool, segment_audio_path, section_dir, chunk_folder, i, used_media_dict, media_lock, mode, avatar, background_video,segments,script_context = segment_info
#     chunk_filename = f"chunk_{i}_{uuid.uuid4()}.mp4"
#     chunk_path = os.path.join(chunk_folder, chunk_filename)
#     print(f"[{i}] Audio path: {segment_audio_path}")
#     prev_segments = get_prev_two_context_text(segments, i)
#     print("These are the 2 previous segments ", prev_segments)

#     logging.info(f"🔹 Processing segment {i}: {segment['text'][:50]}...")

#     try:
#         character_name = segment["character"]
#         segment_tag = segment.get("tag")

#         # --- Commentary: With narration (tagged OR fallback to find_best_media) ---
#         if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is not None:
#             segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#             duration_needed = segment_audio.duration
            
#             segment_audio = segment_audio.set_fps(44100).volumex(2)

#             if segment_tag:
#                 tagged_media = [m for m in media_pool if m["character"] == character_name and m.get("tag") == segment_tag]
#                 if tagged_media:
#                     selected = tagged_media[0]
#                     if selected["type"] == "video":
#                         overlay_avatar_with_audio(
#                             selected["path"],
#                             f"channel_media/{avatar}",
#                             # f"channel_media/{background_video}",
#                             segment_audio_path,
#                             chunk_path
#                         )
#                         return chunk_path

#             # Fallback to find_best_media + avatar overlay
#             logging.info(f"🔁 No tagged video found or no tag — using best media fallback for segment {i}")
#             reset_overlay_usage_for_new_clip()

#             selected_media = find_best_media_combination(
#                 media_pool, segment, character_name, duration_needed, used_media_dict, media_lock,mode,i,script_context,prev_segments
#             )

#             if not selected_media:
#                 logging.error(f"❌ No fallback media found for commentary segment {i}")
#                 return None

#             temp_clip_path = os.path.join(chunk_folder, f"temp_clip_{i}_{uuid.uuid4()}.mp4")
#             segment_clips = []
#             for item in selected_media:
#                 media = item["media"]
#                 dur = item["duration"]
#                 try:
#                     if media["type"] == "video":
#                         clip = VideoFileClip(media["path"]).subclip(0, min(dur, media["duration"])).resize((1920, 1080))
#                         if media.get("mute", False):
#                             clip = clip.without_audio()
#                     else:
#                         clip = create_blurred_background_image(media["path"], dur,mode)
#                     clip = clip.set_duration(dur)
#                     segment_clips.append(clip)
#                 except Exception as e:
#                     logging.error(f"⚠️ Error processing fallback media {media['path']}: {e}")
#                     continue

#             final_clip = concatenate_videoclips(segment_clips, method="compose")
#             final_clip.set_audio(segment_audio.set_duration(final_clip.duration))

#             final_clip.write_videofile(
#                 temp_clip_path,
#                 codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                 audio_codec="aac",
#                 audio_fps=44100,
#                 threads=4,
#                 fps=30,
#                 preset="p3",
#                 ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart", "-shortest"]
#             )
            


#             final_overlayed_path = overlay_avatar_with_audio(
#                 resized_clip=temp_clip_path,
#                 avatar_path=f"channel_media/{avatar}",
#                 # background_path=f"channel_media/{background_video}",
#                 audio_path=segment_audio_path,
#                 output_path=chunk_path
#             )

#             os.remove(temp_clip_path)
#             return final_overlayed_path

#         # --- Commentary: Silent overlay ---
#         if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is None and segment_tag:
#             tagged_media = [m for m in media_pool if m["character"] == character_name and m.get("tag") == segment_tag]
#             if tagged_media:
#                 selected = tagged_media[0]
#                 if selected["type"] == "video":
#                     overlay_avatar_no_audio(
#     selected["path"],
#     f"channel_media/{avatar}",
#     f"channel_media/{background_video}",
#     chunk_path,
#     segment_tag,
#     media_pool=media_pool,
#     character_name=character_name
# )


#                     return chunk_path

#         # --- Non-commentary mode with tag but no audio ---
#         if segment_audio_path is None and segment_tag:
#             tagged_media = [m for m in media_pool if m["character"] == character_name and m.get("tag") == segment_tag]
#             if tagged_media:
#                 selected = tagged_media[0]
#                 if selected["type"] == "video":
#                     full_video = VideoFileClip(selected["path"])
#                     duration = full_video.duration
#                     audio = full_video.audio.set_fps(44100)

#                     if duration <= 10:
#                         logging.info(f"⏱️ Duration {duration:.2f}s ≤ 10s — encoding directly without overlay.")
#                         final_clip = full_video.set_audio(audio.set_duration(duration))
#                         final_clip.write_videofile(
#                             chunk_path,
#                             codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                             audio_codec="aac",
#                             audio_fps=44100,
#                             threads=4,
#                             fps=30,
#                             ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "p5", "-movflags", "faststart", "-shortest"]
#                         )
                        


#                         full_video.close()
#                         return chunk_path

#                     try:
#                         reset_overlay_usage_for_new_clip()
#                         segment_plan = split_segments(duration)
#                         logging.info(f"📊 Segment plan: {segment_plan}")
#                     except RuntimeError as e:
#                         logging.error(str(e))
#                         return None

#                     t = 0
#                     final_parts = []
#                     for seg_type, seg_duration in segment_plan:
#                         if seg_type == "video":
#                             clip = full_video.subclip(t, t + seg_duration)
#                             final_parts.append(clip)
#                         else:
#                             try:
#                                 overlay = get_next_overlay_image(media_pool, character_name)
#                             except RuntimeError as e:
#                                 logging.error(f"⚠️ Could not get new overlay image: {str(e)}")
#                                 continue

#                             overlay_img = create_zoomed_overlay_with_blur_background(
#                                 overlay["path"], seg_duration, target_size=full_video.size
#                             ).set_audio(audio.subclip(t, t + seg_duration).set_fps(44100).volumex(1.5))

#                             final_parts.append(overlay_img)
#                         t += seg_duration

#                     final_clip = concatenate_videoclips(final_parts, method="compose")
#                     final_clip = final_clip.set_audio(audio.set_duration(final_clip.duration).set_fps(44100).volumex(1.5))

#                     final_clip.write_videofile(
#                         chunk_path,
#                         codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                         audio_codec="aac",
#                         audio_fps=44100,
#                         threads=4,
#                         fps=30,
#                         ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "p5", "-movflags", "faststart", "-shortest"]
#                     )
                    


#                     full_video.close()
#                     return chunk_path

#         # --- Default: Normal audio-based segment ---
#         segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#         duration_needed = segment_audio.duration


#         segment_audio = segment_audio.set_fps(44100).volumex(2)

#         selected_media = find_best_media_combination(
#             media_pool, segment, character_name, duration_needed, used_media_dict, media_lock,mode,i,script_context
#         )

#         if not selected_media:
#             logging.error(f"❌ No media selected for segment {i}")
#             return None

#         segment_clips = []
#         for item in selected_media:
#             media = item["media"]
#             dur = item["duration"]
#             try:
#                 if media["type"] == "video":
#                     clip = VideoFileClip(media["path"]).subclip(0, min(dur, media["duration"]))
#                     clip = clip.resize((1920, 1080)).set_duration(dur)
#                     if media.get("mute", False):
#                         clip = clip.without_audio()
#                     logging.info(f"🎥 Video: {media['path']} for {dur:.2f}s")
#                 else:
#                     clip = create_blurred_background_image(media["path"], dur,mode)
#                     logging.info(f"🖼️ Image: {media['path']} for {dur:.2f}s")
#                 segment_clips.append(clip)
#             except Exception as e:
#                 logging.error(f"⚠️ Failed to process {media['path']}: {str(e)}")
#                 continue

#         final_clip = concatenate_videoclips(segment_clips, method="compose")
#         final_clip = final_clip.set_audio(segment_audio.set_duration(final_clip.duration)).set_duration(final_clip.duration)

#         final_clip.write_videofile(
#             chunk_path,
#             codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#             audio_codec="aac",
#             audio_fps=44100,
#             threads=4,
#             fps=30,
#             preset="p3",
#             ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart", "-shortest"]
#         )
        
#         return chunk_path

#     except Exception as e:
#         logging.error(f"❌ Segment {i} failed: {str(e)}", exc_info=True)
#         return None

#     finally:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()

# def encode_video_segment(segment_info):
#     (
#         segment,
#         media_pool,
#         segment_audio_path,
#         section_dir,
#         chunk_folder,
#         i,
#         used_media_dict,
#         media_lock,
#         mode,
#         avatar,
#         background_video,
#         segments,
#         script_context,

#         # 🔥 NEW — values from main process
#         orientation,
#         storyboard,
#         kenburns,
#         avatar_active
#     ) = segment_info

#     # ---------------------------------------------------------
#     # 🔥 APPLY SETTINGS INSIDE WORKER (GLOBAL OVERRIDE HERE)
#     # ---------------------------------------------------------
#     global CURRENT_WIDTH, CURRENT_HEIGHT, CURRENT_ORIENTATION
#     global CURRENT_STORYBOARD_ENABLED, CURRENT_KENBURNS_ENABLED

#     CURRENT_ORIENTATION = orientation
#     CURRENT_STORYBOARD_ENABLED = bool(storyboard)
#     CURRENT_KENBURNS_ENABLED = bool(kenburns)

#     # Apply resolution based on orientation
#     if orientation == "portrait":
#         CURRENT_WIDTH, CURRENT_HEIGHT = 1080, 1920
#     else:
#         CURRENT_WIDTH, CURRENT_HEIGHT = 1920, 1080

#     canvas_w = CURRENT_WIDTH
#     canvas_h = CURRENT_HEIGHT

#     # ---------------------------------------------------------
#     # Existing logic continues from here
#     # ---------------------------------------------------------

#     chunk_filename = f"chunk_{i}_{uuid.uuid4()}.mp4"
#     chunk_path = os.path.join(chunk_folder, chunk_filename)

#     print(f"[{i}] Audio path: {segment_audio_path}")
#     prev_segments = get_prev_two_context_text(segments, i)
#     print("These are the 2 previous segments ", prev_segments)

#     logging.info(f"🔹 Processing segment {i}: {segment['text'][:50]}...")

#     try:
#         character_name = segment["character"]
#         segment_tag = segment.get("tag")

#         if character_name.lower() == "text":
#             logging.info(f"📝 Text-only segment {i} detected")
            
#             # Get duration from audio if available
#             if segment_audio_path and os.path.exists(segment_audio_path):
#                 segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#                 duration_needed = segment_audio.duration
#                 segment_audio = segment_audio.set_fps(44100).volumex(2)
#             else:
#                 # Estimate duration based on text length (reading speed)
#                 words = len(segment['text'].split())
#                 duration_needed = max(3.0, words * 0.35)  # ~170 words per minute
#                 segment_audio = None
            
#             # Get custom styling from segment (with defaults)
#             bg_color = segment.get("text_bg_color", (178, 34, 34))  # Red default
#             text_color = segment.get("text_color", "white")
#             fontsize = segment.get("text_fontsize", None)  # Auto-calculated if None
#             fade_duration = segment.get("text_fade_duration", 0.8)
            
#             # Create the text segment with lines
#             text_clip = create_text_segment_with_lines(
#                 text=segment['text'],
#                 duration=duration_needed,
#                 canvas_w=canvas_w,
#                 canvas_h=canvas_h,
#                 bg_color=bg_color,
#                 text_color=text_color,
#                 fontsize=fontsize,
#                 fade_duration=fade_duration
#             )
            
#             # Add audio if available
#             if segment_audio:
#                 text_clip = text_clip.set_audio(segment_audio)
            
#             # Write the video
#             text_clip.write_videofile(
#                 chunk_path,
#                 codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                 audio_codec="aac" if segment_audio else None,
#                 audio_fps=44100 if segment_audio else None,
#                 threads=4,
#                 fps=30,
#                 preset="p3",
#                 ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"]
#             )
            
#             # Cleanup
#             text_clip.close()
#             if segment_audio:
#                 segment_audio.close()
            
#             logging.info(f"✅ Text segment {i} completed: {chunk_path}")
#             return chunk_path


#         # ---------------------------------------------------------
#         # 🟦 Mode: Commentary WITH narration
#         # ---------------------------------------------------------
#         if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is not None:

#             segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#             duration_needed = segment_audio.duration
#             segment_audio = segment_audio.set_fps(44100).volumex(2)

#             # 1) TAGGED CASE
#             if segment_tag:
#                 tagged_media = [
#                     m for m in media_pool
#                     if m["character"] == character_name and m.get("tag") == segment_tag
#                 ]
#                 if tagged_media:
#                     selected = tagged_media[0]
#                     if selected["type"] == "video":
#                         overlay_avatar_with_audio(
#                             resized_clip=selected["path"],
#                             avatar_path=f"channel_media/{avatar}",
#                             audio_path=segment_audio_path,
#                             output_path=chunk_path,
#                             avatar_active=avatar_active
#                         )
#                         return chunk_path

#             # 2) FALLBACK: BEST MEDIA
#             logging.info(f"🔁 No tag — running best-media fallback for segment {i}")
#             reset_overlay_usage_for_new_clip()

#             selected_media = find_best_media_combination(
#                 media_pool, segment, character_name, duration_needed,
#                 used_media_dict, media_lock, mode, i,
#                 script_context, prev_segments
#             )

#             if not selected_media:
#                 logging.error(f"❌ No fallback media for segment {i}")
#                 return None

#             temp_clip_path = os.path.join(
#                 chunk_folder,
#                 f"temp_clip_{i}_{uuid.uuid4()}.mp4"
#             )

#             segment_clips = []

#             # Build fallback clip list
#             for item in selected_media:
#                 media = item["media"]
#                 dur = item["duration"]

#                 try:
#                     if media["type"] == "video":
#                         clip = (
#                             VideoFileClip(media["path"])
#                             .subclip(0, min(dur, media["duration"]))
#                             .resize((canvas_w, canvas_h))
#                         )
#                         if media.get("mute", False):
#                             clip = clip.without_audio()
#                     else:
#                         clip = create_blurred_background_image(
#                             media["path"], dur, mode
#                         )

#                     clip = clip.set_duration(dur)
#                     segment_clips.append(clip)

#                 except Exception as e:
#                     logging.error(f"⚠️ Fallback media failed: {media['path']} | {e}")

#             if CURRENT_STORYBOARD_ENABLED and len(segment_clips) > 1:
#                 final_clip = apply_storyboard_transitions(segment_clips)
#             else:
#                 final_clip = concatenate_videoclips(segment_clips, method="compose")
#             final_clip = final_clip.set_audio(segment_audio.set_duration(final_clip.duration))

#             # Write fallback clip
#             final_clip.write_videofile(
#                 temp_clip_path,
#                 codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                 audio_codec="aac",
#                 audio_fps=44100,
#                 threads=4,
#                 fps=30,
#                 preset="p3",
#                 ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart", "-shortest"]
#             )

#             # Final overlay with avatar
#             final_overlayed_path = overlay_avatar_with_audio(
#                 resized_clip=temp_clip_path,
#                 avatar_path=f"channel_media/{avatar}",
#                 audio_path=segment_audio_path,
#                 output_path=chunk_path,
#                 avatar_active=_active_operations
#             )

#             os.remove(temp_clip_path)
#             return final_overlayed_path

#         # ---------------------------------------------------------
#         # 🟪 Mode: Commentary WITHOUT narration (Silent overlay)
#         # ---------------------------------------------------------
#         if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is None and segment_tag:
#             tagged_media = [
#                 m for m in media_pool
#                 if m["character"] == character_name and m.get("tag") == segment_tag
#             ]
#             if tagged_media:
#                 selected = tagged_media[0]
#                 if selected["type"] == "video":
#                     overlay_avatar_no_audio(
#                         selected["path"],
#                         f"channel_media/{avatar}",
#                         f"channel_media/{background_video}",
#                         chunk_path,
#                         segment_tag,
#                         media_pool=media_pool,
#                         character_name=character_name
#                     )
#                     return chunk_path

#         # ---------------------------------------------------------
#         # 🟧 Non-commentary mode with tag but no audio
#         # ---------------------------------------------------------
#         if segment_audio_path is None and segment_tag:
#             tagged_media = [
#                 m for m in media_pool
#                 if m["character"] == character_name and m.get("tag") == segment_tag
#             ]
#             if tagged_media:
#                 selected = tagged_media[0]
#                 if selected["type"] == "video":
#                     full_video = VideoFileClip(selected["path"])
#                     duration = full_video.duration
#                     audio = full_video.audio.set_fps(44100)

                    
#                 logging.info(f"⏱️ Duration {duration:.2f}s ≤ 10s — encoding directly without overlay.")
#                 final_clip = full_video.set_audio(audio.set_duration(duration))
#                 final_clip.write_videofile(
#                             chunk_path,
#                             codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                             audio_codec="aac",
#                             audio_fps=44100,
#                             threads=4,
#                             fps=30,
#                             ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "p5", "-movflags", "faststart", "-shortest"]
#                         )

#                 full_video.close()
#                 return chunk_path

#                     # try:
#                     #     reset_overlay_usage_for_new_clip()
#                     #     segment_plan = split_segments(duration)
#                     #     logging.info(f"📊 Segment plan: {segment_plan}")
#                     # except RuntimeError as e:
#                     #     logging.error(str(e))
#                     #     return None

#                     # t = 0
#                     # final_parts = []
#                     # for seg_type, seg_duration in segment_plan:
#                     #     if seg_type == "video":
#                     #         clip = full_video.subclip(t, t + seg_duration)
#                     #         final_parts.append(clip)
#                     #     else:
#                     #         try:
#                     #             overlay = get_next_overlay_image(media_pool, character_name)
#                     #         except RuntimeError as e:
#                     #             logging.error(f"⚠️ Could not get new overlay image: {str(e)}")
#                     #             continue

#                     #         overlay_img = create_zoomed_overlay_with_blur_background(
#                     #             overlay["path"], seg_duration, target_size=full_video.size
#                     #         ).set_audio(audio.subclip(t, t + seg_duration).set_fps(44100).volumex(1.5))

#                     #         final_parts.append(overlay_img)
#                     #     t += seg_duration

#                     # final_clip = concatenate_videoclips(final_parts, method="compose")
#                     # final_clip = final_clip.set_audio(audio.set_duration(final_clip.duration).set_fps(44100).volumex(1.5))

#                     # final_clip.write_videofile(
#                     #     chunk_path,
#                     #     codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                     #     audio_codec="aac",
#                     #     audio_fps=44100,
#                     #     threads=4,
#                     #     fps=30,
#                     #     ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "p5", "-movflags", "faststart", "-shortest"]
#                     # )

#                     # full_video.close()
#                     # return chunk_path

#         # ---------------------------------------------------------
#         # 🟩 Default: Normal audio-based segment
#         # ---------------------------------------------------------
#         segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#         duration_needed = segment_audio.duration
#         segment_audio = segment_audio.set_fps(44100).volumex(2)

#         selected_media = find_best_media_combination(
#             media_pool, segment, character_name,
#             duration_needed, used_media_dict,
#             media_lock, mode, i, script_context
#         )

#         if not selected_media:
#             logging.error(f"❌ No media selected for segment {i}")
#             return None

#         segment_clips = []

#         for item in selected_media:
#             media = item["media"]
#             dur = item["duration"]

#             try:
#                 if media["type"] == "video":
#                     clip = (
#                         VideoFileClip(media["path"])
#                         .subclip(0, min(dur, media["duration"]))
#                         .resize((canvas_w, canvas_h))
#                     )
#                     if media.get("mute", False):
#                         clip = clip.without_audio()
#                     logging.info(f"🎥 Video: {media['path']} for {dur:.2f}s")
#                 else:
#                     clip = create_blurred_background_image(
#                         media["path"], dur, mode
#                     )
#                     logging.info(f"🖼️ Image: {media['path']} for {dur:.2f}s")

#                 clip = clip.set_duration(dur)
#                 segment_clips.append(clip)

#             except Exception as e:
#                 logging.error(f"⚠️ Media failed: {media['path']} | {e}")

#         final_clip = concatenate_videoclips(segment_clips, method="compose")
#         final_clip = final_clip.set_audio(segment_audio.set_duration(final_clip.duration))

#         # FINAL WRITE
#         final_clip.write_videofile(
#             chunk_path,
#             codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#             audio_codec="aac",
#             audio_fps=44100,
#             threads=4,
#             fps=30,
#             preset="p3",
#             ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart", "-shortest"]
#         )

#         return chunk_path

#     except Exception as e:
#         logging.error(f"❌ Segment {i} crashed: {e}", exc_info=True)
#         return None

#     finally:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()
# def encode_video_segment(segment_info):
#     (
#         segment,
#         media_pool,
#         segment_audio_path,
#         section_dir,
#         chunk_folder,
#         i,
#         used_media_dict,
#         media_lock,
#         mode,
#         avatar,
#         background_video,
#         segments,
#         script_context,
#         orientation,
#         storyboard,
#         kenburns,
#         avatar_active
#     ) = segment_info

#     # Apply settings inside worker
#     global CURRENT_WIDTH, CURRENT_HEIGHT, CURRENT_ORIENTATION
#     global CURRENT_STORYBOARD_ENABLED, CURRENT_KENBURNS_ENABLED

#     CURRENT_ORIENTATION = orientation
#     CURRENT_STORYBOARD_ENABLED = bool(storyboard)
#     CURRENT_KENBURNS_ENABLED = bool(kenburns)

#     if orientation == "portrait":
#         CURRENT_WIDTH, CURRENT_HEIGHT = 1080, 1920
#     else:
#         CURRENT_WIDTH, CURRENT_HEIGHT = 1920, 1080

#     canvas_w = CURRENT_WIDTH
#     canvas_h = CURRENT_HEIGHT

#     chunk_filename = f"chunk_{i}_{uuid.uuid4()}.mp4"
#     chunk_path = os.path.join(chunk_folder, chunk_filename)

#     print(f"[{i}] Audio path: {segment_audio_path}")
#     prev_segments = get_prev_two_context_text(segments, i)
#     print("These are the 2 previous segments ", prev_segments)

#     logging.info(f"🔹 Processing segment {i}: {segment['text'][:50]}...")

#     try:
#         character_name = segment["character"]
#         segment_tag = segment.get("tag")

#         # TEXT-ONLY SEGMENT
#         if character_name.lower() == "text":
#             logging.info(f"📝 Text-only segment {i} detected")
            
#             if segment_audio_path and os.path.exists(segment_audio_path):
#                 segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#                 duration_needed = segment_audio.duration
#                 segment_audio = segment_audio.set_fps(44100).volumex(2)
#             else:
#                 words = len(segment['text'].split())
#                 duration_needed = max(3.0, words * 0.35)
#                 segment_audio = None
            
#             bg_color = segment.get("text_bg_color", (178, 34, 34))
#             text_color = segment.get("text_color", "white")
#             fontsize = segment.get("text_fontsize", None)
#             fade_duration = segment.get("text_fade_duration", 0.8)
            
#             text_clip = create_text_segment_with_lines(
#                 text=segment['text'],
#                 duration=duration_needed,
#                 canvas_w=canvas_w,
#                 canvas_h=canvas_h,
#                 bg_color=bg_color,
#                 text_color=text_color,
#                 fontsize=fontsize,
#                 fade_duration=fade_duration
#             )
            
#             if segment_audio:
#                 text_clip = text_clip.set_audio(segment_audio)
            
#             text_clip.write_videofile(
#                 chunk_path,
#                 codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                 audio_codec="aac" if segment_audio else None,
#                 audio_fps=44100 if segment_audio else None,
#                 threads=4,
#                 fps=30,
#                 preset="p3",
#                 ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"]
#             )
            
#             text_clip.close()
#             if segment_audio:
#                 segment_audio.close()
            
#             logging.info(f"✅ Text segment {i} completed: {chunk_path}")
#             return chunk_path

#         # COMMENTARY MODE WITH NARRATION
#         if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is not None:
#             segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#             duration_needed = segment_audio.duration
#             segment_audio = segment_audio.set_fps(44100).volumex(2)

#             # Tagged case
#             if segment_tag:
#                 tagged_media = [
#                     m for m in media_pool
#                     if m["character"] == character_name and m.get("tag") == segment_tag
#                 ]
#                 if tagged_media:
#                     selected = tagged_media[0]
#                     if selected["type"] == "video":
#                         overlay_avatar_with_audio(
#                             resized_clip=selected["path"],
#                             avatar_path=f"avatar.mp4",
#                             audio_path=segment_audio_path,
#                             output_path=chunk_path,
#                             avatar_active=avatar_active
#                         )
#                         return chunk_path

#             # Fallback
#             logging.info(f"🔁 No tag — running best-media fallback for segment {i}")
#             reset_overlay_usage_for_new_clip()

#             selected_media = find_best_media_combination(
#                 media_pool, segment, character_name, duration_needed,
#                 used_media_dict, media_lock, mode, i,
#                 script_context, prev_segments
#             )

#             if not selected_media:
#                 logging.error(f"❌ No fallback media for segment {i}")
#                 return None

#             temp_clip_path = os.path.join(chunk_folder, f"temp_clip_{i}_{uuid.uuid4()}.mp4")
            
#             # 🔥 PER-CLIP VALIDATION WITH BLACK FRAME FALLBACK
#             segment_clips = []
#             corrupt_clips_count = 0

#             for item_idx, item in enumerate(selected_media):
#                 media = item["media"]
#                 dur = item["duration"]
#                 clip = None
#                 clip_created = False

#                 try:
#                     if media["type"] == "video":
#                         video_path = media["path"]
                        
#                         # Pre-validate
#                         logging.debug(f"🔍 Validating video clip {item_idx + 1}/{len(selected_media)}")
                        
#                         if not os.path.exists(video_path):
#                             raise ValueError(f"Video file missing: {video_path}")
                        
#                         file_size = os.path.getsize(video_path)
#                         if file_size < 10000:
#                             raise ValueError(f"Video file too small: {file_size} bytes")
                        
#                         # Quick MoviePy validation
#                         test_clip = VideoFileClip(video_path)
#                         test_duration = test_clip.duration
#                         test_w = test_clip.w
#                         test_h = test_clip.h
#                         test_clip.close()
                        
#                         if test_duration <= 0:
#                             raise ValueError(f"Video has zero duration")
#                         if test_w <= 0 or test_h <= 0:
#                             raise ValueError(f"Invalid dimensions: {test_w}x{test_h}")
                        
#                         # Process validated video
#                         logging.debug(f"✅ Video validated, processing...")
#                         clip = (
#                             VideoFileClip(video_path)
#                             .subclip(0, min(dur, media["duration"]))
#                             .resize((canvas_w, canvas_h))
#                         )
                        
#                         if media.get("mute", False):
#                             clip = clip.without_audio()
                        
#                         clip = clip.set_duration(dur)
#                         clip_created = True
#                         logging.info(f"🎥 Video clip {item_idx + 1}: {os.path.basename(video_path)} ({dur:.2f}s)")
                        
#                     else:
#                         # Image processing
#                         clip = create_blurred_background_image(media["path"], dur, mode)
#                         clip = clip.set_duration(dur)
#                         clip_created = True
#                         logging.info(f"🖼️ Image clip {item_idx + 1}: {os.path.basename(media['path'])} ({dur:.2f}s)")

#                 except Exception as e:
#                     # 🔥 CREATE BLACK FRAME REPLACEMENT FOR THIS CLIP
#                     corrupt_clips_count += 1
#                     logging.error(f"❌ Clip {item_idx + 1}/{len(selected_media)} FAILED: {e}")
#                     logging.warning(f"🎬 Replacing with black frame placeholder ({dur:.2f}s)")
                    
#                     try:
#                         black_frame_path = create_black_frame_image(canvas_w, canvas_h, output_dir=chunk_folder)
                        
#                         if not black_frame_path or not os.path.exists(black_frame_path):
#                             raise RuntimeError("Black frame creation failed")
                        
#                         clip = create_blurred_background_image(black_frame_path, dur, mode)
#                         clip = clip.set_duration(dur)
#                         clip_created = True
                        
#                         logging.info(f"✅ Black frame replacement created for clip {item_idx + 1}")
                        
#                     except Exception as fallback_err:
#                         logging.error(f"💀 Black frame fallback also failed for clip {item_idx + 1}: {fallback_err}")
#                         # Last resort - numpy black clip
#                         try:
#                             import numpy as np
#                             black_array = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 10
#                             clip = ImageClip(black_array, duration=dur)
#                             clip_created = True
#                             logging.info(f"✅ Emergency numpy black clip created for clip {item_idx + 1}")
#                         except Exception as emergency_err:
#                             logging.critical(f"💥 All fallbacks failed for clip {item_idx + 1}: {emergency_err}")
#                             clip_created = False
                
#                 if clip_created and clip is not None:
#                     segment_clips.append(clip)
#                 else:
#                     logging.critical(f"💥 Could not create ANY clip for item {item_idx + 1}")

#             if not segment_clips:
#                 logging.error(f"❌ No valid clips for segment {i} - ALL clips failed")
#                 raise RuntimeError(f"All clips failed for segment {i}")
            
#             if corrupt_clips_count > 0:
#                 logging.warning(f"⚠️ Segment {i}: {corrupt_clips_count}/{len(selected_media)} clips replaced with black frames")
            
#             # Concatenate clips
#             if CURRENT_STORYBOARD_ENABLED and len(segment_clips) > 1:
#                 final_clip = apply_storyboard_transitions(segment_clips)
#             else:
#                 final_clip = concatenate_videoclips(segment_clips, method="compose")
            
#             final_clip = final_clip.set_audio(segment_audio.set_duration(final_clip.duration))

#             # Write
#             final_clip.write_videofile(
#                 temp_clip_path,
#                 codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                 audio_codec="aac",
#                 audio_fps=44100,
#                 threads=4,
#                 fps=30,
#                 preset="p3",
#                 ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart", "-shortest"]
#             )

#             # Overlay avatar
#             final_overlayed_path = overlay_avatar_with_audio(
#                 resized_clip=temp_clip_path,
#                 avatar_path=f"avatar.mp4",
#                 audio_path=segment_audio_path,
#                 output_path=chunk_path,
#                 avatar_active=avatar_active
#             )

#             os.remove(temp_clip_path)
#             return final_overlayed_path

#         # COMMENTARY WITHOUT NARRATION
#         if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is None and segment_tag:
#             tagged_media = [
#                 m for m in media_pool
#                 if m["character"] == character_name and m.get("tag") == segment_tag
#             ]
#             if tagged_media:
#                 selected = tagged_media[0]
#                 if selected["type"] == "video":
#                     overlay_avatar_no_audio(
#                         selected["path"],
#                         f"channel_media/{avatar}",
#                         f"channel_media/{background_video}",
#                         chunk_path,
#                         segment_tag,
#                         media_pool=media_pool,
#                         character_name=character_name
#                     )
#                     return chunk_path

#         # NON-COMMENTARY WITH TAG BUT NO AUDIO
#         if segment_audio_path is None and segment_tag:
#             tagged_media = [
#                 m for m in media_pool
#                 if m["character"] == character_name and m.get("tag") == segment_tag
#             ]
#             if tagged_media:
#                 selected = tagged_media[0]
#                 if selected["type"] == "video":
#                     full_video = VideoFileClip(selected["path"])
#                     duration = full_video.duration
#                     audio = full_video.audio.set_fps(44100)
                    
#                     logging.info(f"⏱️ Duration {duration:.2f}s ≤ 10s — encoding directly without overlay.")
#                     final_clip = full_video.set_audio(audio.set_duration(duration))
#                     final_clip.write_videofile(
#                         chunk_path,
#                         codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#                         audio_codec="aac",
#                         audio_fps=44100,
#                         threads=4,
#                         fps=30,
#                         ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "p5", "-movflags", "faststart", "-shortest"]
#                     )

#                     full_video.close()
#                     return chunk_path

#         # DEFAULT: NORMAL AUDIO-BASED SEGMENT
#         segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#         duration_needed = segment_audio.duration
#         segment_audio = segment_audio.set_fps(44100).volumex(2)

#         selected_media = find_best_media_combination(
#             media_pool, segment, character_name,
#             duration_needed, used_media_dict,
#             media_lock, mode, i, script_context, prev_segments
#         )

#         if not selected_media:
#             logging.error(f"❌ No media selected for segment {i}")
#             return None

#         # 🔥 PER-CLIP VALIDATION WITH BLACK FRAME FALLBACK
#         segment_clips = []
#         corrupt_clips_count = 0

#         for item_idx, item in enumerate(selected_media):
#             media = item["media"]
#             dur = item["duration"]
#             clip = None
#             clip_created = False

#             try:
#                 if media["type"] == "video":
#                     video_path = media["path"]
                    
#                     # Pre-validate
#                     logging.debug(f"🔍 Validating video clip {item_idx + 1}/{len(selected_media)}")
                    
#                     if not os.path.exists(video_path):
#                         raise ValueError(f"Video file missing: {video_path}")
                    
#                     file_size = os.path.getsize(video_path)
#                     if file_size < 10000:
#                         raise ValueError(f"Video file too small: {file_size} bytes")
                    
#                     # Quick MoviePy validation
#                     test_clip = VideoFileClip(video_path)
#                     test_duration = test_clip.duration
#                     test_w = test_clip.w
#                     test_h = test_clip.h
#                     test_clip.close()
                    
#                     if test_duration <= 0:
#                         raise ValueError(f"Video has zero duration")
#                     if test_w <= 0 or test_h <= 0:
#                         raise ValueError(f"Invalid dimensions: {test_w}x{test_h}")
                    
#                     # Process validated video
#                     logging.debug(f"✅ Video validated, processing...")
#                     clip = (
#                         VideoFileClip(video_path)
#                         .subclip(0, min(dur, media["duration"]))
#                         .resize((canvas_w, canvas_h))
#                     )
                    
#                     if media.get("mute", False):
#                         clip = clip.without_audio()
                    
#                     clip = clip.set_duration(dur)
#                     clip_created = True
#                     logging.info(f"🎥 Video clip {item_idx + 1}: {os.path.basename(video_path)} ({dur:.2f}s)")
                    
#                 else:
#                     # Image processing
#                     clip = create_blurred_background_image(media["path"], dur, mode)
#                     clip = clip.set_duration(dur)
#                     clip_created = True
#                     logging.info(f"🖼️ Image clip {item_idx + 1}: {os.path.basename(media['path'])} ({dur:.2f}s)")

#             except Exception as e:
#                 # 🔥 CREATE BLACK FRAME REPLACEMENT FOR THIS CLIP
#                 corrupt_clips_count += 1
#                 logging.error(f"❌ Clip {item_idx + 1}/{len(selected_media)} FAILED: {e}")
#                 logging.warning(f"🎬 Replacing with black frame placeholder ({dur:.2f}s)")
                
#                 try:
#                     black_frame_path = create_black_frame_image(canvas_w, canvas_h, output_dir=chunk_folder)
                    
#                     if not black_frame_path or not os.path.exists(black_frame_path):
#                         raise RuntimeError("Black frame creation failed")
                    
#                     clip = create_blurred_background_image(black_frame_path, dur, mode)
#                     clip = clip.set_duration(dur)
#                     clip_created = True
                    
#                     logging.info(f"✅ Black frame replacement created for clip {item_idx + 1}")
                    
#                 except Exception as fallback_err:
#                     logging.error(f"💀 Black frame fallback also failed for clip {item_idx + 1}: {fallback_err}")
#                     # Last resort - numpy black clip
#                     try:
#                         import numpy as np
#                         black_array = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 10
#                         clip = ImageClip(black_array, duration=dur)
#                         clip_created = True
#                         logging.info(f"✅ Emergency numpy black clip created for clip {item_idx + 1}")
#                     except Exception as emergency_err:
#                         logging.critical(f"💥 All fallbacks failed for clip {item_idx + 1}: {emergency_err}")
#                         clip_created = False
            
#             if clip_created and clip is not None:
#                 segment_clips.append(clip)
#             else:
#                 logging.critical(f"💥 Could not create ANY clip for item {item_idx + 1}")

#         if not segment_clips:
#             logging.error(f"❌ No valid clips for segment {i} - ALL clips failed")
#             raise RuntimeError(f"All clips failed for segment {i}")
        
#         if corrupt_clips_count > 0:
#             logging.warning(f"⚠️ Segment {i}: {corrupt_clips_count}/{len(selected_media)} clips replaced with black frames")

#         # Concatenate
#         if CURRENT_STORYBOARD_ENABLED and len(segment_clips) > 1:
#             final_clip = apply_storyboard_transitions(segment_clips)
#         else:
#             final_clip = concatenate_videoclips(segment_clips, method="compose")
        
#         final_clip = final_clip.set_audio(segment_audio.set_duration(final_clip.duration))

#         # Write
#         final_clip.write_videofile(
#             chunk_path,
#             codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
#             audio_codec="aac",
#             audio_fps=44100,
#             threads=4,
#             fps=30,
#             preset="p3",
#             ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart", "-shortest"]
#         )

#         return chunk_path

#     except Exception as e:
#         logging.error(f"❌ Segment {i} crashed: {e}", exc_info=True)
#         return None

#     finally:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()

import threading
import gc
import logging
import os
import subprocess
import uuid

from moviepy.editor import AudioFileClip, VideoFileClip


FFMPEG_SEGMENT_TIMEOUT = int(os.getenv("FFMPEG_SEGMENT_TIMEOUT", "600"))
FFMPEG_CHUNK_PREP_TIMEOUT = int(os.getenv("FFMPEG_CHUNK_PREP_TIMEOUT", "120"))
# EBU R128 target for avatar vs clip loudness matching (talking-heads + clips).
PIPELINE_AUDIO_TARGET_LUFS = float(os.getenv("PIPELINE_AUDIO_TARGET_LUFS", "-16"))
VIDEO_PROBE_TIMEOUT = int(os.getenv("VIDEO_PROBE_TIMEOUT", "120"))


def _default_nvenc_concurrency():
    """NVENC session slots — tuned for VRAM; override with NVENC_CONCURRENCY."""
    env_val = os.getenv("NVENC_CONCURRENCY")
    if env_val is not None:
        return max(1, int(env_val))
    if not torch.cuda.is_available():
        return 1
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 14:
            return 4   # A4000 / 4080 / 16GB class
        if vram_gb >= 10:
            return 3
        if vram_gb >= 6:
            return 2
        return 1
    except Exception:
        return 3


def _default_gpu_concurrent_jobs(kenburns_enabled: bool) -> int:
    """
    Max simultaneous GPU jobs (Ken Burns + NVENC share this lock).
    Delegates to worker_config; Ken Burns-off uses higher NVENC-only cap when set.
    """
    from worker_config import get_worker_limits

    limits = get_worker_limits()
    if not kenburns_enabled:
        env_val = os.getenv("NVENC_CONCURRENCY")
        if env_val is not None:
            return max(1, int(env_val))
        return min(4, limits.encode_pool_workers // 4)
    return limits.gpu_concurrent_jobs


def _default_kenburns_concurrency():
    """Ken Burns slots — default 1 (exclusive full GPU). Override with KENBURNS_GPU_CONCURRENCY."""
    env_val = os.getenv("KENBURNS_GPU_CONCURRENCY")
    if env_val is not None:
        return max(1, int(env_val))
    return 1


def _default_nvenc_concurrency_for_job(kenburns_enabled: bool) -> int:
    if kenburns_enabled:
        return _default_gpu_concurrent_jobs(True)
    return _default_nvenc_concurrency()



def _acquire_gpu_lock(gpu_semaphore, label: str, timeout_sec: int | None = None) -> None:
    """Log while waiting — with GPU_CONCURRENT_JOBS=1 many workers queue here silently."""
    if timeout_sec is None:
        timeout_sec = int(os.getenv("GPU_LOCK_TIMEOUT_SEC", "900"))
    logging.info(f"⏳ {label}: waiting for GPU slot...")
    acquired = gpu_semaphore.acquire(timeout=timeout_sec)
    if not acquired:
        raise TimeoutError(
            f"{label}: GPU slot unavailable after {timeout_sec}s "
            f"(prior worker may have died while holding the lock — restart wsgi.py)"
        )
    logging.info(f"🔒 {label}: GPU slot acquired")


def _kenburns_needs_gpu_lock() -> bool:
    from gpu_kenburns_simple import kenburns_uses_gpu_lock

    return kenburns_uses_gpu_lock()


@lru_cache(maxsize=1)
def _segment_rate_args(video_codec) -> list:
    """
    Quality settings for segment encodes. Without these NVENC falls back to its
    low default bitrate (~2 Mbps) — blocky 1080p, especially Ken Burns motion.
    Constant-quality VBR: same speed, much sharper. Tunable via env.
    """
    if "nvenc" in str(video_codec):
        return [
            "-tune", "hq", "-rc", "vbr",
            "-cq", os.getenv("SEGMENT_NVENC_CQ", "19"), "-b:v", "0",
            "-maxrate", os.getenv("SEGMENT_NVENC_MAXRATE", "20M"),
            "-bufsize", os.getenv("SEGMENT_NVENC_BUFSIZE", "40M"),
        ]
    return ["-crf", os.getenv("SEGMENT_X264_CRF", "18")]


def _segment_use_nvenc() -> bool:
    """
    Pick NVENC without touching torch during OpenAI/ImageFX (avoids CUDA init per worker).
    Set USE_NVENC=1 on GPU servers; USE_NVENC=0 on CPU-only boxes.
    """
    env = os.getenv("USE_NVENC", "1").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    return torch.cuda.is_available()


def _default_encode_pool_workers(kenburns_enabled: bool = False) -> int:
    from worker_config import get_worker_limits

    return get_worker_limits().encode_pool_workers


def _uses_nvenc_in_cmd(cmd):
    return any("nvenc" in str(arg) for arg in cmd)


def _ffmpeg_run(cmd, timeout=FFMPEG_SEGMENT_TIMEOUT, label="", nvenc_semaphore=None, uses_nvenc=None):
    """Run ffmpeg with timeout; limit concurrent NVENC via counting semaphore."""
    if uses_nvenc is None:
        uses_nvenc = _uses_nvenc_in_cmd(cmd)
    if label:
        logging.debug(f"🎬 FFmpeg start | {label} | timeout={timeout}s")

    def _execute():
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    try:
        if uses_nvenc and nvenc_semaphore is not None:
            nvenc_timeout = int(os.getenv("GPU_LOCK_TIMEOUT_SEC", "900"))
            if not nvenc_semaphore.acquire(timeout=nvenc_timeout):
                logging.error(
                    f"⏱️ NVENC slot TIMEOUT ({nvenc_timeout}s) | {label} — "
                    f"restart wsgi.py if pool is stuck"
                )
                return None
            try:
                result = _execute()
            finally:
                nvenc_semaphore.release()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            result = _execute()
    except subprocess.TimeoutExpired as exc:
        logging.error(f"⏱️ FFmpeg TIMEOUT ({timeout}s) | {label}")
        stderr = exc.stderr
        if stderr:
            tail = stderr[-500:] if isinstance(stderr, str) else stderr.decode(errors="replace")[-500:]
            logging.error(f"stderr tail: {tail}")
        if uses_nvenc and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    except Exception as exc:
        logging.error(f"❌ FFmpeg error | {label}: {exc}")
        return None

    if label:
        if result.returncode == 0:
            logging.debug(f"✅ FFmpeg done | {label}")
        else:
            logging.error(
                f"❌ FFmpeg failed (rc={result.returncode}) | {label}\n"
                f"{result.stderr[-800:]}"
            )
    return result


def _probe_video_duration_safe(path, timeout=VIDEO_PROBE_TIMEOUT):
    """Probe duration without blocking forever on corrupt/huge files."""
    container = {}
    error_container = {}

    def _work():
        try:
            clip = VideoFileClip(path)
            container["duration"] = clip.duration
            clip.close()
            del clip
        except Exception as exc:
            error_container["error"] = exc

    probe_thread = threading.Thread(
        target=_work,
        name=f"probe-{os.path.basename(path)}",
        daemon=True,
    )
    probe_thread.start()
    probe_thread.join(timeout)
    if probe_thread.is_alive():
        raise TimeoutError(f"VideoFileClip probe timed out after {timeout}s: {path}")
    if "error" in error_container:
        raise error_container["error"]
    return container.get("duration", 0)


def overlay_with_layout(
    content_path,
    avatar_path,
    avatar_start,
    audio_path,
    output_path,
    layout_config,
    canvas_w,
    canvas_h,
):
    """
    Three-layer talking heads compositor.

    Layers (bottom to top):
      1. Background  — dark canvas or blurred content fill
      2. Content     — main footage scaled to content_layer coords
      3. Avatar      — talking head scaled to avatar_layer coords
                       seeked to avatar_start in the batch file

    Layout is driven entirely by layout_config from TalkingHeadsStudio.
    No hardcoded positions — everything comes from the frontend.

    Handles all layout types:
      full_avatar    — avatar fills canvas, no content panel
      full_content   — content fills canvas, no avatar
      pip_*          — content fills canvas, avatar inset at coords
      split_*        — both layers side-by-side on dark background
      center_pip     — content fills canvas, avatar centered
      news_*         — broadcast style with avatar inset
    """
    layout_id     = layout_config.get("layout_id",      "pip_bottom_left")
    avatar_layer  = layout_config.get("avatar_layer",   None)
    content_layer = layout_config.get("content_layer",  None)

    use_nvenc   = torch.cuda.is_available()
    video_codec = "h264_nvenc" if use_nvenc else "libx264"
    preset_val  = "p5"         if use_nvenc else "medium"

    logging.info(
        f"🎭 overlay_with_layout | layout={layout_id} | "
        f"avatar_start={avatar_start:.3f}s | "
        f"canvas={canvas_w}x{canvas_h}"
    )

    # ── Validate inputs ───────────────────────────────────────
    if not os.path.exists(content_path):
        logging.error(f"❌ overlay_with_layout: content not found: {content_path}")
        return None
    if not os.path.exists(avatar_path):
        logging.error(f"❌ overlay_with_layout: avatar not found: {avatar_path}")
        return None
    if not os.path.exists(audio_path):
        logging.error(f"❌ overlay_with_layout: audio not found: {audio_path}")
        return None

    # ── Helper: build scale+pad filter for a layer ───────────
    def _scale_pad(input_label, out_w, out_h, out_label):
        """Scale input to fit within out_w x out_h, pad to exact size."""
        return (
            f"[{input_label}]"
            f"scale={out_w}:{out_h}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1"
            f"[{out_label}]"
        )

    def _scale_fill(input_label, out_w, out_h, out_label):
        """Scale input to fill out_w x out_h (may crop), setsar=1."""
        return (
            f"[{input_label}]"
            f"scale={out_w}:{out_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},"
            f"setsar=1"
            f"[{out_label}]"
        )

    try:

        # ════════════════════════════════════════════════════════
        # CASE 1 — FULL AVATAR
        # Avatar fills entire canvas. No content panel.
        # Input 0: avatar   Input 1: audio
        # ════════════════════════════════════════════════════════
        if layout_id == "full_avatar" or avatar_layer is None:
            logging.info(f"🎭 Layout: full_avatar — avatar fills canvas")

            filter_complex = _scale_pad("0:v", canvas_w, canvas_h, "v")

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(round(avatar_start, 4)),
                "-i", avatar_path,     # input 0
                "-i", audio_path,      # input 1
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "1:a:0",
                "-c:v", video_codec, "-preset", preset_val,
                "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart",
                "-shortest",
                output_path,
            ]

        # ════════════════════════════════════════════════════════
        # CASE 2 — FULL CONTENT
        # Content fills entire canvas. No avatar.
        # Input 0: content   Input 1: audio
        # ════════════════════════════════════════════════════════
        elif layout_id == "full_content" or content_layer is None:
            logging.info(f"🎭 Layout: full_content — content fills canvas")

            filter_complex = _scale_fill("0:v", canvas_w, canvas_h, "v")

            cmd = [
                "ffmpeg", "-y",
                "-i", content_path,    # input 0
                "-i", audio_path,      # input 1
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "1:a:0",
                "-c:v", video_codec, "-preset", preset_val,
                "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart",
                "-shortest",
                output_path,
            ]

        # ════════════════════════════════════════════════════════
        # CASE 3 — PIP / NEWS / CENTER layouts
        # Content fills canvas as background.
        # Avatar overlaid at avatar_layer coordinates.
        # Input 0: content   Input 1: avatar   Input 2: audio
        # ════════════════════════════════════════════════════════
        elif layout_id in (
            "pip_bottom_left",  "pip_bottom_right",
            "pip_top_left",     "pip_top_right",
            "pip_bottom_center","pip_top_center",
            "center_pip",
            "news_broadcast",   "news_broadcast_br",
            "news_top_left",    "news_portrait_bottom",
            "news_portrait_top",
        ):
            av_x = avatar_layer["x"]
            av_y = avatar_layer["y"]
            av_w = avatar_layer["width"]
            av_h = avatar_layer["height"]

            logging.info(
                f"🎭 Layout: {layout_id} | "
                f"avatar at ({av_x},{av_y}) size {av_w}x{av_h}"
            )

            filter_complex = ";".join([
                # Layer 1: content fills canvas
                _scale_fill("0:v", canvas_w, canvas_h, "bg"),
                # Layer 2: avatar scaled to layout coords
                _scale_pad("1:v", av_w, av_h, "av"),
                # Composite: avatar overlaid on content
                f"[bg][av]overlay={av_x}:{av_y}[v]",
            ])

            cmd = [
                "ffmpeg", "-y",
                "-i", content_path,                    # input 0
                "-ss", str(round(avatar_start, 4)),
                "-i", avatar_path,                     # input 1 (seeked)
                "-i", audio_path,                      # input 2
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "2:a:0",
                "-c:v", video_codec, "-preset", preset_val,
                "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart",
                "-shortest",
                output_path,
            ]

        # ════════════════════════════════════════════════════════
        # CASE 4 — SPLIT layouts
        # Both content and avatar occupy separate regions
        # on a dark background canvas.
        # Input 0: content   Input 1: avatar   Input 2: audio
        # ════════════════════════════════════════════════════════
        elif layout_id in (
            "split_left",    "split_right",
            "split_equal",
            "split_top",     "split_bottom",
            "split_portrait",
        ):
            av_x = avatar_layer["x"]
            av_y = avatar_layer["y"]
            av_w = avatar_layer["width"]
            av_h = avatar_layer["height"]

            ct_x = content_layer["x"]
            ct_y = content_layer["y"]
            ct_w = content_layer["width"]
            ct_h = content_layer["height"]

            logging.info(
                f"🎭 Layout: {layout_id} | "
                f"content at ({ct_x},{ct_y}) {ct_w}x{ct_h} | "
                f"avatar at ({av_x},{av_y}) {av_w}x{av_h}"
            )

            filter_complex = ";".join([
                # Dark background
                f"color=c=#0d0d18:s={canvas_w}x{canvas_h}:r=30[bg]",
                # Content scaled to its region
                _scale_pad("0:v", ct_w, ct_h, "ct"),
                # Avatar scaled to its region
                _scale_pad("1:v", av_w, av_h, "av"),
                # Place content on background
                f"[bg][ct]overlay={ct_x}:{ct_y}[tmp]",
                # Place avatar on top
                f"[tmp][av]overlay={av_x}:{av_y}[v]",
            ])

            cmd = [
                "ffmpeg", "-y",
                "-i", content_path,                    # input 0
                "-ss", str(round(avatar_start, 4)),
                "-i", avatar_path,                     # input 1 (seeked)
                "-i", audio_path,                      # input 2
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "2:a:0",
                "-c:v", video_codec, "-preset", preset_val,
                "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart",
                "-shortest",
                output_path,
            ]

        # ════════════════════════════════════════════════════════
        # CASE 5 — CENTER FOCUS
        # Content fills canvas (used as background).
        # Avatar centered on top at medium size.
        # Same filter as PIP but kept separate for clarity.
        # Input 0: content   Input 1: avatar   Input 2: audio
        # ════════════════════════════════════════════════════════
        elif layout_id == "center_focus":
            av_x = avatar_layer["x"]
            av_y = avatar_layer["y"]
            av_w = avatar_layer["width"]
            av_h = avatar_layer["height"]

            logging.info(
                f"🎭 Layout: center_focus | "
                f"avatar at ({av_x},{av_y}) size {av_w}x{av_h}"
            )

            filter_complex = ";".join([
                _scale_fill("0:v", canvas_w, canvas_h, "bg"),
                _scale_pad("1:v", av_w, av_h, "av"),
                f"[bg][av]overlay={av_x}:{av_y}[v]",
            ])

            cmd = [
                "ffmpeg", "-y",
                "-i", content_path,
                "-ss", str(round(avatar_start, 4)),
                "-i", avatar_path,
                "-i", audio_path,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "2:a:0",
                "-c:v", video_codec, "-preset", preset_val,
                "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart",
                "-shortest",
                output_path,
            ]

        # ════════════════════════════════════════════════════════
        # CASE 6 — CONTENT MAIN
        # Content fills canvas, avatar small inset (bottom right).
        # Coordinates come from layout_config as usual.
        # Same filter as PIP — content fill + avatar overlay.
        # Input 0: content   Input 1: avatar   Input 2: audio
        # ════════════════════════════════════════════════════════
        elif layout_id == "content_main":
            av_x = avatar_layer["x"]
            av_y = avatar_layer["y"]
            av_w = avatar_layer["width"]
            av_h = avatar_layer["height"]

            logging.info(
                f"🎭 Layout: content_main | "
                f"avatar small inset at ({av_x},{av_y}) {av_w}x{av_h}"
            )

            filter_complex = ";".join([
                _scale_fill("0:v", canvas_w, canvas_h, "bg"),
                _scale_pad("1:v", av_w, av_h, "av"),
                f"[bg][av]overlay={av_x}:{av_y}[v]",
            ])

            cmd = [
                "ffmpeg", "-y",
                "-i", content_path,
                "-ss", str(round(avatar_start, 4)),
                "-i", avatar_path,
                "-i", audio_path,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "2:a:0",
                "-c:v", video_codec, "-preset", preset_val,
                "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart",
                "-shortest",
                output_path,
            ]

        # ════════════════════════════════════════════════════════
        # FALLBACK — unknown layout_id
        # Treat as PIP bottom left using avatar_layer coords.
        # If avatar_layer missing, just scale content to canvas.
        # ════════════════════════════════════════════════════════
        else:
            logging.warning(
                f"⚠️ overlay_with_layout: unknown layout_id '{layout_id}' "
                f"— falling back to PIP"
            )

            if avatar_layer:
                av_x = avatar_layer["x"]
                av_y = avatar_layer["y"]
                av_w = avatar_layer["width"]
                av_h = avatar_layer["height"]

                filter_complex = ";".join([
                    _scale_fill("0:v", canvas_w, canvas_h, "bg"),
                    _scale_pad("1:v", av_w, av_h, "av"),
                    f"[bg][av]overlay={av_x}:{av_y}[v]",
                ])

                cmd = [
                    "ffmpeg", "-y",
                    "-i", content_path,
                    "-ss", str(round(avatar_start, 4)),
                    "-i", avatar_path,
                    "-i", audio_path,
                    "-filter_complex", filter_complex,
                    "-map", "[v]",
                    "-map", "2:a:0",
                    "-c:v", video_codec, "-preset", preset_val,
                    "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart",
                    "-shortest",
                    output_path,
                ]
            else:
                # No avatar layer at all — just content + audio
                filter_complex = _scale_fill("0:v", canvas_w, canvas_h, "v")

                cmd = [
                    "ffmpeg", "-y",
                    "-i", content_path,
                    "-i", audio_path,
                    "-filter_complex", filter_complex,
                    "-map", "[v]",
                    "-map", "1:a:0",
                    "-c:v", video_codec, "-preset", preset_val,
                    "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart",
                    "-shortest",
                    output_path,
                ]

        # ── Run FFmpeg ────────────────────────────────────────
        logging.info(
            f"🎬 overlay_with_layout: running FFmpeg for segment "
            f"[layout={layout_id}]"
        )
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(
                f"❌ overlay_with_layout FFmpeg failed "
                f"[layout={layout_id}]: {result.stderr[-600:]}"
            )
            return None

        if not os.path.exists(output_path):
            logging.error(
                f"❌ overlay_with_layout: output not created: {output_path}"
            )
            return None

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(
            f"✅ overlay_with_layout done: {output_path} "
            f"({file_size:.1f}MB) [layout={layout_id}]"
        )
        return output_path

    except Exception as e:
        logging.error(
            f"❌ overlay_with_layout crashed [layout={layout_id}]: {e}",
            exc_info=True,
        )
        return None


# def encode_video_segment(segment_info):
#     (
#         segment,
#         media_pool,
#         segment_audio_path,
#         section_dir,
#         chunk_folder,
#         i,
#         used_media_dict,
#         media_lock,
#         mode,
#         avatar,
#         background_video,
#         segments,
#         script_context,
#         orientation,
#         storyboard,
#         kenburns,
#         avatar_active,
#         gpu_semaphore,
#         effect_assignment,
#         whisk_shared_state,
#         head_freeze_sec,
#         talking_heads_mode,
#     ) = segment_info

#     global CURRENT_WIDTH, CURRENT_HEIGHT, CURRENT_ORIENTATION
#     global CURRENT_STORYBOARD_ENABLED, CURRENT_KENBURNS_ENABLED

#     CURRENT_ORIENTATION = orientation
#     CURRENT_STORYBOARD_ENABLED = bool(storyboard)
#     CURRENT_KENBURNS_ENABLED = bool(kenburns)

#     if orientation == "portrait":
#         CURRENT_WIDTH, CURRENT_HEIGHT = 1080, 1920
#     else:
#         CURRENT_WIDTH, CURRENT_HEIGHT = 1920, 1080

#     canvas_w = CURRENT_WIDTH
#     canvas_h = CURRENT_HEIGHT

#     chunk_filename = f"chunk_{i}_{uuid.uuid4()}.mp4"
#     chunk_path = os.path.join(chunk_folder, chunk_filename)

#     print(f"[{i}] Audio path: {segment_audio_path}")
#     prev_segments = get_prev_two_context_text(segments, i)
#     print("These are the 2 previous segments ", prev_segments)

#     logging.info(f"🔹 Processing segment {i}: {segment['text'][:50]}...")

#     use_nvenc = torch.cuda.is_available()
#     video_codec = "h264_nvenc" if use_nvenc else "libx264"
#     preset_val = "p3" if use_nvenc else "medium"

#     try:
#         character_name = segment["character"]
#         segment_tag = segment.get("tag")

#         # ============================================================
#         # TEXT-ONLY SEGMENT
#         # ============================================================
#         if character_name.lower() == "text":
#             logging.info(f"📝 Text-only segment {i} detected")

#             if segment_audio_path and os.path.exists(segment_audio_path):
#                 segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#                 duration_needed = segment_audio.duration
#                 segment_audio = segment_audio.set_fps(44100).volumex(2)
#             else:
#                 words = len(segment['text'].split())
#                 duration_needed = max(3.0, words * 0.35)
#                 segment_audio = None

#             bg_color = segment.get("text_bg_color", (178, 34, 34))
#             text_color = segment.get("text_color", "white")
#             fontsize = segment.get("text_fontsize", None)
#             fade_duration = segment.get("text_fade_duration", 0.8)

#             text_clip = create_text_segment_with_lines(
#                 text=segment['text'],
#                 duration=duration_needed,
#                 canvas_w=canvas_w,
#                 canvas_h=canvas_h,
#                 bg_color=bg_color,
#                 text_color=text_color,
#                 fontsize=fontsize,
#                 fade_duration=fade_duration
#             )

#             if segment_audio:
#                 text_clip = text_clip.set_audio(segment_audio)

#             text_clip.write_videofile(
#                 chunk_path,
#                 codec=video_codec,
#                 audio_codec="aac" if segment_audio else None,
#                 audio_fps=44100 if segment_audio else None,
#                 threads=4,
#                 fps=30,
#                 preset=preset_val,
#                 ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"]
#             )

#             text_clip.close()
#             if segment_audio:
#                 segment_audio.close()

#             logging.info(f"✅ Text segment {i} completed: {chunk_path}")
#             return chunk_path

#         # ============================================================
#         # CLIP SEGMENT — use downloaded video's original audio
#         # (unchanged: already content-only, no avatar/bg dependency)
#         # ============================================================
#         segment_text = segment.get("text", "").strip()
#         is_clip_segment = segment_text.startswith("[CLIP:") and segment_text.endswith("]")

#         if is_clip_segment:
#             logging.info(f"🎬 CLIP segment {i} detected")

#             if not segment_tag:
#                 logging.error(f"❌ CLIP segment {i} has no tag — cannot find video in library")
#                 return None

#             tagged_media = [
#                 m for m in media_pool
#                 if m["character"] == character_name and m.get("tag") == segment_tag
#             ]

#             if not tagged_media:
#                 logging.error(f"❌ CLIP segment {i}: no media found in library with tag '{segment_tag}'")
#                 return None

#             selected = tagged_media[0]

#             if selected["type"] != "video":
#                 logging.error(f"❌ CLIP segment {i}: tagged media is not a video")
#                 return None

#             cmd = [
#                 'ffmpeg', '-y',
#                 '-i', selected["path"],
#                 '-c:v', video_codec, '-preset', preset_val,
#                 '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
#                 '-pix_fmt', 'yuv420p',
#                 '-movflags', 'faststart',
#                 chunk_path
#             ]
#             result = subprocess.run(cmd, capture_output=True, text=True)
#             if result.returncode != 0:
#                 logging.error(f"❌ CLIP segment {i} FFmpeg failed: {result.stderr[-500:]}")
#                 return None

#             logging.info(f"✅ CLIP segment {i} completed using original video audio: {chunk_path}")
#             return chunk_path

#         # ============================================================
#         # COMMENTARY MODE WITH NARRATION
#         # ============================================================
#         if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is not None:
#             segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#             duration_needed = segment_audio.duration
#             segment_audio = segment_audio.set_fps(44100).volumex(2)

#             if segment_tag:
#                 tagged_media = [
#                     m for m in media_pool
#                     if m["character"] == character_name and m.get("tag") == segment_tag
#                 ]
#                 if tagged_media:
#                     selected = tagged_media[0]
#                     if selected["type"] == "video":

#                         if talking_heads_mode:
#                             # ────────────────────────────────────────────
#                             # RULE 1 — IRON RULE:
#                             # output duration == narration audio duration,
#                             # exactly. No avatar/border/background — those
#                             # are added in the post-assembly composite.
#                             # Source video audio fully discarded (muted).
#                             # -stream_loop -1 (input-level) loops source
#                             # if shorter than narration; trim cuts to exact
#                             # length; setpts zeroes timestamps so video and
#                             # audio both start at PTS=0 with no lead-in gap.
#                             # ────────────────────────────────────────────
#                             cmd = [
#                                 'ffmpeg', '-y',
#                                 '-stream_loop', '-1', '-i', selected["path"],
#                                 '-i', segment_audio_path,
#                                 '-filter_complex',
#                                 f"[0:v]trim=duration={round(duration_needed,4)},"
#                                 f"setpts=PTS-STARTPTS,"
#                                 f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
#                                 f"crop={canvas_w}:{canvas_h},"
#                                 f"format=yuv420p[v]",
#                                 '-map', '[v]',
#                                 '-map', '1:a:0',
#                                 '-c:v', video_codec, '-preset', preset_val,
#                                 '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
#                                 '-r', '30', '-vsync', 'cfr',
#                                 '-shortest',
#                                 '-pix_fmt', 'yuv420p',
#                                 '-movflags', 'faststart',
#                                 chunk_path
#                             ]
#                             result = subprocess.run(cmd, capture_output=True, text=True)
#                             if result.returncode == 0 and os.path.exists(chunk_path):
#                                 logging.info(
#                                     f"✅ Tagged narration segment {i} (talking-heads, content-only): {chunk_path}"
#                                 )
#                                 segment_audio.close()
#                                 return chunk_path
#                             else:
#                                 logging.warning(
#                                     f"⚠️ Talking-heads content-only encode failed for segment {i}: "
#                                     f"{result.stderr[-300:]} — falling through to best-media fallback"
#                                 )

#                         else:
#                             result = overlay_avatar_with_audio(
#                                 resized_clip=selected["path"],
#                                 avatar_path="avatar.mp4",
#                                 audio_path=segment_audio_path,
#                                 output_path=chunk_path,
#                                 avatar_active=avatar_active
#                             )
#                             if result and os.path.exists(result):
#                                 logging.info(f"✅ Tagged narration segment {i} completed: {result}")
#                                 segment_audio.close()
#                                 return result
#                             else:
#                                 logging.warning(f"⚠️ overlay_avatar_with_audio failed for tagged segment {i} — falling through to best-media fallback")

#             logging.info(f"🔁 No tag or tag fallback — running best-media for segment {i}")
#             reset_overlay_usage_for_new_clip()

#             selected_media = find_best_media_combination(
#                 media_pool, segment, character_name, duration_needed,
#                 used_media_dict, media_lock, mode, i,
#                 script_context, prev_segments, whisk_shared_state
#             )

#             if not selected_media:
#                 logging.error(f"❌ No fallback media for segment {i}")
#                 return None

#             temp_clip_path = os.path.join(chunk_folder, f"temp_clip_{i}_{uuid.uuid4()}.mp4")
#             processed_clips = []
#             corrupt_clips_count = 0

#             for item_idx, item in enumerate(selected_media):
#                 media = item["media"]
#                 dur = item["duration"]
#                 clip_output = os.path.join(chunk_folder, f"clip_{i}_{item_idx}_{uuid.uuid4()}.mp4")
#                 clip_created = False

#                 try:
#                     if media["type"] == "video":
#                         video_path = media["path"]

#                         if not os.path.exists(video_path):
#                             raise ValueError(f"Video file missing: {video_path}")
#                         file_size = os.path.getsize(video_path)
#                         if file_size < 10000:
#                             raise ValueError(f"Video file too small: {file_size} bytes")

#                         test_clip = VideoFileClip(video_path)
#                         test_duration = test_clip.duration
#                         test_w = test_clip.w
#                         test_h = test_clip.h
#                         test_clip.close()

#                         if test_duration <= 0:
#                             raise ValueError(f"Video has zero duration")
#                         if test_w <= 0 or test_h <= 0:
#                             raise ValueError(f"Invalid dimensions: {test_w}x{test_h}")

#                         subclip_dur = min(dur, media["duration"])

#                         cmd = [
#                             'ffmpeg', '-y',
#                             '-i', video_path,
#                             '-t', str(round(subclip_dur, 4)),
#                             '-vf', f'fps=30,scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease,'
#                                    f'pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2',
#                             '-c:v', video_codec, '-preset', preset_val,
#                             '-r', '30', '-vsync', 'cfr',
#                             '-avoid_negative_ts', 'make_zero',
#                             '-g', '30', '-bf', '0',
#                         ]
#                         if media.get("mute", False):
#                             cmd.extend(['-an'])
#                         else:
#                             cmd.extend(['-c:a', 'aac', '-b:a', '192k', '-ar', '44100'])
#                         cmd.extend(['-pix_fmt', 'yuv420p', '-movflags', 'faststart', clip_output])

#                         result = subprocess.run(cmd, capture_output=True, text=True)
#                         if result.returncode != 0:
#                             raise RuntimeError(f"FFmpeg video encoding failed: {result.stderr[-500:]}")

#                         clip_created = True
#                         logging.info(f"🎥 Video clip {item_idx + 1}: {os.path.basename(video_path)} ({dur:.2f}s)")

#                     else:
#                         from gpu_kenburns_simple import get_secondary_effect, get_motion_speed

#                         _primary_effect  = effect_assignment["primary_effect"]
#                         _video_seed      = effect_assignment["video_seed"]
#                         _global_clip_idx = effect_assignment["global_clip_idx"]

#                         if item_idx == 0:
#                             _chosen_effect = _primary_effect
#                         else:
#                             _chosen_effect = get_secondary_effect(
#                                 primary_effect=_primary_effect,
#                                 item_idx=item_idx,
#                                 global_clip_idx=_global_clip_idx,
#                                 video_seed=_video_seed,
#                             )

#                         _speed = get_motion_speed(
#                             duration=dur,
#                             segment_text=segment.get("text", ""),
#                             effect_type=_chosen_effect,
#                             segment_index=_global_clip_idx,
#                             video_seed=_video_seed,
#                         )

#                         _clip_output = clip_output
#                         _image_path  = media["path"]
#                         _freeze = head_freeze_sec if item_idx == 0 else 0.0

#                         def _render_commentary():
#                             return create_image_video_with_effects(
#                                 image_path=_image_path,
#                                 duration=dur,
#                                 canvas_w=canvas_w,
#                                 canvas_h=canvas_h,
#                                 output_path=_clip_output,
#                                 mode=mode,
#                                 index=_global_clip_idx,
#                                 kenburns_enabled=CURRENT_KENBURNS_ENABLED,
#                                 storyboard_enabled=CURRENT_STORYBOARD_ENABLED,
#                                 fps=30,
#                                 effect_override=_chosen_effect,
#                                 speed_override=_speed,
#                                 segment_text=segment.get("text", ""),
#                                 head_freeze_sec=_freeze,
#                             )

#                         _gpu_acquired = False
#                         try:
#                             logging.info(f"🔒 Seg {i} item {item_idx}: acquiring gpu_semaphore...")
#                             gpu_semaphore.acquire()
#                             _gpu_acquired = True
#                             logging.info(f"🟢 Seg {i} item {item_idx}: gpu_semaphore acquired")

#                             render_result, render_error = _gpu_render_with_timeout(_render_commentary, timeout=600)

#                             if render_error is not None:
#                                 raise render_error

#                             clip_output = render_result
#                             clip_created = True
#                             logging.info(f"🖼️ Image clip {item_idx + 1}: {os.path.basename(media['path'])} ({dur:.2f}s)")

#                         finally:
#                             if _gpu_acquired:
#                                 gpu_semaphore.release()
#                                 logging.info(f"🔓 Seg {i} item {item_idx}: gpu_semaphore released")

#                 except Exception as e:
#                     corrupt_clips_count += 1
#                     logging.error(f"❌ Clip {item_idx + 1}/{len(selected_media)} FAILED: {e}")
#                     logging.warning(f"🎬 Replacing with black frame placeholder ({dur:.2f}s)")

#                     try:
#                         cmd = [
#                             'ffmpeg', '-y', '-f', 'lavfi',
#                             '-i', f'color=c=black:s={canvas_w}x{canvas_h}:d={round(dur, 4)}:r=30',
#                             '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
#                             '-preset', 'ultrafast', clip_output
#                         ]
#                         result = subprocess.run(cmd, capture_output=True, text=True)
#                         if result.returncode != 0:
#                             raise RuntimeError("FFmpeg black frame failed")
#                         clip_created = True
#                         logging.info(f"✅ Black frame replacement created for clip {item_idx + 1}")

#                     except Exception as fallback_err:
#                         logging.error(f"💀 Black frame fallback also failed: {fallback_err}")
#                         try:
#                             import numpy as np
#                             black_array = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 10
#                             frame_bytes = black_array.tobytes()
#                             total_frames = int(dur * 30)
#                             cmd = [
#                                 'ffmpeg', '-y',
#                                 '-f', 'rawvideo', '-vcodec', 'rawvideo',
#                                 '-s', f'{canvas_w}x{canvas_h}',
#                                 '-pix_fmt', 'rgb24', '-r', '30',
#                                 '-i', '-',
#                                 '-t', str(round(dur, 4)),
#                                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
#                                 '-preset', 'ultrafast', clip_output
#                             ]
#                             proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
#                                                     stdout=subprocess.DEVNULL,
#                                                     stderr=subprocess.DEVNULL)
#                             for _ in range(total_frames):
#                                 proc.stdin.write(frame_bytes)
#                             proc.stdin.close()
#                             proc.wait()
#                             if proc.returncode == 0:
#                                 clip_created = True
#                                 logging.info(f"✅ Numpy emergency black clip created for clip {item_idx + 1}")
#                             else:
#                                 clip_created = False
#                         except Exception as emergency_err:
#                             logging.critical(f"💥 All fallbacks failed for clip {item_idx + 1}: {emergency_err}")
#                             clip_created = False

#                 if clip_created and os.path.exists(clip_output):
#                     processed_clips.append(clip_output)
#                 else:
#                     logging.critical(f"💥 Could not create ANY clip for item {item_idx + 1}")

#             if not processed_clips:
#                 logging.error(f"❌ No valid clips for segment {i} - ALL clips failed")
#                 raise RuntimeError(f"All clips failed for segment {i}")

#             if corrupt_clips_count > 0:
#                 logging.warning(f"⚠️ Segment {i}: {corrupt_clips_count}/{len(selected_media)} clips replaced with black frames")

#             if CURRENT_STORYBOARD_ENABLED and len(processed_clips) > 1:
#                 inputs = []
#                 for cp in processed_clips:
#                     inputs.extend(['-i', cp])

#                 durations = []
#                 for cp in processed_clips:
#                     p = subprocess.run(
#                         ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
#                          '-of', 'default=noprint_wrappers=1:nokey=1', cp],
#                         capture_output=True, text=True
#                     )
#                     durations.append(float(p.stdout.strip()))

#                 transition_duration = 0.5
#                 filter_complex = ""
#                 offset = 0.0

#                 for idx in range(len(processed_clips) - 1):
#                     if idx == 0:
#                         offset = durations[0] - transition_duration
#                         filter_complex += (
#                             f"[0:v][1:v]xfade=transition=fade:"
#                             f"duration={transition_duration}:"
#                             f"offset={round(offset, 4)}[v{idx}];"
#                         )
#                     else:
#                         offset += durations[idx] - transition_duration
#                         filter_complex += (
#                             f"[v{idx-1}][{idx+1}:v]xfade=transition=fade:"
#                             f"duration={transition_duration}:"
#                             f"offset={round(offset, 4)}[v{idx}];"
#                         )

#                 final_label = f"[v{len(processed_clips) - 2}]" if len(processed_clips) > 2 else "[v0]"
#                 cmd = [
#                     'ffmpeg', '-y',
#                     *inputs,
#                     '-filter_complex', filter_complex,
#                     '-map', final_label,
#                     '-c:v', video_codec, '-preset', preset_val,
#                     '-t', str(round(duration_needed, 4)),
#                     '-pix_fmt', 'yuv420p',
#                     '-movflags', 'faststart',
#                     temp_clip_path
#                 ]
#             elif len(processed_clips) == 1:
#                 cmd = [
#                     'ffmpeg', '-y',
#                     '-i', processed_clips[0],
#                     '-c:v', video_codec, '-preset', preset_val,
#                     '-t', str(round(duration_needed, 4)),
#                     '-pix_fmt', 'yuv420p',
#                     '-movflags', 'faststart',
#                     temp_clip_path
#                 ]
#             else:
#                 concat_list_path = os.path.join(chunk_folder, f"concat_{uuid.uuid4()}.txt")
#                 with open(concat_list_path, 'w') as f:
#                     for cp in processed_clips:
#                         f.write(f"file '{os.path.abspath(cp)}'\n")
#                 cmd = [
#                     'ffmpeg', '-y',
#                     '-f', 'concat', '-safe', '0',
#                     '-i', concat_list_path,
#                     '-c:v', video_codec, '-preset', preset_val,
#                     '-t', str(round(duration_needed, 4)),
#                     '-pix_fmt', 'yuv420p',
#                     '-movflags', 'faststart',
#                     temp_clip_path
#                 ]

#             result = subprocess.run(cmd, capture_output=True, text=True)
#             if result.returncode != 0:
#                 logging.error(f"FFmpeg concat STDERR: {result.stderr[-1000:]}")
#                 return None

#             for cp in processed_clips:
#                 if os.path.exists(cp):
#                     os.remove(cp)

#             if talking_heads_mode:
#                 # Content-only finalize: re-encode temp_clip_path to lock
#                 # duration exactly to narration audio, no avatar/border/bg.
#                 cmd = [
#                     'ffmpeg', '-y',
#                     '-stream_loop', '-1', '-i', temp_clip_path,
#                     '-i', segment_audio_path,
#                     '-filter_complex',
#                     f"[0:v]trim=duration={round(duration_needed,4)},"
#                     f"setpts=PTS-STARTPTS,format=yuv420p[v]",
#                     '-map', '[v]',
#                     '-map', '1:a:0',
#                     '-c:v', video_codec, '-preset', preset_val,
#                     '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
#                     '-r', '30', '-vsync', 'cfr',
#                     '-shortest',
#                     '-pix_fmt', 'yuv420p',
#                     '-movflags', 'faststart',
#                     chunk_path
#                 ]
#                 result = subprocess.run(cmd, capture_output=True, text=True)
#                 if os.path.exists(temp_clip_path):
#                     os.remove(temp_clip_path)

#                 if result.returncode == 0 and os.path.exists(chunk_path):
#                     logging.info(f"✅ Fallback narration segment {i} (talking-heads, content-only): {chunk_path}")
#                     return chunk_path
#                 else:
#                     logging.error(f"❌ Talking-heads fallback finalize failed for segment {i}: {result.stderr[-300:]}")
#                     return None

#             final_overlayed_path = overlay_avatar_with_audio(
#                 resized_clip=temp_clip_path,
#                 avatar_path="avatar.mp4",
#                 audio_path=segment_audio_path,
#                 output_path=chunk_path,
#                 avatar_active=avatar_active
#             )

#             if os.path.exists(temp_clip_path):
#                 os.remove(temp_clip_path)

#             return final_overlayed_path

#         # ============================================================
#         # COMMENTARY WITHOUT NARRATION
#         # ============================================================
#         if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is None and segment_tag:
#             tagged_media = [
#                 m for m in media_pool
#                 if m["character"] == character_name and m.get("tag") == segment_tag
#             ]
#             if tagged_media:
#                 selected = tagged_media[0]
#                 if selected["type"] == "video":
#                     overlay_avatar_no_audio(
#                         selected["path"],
#                         f"channel_media/{avatar}",
#                         f"channel_media/{background_video}",
#                         chunk_path,
#                         segment_tag,
#                         media_pool=media_pool,
#                         character_name=character_name,
#                         talking_heads_mode=talking_heads_mode,
#                     )
#                     return chunk_path

#         # ============================================================
#         # NON-COMMENTARY WITH TAG BUT NO AUDIO
#         # ============================================================
#         if segment_audio_path is None and segment_tag:
#             tagged_media = [
#                 m for m in media_pool
#                 if m["character"] == character_name and m.get("tag") == segment_tag
#             ]
#             if tagged_media:
#                 selected = tagged_media[0]
#                 if selected["type"] == "video":
#                     full_video = VideoFileClip(selected["path"])
#                     duration = full_video.duration
#                     audio = full_video.audio.set_fps(44100)

#                     logging.info(f"⏱️ Duration {duration:.2f}s — encoding directly without overlay.")
#                     final_clip = full_video.set_audio(audio.set_duration(duration))
#                     final_clip.write_videofile(
#                         chunk_path,
#                         codec=video_codec,
#                         audio_codec="aac",
#                         audio_fps=44100,
#                         threads=4,
#                         fps=30,
#                         ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", preset_val,
#                                        "-movflags", "faststart", "-shortest"]
#                     )
#                     full_video.close()
#                     return chunk_path

#         # ============================================================
#         # DEFAULT: NORMAL AUDIO-BASED SEGMENT
#         # ============================================================
#         if not segment_audio_path or not os.path.exists(segment_audio_path):
#             logging.error(f"❌ No audio file for segment {i}")
#             return None

#         segment_audio = AudioFileClip(segment_audio_path, fps=44100)
#         duration_needed = segment_audio.duration
#         segment_audio = segment_audio.set_fps(44100).volumex(2)

#         selected_media = find_best_media_combination(
#             media_pool, segment, character_name,
#             duration_needed, used_media_dict,
#             media_lock, mode, i, script_context, prev_segments, whisk_shared_state
#         )

#         if not selected_media:
#             logging.error(f"❌ No media selected for segment {i}")
#             return None

#         processed_clips = []
#         corrupt_clips_count = 0

#         for item_idx, item in enumerate(selected_media):
#             media = item["media"]
#             dur = item["duration"]
#             clip_output = os.path.join(chunk_folder, f"clip_{i}_{item_idx}_{uuid.uuid4()}.mp4")
#             clip_created = False

#             try:
#                 if media["type"] == "video":
#                     video_path = media["path"]

#                     if not os.path.exists(video_path):
#                         raise ValueError(f"Video file missing: {video_path}")
#                     file_size = os.path.getsize(video_path)
#                     if file_size < 10000:
#                         raise ValueError(f"Video file too small: {file_size} bytes")

#                     test_clip = VideoFileClip(video_path)
#                     test_duration = test_clip.duration
#                     test_w = test_clip.w
#                     test_h = test_clip.h
#                     test_clip.close()

#                     if test_duration <= 0:
#                         raise ValueError(f"Video has zero duration")
#                     if test_w <= 0 or test_h <= 0:
#                         raise ValueError(f"Invalid dimensions: {test_w}x{test_h}")

#                     subclip_dur = min(dur, media["duration"])

#                     cmd = [
#                         'ffmpeg', '-y',
#                         '-i', video_path,
#                         '-t', str(round(subclip_dur, 4)),
#                         '-vf', f'fps=30,scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease,'
#                                f'pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2',
#                         '-c:v', video_codec, '-preset', preset_val,
#                         '-r', '30', '-vsync', 'cfr',
#                         '-avoid_negative_ts', 'make_zero',
#                         '-g', '30', '-bf', '0',
#                     ]
#                     if media.get("mute", False):
#                         cmd.extend(['-an'])
#                     else:
#                         cmd.extend(['-c:a', 'aac', '-b:a', '192k', '-ar', '44100'])
#                     cmd.extend(['-pix_fmt', 'yuv420p', '-movflags', 'faststart', clip_output])

#                     result = subprocess.run(cmd, capture_output=True, text=True)
#                     if result.returncode != 0:
#                         raise RuntimeError(f"FFmpeg video encoding failed: {result.stderr[-500:]}")

#                     clip_created = True
#                     logging.info(f"🎥 Video clip {item_idx + 1}: {os.path.basename(video_path)} ({dur:.2f}s)")

#                 else:
#                     from gpu_kenburns_simple import get_secondary_effect, get_motion_speed

#                     _primary_effect  = effect_assignment["primary_effect"]
#                     _video_seed      = effect_assignment["video_seed"]
#                     _global_clip_idx = effect_assignment["global_clip_idx"]

#                     if item_idx == 0:
#                         _chosen_effect = _primary_effect
#                     else:
#                         _chosen_effect = get_secondary_effect(
#                             primary_effect=_primary_effect,
#                             item_idx=item_idx,
#                             global_clip_idx=_global_clip_idx,
#                             video_seed=_video_seed,
#                         )

#                     _speed = get_motion_speed(
#                         duration=dur,
#                         segment_text=segment.get("text", ""),
#                         effect_type=_chosen_effect,
#                         segment_index=_global_clip_idx,
#                         video_seed=_video_seed,
#                     )

#                     _clip_output = clip_output
#                     _image_path  = media["path"]
#                     _freeze = head_freeze_sec if item_idx == 0 else 0.0

#                     def _render_default():
#                         return create_image_video_with_effects(
#                             image_path=_image_path,
#                             duration=dur,
#                             canvas_w=canvas_w,
#                             canvas_h=canvas_h,
#                             output_path=_clip_output,
#                             mode=mode,
#                             index=_global_clip_idx,
#                             kenburns_enabled=CURRENT_KENBURNS_ENABLED,
#                             storyboard_enabled=CURRENT_STORYBOARD_ENABLED,
#                             fps=30,
#                             effect_override=_chosen_effect,
#                             speed_override=_speed,
#                             segment_text=segment.get("text", ""),
#                             head_freeze_sec=_freeze,
#                         )

#                     _gpu_acquired = False
#                     try:
#                         logging.info(f"🔒 Seg {i} item {item_idx}: acquiring gpu_semaphore...")
#                         gpu_semaphore.acquire()
#                         _gpu_acquired = True
#                         logging.info(f"🟢 Seg {i} item {item_idx}: gpu_semaphore acquired")

#                         render_result, render_error = _gpu_render_with_timeout(_render_default, timeout=600)

#                         if render_error is not None:
#                             raise render_error

#                         clip_output = render_result
#                         clip_created = True
#                         logging.info(f"🖼️ Image clip {item_idx + 1}: {os.path.basename(media['path'])} ({dur:.2f}s)")

#                     finally:
#                         if _gpu_acquired:
#                             gpu_semaphore.release()
#                             logging.info(f"🔓 Seg {i} item {item_idx}: gpu_semaphore released")

#             except Exception as e:
#                 corrupt_clips_count += 1
#                 logging.error(f"❌ Clip {item_idx + 1}/{len(selected_media)} FAILED: {e}")
#                 logging.warning(f"🎬 Replacing with black frame placeholder ({dur:.2f}s)")

#                 try:
#                     cmd = [
#                         'ffmpeg', '-y', '-f', 'lavfi',
#                         '-i', f'color=c=black:s={canvas_w}x{canvas_h}:d={round(dur, 4)}:r=30',
#                         '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
#                         '-preset', 'ultrafast', clip_output
#                     ]
#                     result = subprocess.run(cmd, capture_output=True, text=True)
#                     if result.returncode != 0:
#                         raise RuntimeError("FFmpeg black frame failed")
#                     clip_created = True
#                     logging.info(f"✅ Black frame replacement created for clip {item_idx + 1}")

#                 except Exception as fallback_err:
#                     logging.error(f"💀 Black frame fallback also failed: {fallback_err}")
#                     try:
#                         import numpy as np
#                         black_array = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 10
#                         frame_bytes = black_array.tobytes()
#                         total_frames = int(dur * 30)
#                         cmd = [
#                             'ffmpeg', '-y',
#                             '-f', 'rawvideo', '-vcodec', 'rawvideo',
#                             '-s', f'{canvas_w}x{canvas_h}',
#                             '-pix_fmt', 'rgb24', '-r', '30',
#                             '-i', '-',
#                             '-t', str(round(dur, 4)),
#                             '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
#                             '-preset', 'ultrafast', clip_output
#                         ]
#                         proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
#                                                 stdout=subprocess.DEVNULL,
#                                                 stderr=subprocess.DEVNULL)
#                         for _ in range(total_frames):
#                             proc.stdin.write(frame_bytes)
#                         proc.stdin.close()
#                         proc.wait()
#                         if proc.returncode == 0:
#                             clip_created = True
#                             logging.info(f"✅ Numpy emergency black clip created for clip {item_idx + 1}")
#                         else:
#                             clip_created = False
#                     except Exception as emergency_err:
#                         logging.critical(f"💥 All fallbacks failed for clip {item_idx + 1}: {emergency_err}")
#                         clip_created = False

#             if clip_created and os.path.exists(clip_output):
#                 processed_clips.append(clip_output)
#             else:
#                 logging.critical(f"💥 Could not create ANY clip for item {item_idx + 1}")

#         if not processed_clips:
#             logging.error(f"❌ No valid clips for segment {i} - ALL clips failed")
#             raise RuntimeError(f"All clips failed for segment {i}")

#         if corrupt_clips_count > 0:
#             logging.warning(f"⚠️ Segment {i}: {corrupt_clips_count}/{len(selected_media)} clips replaced with black frames")

#         if CURRENT_STORYBOARD_ENABLED and len(processed_clips) > 1:
#             inputs = []
#             for cp in processed_clips:
#                 inputs.extend(['-i', cp])
#             inputs.extend(['-i', segment_audio_path])

#             durations = []
#             for cp in processed_clips:
#                 p = subprocess.run(
#                     ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
#                      '-of', 'default=noprint_wrappers=1:nokey=1', cp],
#                     capture_output=True, text=True
#                 )
#                 durations.append(float(p.stdout.strip()))

#             transition_duration = 0.5
#             filter_complex = ""
#             offset = 0.0

#             for idx in range(len(processed_clips) - 1):
#                 if idx == 0:
#                     offset = durations[0] - transition_duration
#                     filter_complex += (
#                         f"[0:v][1:v]xfade=transition=fade:"
#                         f"duration={transition_duration}:"
#                         f"offset={round(offset, 4)}[v{idx}];"
#                     )
#                 else:
#                     offset += durations[idx] - transition_duration
#                     filter_complex += (
#                         f"[v{idx-1}][{idx+1}:v]xfade=transition=fade:"
#                         f"duration={transition_duration}:"
#                         f"offset={round(offset, 4)}[v{idx}];"
#                     )

#             final_label = f"[v{len(processed_clips) - 2}]" if len(processed_clips) > 2 else "[v0]"
#             n_video_inputs = len(processed_clips)
#             cmd = [
#                 'ffmpeg', '-y',
#                 *inputs,
#                 '-filter_complex', filter_complex,
#                 '-map', final_label,
#                 '-map', f'{n_video_inputs}:a:0',
#                 '-c:v', video_codec, '-preset', preset_val,
#                 '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
#                 '-t', str(round(duration_needed, 4)),
#                 '-pix_fmt', 'yuv420p',
#                 '-movflags', 'faststart',
#                 chunk_path
#             ]

#         elif len(processed_clips) == 1:
#             cmd = [
#                 'ffmpeg', '-y',
#                 '-i', processed_clips[0],
#                 '-i', segment_audio_path,
#                 '-c:v', video_codec, '-preset', preset_val,
#                 '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
#                 '-map', '0:v:0', '-map', '1:a:0',
#                 '-t', str(round(duration_needed, 4)),
#                 '-pix_fmt', 'yuv420p',
#                 '-movflags', 'faststart',
#                 chunk_path
#             ]

#         else:
#             concat_list_path = os.path.join(chunk_folder, f"concat_{uuid.uuid4()}.txt")
#             with open(concat_list_path, 'w') as f:
#                 for cp in processed_clips:
#                     f.write(f"file '{os.path.abspath(cp)}'\n")
#             cmd = [
#                 'ffmpeg', '-y',
#                 '-f', 'concat', '-safe', '0',
#                 '-i', concat_list_path,
#                 '-i', segment_audio_path,
#                 '-c:v', video_codec, '-preset', preset_val,
#                 '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
#                 '-map', '0:v:0', '-map', '1:a:0',
#                 '-t', str(round(duration_needed, 4)),
#                 '-pix_fmt', 'yuv420p',
#                 '-movflags', 'faststart',
#                 chunk_path
#             ]

#         logging.debug(f"Final encode: {' '.join(cmd[:10])}...")
#         result = subprocess.run(cmd, capture_output=True, text=True)
#         if result.returncode != 0:
#             logging.error(f"FFmpeg final encode STDERR: {result.stderr[-1000:]}")
#             return None

#         for cp in processed_clips:
#             if os.path.exists(cp):
#                 os.remove(cp)
#         if 'concat_list_path' in locals() and os.path.exists(concat_list_path):
#             os.remove(concat_list_path)

#         segment_audio.close()

#         logging.info(f"✅ Segment {i} completed: {chunk_path}")
#         return chunk_path

#     except Exception as e:
#         logging.error(f"❌ Segment {i} crashed: {e}", exc_info=True)
#         return None

#     finally:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()


def encode_video_segment(segment_info):
    (
        segment,
        media_pool,
        segment_audio_path,
        section_dir,
        chunk_folder,
        i,
        used_media_dict,
        media_lock,
        mode,
        avatar,
        background_video,
        segments,
        script_context,
        orientation,
        storyboard,
        kenburns,
        avatar_active,
        gpu_semaphore,
        nvenc_semaphore,
        effect_assignment,
        whisk_shared_state,
        head_freeze_sec,
        talking_heads_mode,
        intro_teaser_config,
        narration_images_config,
        section_type,
    ) = segment_info[:26]

    storyboard_semaphore = segment_info[26] if len(segment_info) > 26 else _storyboard_sem_fallback
    layout_config = segment_info[27] if len(segment_info) > 27 else None
    _register_storyboard_semaphore(storyboard_semaphore)

    from helper.teaser_backend import (
        is_intro_tag,
        resolve_narration_image,
        download_image_to_local,
        apply_intro_teaser_overlay,
        narration_override_mode,
    )

    global CURRENT_WIDTH, CURRENT_HEIGHT, CURRENT_ORIENTATION
    global CURRENT_STORYBOARD_ENABLED, CURRENT_KENBURNS_ENABLED

    CURRENT_ORIENTATION = orientation
    CURRENT_STORYBOARD_ENABLED = bool(storyboard)
    CURRENT_KENBURNS_ENABLED = bool(kenburns)

    if orientation == "portrait":
        CURRENT_WIDTH, CURRENT_HEIGHT = 1080, 1920
    else:
        CURRENT_WIDTH, CURRENT_HEIGHT = 1920, 1080

    canvas_w = CURRENT_WIDTH
    canvas_h = CURRENT_HEIGHT

    chunk_filename = f"chunk_{i}_{uuid.uuid4()}.mp4"
    chunk_path = os.path.join(chunk_folder, chunk_filename)

    prev_segments = get_prev_two_context_text(segments, i)
    logging.info(f"🔹 Processing segment {i}: {segment['text'][:50]}...")

    use_nvenc = _segment_use_nvenc()
    video_codec = "h264_nvenc" if use_nvenc else "libx264"
    preset_val = "p3" if use_nvenc else "medium"

    def _run_ffmpeg(cmd, label):
        return _ffmpeg_run(
            cmd,
            timeout=FFMPEG_SEGMENT_TIMEOUT,
            label=f"segment {i}: {label}",
            nvenc_semaphore=nvenc_semaphore,
            uses_nvenc=use_nvenc,
        )

    segment_audio = None
    override_mode = narration_override_mode(narration_images_config)
    has_segment_tag = bool((segment.get("tag") or "").strip())

    def _full_avatar_skip_narration_media() -> bool:
        return bool(
            talking_heads_mode
            and isinstance(layout_config, dict)
            and (layout_config.get("layout_id") or "") == "full_avatar"
        )

    def _encode_full_avatar_narration_placeholder(duration_needed: float):
        """Black video + narration audio only — no Pexels / AI / downloads."""
        if duration_needed <= 0:
            return None
        if not segment_audio_path or not os.path.exists(segment_audio_path):
            logging.warning(
                f"⚠️ Segment {i}: full_avatar skip needs narration audio — none found"
            )
            return None
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"color=c=black:s={canvas_w}x{canvas_h}:d={duration_needed:.4f}:r=30",
            "-i", segment_audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", video_codec, "-preset", preset_val, *_segment_rate_args(video_codec),
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-r", "30", "-vsync", "cfr",
            "-shortest",
            "-avoid_negative_ts", "make_zero",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            chunk_path,
        ]
        r = _run_ffmpeg(cmd, "full_avatar narration placeholder")
        if not r or r.returncode != 0 or not os.path.exists(chunk_path):
            return None
        logging.info(
            f"⚡ Segment {i}: full_avatar — skipped narration media "
            f"({duration_needed:.2f}s placeholder)"
        )
        return chunk_path

    def _maybe_apply_intro_teaser(raw_path):
        if not raw_path or not os.path.exists(raw_path):
            return raw_path
        if not is_intro_tag(segment.get("tag")):
            return raw_path

        teaser_out = os.path.join(chunk_folder, f"intro_teaser_{i}_{uuid.uuid4()}.mp4")
        logging.info(f"📺 Applying intro teaser overlay to segment {i}")
        try:
            apply_intro_teaser_overlay(
                input_path=raw_path,
                output_path=teaser_out,
                background_path=None,
                teaser_config=intro_teaser_config,
            )
        except Exception as exc:
            logging.error(f"❌ Intro teaser overlay failed for segment {i}: {exc}")
            if os.path.isfile(teaser_out):
                try:
                    os.remove(teaser_out)
                except OSError:
                    pass
            return raw_path
        return teaser_out

    def _render_forced_narration_image(duration_needed):
        forced_url = resolve_narration_image(narration_images_config, section_type, i)
        if not forced_url:
            return None

        local_img = download_image_to_local(
            forced_url,
            os.path.join(section_dir, f"forced_{section_type}_{i}.jpg"),
        )
        if not local_img:
            logging.warning(f"⚠️ Could not resolve forced narration image for segment {i}")
            return None

        # Global narration image stays fixed on screen — blur/letterbox only.
        force_stationary = override_mode == "global"
        kb_enabled = False if force_stationary else CURRENT_KENBURNS_ENABLED
        sb_enabled = False if force_stationary else CURRENT_STORYBOARD_ENABLED

        if force_stationary:
            logging.info(
                f"📌 Segment {i}: global narration image — static "
                f"(Ken Burns/storyboard disabled; blur/layout still applied)"
            )

        _primary_effect = effect_assignment["primary_effect"]
        _video_seed = effect_assignment["video_seed"]
        _global_clip_idx = effect_assignment["global_clip_idx"]
        clip_output = os.path.join(chunk_folder, f"forced_{i}_{uuid.uuid4()}.mp4")
        _speed = None

        def _render():
            return create_image_video_with_effects(
                image_path=local_img,
                duration=duration_needed,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                output_path=clip_output,
                mode=mode,
                index=_global_clip_idx,
                kenburns_enabled=kb_enabled,
                storyboard_enabled=sb_enabled,
                fps=30,
                effect_override=_primary_effect,
                speed_override=_speed,
                segment_text=segment.get("text", ""),
                head_freeze_sec=head_freeze_sec,
                video_seed=_video_seed,
                nvenc_semaphore=nvenc_semaphore,
            )

        _kb_locked = _kenburns_needs_gpu_lock() if kb_enabled else False
        if _kb_locked:
            _acquire_gpu_lock(gpu_semaphore, f"Segment {i} forced narration")
        elif kb_enabled:
            logging.info(f"🎬 Segment {i}: Ken Burns starting (ffmpeg)...")
        try:
            if kb_enabled:
                from gpu_kenburns_simple import get_motion_speed

                _speed = get_motion_speed(
                    duration=duration_needed,
                    segment_text=segment.get("text", ""),
                    effect_type=_primary_effect,
                    segment_index=_global_clip_idx,
                    video_seed=_video_seed,
                )
            return _render()
        finally:
            if _kb_locked:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gpu_semaphore.release()
                logging.info(f"🔓 Segment {i}: GPU lock released")

    def _encode_forced_narration(duration_needed):
        forced_src = _render_forced_narration_image(duration_needed)
        if not forced_src:
            if override_mode == "global":
                logging.error(
                    f"❌ GLOBAL narration image failed for segment {i} | "
                    f"url={narration_images_config.get('globalImage')!r} | "
                    f"NO character fallback allowed"
                )
            elif override_mode == "per_segment":
                logging.error(
                    f"❌ Per-segment narration image failed for segment {i} | "
                    f"section={section_type} | NO character fallback allowed"
                )
            return None

        out = _encode_narration_to_audio_length(
            video_path=forced_src,
            segment_audio_path=segment_audio_path,
            duration_needed=duration_needed,
            chunk_path=chunk_path,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            video_codec=video_codec,
            preset_val=preset_val,
            mute_source=True,
            nvenc_semaphore=nvenc_semaphore,
            segment_index=i,
        )
        return _maybe_apply_intro_teaser(out)

    try:
        character_name = segment["character"]
        segment_tag = segment.get("tag")
        segment_text = (segment.get("text") or "").strip()
        is_clip_segment = segment_text.startswith("[CLIP:") and segment_text.endswith("]")

        # TEXT-ONLY
        if character_name.lower() == "text":
            if segment_audio_path and os.path.exists(segment_audio_path):
                segment_audio = AudioFileClip(segment_audio_path, fps=44100)
                duration_needed = segment_audio.duration
                segment_audio = segment_audio.set_fps(44100).volumex(2)
            else:
                words = len(segment["text"].split())
                duration_needed = max(3.0, words * 0.35)
                segment_audio = None

            text_clip = create_text_segment_with_lines(
                text=segment["text"],
                duration=duration_needed,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                bg_color=segment.get("text_bg_color", (178, 34, 34)),
                text_color=segment.get("text_color", "white"),
                fontsize=segment.get("text_fontsize", None),
                fade_duration=segment.get("text_fade_duration", 0.8),
            )

            if segment_audio:
                text_clip = text_clip.set_audio(segment_audio)

            text_clip.write_videofile(
                chunk_path,
                codec=video_codec,
                audio_codec="aac" if segment_audio else None,
                audio_fps=44100 if segment_audio else None,
                threads=4,
                fps=30,
                preset=preset_val,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
            )
            text_clip.close()
            if segment_audio:
                segment_audio.close()
            return _maybe_apply_intro_teaser(chunk_path)

        # CLIP SEGMENT
        if is_clip_segment:
            if not segment_tag:
                logging.error(f"❌ CLIP segment {i}: no tag")
                return None

            tagged_media = [
                m for m in media_pool
                if m["character"] == character_name and m.get("tag") == segment_tag
            ]
            if not tagged_media or tagged_media[0]["type"] != "video":
                logging.error(f"❌ CLIP segment {i}: tagged video not found")
                return None

            selected = tagged_media[0]
            out = _encode_clip_segment(
                src=selected["path"],
                chunk_path=chunk_path,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                video_codec=video_codec,
                preset_val=preset_val,
                mute=selected.get("mute", False),
                nvenc_semaphore=nvenc_semaphore,
                segment_index=i,
            )
            return _maybe_apply_intro_teaser(out)

        # COMMENTARY WITH NARRATION
        if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is not None:
            segment_audio = AudioFileClip(segment_audio_path, fps=44100)
            duration_needed = segment_audio.duration
            segment_audio.close()
            segment_audio = None

            if _full_avatar_skip_narration_media() and not is_clip_segment:
                placeholder = _encode_full_avatar_narration_placeholder(duration_needed)
                if placeholder:
                    return _maybe_apply_intro_teaser(placeholder)

            if override_mode == "global" and not has_segment_tag:
                return _encode_forced_narration(duration_needed)

            if override_mode == "per_segment" and not has_segment_tag:
                return _encode_forced_narration(duration_needed)


            if segment_tag:
                tagged_media = [
                    m for m in media_pool
                    if m["character"] == character_name and m.get("tag") == segment_tag
                ]
                if tagged_media and tagged_media[0]["type"] == "video":
                    selected = tagged_media[0]
                    out = _encode_narration_to_audio_length(
                        video_path=selected["path"],
                        segment_audio_path=segment_audio_path,
                        duration_needed=duration_needed,
                        chunk_path=chunk_path,
                        canvas_w=canvas_w,
                        canvas_h=canvas_h,
                        video_codec=video_codec,
                        preset_val=preset_val,
                        mute_source=True,
                        nvenc_semaphore=nvenc_semaphore,
                        segment_index=i,
                    )
                    if out:
                        logging.info(f"✅ Tagged narration segment {i}: {out}")
                        return _maybe_apply_intro_teaser(out)

            logging.info(f"🔁 best-media fallback for segment {i}")
            reset_overlay_usage_for_new_clip()

            selected_media = find_best_media_combination(
                media_pool, segment, character_name, duration_needed,
                used_media_dict, media_lock, mode, i,
                script_context, prev_segments, whisk_shared_state,
            )
            if not selected_media:
                return None

            logging.info(
                f"🎞️ Segment {i}: encoding {len(selected_media)} media clip(s) to chunk..."
            )

            processed_clips = []
            for item_idx, item in enumerate(selected_media):
                media = item["media"]
                dur = item["duration"]
                clip_output = os.path.join(chunk_folder, f"clip_{i}_{item_idx}_{uuid.uuid4()}.mp4")

                if media["type"] == "video":
                    subclip_dur = min(dur, media["duration"])
                    cmd = [
                        "ffmpeg", "-y", "-i", media["path"],
                        "-t", str(round(subclip_dur, 4)),
                        "-vf",
                        (
                            f"fps=30,setpts=PTS-STARTPTS,"
                            f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease,"
                            f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2"
                        ),
                        "-c:v", video_codec, "-preset", preset_val, *_segment_rate_args(video_codec),
                        "-r", "30", "-vsync", "cfr",
                        "-avoid_negative_ts", "make_zero",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-an",
                        clip_output,
                    ]
                    r = _run_ffmpeg(cmd, f"clip {item_idx} trim/scale")
                    if not r or r.returncode != 0:
                        continue
                    processed_clips.append(clip_output)
                else:
                    _kb_locked = _kenburns_needs_gpu_lock()
                    if _kb_locked:
                        _acquire_gpu_lock(
                            gpu_semaphore, f"Segment {i} clip {item_idx}"
                        )
                    logging.info(
                        f"🎬 Segment {i} clip {item_idx}: Ken Burns..."
                    )
                    try:
                        from gpu_kenburns_simple import (
                            get_secondary_effect,
                            get_motion_speed,
                        )

                        _primary_effect = effect_assignment["primary_effect"]
                        _video_seed = effect_assignment["video_seed"]
                        _global_clip_idx = effect_assignment["global_clip_idx"]
                        _chosen_effect = _primary_effect if item_idx == 0 else get_secondary_effect(
                            primary_effect=_primary_effect,
                            item_idx=item_idx,
                            global_clip_idx=_global_clip_idx,
                            video_seed=_video_seed,
                        )
                        _speed = get_motion_speed(
                            duration=dur,
                            segment_text=segment.get("text", ""),
                            effect_type=_chosen_effect,
                            segment_index=_global_clip_idx,
                            video_seed=_video_seed,
                        )
                        _freeze = head_freeze_sec if item_idx == 0 else 0.0

                        def _render():
                            return create_image_video_with_effects(
                                image_path=media["path"],
                                duration=dur,
                                canvas_w=canvas_w,
                                canvas_h=canvas_h,
                                output_path=clip_output,
                                mode=mode,
                                index=_global_clip_idx,
                                kenburns_enabled=CURRENT_KENBURNS_ENABLED,
                                storyboard_enabled=CURRENT_STORYBOARD_ENABLED,
                                fps=30,
                                effect_override=_chosen_effect,
                                speed_override=_speed,
                                segment_text=segment.get("text", ""),
                                head_freeze_sec=_freeze,
                                video_seed=_video_seed,
                                nvenc_semaphore=nvenc_semaphore,
                            )

                        render_result = _render()
                        processed_clips.append(render_result)
                    finally:
                        if _kb_locked:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gpu_semaphore.release()
                            logging.info(
                                f"🔓 Segment {i} clip {item_idx}: GPU slot released"
                            )

            if not processed_clips:
                return None

            if len(processed_clips) == 1:
                content_src = processed_clips[0]
            else:
                concat_list_path = os.path.join(chunk_folder, f"concat_{uuid.uuid4()}.txt")
                with open(concat_list_path, "w") as f:
                    for cp in processed_clips:
                        f.write(f"file '{os.path.abspath(cp).replace(os.sep, '/')}'\n")
                content_src = os.path.join(chunk_folder, f"merged_{i}_{uuid.uuid4()}.mp4")
                r = _run_ffmpeg([
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_list_path,
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    content_src,
                ], "intra-segment concat")
                if not r or r.returncode != 0:
                    return None

            logging.info(f"🎞️ Segment {i}: muxing narration audio ({duration_needed:.2f}s)...")
            out = _encode_narration_to_audio_length(
                video_path=content_src,
                segment_audio_path=segment_audio_path,
                duration_needed=duration_needed,
                chunk_path=chunk_path,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                video_codec=video_codec,
                preset_val=preset_val,
                mute_source=True,
                nvenc_semaphore=nvenc_semaphore,
                segment_index=i,
            )

            for cp in processed_clips:
                try:
                    if os.path.exists(cp):
                        os.remove(cp)
                except Exception:
                    pass

            return _maybe_apply_intro_teaser(out)

        # COMMENTARY CLIP WITHOUT NARRATION AUDIO
        if mode in ["commentary", "regenerate_commentary"] and segment_audio_path is None and segment_tag:
            tagged_media = [
                m for m in media_pool
                if m["character"] == character_name and m.get("tag") == segment_tag
            ]
            if tagged_media and tagged_media[0]["type"] == "video":
                selected = tagged_media[0]
                out = _encode_clip_segment(
                    src=selected["path"],
                    chunk_path=chunk_path,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    video_codec=video_codec,
                    preset_val=preset_val,
                    mute=selected.get("mute", False),
                    nvenc_semaphore=nvenc_semaphore,
                    segment_index=i,
                )
                return _maybe_apply_intro_teaser(out)

        # DEFAULT AUDIO-BASED
        if not segment_audio_path or not os.path.exists(segment_audio_path):
            return None

        segment_audio = AudioFileClip(segment_audio_path, fps=44100)
        duration_needed = segment_audio.duration
        segment_audio.close()
        segment_audio = None

        if _full_avatar_skip_narration_media() and not is_clip_segment:
            placeholder = _encode_full_avatar_narration_placeholder(duration_needed)
            if placeholder:
                return _maybe_apply_intro_teaser(placeholder)

        if override_mode == "global" and not has_segment_tag:
            return _encode_forced_narration(duration_needed)
        
        if override_mode == "per_segment" and not has_segment_tag:
            return _encode_forced_narration(duration_needed)
        
     

        selected_media = find_best_media_combination(
            media_pool, segment, character_name,
            duration_needed, used_media_dict,
            media_lock, mode, i, script_context, prev_segments, whisk_shared_state,
        )
        if not selected_media:
            return None

        logging.info(
            f"🎞️ Segment {i}: encoding {len(selected_media)} media clip(s) to chunk..."
        )

        processed_clips = []
        for item_idx, item in enumerate(selected_media):
            media = item["media"]
            dur = item["duration"]
            clip_output = os.path.join(chunk_folder, f"clip_{i}_{item_idx}_{uuid.uuid4()}.mp4")

            if media["type"] == "video":
                subclip_dur = min(dur, media["duration"])
                cmd = [
                    "ffmpeg", "-y", "-i", media["path"],
                    "-t", str(round(subclip_dur, 4)),
                    "-vf",
                    (
                        f"fps=30,setpts=PTS-STARTPTS,"
                        f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease,"
                        f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2"
                    ),
                    "-c:v", video_codec, "-preset", preset_val, *_segment_rate_args(video_codec),
                    "-r", "30", "-vsync", "cfr",
                    "-avoid_negative_ts", "make_zero",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    clip_output,
                ]
                r = _run_ffmpeg(cmd, f"clip {item_idx} trim/scale")
                if r and r.returncode == 0:
                    processed_clips.append(clip_output)
            else:
                _kb_locked = _kenburns_needs_gpu_lock()
                if _kb_locked:
                    _acquire_gpu_lock(gpu_semaphore, f"Segment {i} clip {item_idx}")
                logging.info(f"🎬 Segment {i} clip {item_idx}: Ken Burns...")
                try:
                    from gpu_kenburns_simple import get_secondary_effect, get_motion_speed

                    _primary_effect = effect_assignment["primary_effect"]
                    _video_seed = effect_assignment["video_seed"]
                    _global_clip_idx = effect_assignment["global_clip_idx"]
                    _chosen_effect = _primary_effect if item_idx == 0 else get_secondary_effect(
                        primary_effect=_primary_effect,
                        item_idx=item_idx,
                        global_clip_idx=_global_clip_idx,
                        video_seed=_video_seed,
                    )
                    _speed = get_motion_speed(
                        duration=dur,
                        segment_text=segment.get("text", ""),
                        effect_type=_chosen_effect,
                        segment_index=_global_clip_idx,
                        video_seed=_video_seed,
                    )
                    _freeze = head_freeze_sec if item_idx == 0 else 0.0

                    def _render_default():
                        return create_image_video_with_effects(
                            image_path=media["path"],
                            duration=dur,
                            canvas_w=canvas_w,
                            canvas_h=canvas_h,
                            output_path=clip_output,
                            mode=mode,
                            index=_global_clip_idx,
                            kenburns_enabled=CURRENT_KENBURNS_ENABLED,
                            storyboard_enabled=CURRENT_STORYBOARD_ENABLED,
                            fps=30,
                            effect_override=_chosen_effect,
                            speed_override=_speed,
                            segment_text=segment.get("text", ""),
                            head_freeze_sec=_freeze,
                            video_seed=_video_seed,
                            nvenc_semaphore=nvenc_semaphore,
                        )

                    render_result = _render_default()
                    processed_clips.append(render_result)
                finally:
                    if _kb_locked:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gpu_semaphore.release()
                        logging.info(f"🔓 Segment {i} clip {item_idx}: GPU slot released")

        if not processed_clips:
            return None

        if len(processed_clips) == 1:
            content_src = processed_clips[0]
        else:
            concat_list_path = os.path.join(chunk_folder, f"concat_{uuid.uuid4()}.txt")
            with open(concat_list_path, "w") as f:
                for cp in processed_clips:
                    f.write(f"file '{os.path.abspath(cp).replace(os.sep, '/')}'\n")
            content_src = os.path.join(chunk_folder, f"merged_{i}_{uuid.uuid4()}.mp4")
            r = _run_ffmpeg([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                content_src,
            ], "intra-segment concat")
            if not r or r.returncode != 0:
                return None

        logging.info(f"🎞️ Segment {i}: muxing narration audio ({duration_needed:.2f}s)...")
        out = _encode_narration_to_audio_length(
            video_path=content_src,
            segment_audio_path=segment_audio_path,
            duration_needed=duration_needed,
            chunk_path=chunk_path,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            video_codec=video_codec,
            preset_val=preset_val,
            mute_source=True,
            nvenc_semaphore=nvenc_semaphore,
            segment_index=i,
        )
        return _maybe_apply_intro_teaser(out)

    except Exception as e:
        logging.error(f"❌ Segment {i} crashed: {e}", exc_info=True)
        return None

    finally:
        if segment_audio:
            try:
                segment_audio.close()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _encode_segment_pool_worker(segment_info):
    """Pool worker — returns (segment_index, chunk_path) for progress tracking."""
    segment_index = segment_info[5]
    return segment_index, encode_video_segment_with_retry(segment_info)


def encode_video_segment_with_retry(segment_info, max_retries=5, base_delay=3):
    """
    Retry wrapper for encode_video_segment with exponential backoff.
    
    Fallback hierarchy:
    1. encode_video_segment (up to max_retries attempts)
    2. create_full_segment_fallback (black video with audio)
    3. Dark frame image fallback (absolute last resort if audio-based fallback also fails)
    """
    segment_index = segment_info[5]
    segment = segment_info[0]
    segment_audio_path = segment_info[2]
    chunk_folder = segment_info[4]
    orientation = segment_info[13]

    # Determine canvas dimensions
    if orientation == "portrait":
        canvas_w, canvas_h = 1080, 1920
    else:
        canvas_w, canvas_h = 1920, 1080

    # Determine expected duration
    expected_duration = 5.0
    if segment_audio_path and os.path.exists(segment_audio_path):
        try:
            audio_clip = AudioFileClip(segment_audio_path)
            expected_duration = audio_clip.duration
            audio_clip.close()
            del audio_clip
        except Exception as e:
            logging.warning(f"⚠️ Could not read audio duration: {e}")
            words = len(segment.get('text', '').split())
            expected_duration = max(3.0, words * 0.35)
    else:
        words = len(segment.get('text', '').split())
        expected_duration = max(3.0, words * 0.35)

    last_error = None

    # ─────────────────────────────────────────────────────────
    # TIER 1: Normal encode with retries
    # ─────────────────────────────────────────────────────────
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"🔄 Segment {segment_index} - Attempt {attempt}/{max_retries}")

            result = encode_video_segment(segment_info)

            if result and os.path.exists(result):
                file_size = os.path.getsize(result)

                if file_size > 10000:
                    try:
                        test_duration = _probe_video_duration_safe(result)

                        if test_duration > 0:
                            logging.info(f"✅ Segment {segment_index} completed on attempt {attempt}")
                            return result
                        else:
                            logging.warning(f"⚠️ Invalid duration: {test_duration}s")
                            os.remove(result)
                            last_error = f"Invalid duration: {test_duration}s"
                    except TimeoutError as e:
                        logging.warning(f"⚠️ Validation timed out: {e}")
                        if os.path.exists(result):
                            os.remove(result)
                        last_error = str(e)
                    except Exception as e:
                        logging.warning(f"⚠️ Validation failed: {e}")
                        if os.path.exists(result):
                            os.remove(result)
                        last_error = f"Validation error: {str(e)}"
                else:
                    logging.warning(f"⚠️ Output too small: {file_size} bytes")
                    os.remove(result)
                    last_error = f"File too small: {file_size} bytes"
            else:
                last_error = "encode_video_segment returned None"
                raise RuntimeError(last_error)

        except Exception as e:
            last_error = str(e)
            logging.error(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                delay = base_delay * attempt + random.uniform(0, 1)
                logging.info(f"⏳ Retrying in {delay:.1f}s...")
                time.sleep(delay)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # ─────────────────────────────────────────────────────────
    # TIER 2: Full segment fallback (black video + audio sync)
    # ─────────────────────────────────────────────────────────
    logging.error(f"💀 All {max_retries} retries failed for segment {segment_index}. Last error: {last_error}")
    logging.warning(f"🎬 [TIER 2] Attempting full segment black placeholder with audio...")

    try:
        fallback_path = create_full_segment_fallback(
            duration=expected_duration,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            segment_index=segment_index,
            audio_path=segment_audio_path,
            chunk_folder=chunk_folder,
            reason=last_error
        )

        if fallback_path and os.path.exists(fallback_path):
            logging.info(f"✅ [TIER 2] Full segment fallback created for segment {segment_index}")
            return fallback_path
        else:
            logging.error(f"❌ [TIER 2] Full segment fallback returned invalid path")

    except Exception as fallback_error:
        logging.critical(f"💥 [TIER 2] Full segment fallback crashed: {fallback_error}")
        logging.error(traceback.format_exc())

    # ─────────────────────────────────────────────────────────
    # TIER 3: Dark frame image — absolute last resort
    # Only reached if BOTH encode AND create_full_segment_fallback fail
    # ─────────────────────────────────────────────────────────
    logging.critical(f"🚨 [TIER 3] All fallbacks failed for segment {segment_index}. Creating dark frame as last resort...")

    try:
        from PIL import Image as PILImage

        dark_frame_dir = os.path.join(chunk_folder, f"dark_frame_{segment_index}")
        os.makedirs(dark_frame_dir, exist_ok=True)
        dark_frame_png = os.path.join(dark_frame_dir, "dark_frame.png")
        dark_video_path = os.path.join(chunk_folder, f"dark_frame_segment_{segment_index}.mp4")

        # Create dark image matching canvas dimensions
        dark_img = PILImage.new('RGB', (canvas_w, canvas_h), color=(10, 10, 10))
        dark_img.save(dark_frame_png)

        # Convert dark image to video with or without audio
        if segment_audio_path and os.path.exists(segment_audio_path):
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1', '-i', dark_frame_png,
                '-i', segment_audio_path,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-c:a', 'aac', '-b:a', '192k',
                '-pix_fmt', 'yuv420p',
                '-shortest',
                '-movflags', 'faststart',
                dark_video_path
            ]
        else:
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1', '-i', dark_frame_png,
                '-t', str(expected_duration),
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p',
                '-r', '30',
                '-movflags', 'faststart',
                dark_video_path
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(dark_video_path):
            logging.info(f"✅ [TIER 3] Dark frame video created for segment {segment_index}: {dark_video_path}")
            return dark_video_path
        else:
            logging.critical(f"💥 [TIER 3] Dark frame FFmpeg failed: {result.stderr[-500:]}")

    except Exception as dark_error:
        logging.critical(f"💥 [TIER 3] Dark frame creation crashed: {dark_error}")
        logging.error(traceback.format_exc())

    # Nothing worked at all
    logging.critical(f"🔴 FATAL: Segment {segment_index} could not be created by any method. Returning None.")
    return None



def merge_audios(audio_paths, output_path=None):
    try:
        if not audio_paths or any(path is None or not os.path.exists(path) for path in audio_paths):
            logging.error(f"Invalid or missing audio paths: {audio_paths}")
            return None

        if output_path is None:
            output_path = os.path.join("temp_audio", f"merged_{uuid.uuid4()}.mp3")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        clips = []
        for path in audio_paths:
            try:
                clip = AudioFileClip(path)
                if clip.duration == 0:
                    logging.error(f"Audio file has zero duration: {path}")
                    clip.close()
                    for c in clips: c.close()
                    return None
                clips.append(clip)
                logging.info(f"✅ Loaded clip: {path} | Duration: {clip.duration:.2f}s")
            except Exception as e:
                logging.error(f"❌ Failed to load audio: {path} | Error: {e}")
                for c in clips: c.close()
                return None

        final_audio = concatenate_audioclips(clips)

        # Write the file using a stable codec
        logging.info(f"🎧 Writing merged audio to: {output_path}")
        final_audio.write_audiofile(output_path, codec="libmp3lame")  # Change to libmp3lame

        # Cleanup
        for clip in clips: clip.close()
        final_audio.close()

        # Check output file size
        if not os.path.exists(output_path):
            logging.error(f"❌ Output file was not created: {output_path}")
            return None
        final_size = os.path.getsize(output_path)
        if final_size == 0:
            logging.error(f"⚠️ Output file is 0 bytes: {output_path}")
            return None

        logging.info(f"✅ Merge successful. File size: {final_size} bytes")
        return output_path

    except Exception as e:
        logging.error(f"🔥 Exception in merge_audios: {e}")
        return None



def render_segment_with_ffmpeg_fast(clip, output_path, temp_audio_path):
    try:
        width, height = clip.size
        fps = int(clip.fps or 30)
        has_audio = clip.audio is not None

        if has_audio:
            clip.audio.write_audiofile(temp_audio_path, fps=44100, nbytes=2, codec='aac')

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", "1920x1080",
            # "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-"
        ]

        if has_audio:
            ffmpeg_cmd.extend(["-i", temp_audio_path])

        ffmpeg_cmd.extend([
    "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
    "-rc", "vbr", "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
    "-profile:v", "high", "-pix_fmt", "yuv420p",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-gpu", "0"
])


        if has_audio:
            ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            ffmpeg_cmd.append("-an")

        ffmpeg_cmd.append(output_path)

        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        try:
            for frame in clip.iter_frames(fps=fps, dtype="uint8"):
                process.stdin.write(frame.tobytes())
        except Exception as stream_err:
            logging.error(f"❌ Frame stream error: {stream_err}")
        finally:
            process.stdin.close()
            process.wait()

    except Exception as e:
        logging.error(f"Failed to render segment {output_path}: {e}")
        raise
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
        if clip:
            clip.close()

def stream_segments_to_ffmpeg(final_clips, output_path, max_parallel=8):
    with tempfile.TemporaryDirectory() as tmp_dir:
        concat_txt_path = os.path.join(tmp_dir, "segments.txt")
        segment_paths = []

        for batch_idx in range(0, len(final_clips), max_parallel):
            batch_clips = final_clips[batch_idx:batch_idx + max_parallel]
            futures = []

            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                for i, clip in enumerate(batch_clips):
                    clip_idx = batch_idx + i
                    temp_segment_path = os.path.join(tmp_dir, f"segment_{clip_idx:04}.mp4")
                    temp_audio_path = os.path.join(tmp_dir, f"audio_{clip_idx:04}.wav")
                    future = executor.submit(
                        render_segment_with_ffmpeg_fast,
                        clip,
                        temp_segment_path,
                        temp_audio_path
                    )
                    futures.append((future, temp_segment_path))

                for future, path in futures:
                    try:
                        future.result()
                        if os.path.getsize(path) >= 10000:
                            segment_paths.append(path)
                        else:
                            logging.warning(f"⚠️ Skipping corrupted or too short chunk: {path}")
                    except Exception as e:
                        logging.error(f"Error rendering segment {path}: {e}")
                        raise

        with open(concat_txt_path, "w") as f:
            for path in segment_paths:
                f.write(f"file '{os.path.abspath(path).replace(os.sep, '/')}" + "'\n")

        ffmpeg_cmd = [
    "ffmpeg", "-f", "concat", "-safe", "0",
    "-i", concat_txt_path,
    "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
    "-rc", "vbr", "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
    "-profile:v", "high", "-pix_fmt", "yuv420p",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart",
    "-y", output_path
]


        subprocess.run(ffmpeg_cmd, check=True)

        for clip in final_clips:
            if clip:
                clip.close()


from multiprocessing import Pool, cpu_count
import os
import shutil
import subprocess
import logging


def add_background_music(video_path, background, volume=1):
    """
    Mix background music into a video without re-encoding the video stream.
    Loops background music to match the duration of the main video's audio.

    Args:
        video_path (str): Path to the final video file.
        background (str): Name of the background track, with or without '.mp3'.
        volume (float): Volume level for background music.
    
    Returns:
        str: The final path to the updated video.
    """
    if not background:
        logging.warning("⚠️ No background music provided. Skipping music addition.")
        return video_path

    # Ensure .mp3 extension
    if not background.lower().endswith('.mp3'):
        background += '.mp3'

    # Primary folder
    music_path = os.path.join("background_music", background)

    # Check fallback folder if not found
    if not os.path.isfile(music_path):
        alt_path = os.path.join("bg_static", "music", background)
        if os.path.isfile(alt_path):
            music_path = alt_path
        else:
            logging.warning(f"⚠️ Background music not found in either folder: {background}. Skipping music addition.")
            return video_path

    temp_output = video_path.replace(".mp4", "_with_music.mp4")

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=3",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-y", temp_output
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        shutil.move(temp_output, video_path)
        logging.info(f"🎵 Background music '{background}' mixed successfully from {music_path}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to mix background music: {e}")

    return video_path




# Font mapping for FFmpeg compatibility
FONT_MAPPING = {
    'Montserrat': 'Arial',
    'Roboto': 'Helvetica',
    'Open Sans': 'Arial',
    'Lato': 'Arial',
    'Oswald': 'Impact',
    'Raleway': 'Helvetica',
    'Bebas Neue': 'Impact',
    'Poppins': 'Arial',
    'Playfair Display': 'Georgia',
    'Merriweather': 'Georgia',
    'Trebuchet MS': 'Trebuchet MS',
    'Palatino': 'Palatino',
    'Garamond': 'Garamond',
    'Bookman': 'Bookman',
    'Avant Garde': 'Avant Garde',
    'Arial': 'Arial',
    'Helvetica': 'Helvetica',
    'Impact': 'Impact',
    'Georgia': 'Georgia',
    'Times New Roman': 'Times New Roman',
    'Courier New': 'Courier New',
    'Verdana': 'Verdana',
    'Comic Sans MS': 'Comic Sans MS',
}

def get_safe_font(requested_font):
    """Get FFmpeg-compatible font name"""
    return FONT_MAPPING.get(requested_font, 'Arial')

def hex_to_ass_color(hex_color):
    """
    Convert hex color (#RRGGBB) to ASS color format (&HBBGGRR)
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"

def get_font_weight_value(weight_str):
    """Convert font weight string to ASS bold value"""
    weight_map = {
        'normal': 0,
        'bold': -1,
        'bolder': -1,
        'lighter': 0
    }
    return weight_map.get(weight_str, -1)

def _srt_time_to_ms(time_str):
    """Convert SRT timestamp to milliseconds"""
    try:
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        
        if ',' in parts[2]:
            seconds, milliseconds = parts[2].split(',')
        else:
            seconds, milliseconds = parts[2].split('.')
            
        seconds = int(seconds)
        milliseconds = int(milliseconds)
        
        return hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds
    except Exception as e:
        logging.error(f"Error parsing time {time_str}: {e}")
        return 0


def _ms_to_ass_time(ms):
    """Convert milliseconds to ASS timestamp format (H:MM:SS.CC)"""
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    centiseconds = (ms % 1000) // 10
    
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def group_srt_by_exact_timing(input_srt, output_srt, words_per_group=6):
    """
    SIMPLE grouping: Start when first word starts, end when last word ends.
    """
    if not os.path.exists(input_srt):
        logging.error(f"Input SRT not found: {input_srt}")
        return False
    
    try:
        with open(input_srt, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse SRT entries
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            logging.error("No SRT entries found")
            return False
        
        # Group words
        grouped_entries = []
        entry_num = 1
        i = 0
        
        while i < len(matches):
            # Collect N words
            group_words = []
            group_start = matches[i][1]  # First word start
            group_end = None
            
            # Take up to words_per_group words
            for j in range(i, min(i + words_per_group, len(matches))):
                group_words.append(matches[j][3].strip())
                group_end = matches[j][2]  # Keep updating to last word's end
            
            grouped_entries.append({
                'num': entry_num,
                'start': group_start,
                'end': group_end,
                'text': ' '.join(group_words)
            })
            
            entry_num += 1
            i += words_per_group
        
        # Write SRT
        with open(output_srt, 'w', encoding='utf-8') as f:
            for entry in grouped_entries:
                f.write(f"{entry['num']}\n")
                f.write(f"{entry['start']} --> {entry['end']}\n")
                f.write(f"{entry['text']}\n\n")
        
        logging.info(f"✅ Grouped SRT: {output_srt} ({len(grouped_entries)} entries)")
        return True
        
    except Exception as e:
        logging.error(f"Failed to group SRT: {e}")
        logging.error(traceback.format_exc())
        return False


def create_karaoke_ass_exact_timing(word_level_srt, output_ass, words_per_line=6, 
                                    position="bottom", caption_settings=None):
    """
    Karaoke: Each word highlights EXACTLY when it's spoken.
    Optimized for long videos with better memory management.
    """
    if not os.path.exists(word_level_srt):
        logging.error(f"Input SRT not found: {word_level_srt}")
        return False
    
    try:
        logging.info(f"📝 Creating karaoke ASS from {word_level_srt}...")
        
        # Read and parse SRT (streaming to avoid memory issues)
        matches = []
        with open(word_level_srt, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            logging.error("No SRT entries found")
            return False
        
        total_words = len(matches)
        logging.info(f"📊 Processing {total_words} words for karaoke...")
        
        # Use caption_settings if provided, otherwise use defaults
        if caption_settings:
            position = caption_settings.get('position', 'bottom')
            font_size = caption_settings.get('fontSize', 52)
            font_family = get_safe_font(caption_settings.get('fontFamily', 'Arial'))
            font_weight = get_font_weight_value(caption_settings.get('fontWeight', 'bold'))
            
            text_color = hex_to_ass_color(caption_settings.get('textColor', '#FFFFFF'))
            highlight_color = hex_to_ass_color(caption_settings.get('highlightColor','#FFD700'))
            outline_color = hex_to_ass_color(caption_settings.get('outlineColor', '#000000'))
            bg_color = hex_to_ass_color(caption_settings.get('backgroundColor', '#000000'))
            
            bg_opacity = caption_settings.get('backgroundOpacity', 0)
            bg_alpha = format(int((bg_opacity / 100) * 255), '02X')
            
            outline_width = caption_settings.get('outlineWidth', 3)
            words_per_line = caption_settings.get('wordsPerGroup', 6)
            
            alignment_map = {
                'left': {'top': 7, 'middle': 4, 'bottom': 1},
                'center': {'top': 8, 'middle': 5, 'bottom': 2},
                'right': {'top': 9, 'middle': 6, 'bottom': 3}
            }
            text_alignment = caption_settings.get('alignment', 'center')
            alignment = alignment_map[text_alignment].get(position, 2)
        else:
            # Default values
            font_size = 52
            font_family = 'Arial'
            font_weight = -1
            text_color = '&H00FFFFFF'
            highlight_color = '&H0000FFFF'
            outline_color = '&H00000000'
            bg_color = '&H80000000'
            bg_alpha = '80'
            outline_width = 3
            
            if position == "top":
                alignment = 8
            elif position == "middle":
                alignment = 5
            else:
                alignment = 2
        
        margin_v = 60 if position in ['top', 'bottom'] else 0
        
        # ASS header
        ass_header = f"""[Script Info]
Title: Karaoke Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},{text_color},{highlight_color},{outline_color},{bg_color}{bg_alpha},{font_weight},0,0,0,100,100,0,0,1,{outline_width},2,{alignment},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # 🔥 Write to file incrementally to avoid memory issues
        with open(output_ass, 'w', encoding='utf-8') as ass_file:
            ass_file.write(ass_header)
            
            # Process lines in batches
            i = 0
            lines_processed = 0
            
            while i < len(matches):
                line_words = []
                for j in range(i, min(i + words_per_line, len(matches))):
                    line_words.append({
                        'text': matches[j][3].strip(),
                        'start_ms': _srt_time_to_ms(matches[j][1]),
                        'end_ms': _srt_time_to_ms(matches[j][2])
                    })
                
                if not line_words:
                    break
                
                line_start_ms = line_words[0]['start_ms']
                line_end_ms = line_words[-1]['end_ms']
                
                line_start = _ms_to_ass_time(line_start_ms)
                line_end = _ms_to_ass_time(line_end_ms)
                
                karaoke_text = ""
                
                for idx, word in enumerate(line_words):
                    duration_ms = word['end_ms'] - word['start_ms']
                    karaoke_text += f"{{\\k{duration_ms//10}}}{word['text']}"
                    
                    if idx < len(line_words) - 1:
                        karaoke_text += " "
                
                # Write line immediately
                ass_file.write(f"Dialogue: 0,{line_start},{line_end},Default,,0,0,0,,{karaoke_text}\n")
                
                lines_processed += 1
                i += words_per_line
                
                # Progress logging every 100 lines
                if lines_processed % 100 == 0:
                    progress = (i / total_words) * 100
                    logging.info(f"   Progress: {progress:.1f}% ({i}/{total_words} words)")
        
        logging.info(f"✅ Karaoke ASS created: {output_ass} ({lines_processed} dialogue lines)")
        return True
        
    except Exception as e:
        logging.error(f"Failed to create karaoke ASS: {e}")
        logging.error(traceback.format_exc())
        return False


def _convert_srt_to_positioned_ass(srt_file, ass_file, position="bottom", caption_settings=None):
    """
    Convert SRT to ASS with custom position and settings.
    Optimized for large files.
    """
    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        logging.info(f"📝 Converting {len(matches)} SRT entries to ASS...")
        
        # Use caption_settings if provided
        if caption_settings:
            position = caption_settings.get('position', 'bottom')
            font_size = caption_settings.get('fontSize', 52)
            font_family = get_safe_font(caption_settings.get('fontFamily', 'Arial'))
            font_weight = get_font_weight_value(caption_settings.get('fontWeight', 'bold'))
            
            text_color = hex_to_ass_color(caption_settings.get('textColor', '#FFFFFF'))
            outline_color = hex_to_ass_color(caption_settings.get('outlineColor', '#000000'))
            bg_color = hex_to_ass_color(caption_settings.get('backgroundColor', '#000000'))
            
            bg_opacity = caption_settings.get('backgroundOpacity', 0)
            bg_alpha = format(int((bg_opacity / 100) * 255), '02X')
            
            outline_width = caption_settings.get('outlineWidth', 3)
            
            alignment_map = {
                'left': {'top': 7, 'middle': 4, 'bottom': 1},
                'center': {'top': 8, 'middle': 5, 'bottom': 2},
                'right': {'top': 9, 'middle': 6, 'bottom': 3}
            }
            text_alignment = caption_settings.get('alignment', 'center')
            alignment = alignment_map[text_alignment].get(position, 2)
        else:
            # Defaults
            font_size = 52
            font_family = 'Arial'
            font_weight = -1
            text_color = '&H00FFFFFF'
            outline_color = '&H00000000'
            bg_color = '&H80000000'
            bg_alpha = '80'
            outline_width = 3
            
            if position == "top":
                alignment = 8
            elif position == "middle":
                alignment = 5
            else:
                alignment = 2
        
        margin_v = 60 if position in ['top', 'bottom'] else 0
        
        # Create ASS header
        ass_header = f"""[Script Info]
Title: Positioned Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},{text_color},{text_color},{outline_color},{bg_color}{bg_alpha},{font_weight},0,0,0,100,100,0,0,1,{outline_width},2,{alignment},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Write incrementally
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(ass_header)
            
            for idx, match in enumerate(matches):
                start_ms = _srt_time_to_ms(match[1])
                end_ms = _srt_time_to_ms(match[2])
                start_ass = _ms_to_ass_time(start_ms)
                end_ass = _ms_to_ass_time(end_ms)
                text = match[3].strip().replace('\n', ' ')
                
                f.write(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}\n")
                
                # Progress logging every 500 entries
                if (idx + 1) % 500 == 0:
                    progress = ((idx + 1) / len(matches)) * 100
                    logging.info(f"   Progress: {progress:.1f}% ({idx + 1}/{len(matches)} entries)")
        
        logging.info(f"✅ ASS conversion complete: {ass_file}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to convert SRT to ASS: {e}")
        logging.error(traceback.format_exc())
        return False


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logging.error(f"Failed to get video duration: {e}")
        return 0


def burn_srt_to_video(input_video, srt_file, output_video, 
                      subtitle_style="grouped", words_per_group=3, 
                      position="top", caption_settings=None):
    """
    Burn subtitles to video with dynamic timeout based on video length.
    Now supports caption_settings dict from frontend.
    
    Args:
        caption_settings: Dict from React component (optional, overrides other params)
    """
    if not os.path.exists(srt_file):
        logging.error(f"❌ SRT file not found: {srt_file}")
        return False
    
    if not os.path.exists(input_video):
        logging.error(f"❌ Input video not found: {input_video}")
        return False
    
    # Use caption_settings if provided
    if caption_settings:
        subtitle_style = caption_settings.get('style', 'karaoke')
        words_per_group = caption_settings.get('wordsPerGroup', 3)
        position = caption_settings.get('position', 'top')
    
    logging.info(f"🔥 Burning subtitles: {subtitle_style}")
    
    # 🔥 Calculate dynamic timeout based on video length
    video_duration = get_video_duration(input_video)
    if video_duration > 0:
        # Base timeout: 30 seconds per minute of video, minimum 180 seconds
        timeout_seconds = max(180, int(video_duration * 0.5) + 120)
        logging.info(f"⏱️  Video duration: {video_duration:.1f}s, timeout set to: {timeout_seconds}s")
    else:
        timeout_seconds = 600  # 10 minutes default if duration unknown
        logging.warning(f"⚠️  Could not determine video duration, using default timeout: {timeout_seconds}s")
    
    output_dir = os.path.dirname(output_video)
    temp_subtitle = None
    temp_ass = None
    
    try:
        if subtitle_style == "karaoke":
            temp_subtitle = os.path.join(output_dir, "temp_karaoke.ass")
            
            logging.info("🎤 Creating karaoke ASS file...")
            success = create_karaoke_ass_exact_timing(
                srt_file, 
                temp_subtitle,
                words_per_line=words_per_group,
                position=position,
                caption_settings=caption_settings
            )
            
            if not success:
                logging.error("❌ Failed to create karaoke ASS")
                return False
            
            # Verify ASS file was created and has content
            if not os.path.exists(temp_subtitle) or os.path.getsize(temp_subtitle) < 100:
                logging.error("❌ Karaoke ASS file is missing or empty")
                return False
            
            subtitle_filter = f"ass={temp_subtitle.replace(chr(92), '/')}"
            
        elif subtitle_style == "grouped":
            temp_subtitle = os.path.join(output_dir, "temp_grouped.srt")
            
            logging.info("📦 Grouping words...")
            success = group_srt_by_exact_timing(
                srt_file,
                temp_subtitle,
                words_per_group=words_per_group
            )
            
            if not success:
                logging.error("❌ Failed to group SRT")
                return False
            
            # Convert to ASS with settings
            temp_ass = os.path.join(output_dir, "temp_positioned.ass")
            logging.info("🎨 Converting to positioned ASS...")
            _convert_srt_to_positioned_ass(temp_subtitle, temp_ass, position, caption_settings)
            
            if not os.path.exists(temp_ass) or os.path.getsize(temp_ass) < 100:
                logging.error("❌ Positioned ASS file is missing or empty")
                return False
            
            subtitle_filter = f"ass={temp_ass.replace(chr(92), '/')}"
            
        else:  # word-by-word
            temp_subtitle = os.path.join(output_dir, "temp_subs.srt")
            shutil.copy2(srt_file, temp_subtitle)
            
            # Convert to ASS with settings
            temp_ass = os.path.join(output_dir, "temp_word.ass")
            logging.info("🎨 Converting to word-by-word ASS...")
            _convert_srt_to_positioned_ass(temp_subtitle, temp_ass, position, caption_settings)
            
            if not os.path.exists(temp_ass) or os.path.getsize(temp_ass) < 100:
                logging.error("❌ Word ASS file is missing or empty")
                return False
            
            subtitle_filter = f"ass={temp_ass.replace(chr(92), '/')}"
        
        logging.info("🎬 Running FFmpeg subtitle burn...")
        
        ffmpeg_cmd = [
            "ffmpeg", 
            "-y",
            "-i", input_video,
            "-vf", subtitle_filter,
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-tune", "hq",
            "-rc", "vbr",
            "-b:v", "8M",
            "-maxrate", "12M",
            "-bufsize", "24M",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_video
        ]
        
        logging.info(f"⏳ Processing (timeout: {timeout_seconds}s)...")
        
        # 🔥 Use dynamic timeout
        result = subprocess.run(
            ffmpeg_cmd, 
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds  # Dynamic timeout!
        )
        
        # Verify output
        if os.path.exists(output_video) and os.path.getsize(output_video) > 10000:
            output_size = os.path.getsize(output_video) / (1024 * 1024)
            logging.info(f"🎉 Video with subtitles created: {output_video} ({output_size:.2f} MB)")
            return True
        else:
            logging.error("❌ Output video failed validation")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"⏱️  FFmpeg subtitle burn TIMED OUT after {timeout_seconds} seconds")
        logging.error(f"   Video duration: {video_duration:.1f}s")
        logging.error(f"   This suggests the subtitle file may be too large or complex")
        return False
        
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ FFmpeg subtitle burn failed: {e}")
        if e.stderr:
            logging.error(f"   FFmpeg error: {e.stderr}")
        return False
        
    except Exception as e:
        logging.error(f"❌ Subtitle burn error: {e}")
        logging.error(traceback.format_exc())
        return False
    
    finally:
        # Cleanup temp files
        try:
            if temp_subtitle and os.path.exists(temp_subtitle):
                os.remove(temp_subtitle)
                logging.debug(f"🧹 Cleaned up: {temp_subtitle}")
            if temp_ass and os.path.exists(temp_ass):
                os.remove(temp_ass)
                logging.debug(f"🧹 Cleaned up: {temp_ass}")
        except Exception as cleanup_error:
            logging.warning(f"⚠️  Cleanup error: {cleanup_error}")


def compute_segment_durations(segment_audio_paths):
    durations = []
    for p in segment_audio_paths:
        try:
            clip = AudioFileClip(p)
            durations.append(float(clip.duration or 0.0))
            clip.close()
        except Exception:
            durations.append(0.0)
    return durations


def assign_timeline_phases(
    segments,
    segment_audio_paths,
    phase_media: dict | None = None,
    intro_sec: int = 600,
    finale_sec: int = 600,
):
    """
    Phasing with Whisk character support:
      - Intro from start until crossing intro_sec
      - Finale from end until crossing finale_sec
      - Body is the middle

    Per segment adds:
      seg["duration_needed"], seg["phase"] in {"intro","body","finale"}
      Exactly one of:
        seg["pexels_required"]       (Stock Video)
        seg["pexels_image_required"] (Stock Image)
        seg["imagefx_required"]      (AI Image)
        seg["whisk_required"]        (Whisk Character) ← NEW
      
      If whisk_required: seg["character_id"] is added
      Optional seg["ai_prompt"] when AI Image is chosen and a prompt is provided.
      Use style key "space" for space-documentary painterly mode (visual bible + layered prompts).
    """
    def select_flags_for_type(media_type: str, character_id: str = None):
        flags = {
            "pexels_required":       media_type == "stock_video",
            "pexels_image_required": media_type == "stock_image",
            "imagefx_required":      media_type == "ai_image",
            "whisk_required":        media_type == "whisk_character",  # ← NEW
        }
        if flags["whisk_required"] and character_id:
            flags["character_id"] = character_id
        return flags

    print(phase_media)

    # durations
    durs = compute_segment_durations(segment_audio_paths)
    n = len(segments)
    for i, seg in enumerate(segments):
        seg["duration_needed"] = durs[i]

    # intro indices
    intro_idxs, cum = [], 0.0
    for i, dur in enumerate(durs):
        if cum >= intro_sec: break
        intro_idxs.append(i)
        cum += dur
        if cum >= intro_sec: break

    # finale indices
    finale_idxs, cum = [], 0.0
    for j in range(n - 1, -1, -1):
        if j in intro_idxs: break
        if cum >= finale_sec: break
        finale_idxs.append(j)
        cum += durs[j]
        if cum >= finale_sec: break
    finale_idxs = set(finale_idxs)

    # ensure no overlap
    last_intro = max(intro_idxs) if intro_idxs else -1
    finale_idxs = {j for j in finale_idxs if j > last_intro}

    # assign phases + reset flags
    for i, seg in enumerate(segments):
        if i in intro_idxs:
            seg["phase"] = "intro"
        elif i in finale_idxs:
            seg["phase"] = "finale"
        else:
            seg["phase"] = "body"
        
        # Clear old flags
        for k in ("pexels_required", "pexels_image_required", "imagefx_required", 
                  "whisk_required", "ai_prompt", "character_id"):
            seg.pop(k, None)

    # apply per-phase media
    for seg in segments:
        ph = seg["phase"]
        choice = phase_media.get(ph, {"mediaType": "stock_video"})
        mt = choice.get("mediaType", "stock_video")
        char_id = choice.get("characterId")  # ← NEW
        
        flags = select_flags_for_type(mt, char_id)
        seg.update(flags)
        
        # Add AI prompt if applicable
        if flags.get("imagefx_required") and choice.get("prompt"):
            seg["ai_prompt"] = choice["prompt"]

    return segments

from helper.transition_engine import apply_transitions_and_concat

from whisk_character_handler import (
    initialize_whisk_character,
    whisk_generate_for_segment,
    Whisk401Error
)


"""
BACKEND PATCH  —  paste these two blocks into create_video.py

1. Replace the canvas_w / canvas_h block inside create_video()
2. Replace _composite_avatar_over_full_video() entirely

Changes vs original
───────────────────
• canvas_w / canvas_h now read from layout_config["canvas"] first,
  then fall back to the orientation preset — so free-form canvas
  sizes sent by the frontend are honoured exactly.

• Border is read from avatar_layer / content_layer dicts.
  Fields:  width (int px), color (hex str), style (str, ignored by
  FFmpeg — drawbox only supports solid), radius (int px).
  radius is applied with a rounded_rect workaround (overlay approach).

• background="image:<url>" and "video:<url>" are handled:
  the media is fetched via the url and used as a static overlay.

• No silent clamping anywhere — what the frontend sends is exactly
  what FFmpeg receives.
"""

# ─────────────────────────────────────────────────────────────────
# 1. CANVAS SIZE BLOCK  (replaces the CURRENT_WIDTH/HEIGHT block)
#    inside create_video(), after the orientation is set
# ─────────────────────────────────────────────────────────────────

# REPLACE THIS BLOCK in create_video():
#
#   CURRENT_ORIENTATION = orientation.lower().strip()
#   if CURRENT_ORIENTATION == "portrait":
#       CURRENT_WIDTH  = 1080
#       CURRENT_HEIGHT = 1920
#   else:
#       CURRENT_WIDTH  = 1920
#       CURRENT_HEIGHT = 1080
#   canvas_w = CURRENT_WIDTH
#   canvas_h = CURRENT_HEIGHT
#
# WITH:

def _resolve_canvas(orientation, layout_config):
    """
    Canvas dimensions in priority order:
      1. layout_config["canvas"] sent by frontend (free-form)
      2. Standard orientation preset
    Always integers, always >= 320 on each axis.
    """
    if layout_config and "canvas" in layout_config:
        w = int(layout_config["canvas"].get("width",  0))
        h = int(layout_config["canvas"].get("height", 0))
        if w >= 320 and h >= 320:
            return w, h

    # Fallback to orientation preset
    orient = (orientation or "landscape").lower().strip()
    if orient == "portrait":
        return 1080, 1920
    return 1920, 1080

"""
BACKEND PATCH  —  paste these two blocks into create_video.py

1. Replace the canvas_w / canvas_h block inside create_video()
2. Replace _composite_avatar_over_full_video() entirely

Changes vs original
───────────────────
• canvas_w / canvas_h now read from layout_config["canvas"] first,
  then fall back to the orientation preset — so free-form canvas
  sizes sent by the frontend are honoured exactly.

• Border is read from avatar_layer / content_layer dicts.
  Fields:  width (int px), color (hex str), style (str, ignored by
  FFmpeg — drawbox only supports solid), radius (int px).
  radius is applied with a rounded_rect workaround (overlay approach).

• background="image:<url>" and "video:<url>" are handled:
  the media is fetched via the url and used as a static overlay.

• No silent clamping anywhere — what the frontend sends is exactly
  what FFmpeg receives.
"""

# ─────────────────────────────────────────────────────────────────
# 1. CANVAS SIZE BLOCK  (replaces the CURRENT_WIDTH/HEIGHT block)
#    inside create_video(), after the orientation is set
# ─────────────────────────────────────────────────────────────────

# REPLACE THIS BLOCK in create_video():
#
#   CURRENT_ORIENTATION = orientation.lower().strip()
#   if CURRENT_ORIENTATION == "portrait":
#       CURRENT_WIDTH  = 1080
#       CURRENT_HEIGHT = 1920
#   else:
#       CURRENT_WIDTH  = 1920
#       CURRENT_HEIGHT = 1080
#   canvas_w = CURRENT_WIDTH
#   canvas_h = CURRENT_HEIGHT
#
# WITH:

def _resolve_canvas(orientation, layout_config):
    """
    Canvas dimensions in priority order:
      1. layout_config["canvas"] sent by frontend (free-form)
      2. Standard orientation preset
    Always integers, always >= 320 on each axis.
    """
    if layout_config and "canvas" in layout_config:
        w = int(layout_config["canvas"].get("width",  0))
        h = int(layout_config["canvas"].get("height", 0))
        if w >= 320 and h >= 320:
            return w, h

    # Fallback to orientation preset
    orient = (orientation or "landscape").lower().strip()
    if orient == "portrait":
        return 1080, 1920
    return 1920, 1080

"""
BACKEND PATCH  —  paste these two blocks into create_video.py

1. Replace the canvas_w / canvas_h block inside create_video()
2. Replace _composite_avatar_over_full_video() entirely

Changes vs original
───────────────────
• canvas_w / canvas_h now read from layout_config["canvas"] first,
  then fall back to the orientation preset — so free-form canvas
  sizes sent by the frontend are honoured exactly.

• Border is read from avatar_layer / content_layer dicts.
  Fields:  width (int px), color (hex str), style (str, ignored by
  FFmpeg — drawbox only supports solid), radius (int px).
  radius is applied with a rounded_rect workaround (overlay approach).

• background="image:<url>" and "video:<url>" are handled:
  the media is fetched via the url and used as a static overlay.

• No silent clamping anywhere — what the frontend sends is exactly
  what FFmpeg receives.
"""

# ─────────────────────────────────────────────────────────────────
# 1. CANVAS SIZE BLOCK  (replaces the CURRENT_WIDTH/HEIGHT block)
#    inside create_video(), after the orientation is set
# ─────────────────────────────────────────────────────────────────

# REPLACE THIS BLOCK in create_video():
#
#   CURRENT_ORIENTATION = orientation.lower().strip()
#   if CURRENT_ORIENTATION == "portrait":
#       CURRENT_WIDTH  = 1080
#       CURRENT_HEIGHT = 1920
#   else:
#       CURRENT_WIDTH  = 1920
#       CURRENT_HEIGHT = 1080
#   canvas_w = CURRENT_WIDTH
#   canvas_h = CURRENT_HEIGHT
#
# WITH:

def _resolve_canvas(orientation, layout_config):
    """
    Canvas dimensions in priority order:
      1. layout_config["canvas"] sent by frontend (free-form)
      2. Standard orientation preset
    Always integers, always >= 320 on each axis.
    """
    if layout_config and "canvas" in layout_config:
        w = int(layout_config["canvas"].get("width",  0))
        h = int(layout_config["canvas"].get("height", 0))
        if w >= 320 and h >= 320:
            return w, h

    # Fallback to orientation preset
    orient = (orientation or "landscape").lower().strip()
    if orient == "portrait":
        return 1080, 1920
    return 1920, 1080


import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import time
import logging
import tempfile
import subprocess

def _dreamface_skip_narration_media_pool(layout_config, avatar_segment_map) -> bool:
    """Full-screen avatar renders do not use Pexels/ImageFX narration assets."""
    return bool(
        avatar_segment_map
        and isinstance(layout_config, dict)
        and (layout_config.get("layout_id") or "") == "full_avatar"
    )


def _quick_video_file_ok(path, min_bytes=10000) -> bool:
    if not path or not os.path.exists(path):
        return False
    if os.path.getsize(path) < min_bytes:
        return False
    return _probe_stream_duration(path, "video") > 0 or _probe_media_duration(path) > 0


def _verify_video_decodable(path, timeout=45) -> bool:
    """True if ffmpeg can decode at least one video frame (catches Grok/NVENC glitches)."""
    if not path or not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1000:
        return False
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostats",
                "-fflags", "+discardcorrupt",
                "-i", path,
                "-map", "0:v:0",
                "-frames:v", "1",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            logging.warning(
                "Video decode check failed | %s | %s",
                os.path.basename(path),
                (r.stderr or "").strip()[-400:],
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logging.warning("Video decode check timed out: %s", path)
        return False
    except Exception as exc:
        logging.warning("Video decode check error for %s: %s", path, exc)
        return False


def _repair_clip_source_for_encode(src, dst) -> bool:
    """Re-mux/transcode a broken clip so segment encode can read it."""
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts+discardcorrupt",
        "-err_detect", "ignore_err",
        "-i", src,
        "-map", "0:v:0?",
        "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        dst,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 1000:
            logging.warning(
                "CLIP source repair failed | %s | %s",
                os.path.basename(src),
                (r.stderr or "").strip()[-500:],
            )
            return False
        if not _verify_video_decodable(dst, timeout=60):
            return False
        logging.info("🔧 Repaired CLIP source for encode | %s", os.path.basename(src))
        return True
    except subprocess.TimeoutExpired:
        logging.warning("CLIP source repair timed out: %s", src)
        return False
    except Exception as exc:
        logging.warning("CLIP source repair error for %s: %s", src, exc)
        return False


def _append_tagged_clip_to_media_pool(media_pool, result, video_path):
    """CLIP/tagged sources only need path + duration — no CLIP embedding."""
    duration = _probe_media_duration(video_path)
    if duration <= 0:
        duration = _probe_stream_duration(video_path, "video") or 0.0
    if duration <= 0:
        return False
    media_pool.append(
        {
            "type": "video",
            "path": video_path,
            "character": result["character"],
            "embedding": None,
            "mute": result.get("mute", False),
            "duration": duration,
            "tag": result.get("tag"),
        }
    )
    return True


def _probe_media_duration(path):
    if not path or not os.path.exists(path):
        return 0.0
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True,
        )
        for s in json.loads(r.stdout).get("streams", []):
            d = float(s.get("duration", 0))
            if d > 0:
                return d
    except Exception as e:
        logging.warning(f"probe_duration failed for {path}: {e}")
    return 0.0

def _is_clip_segment(seg):
    text = (seg.get("text") or "").strip()
    return text.startswith("[CLIP:") and text.endswith("]")


def _has_clip_segments(vid_segments):
    return any(_is_clip_segment(seg) for seg in (vid_segments or []))

def _clip_has_audio(path):
    if not path or not os.path.exists(path):
        return False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True,
        )
        return any(
            s.get("codec_type") == "audio"
            for s in json.loads(r.stdout).get("streams", [])
        )
    except Exception:
        return False

def _probe_stream_duration(path, codec_type=None):
    """Duration of the first matching stream, else container format duration."""
    if not path or not os.path.exists(path):
        return 0.0
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", path,
            ],
            capture_output=True, text=True,
        )
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if codec_type and s.get("codec_type") != codec_type:
                continue
            d = float(s.get("duration") or 0)
            if d > 0:
                return d
        d = float(data.get("format", {}).get("duration") or 0)
        if d > 0:
            return d
    except Exception as e:
        logging.warning(f"probe_stream_duration failed for {path}: {e}")
    return 0.0

def _probe_audio_duration(path):
    d = _probe_stream_duration(path, "audio")
    if d > 0:
        return d
    return _probe_stream_duration(path, "video") or _probe_media_duration(path)

def _drawbox(x, y, w, h, border_w, color_hex, radius):
    t   = max(1, min(border_w, min(w, h) // 2))
    col = f"{color_hex}@1.0"
    if radius <= 0:
        return f"drawbox=x={x}:y={y}:w={w}:h={h}:color={col}:t={t}"
    r = max(1, min(radius, w // 2, h // 2))
    return ",".join([
        f"drawbox=x={x+r}:y={y}:w={w-2*r}:h={t}:color={col}:t=fill",
        f"drawbox=x={x+r}:y={y+h-t}:w={w-2*r}:h={t}:color={col}:t=fill",
        f"drawbox=x={x}:y={y+r}:w={t}:h={h-2*r}:color={col}:t=fill",
        f"drawbox=x={x+w-t}:y={y+r}:w={t}:h={h-2*r}:color={col}:t=fill",
    ])


def _normalize_fit_mode(value, default="cover"):
    mode = str(value or default).lower().strip()
    if mode in ("cover", "contain", "fill"):
        return mode
    return default


def _ffmpeg_content_box_filter(in_label, out_label, w, h, fit_mode="cover", pad_color="black"):
    """
    Scale content into a box — cover (crop), contain, or fill (stretch).

    contain never leaves dark bars: the whole frame is fitted inside the box and
    the empty sides show a blurred, slightly dimmed cover copy of the same
    playing video (pad_color is kept for call compatibility only).
    """
    fit_mode = _normalize_fit_mode(fit_mode, "cover")
    if fit_mode == "contain":
        bw = max(2, (w // 16) // 2 * 2)
        bh = max(2, (h // 16) // 2 * 2)
        o = out_label
        return (
            f"[{in_label}]split=2[{o}_fg][{o}_bg];"
            f"[{o}_bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
            f"scale={bw}:{bh}:flags=area,boxblur=2:1,"
            f"scale={w}:{h}:flags=bicubic,eq=brightness=-0.06:saturation=0.9,"
            f"setsar=1,format=yuv420p[{o}_blur];"
            f"[{o}_fg]scale={w}:{h}:force_original_aspect_ratio=decrease:"
            f"force_divisible_by=2:flags=lanczos,setsar=1,format=yuv420p[{o}_fit];"
            f"[{o}_blur][{o}_fit]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[{o}]"
        )
    if fit_mode == "fill":
        return (
            f"[{in_label}]scale={w}:{h}:flags=lanczos,"
            f"setsar=1,format=yuv420p[{out_label}]"
        )
    return (
        f"[{in_label}]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1,format=yuv420p[{out_label}]"
    )


def _ffmpeg_avatar_box_filter(in_label, out_label, w, h):
    """
    Avatar never stretches — contain in box; letterbox/pillarbox filled with
    a blurred cover crop of the same avatar frame.
    """
    bw = max(w // 8, 2)
    bh = max(h // 8, 2)
    return (
        f"[{in_label}]split=2[{out_label}_fg][{out_label}_bg];"
        f"[{out_label}_bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},"
        f"scale={bw}:{bh}:flags=bilinear,"
        f"scale={w}:{h}:flags=bilinear,"
        f"setsar=1,format=yuv420p[{out_label}_blur];"
        f"[{out_label}_fg]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"setsar=1,format=yuv420p[{out_label}_fgfit];"
        f"[{out_label}_blur][{out_label}_fgfit]overlay=(W-w)/2:(H-h)/2,"
        f"setsar=1,format=yuv420p[{out_label}]"
    )


def _thc_v2_config(layout_geom):
    """layout_config when it carries a Talking Heads canvas design (style_version 2)."""
    cfg = layout_geom.get("layout_config") if isinstance(layout_geom, dict) else None
    try:
        if isinstance(cfg, dict) and int(cfg.get("style_version") or 1) >= 2 and isinstance(cfg.get("design"), dict):
            return cfg
    except (TypeError, ValueError):
        pass
    return None


def _thc_fx(layout_geom, canvas_w, canvas_h, mode, chunk_folder, first_input):
    """CanvasFx for designed canvases (None → classic rendering, unchanged)."""
    cfg = _thc_v2_config(layout_geom)
    if not cfg:
        return None
    try:
        from helper.talking_heads_canvas import CanvasFx

        static_root = os.getenv("STATIC_ROOT") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "static"
        )
        return CanvasFx.for_layout(
            cfg, canvas_w, canvas_h, mode, chunk_folder,
            first_input=first_input, static_root=static_root,
        )
    except Exception as exc:
        logging.warning(f"⚠️  Canvas design unavailable — classic layout used: {exc}")
        return None


def _thc_bg_scale(layout_geom, canvas_w, canvas_h):
    """Image/video backgrounds: designed canvases cover (like the studio preview)."""
    if _thc_v2_config(layout_geom):
        return (
            f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={canvas_w}:{canvas_h}"
        )
    return f"scale={canvas_w}:{canvas_h}:flags=lanczos"


def _thc_count_inputs(args):
    return sum(1 for a in args if a == "-i")


def _extract_layout_geometry(layout_config, canvas_w, canvas_h):
    import re

    layout_id     = layout_config.get("layout_id", "")
    avatar_layer  = layout_config.get("avatar_layer") or {}
    content_layer = layout_config.get("content_layer")
    clip_layer    = layout_config.get("clip_layer")
    background    = layout_config.get("background", "#0a0a0a")
    content_fit   = _normalize_fit_mode(layout_config.get("content_fit"), "cover")
    clip_fit      = _normalize_fit_mode(layout_config.get("clip_fit"), "contain")

    av_x = int(avatar_layer.get("x", 0))
    av_y = int(avatar_layer.get("y", 0))
    av_w = int(avatar_layer.get("width", 320))
    av_h = int(avatar_layer.get("height", 450))

    av_border        = avatar_layer.get("border", {}) or {}
    av_border_w      = max(0, int(av_border.get("width", 0)))
    av_border_color  = av_border.get("color", "#ffffff").lstrip("#")
    av_border_radius = max(0, int(av_border.get("radius", 0)))
    if not re.match(r"^[0-9a-fA-F]{6}$", av_border_color):
        av_border_color = "ffffff"

    ct_x = ct_y = ct_w = ct_h = 0
    ct_border_w = ct_border_radius = 0
    ct_border_color = "ffffff"

    if content_layer:
        ct_x = int(content_layer.get("x", 0))
        ct_y = int(content_layer.get("y", 0))
        ct_w = int(content_layer.get("width", canvas_w))
        ct_h = int(content_layer.get("height", canvas_h))
        ct_border        = content_layer.get("border", {}) or {}
        ct_border_w      = max(0, int(ct_border.get("width", 0)))
        ct_border_color  = ct_border.get("color", "#ffffff").lstrip("#")
        ct_border_radius = max(0, int(ct_border.get("radius", 0)))
        if not re.match(r"^[0-9a-fA-F]{6}$", ct_border_color):
            ct_border_color = "ffffff"

    cl_x = cl_y = cl_w = cl_h = 0
    cl_border_w = cl_border_radius = 0
    cl_border_color = "ffffff"

    if clip_layer:
        cl_x = int(clip_layer.get("x", 0))
        cl_y = int(clip_layer.get("y", 0))
        cl_w = int(clip_layer.get("width", canvas_w))
        cl_h = int(clip_layer.get("height", canvas_h))
        cl_border        = clip_layer.get("border", {}) or {}
        cl_border_w      = max(0, int(cl_border.get("width", 0)))
        cl_border_color  = cl_border.get("color", "#ffffff").lstrip("#")
        cl_border_radius = max(0, int(cl_border.get("radius", 0)))
        if not re.match(r"^[0-9a-fA-F]{6}$", cl_border_color):
            cl_border_color = "ffffff"
    elif content_layer and ct_w > 0 and ct_h > 0:
        cl_w, cl_h = ct_w, ct_h
        cl_x = round((canvas_w - cl_w) / 2)
        cl_y = round((canvas_h - cl_h) / 2)
        cl_border_w = ct_border_w
        cl_border_color = ct_border_color
        cl_border_radius = ct_border_radius
    else:
        cl_w, cl_h = canvas_w, canvas_h
        cl_x = cl_y = 0

    PREVIEW_W    = 560
    border_scale = canvas_w / PREVIEW_W
    av_border_w      = max(0, round(av_border_w * border_scale))
    av_border_radius = max(0, round(av_border_radius * border_scale))
    ct_border_w      = max(0, round(ct_border_w * border_scale))
    ct_border_radius = max(0, round(ct_border_radius * border_scale))
    cl_border_w      = max(0, round(cl_border_w * border_scale))
    cl_border_radius = max(0, round(cl_border_radius * border_scale))

    bg_type  = "color"
    bg_color = "0a0a0a"
    bg_url   = None

    if re.match(r"^#[0-9a-fA-F]{6}$", str(background)):
        bg_type  = "color"
        bg_color = background.lstrip("#")
    elif background == "blurred":
        bg_type = "blurred"
    elif str(background).startswith("image:"):
        bg_type = "image"
        bg_url  = str(background)[6:]
    elif str(background).startswith("video:"):
        bg_type = "bgvideo"
        bg_url  = str(background)[6:]

    logging.info(
        f"🖼️  layout={layout_id} | canvas={canvas_w}x{canvas_h} | bg={background}\n"
        f"   avatar  : ({av_x},{av_y}) {av_w}x{av_h} border={av_border_w}px\n"
        f"   content : ({ct_x},{ct_y}) {ct_w}x{ct_h} border={ct_border_w}px\n"
        f"   clip    : ({cl_x},{cl_y}) {cl_w}x{cl_h} border={cl_border_w}px\n"
        f"   fit     : content={content_fit} clip={clip_fit}"
    )

    return {
        "layout_id":        layout_id,
        "content_fit":      content_fit,
        "clip_fit":         clip_fit,
        "av_x": av_x, "av_y": av_y, "av_w": av_w, "av_h": av_h,
        "av_border_w": av_border_w,
        "av_border_color": av_border_color,
        "av_border_radius": av_border_radius,
        "ct_x": ct_x, "ct_y": ct_y, "ct_w": ct_w, "ct_h": ct_h,
        "ct_border_w": ct_border_w,
        "ct_border_color": ct_border_color,
        "ct_border_radius": ct_border_radius,
        "cl_x": cl_x, "cl_y": cl_y, "cl_w": cl_w, "cl_h": cl_h,
        "cl_border_w": cl_border_w,
        "cl_border_color": cl_border_color,
        "cl_border_radius": cl_border_radius,
        "bg_type": bg_type,
        "bg_color": bg_color,
        "bg_url": bg_url,
        "content_layer": content_layer,
        "clip_layer": clip_layer,
        "layout_config": layout_config,
    }


def _fetch_bg_media(url, suffix, chunk_folder):
    import urllib.request
    try:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=chunk_folder
        )
        urllib.request.urlretrieve(url, tmp.name)
        return tmp.name
    except Exception as e:
        logging.warning(f"⚠️  BG fetch failed: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════
# CLIP VIDEOS — batch-level composite (NEW)
#
# No avatar_start, no per-segment slicing, no duration prediction.
#
# For each avatar batch:
#   1. Group its consecutive narration segments together
#   2. Concat their content chunks into one block
#   3. Take the WHOLE avatar batch file as-is (video + its own audio)
#   4. Compare lengths AFTER THE FACT:
#        avatar shorter than content block → trim content block
#        avatar longer  than content block → loop content block
#        equal                             → nothing extra
#   5. Composite: avatar plays naturally, content fitted behind/around it,
#      avatar's own audio is the ONLY audio for this block.
#
# Clip segments (`[CLIP:...]`) are untouched — layout only, original audio.
# ════════════════════════════════════════════════════════════════════════

def _group_batches_from_map(avatar_segment_map):
    """
    batch_index → sorted list of seg_index values (contiguous run).
    """
    groups = {}
    for raw_idx, info in avatar_segment_map.items():
        seg_idx = int(raw_idx)
        b = info["batch_index"]
        groups.setdefault(b, {"path": info["path"], "seg_indices": []})
        groups[b]["seg_indices"].append(seg_idx)

    for b in groups.values():
        b["seg_indices"].sort()

    return groups


def _concat_chunks(chunk_list, chunk_folder, tag):
    """
    Plain concat of already-uniform chunks (same codec/fps/pix_fmt —
    guaranteed by Rule 1 narration encode). Stream copy.
    """
    if not chunk_list:
        return None

    if len(chunk_list) == 1:
        return chunk_list[0]

    concat_txt = os.path.join(chunk_folder, f"block_concat_{tag}_{int(time.time()*1000)}.txt")
    out_path   = os.path.join(chunk_folder, f"block_{tag}_{int(time.time()*1000)}.mp4")

    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in chunk_list:
            f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")

    r = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ], capture_output=True, text=True)

    try:
        os.remove(concat_txt)
    except Exception:
        pass

    if r.returncode != 0 or not os.path.exists(out_path):
        logging.error(f"❌ Block concat failed ({tag}):\n{r.stderr[-500:]}")
        return None

    return out_path

def _full_avatar_uses_full_bleed(layout_geom, canvas_w, canvas_h) -> bool:
    if not layout_geom:
        return True
    return (
        int(layout_geom.get("av_x", 0)) <= 1
        and int(layout_geom.get("av_y", 0)) <= 1
        and int(layout_geom.get("av_w", canvas_w)) >= canvas_w - 2
        and int(layout_geom.get("av_h", canvas_h)) >= canvas_h - 2
    )


def _composite_full_avatar_from_normalized_avatar(
    avatar_normalized: str,
    canvas_w: int,
    canvas_h: int,
    chunk_folder: str,
    tag: str,
    layout_geom=None,
) -> str | None:
    """Avatar-only composite (no narration B-roll) using avatar_layer box from studio."""
    if not avatar_normalized or not os.path.exists(avatar_normalized):
        return None

    ts = int(time.time() * 1000)
    geom = layout_geom or {}
    avatar_dur = _probe_media_duration(avatar_normalized) or 1.0
    dur_s = f"{avatar_dur:.4f}"

    av_x = int(geom.get("av_x", 0))
    av_y = int(geom.get("av_y", 0))
    av_w = int(geom.get("av_w", canvas_w))
    av_h = int(geom.get("av_h", canvas_h))
    bg_type = geom.get("bg_type", "color")
    bg_color = geom.get("bg_color", "0a0a0a")

    parts: list[str] = []
    current = "base"
    inputs = ["-i", avatar_normalized]
    next_idx = 1
    bg_media = None
    temp_bg = None

    if not _thc_v2_config(geom) and _full_avatar_uses_full_bleed(geom, canvas_w, canvas_h) and bg_type not in (
        "image",
        "bgvideo",
    ):
        filt = _ffmpeg_avatar_box_filter("0:v", "vout", canvas_w, canvas_h)
        filter_complex = filt
        final_label = "vout"
    else:
        if bg_type == "image" and geom.get("bg_url"):
            temp_bg = _fetch_bg_media(geom["bg_url"], ".jpg", chunk_folder)
            bg_media = temp_bg
        elif bg_type == "bgvideo" and geom.get("bg_url"):
            temp_bg = _fetch_bg_media(geom["bg_url"], ".mp4", chunk_folder)
            bg_media = temp_bg

        avatar_in = "0:v"
        if bg_type == "blurred":
            parts.append(
                f"[0:v]split=2[avsrc][bgsrc];"
                f"[bgsrc]scale={canvas_w//8}:{canvas_h//8}:flags=bilinear,"
                f"scale={canvas_w}:{canvas_h}:flags=bilinear,"
                f"setsar=1,format=yuv420p[base]"
            )
            avatar_in = "avsrc"
        elif bg_type == "color":
            parts.append(
                f"color=c=#{bg_color}:s={canvas_w}x{canvas_h}:r=30:d={dur_s},"
                f"format=yuv420p[base]"
            )
        elif bg_type == "image" and bg_media:
            parts.append(
                f"[{next_idx}:v]{_thc_bg_scale(geom, canvas_w, canvas_h)},"
                f"setsar=1,format=yuv420p[base]"
            )
            inputs += ["-i", bg_media]
            next_idx += 1
        elif bg_type == "bgvideo" and bg_media:
            parts.append(
                f"[{next_idx}:v]{_thc_bg_scale(geom, canvas_w, canvas_h)},"
                f"trim=duration={dur_s},setsar=1,format=yuv420p[base]"
            )
            inputs = ["-i", avatar_normalized]
            if bg_type == "bgvideo":
                inputs += ["-stream_loop", "-1"]
            inputs += ["-i", bg_media]
            next_idx = 2
        else:
            parts.append(
                f"color=c=#0a0a0a:s={canvas_w}x{canvas_h}:r=30:d={dur_s},"
                f"format=yuv420p[base]"
            )

        fx = _thc_fx(geom, canvas_w, canvas_h, "narration", chunk_folder, _thc_count_inputs(inputs))
        if fx:
            current = fx.after_base(parts, current)
        parts.append(_ffmpeg_avatar_box_filter(avatar_in, "av", av_w, av_h))
        av_label = fx.mask_layer(parts, "avatar", "av") if fx else "av"
        parts.append(
            f"[{current}][{av_label}]overlay=x={av_x}:y={av_y}:format=yuv420[after_av]"
        )
        current = "after_av"
        if not fx and int(geom.get("av_border_w", 0)) > 0:
            db = _drawbox(
                av_x,
                av_y,
                av_w,
                av_h,
                geom.get("av_border_w", 0),
                geom.get("av_border_color", "ffffff"),
                geom.get("av_border_radius", 0),
            )
            parts.append(f"[{current}]{db}[after_av_border]")
            current = "after_av_border"
        if fx:
            current = fx.finish(parts, current)
            inputs += fx.input_args()
        filter_complex = ";".join(parts)
        final_label = current

    visual_path = os.path.join(chunk_folder, f"fullav_vis_{tag}_{ts}.mp4")

    cmd_visual = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{final_label}]",
        "-an",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "constqp",
        "-qp", "18",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        visual_path,
    ]
    r_visual = subprocess.run(cmd_visual, capture_output=True, text=True)

    if temp_bg and os.path.exists(temp_bg):
        try:
            os.remove(temp_bg)
        except Exception:
            pass
    if r_visual.returncode != 0 or not os.path.exists(visual_path):
        logging.error(
            f"❌ full_avatar visual {tag} failed:\n{(r_visual.stderr or '')[-800:]}"
        )
        return None

    output_path = os.path.join(chunk_folder, f"fullav_{tag}_{ts}.mp4")
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", visual_path,
        "-i", avatar_normalized,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    r_mux = subprocess.run(cmd_mux, capture_output=True, text=True)
    for p in (visual_path,):
        try:
            os.remove(p)
        except Exception:
            pass

    if r_mux.returncode != 0 or not os.path.exists(output_path):
        logging.error(
            f"❌ full_avatar mux {tag} failed:\n{(r_mux.stderr or '')[-800:]}"
        )
        return None

    out_dur = _probe_media_duration(output_path)
    logging.info(
        f"⚡ full_avatar batch {tag} | avatar-only composite | out={out_dur:.3f}s"
    )
    return output_path


def _normalize_avatar_batch_input(avatar_batch_path, chunk_folder, tag):
    """Pass 0 — PTS=0 remux (shared by batch composite paths)."""
    ts = int(time.time() * 1000)
    avatar_normalized = os.path.join(chunk_folder, f"avnorm_{tag}_{ts}.mp4")
    cmd_norm = [
        "ffmpeg", "-y",
        "-i", avatar_batch_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        avatar_normalized,
    ]
    r_norm = subprocess.run(cmd_norm, capture_output=True, text=True)
    if r_norm.returncode != 0 or not os.path.exists(avatar_normalized):
        logging.error(
            f"❌ Avatar normalize failed ({tag}):\n{(r_norm.stderr or '')[-600:]}"
        )
        return None
    return avatar_normalized


def _composite_full_avatar_batch_block(
    avatar_batch_path,
    layout_geom,
    canvas_w,
    canvas_h,
    chunk_folder,
    tag,
):
    avatar_normalized = _normalize_avatar_batch_input(
        avatar_batch_path, chunk_folder, tag
    )
    if not avatar_normalized:
        return None
    out = _composite_full_avatar_from_normalized_avatar(
        avatar_normalized,
        canvas_w,
        canvas_h,
        chunk_folder,
        tag,
        layout_geom=layout_geom,
    )
    try:
        os.remove(avatar_normalized)
    except Exception:
        pass
    return out


def _composite_batch_block(
    content_block,
    avatar_batch_path,
    layout_geom,
    canvas_w,
    canvas_h,
    chunk_folder,
    tag,
):
    """
    One avatar batch ↔ one content block. Mirrors the confirmed
    non-drifting zoom2 reference:

    Pass 0: normalize avatar input to PTS=0 via -c copy
            -avoid_negative_ts make_zero (zoom2 gets this for free
            from concat_avatars; here it's a single-file remux).
    Pass 1: fit content_block to exactly avatar_dur (separate, isolated
            ffmpeg call — loop or trim, no filtergraph, no shared inputs).
    Pass 2: composite content_fitted + avatar video (re-encoded, scaled,
            positioned) — VIDEO ONLY, no audio mapped. Avatar video PTS
            passed through as-is (no setpts=PTS-STARTPTS), same as zoom2's
            [1:v]scale=... .
    Pass 3: mux avatar's normalized audio in via -c:a copy (no re-encode).

    Avatar plays naturally start-to-finish, exactly as DreamFace made it.
    Avatar's own (untouched) audio is the only audio for this block.
    """
    layout_id = layout_geom["layout_id"]

    if layout_id == "full_avatar":
        logging.info(f"⚡ Batch block {tag}: full_avatar — skipping narration content")
        return _composite_full_avatar_batch_block(
            avatar_batch_path,
            layout_geom,
            canvas_w,
            canvas_h,
            chunk_folder,
            tag,
        )

    avatar_dur  = _probe_media_duration(avatar_batch_path)
    content_dur = _probe_media_duration(content_block)

    if avatar_dur <= 0:
        logging.error(f"❌ Batch block {tag}: avatar batch has no duration")
        return None
    if content_dur <= 0:
        logging.error(f"❌ Batch block {tag}: content block has no duration")
        return None

    if layout_id == "full_content":
        return content_block

    ts = int(time.time() * 1000)

    avatar_normalized = _normalize_avatar_batch_input(
        avatar_batch_path, chunk_folder, tag
    )
    if not avatar_normalized:
        logging.error(f"❌ Batch block {tag}: avatar normalize failed")
        return None

    norm_dur = _probe_media_duration(avatar_normalized)
    if norm_dur > 0:
        avatar_dur = norm_dur
    logging.info(f"   {tag}: avatar normalized | {avatar_dur:.3f}s")

    # ────────────────────────────────────────────────────────────────────
    # PASS 1 — fit content_block to exactly avatar_dur
    # Isolated call: single input, no filtergraph shared with avatar.
    # ────────────────────────────────────────────────────────────────────
    content_fitted = os.path.join(chunk_folder, f"fit_{tag}_{ts}.mp4")

    if content_dur < avatar_dur - 0.01:
        cmd_fit = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", content_block,
            "-t", f"{avatar_dur:.4f}",
            "-c:v", "h264_nvenc", "-preset", "p4",
            "-pix_fmt", "yuv420p",
            "-an",
            content_fitted,
        ]
        fit_mode = "loop"
    elif content_dur > avatar_dur + 0.01:
        cmd_fit = [
            "ffmpeg", "-y",
            "-i", content_block,
            "-t", f"{avatar_dur:.4f}",
            "-c:v", "h264_nvenc", "-preset", "p4",
            "-pix_fmt", "yuv420p",
            "-an",
            content_fitted,
        ]
        fit_mode = "trim"
    else:
        cmd_fit = [
            "ffmpeg", "-y",
            "-i", content_block,
            "-t", f"{avatar_dur:.4f}",
            "-c:v", "h264_nvenc", "-preset", "p4",
            "-pix_fmt", "yuv420p",
            "-an",
            content_fitted,
        ]
        fit_mode = "exact"

    r_fit = subprocess.run(cmd_fit, capture_output=True, text=True)
    if r_fit.returncode != 0 or not os.path.exists(content_fitted):
        logging.error(f"❌ Batch block {tag}: content fit ({fit_mode}) failed:\n{r_fit.stderr[-600:]}")
        try:
            os.remove(avatar_normalized)
        except Exception:
            pass
        return None

    fitted_dur = _probe_media_duration(content_fitted)
    logging.info(
        f"   {tag}: content fit ({fit_mode}) | "
        f"in={content_dur:.3f}s → out={fitted_dur:.3f}s (target={avatar_dur:.3f}s)"
    )

    # ────────────────────────────────────────────────────────────────────
    # PASS 2 — visual composite (content_fitted + avatar video), no audio
    # ────────────────────────────────────────────────────────────────────
    av_x, av_y = layout_geom["av_x"], layout_geom["av_y"]
    av_w, av_h = layout_geom["av_w"], layout_geom["av_h"]
    ct_x, ct_y = layout_geom["ct_x"], layout_geom["ct_y"]
    ct_w, ct_h = layout_geom["ct_w"], layout_geom["ct_h"]

    bg_type  = layout_geom["bg_type"]
    bg_color = layout_geom["bg_color"]
    bg_media = None
    temp_bg  = None

    if bg_type == "image":
        temp_bg  = _fetch_bg_media(layout_geom["bg_url"], ".jpg", chunk_folder)
        bg_media = temp_bg
    elif bg_type == "bgvideo":
        temp_bg  = _fetch_bg_media(layout_geom["bg_url"], ".mp4", chunk_folder)
        bg_media = temp_bg

    parts   = []
    current = "base"

    if bg_type == "color":
        parts.append(
            f"color=c=#{bg_color}:s={canvas_w}x{canvas_h}:r=30:d={avatar_dur:.4f},"
            f"format=yuv420p[base]"
        )
    elif bg_type == "blurred":
        parts.append(
            f"[0:v]scale={canvas_w//8}:{canvas_h//8}:flags=bilinear,"
            f"scale={canvas_w}:{canvas_h}:flags=bilinear,"
            f"setsar=1,format=yuv420p[base]"
        )
    elif bg_type == "image" and bg_media:
        parts.append(
            f"[1:v]{_thc_bg_scale(layout_geom, canvas_w, canvas_h)},"
            f"setsar=1,format=yuv420p[base]"
        )
    elif bg_type == "bgvideo" and bg_media:
        parts.append(
            f"[1:v]{_thc_bg_scale(layout_geom, canvas_w, canvas_h)},"
            f"trim=duration={avatar_dur:.4f},setsar=1,format=yuv420p[base]"
        )
    else:
        parts.append(
            f"color=c=#0a0a0a:s={canvas_w}x{canvas_h}:r=30:d={avatar_dur:.4f},"
            f"format=yuv420p[base]"
        )

    # ── Input plan (no -stream_loop anywhere in this call) ─────────────────
    inputs   = ["-i", content_fitted]
    next_idx = 1

    if bg_type in ("image", "bgvideo") and bg_media:
        if bg_type == "bgvideo":
            inputs += ["-stream_loop", "-1"]
        inputs += ["-i", bg_media]
        next_idx += 1

    inputs += ["-i", avatar_normalized]
    avatar_idx = next_idx
    next_idx += 1

    fx = _thc_fx(layout_geom, canvas_w, canvas_h, "narration", chunk_folder, _thc_count_inputs(inputs))
    if fx:
        current = fx.after_base(parts, current)

    if layout_geom["content_layer"]:
        parts.append(
            _ffmpeg_content_box_filter(
                "0:v", "ct", ct_w, ct_h, layout_geom.get("content_fit", "cover")
            )
        )
        ct_label = fx.mask_layer(parts, "content", "ct") if fx else "ct"
        parts.append(
            f"[{current}][{ct_label}]overlay=x={ct_x}:y={ct_y}:format=yuv420[after_ct]"
        )
        current = "after_ct"
        if not fx and layout_geom["ct_border_w"] > 0:
            db = _drawbox(
                ct_x, ct_y, ct_w, ct_h,
                layout_geom["ct_border_w"],
                layout_geom["ct_border_color"],
                layout_geom["ct_border_radius"],
            )
            parts.append(f"[{current}]{db}[after_ct_border]")
            current = "after_ct_border"

    parts.append(
        _ffmpeg_avatar_box_filter(f"{avatar_idx}:v", "av", av_w, av_h)
    )
    av_label = fx.mask_layer(parts, "avatar", "av") if fx else "av"
    parts.append(
        f"[{current}][{av_label}]overlay=x={av_x}:y={av_y}:format=yuv420[after_av]"
    )
    current = "after_av"
    if not fx and layout_geom["av_border_w"] > 0:
        db = _drawbox(
            av_x, av_y, av_w, av_h,
            layout_geom["av_border_w"],
            layout_geom["av_border_color"],
            layout_geom["av_border_radius"],
        )
        parts.append(f"[{current}]{db}[after_av_border]")
        current = "after_av_border"

    if fx:
        current = fx.finish(parts, current)
        inputs += fx.input_args()

    filter_complex = ";".join(parts)
    final_label    = current

    visual_path = os.path.join(chunk_folder, f"visual_{tag}_{ts}.mp4")

    cmd_visual = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{final_label}]",
        "-an",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "constqp",
        "-qp", "18",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        visual_path,
    ]

    r_visual = subprocess.run(cmd_visual, capture_output=True, text=True)

    if temp_bg and os.path.exists(temp_bg):
        try:
            os.remove(temp_bg)
        except Exception:
            pass
    try:
        os.remove(content_fitted)
    except Exception:
        pass

    if r_visual.returncode != 0 or not os.path.exists(visual_path):
        logging.error(f"❌ Batch block {tag}: visual composite failed:\n{r_visual.stderr[-800:]}")
        try:
            os.remove(avatar_normalized)
        except Exception:
            pass
        return None

    # ────────────────────────────────────────────────────────────────────
    # PASS 3 — mux avatar's normalized audio in, stream copy, no re-encode
    # ────────────────────────────────────────────────────────────────────
    output_path = os.path.join(chunk_folder, f"batchblock_{tag}_{ts}.mp4")

    cmd_mux = [
        "ffmpeg", "-y",
        "-i", visual_path,
        "-i", avatar_normalized,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]

    r_mux = subprocess.run(cmd_mux, capture_output=True, text=True)

    try:
        os.remove(visual_path)
    except Exception:
        pass
    try:
        os.remove(avatar_normalized)
    except Exception:
        pass

    if r_mux.returncode != 0 or not os.path.exists(output_path):
        logging.error(f"❌ Batch block {tag}: audio mux failed:\n{r_mux.stderr[-800:]}")
        return None

    out_dur = _probe_media_duration(output_path)
    logging.info(
        f"✅ Batch block {tag} | content_in={content_dur:.3f}s "
        f"avatar={avatar_dur:.3f}s out={out_dur:.3f}s | "
        f"mode={fit_mode} | audio=avatar (copy, untouched) | "
        f"zoom2-style: avoid_negative_ts make_zero + no setpts"
    )
    return output_path
  
# ── FUNCTION 1: clip layout composite (A/V locked) ─────────────────────
def _composite_clip_with_layout(
    clip_chunk,
    layout_geom,
    canvas_w,
    canvas_h,
    chunk_folder,
    tag,
):
    """
    CLIP segment composite — uses clip_layer from layout_config (position +
    size on canvas). Falls back to centered content_layer box when clip_layer
    is omitted (legacy).

    Two-pass (matches batch composite — avoids nvenc filter reinit on long clips):
      1) Fit/trim clip video into the layout box.
      2) Composite fitted clip onto background canvas.
      3) Mux clip audio (stream copy) when present.
    """
    layout_id = layout_geom["layout_id"]
    if layout_id == "full_content":
        return clip_chunk
    has_audio = _clip_has_audio(clip_chunk)
    if has_audio:
        clip_dur = _probe_audio_duration(clip_chunk)
    else:
        clip_dur = _probe_stream_duration(clip_chunk, "video") or _probe_media_duration(clip_chunk)
    if clip_dur <= 0:
        logging.error(f"❌ Clip composite {tag}: clip has no duration")
        return None
    dur_s = f"{clip_dur:.4f}"
    vtrim = f"trim=duration={dur_s},setpts=PTS-STARTPTS"
    cl_w = layout_geom["cl_w"]
    cl_h = layout_geom["cl_h"]
    cl_x = layout_geom["cl_x"]
    cl_y = layout_geom["cl_y"]
    if (layout_id == "full_avatar" and not _thc_v2_config(layout_geom)) or cl_w <= 0 or cl_h <= 0:
        cl_w, cl_h = canvas_w, canvas_h
        cl_x, cl_y = 0, 0
    bg_type  = layout_geom["bg_type"]
    bg_color = layout_geom["bg_color"]
    pad_color = f"0x{bg_color}" if bg_type == "color" else "black"
    clip_fit = layout_geom.get("clip_fit", "contain")
    ts = int(time.time() * 1000)

    # ── PASS 1 — fit clip into layout box (isolated; no shared filtergraph) ──
    clip_fitted = os.path.join(chunk_folder, f"clipfit_{tag}_{ts}.mp4")
    fit_inner = _ffmpeg_content_box_filter("vin", "vout", cl_w, cl_h, clip_fit, pad_color)
    fit_fc = f"[0:v]{vtrim}[vin];{fit_inner}"
    cmd_fit = [
        "ffmpeg", "-y",
        "-i", clip_chunk,
        "-filter_complex", fit_fc,
        "-map", "[vout]",
        "-an",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "constqp",
        "-qp", "18",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-t", dur_s,
        clip_fitted,
    ]
    r_fit = subprocess.run(cmd_fit, capture_output=True, text=True)
    if r_fit.returncode != 0 or not os.path.exists(clip_fitted):
        logging.error(f"❌ Clip composite {tag} pass-1 (fit) failed:\n{r_fit.stderr[-800:]}")
        return None

    fitted_dur = _probe_stream_duration(clip_fitted, "video") or clip_dur
    dur_s = f"{fitted_dur:.4f}"

    # ── PASS 2 — background + overlay fitted clip (video only) ──
    bg_media = None
    temp_bg  = None
    if bg_type == "image":
        temp_bg  = _fetch_bg_media(layout_geom["bg_url"], ".jpg", chunk_folder)
        bg_media = temp_bg
    elif bg_type == "bgvideo":
        temp_bg  = _fetch_bg_media(layout_geom["bg_url"], ".mp4", chunk_folder)
        bg_media = temp_bg

    parts   = []
    current = "base"
    if bg_type == "color":
        parts.append(
            f"color=c=0x{bg_color}:s={canvas_w}x{canvas_h}:r=30:d={dur_s},"
            f"format=yuv420p[base]"
        )
    elif bg_type == "blurred":
        parts.append(
            f"[0:v]split=2[blur_in][fg_in];"
            f"[blur_in]scale={canvas_w//8}:{canvas_h//8}:flags=bilinear,"
            f"scale={canvas_w}:{canvas_h}:flags=bilinear,"
            f"setsar=1,format=yuv420p[base]"
        )
    elif bg_type == "image" and bg_media:
        parts.append(
            f"[1:v]{_thc_bg_scale(layout_geom, canvas_w, canvas_h)},"
            f"setsar=1,format=yuv420p[base]"
        )
    elif bg_type == "bgvideo" and bg_media:
        parts.append(
            f"[1:v]{_thc_bg_scale(layout_geom, canvas_w, canvas_h)},"
            f"trim=duration={dur_s},setsar=1,format=yuv420p[base]"
        )
    else:
        parts.append(
            f"color=c=0x0a0a0a:s={canvas_w}x{canvas_h}:r=30:d={dur_s},"
            f"format=yuv420p[base]"
        )

    inputs = ["-i", clip_fitted]
    if bg_type in ("image", "bgvideo") and bg_media:
        if bg_type == "bgvideo":
            inputs += ["-stream_loop", "-1"]
        inputs += ["-i", bg_media]

    fx = _thc_fx(layout_geom, canvas_w, canvas_h, "clip", chunk_folder, _thc_count_inputs(inputs))
    if fx:
        current = fx.after_base(parts, current)
    clip_src = "fg_in" if bg_type == "blurred" else "0:v"
    clip_label = fx.mask_layer(parts, "clip", clip_src) if fx else clip_src
    parts.append(
        f"[{current}][{clip_label}]overlay=x={cl_x}:y={cl_y}:format=auto[after_ct]"
    )
    current = "after_ct"

    clip_border_w = layout_geom.get("cl_border_w", layout_geom["ct_border_w"])
    clip_border_color = layout_geom.get("cl_border_color", layout_geom["ct_border_color"])
    clip_border_radius = layout_geom.get("cl_border_radius", layout_geom["ct_border_radius"])
    if not fx and clip_border_w > 0:
        db = _drawbox(
            cl_x, cl_y, cl_w, cl_h,
            clip_border_w,
            clip_border_color,
            clip_border_radius,
        )
        parts.append(f"[{current}]{db}[after_ct_border]")
        current = "after_ct_border"

    if fx:
        current = fx.finish(parts, current)
        inputs += fx.input_args()

    filter_complex = ";".join(parts)
    visual_path = os.path.join(chunk_folder, f"clipvis_{tag}_{ts}.mp4")
    cmd_visual = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{current}]",
        "-an",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "constqp",
        "-qp", "18",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        visual_path,
    ]
    r_visual = subprocess.run(cmd_visual, capture_output=True, text=True)
    try:
        os.remove(clip_fitted)
    except Exception:
        pass
    if temp_bg and os.path.exists(temp_bg):
        try:
            os.remove(temp_bg)
        except Exception:
            pass
    if r_visual.returncode != 0 or not os.path.exists(visual_path):
        logging.error(f"❌ Clip composite {tag} pass-2 (visual) failed:\n{r_visual.stderr[-800:]}")
        return None

    output_path = os.path.join(chunk_folder, f"clipcomposited_{tag}_{ts}.mp4")
    if not has_audio:
        try:
            os.rename(visual_path, output_path)
        except Exception:
            shutil.copy2(visual_path, output_path)
            os.remove(visual_path)
    else:
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", visual_path,
            "-i", clip_chunk,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-t", dur_s,
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
        r_mux = subprocess.run(cmd_mux, capture_output=True, text=True)
        try:
            os.remove(visual_path)
        except Exception:
            pass
        if r_mux.returncode != 0 or not os.path.exists(output_path):
            logging.error(f"❌ Clip composite {tag} pass-3 (mux) failed:\n{r_mux.stderr[-800:]}")
            return None

    out_dur = _probe_audio_duration(output_path) if has_audio else _probe_stream_duration(output_path, "video")
    logging.info(
        f"✅ Clip composite {tag} | in={clip_dur:.3f}s out={out_dur:.3f}s | "
        f"at ({cl_x},{cl_y}) {cl_w}x{cl_h} | "
        f"A/V locked ({'audio master' if has_audio else 'video only'})"
    )
    return output_path

def _apply_batch_avatar_composite(
    chunk_paths,
    avatar_segment_map,
    vid_segments,
    layout_config,
    canvas_w,
    canvas_h,
    chunk_folder,
):
    """
    CLIP VIDEOS ONLY — batch-level composite BEFORE final concat.

    Replaces each group of narration segment chunks (one avatar batch's
    worth) with a single composited block: avatar plays naturally,
    content trimmed/looped to match, avatar's own audio is final audio.

    Clip segments get the same layout treatment (background, border,
    content-box size) as narration blocks, but CENTERED on canvas since
    there's no avatar to share space with. Original audio + duration
    preserved.
    """
    layout_id = layout_config.get("layout_id", "")
    if layout_id == "full_content":
        logging.info("full_content — skipping batch avatar composite")
        return chunk_paths

    if not avatar_segment_map:
        logging.error("❌ avatar_segment_map is empty")
        return chunk_paths

    layout_geom = _extract_layout_geometry(layout_config, canvas_w, canvas_h)
    groups      = _group_batches_from_map(avatar_segment_map)

    logging.info(
        f"🎭 Batch composite (clip video) | layout={layout_id} | "
        f"batches={len(groups)} | segments={len(vid_segments)}"
    )

    # Build a lookup: seg_idx → batch_index (for narration segments)
    seg_to_batch = {}
    for bidx, g in groups.items():
        for seg_idx in g["seg_indices"]:
            seg_to_batch[seg_idx] = bidx

    new_paths = []
    i = 0
    n = len(vid_segments)
    batch_count = 0
    clip_count  = 0
    fail_count  = 0

    while i < n:
        seg = vid_segments[i]
        chunk = chunk_paths[i] if i < len(chunk_paths) else None

        if not chunk or not os.path.exists(chunk):
            logging.error(f"❌ Segment {i}: missing chunk, skipping")
            new_paths.append(chunk)
            fail_count += 1
            i += 1
            continue

        if _is_clip_segment(seg):
            # Same layout treatment as narration blocks (background,
            # border, content-box size) but CENTERED — no avatar to
            # share space with. Original audio + duration preserved.
            clip_count += 1
            composited_clip = _composite_clip_with_layout(
                clip_chunk   = chunk,
                layout_geom  = layout_geom,
                canvas_w     = canvas_w,
                canvas_h     = canvas_h,
                chunk_folder = chunk_folder,
                tag          = f"seg{i}",
            )
            if composited_clip:
                logging.info(f"🎬 Segment {i}: CLIP — centered layout composite")
                new_paths.append(composited_clip)
            else:
                logging.warning(f"⚠️ Segment {i}: CLIP composite failed — keeping raw chunk")
                new_paths.append(chunk)
            i += 1
            continue

        bidx = seg_to_batch.get(i)
        if bidx is None:
            logging.warning(f"⚠️ Segment {i}: narration but no batch mapping — keeping raw chunk")
            new_paths.append(chunk)
            i += 1
            continue

        group       = groups[bidx]
        seg_indices = group["seg_indices"]

        if seg_indices[0] != i:
            # Shouldn't happen if groups are contiguous and we walk in order,
            # but guard against malformed maps.
            logging.warning(
                f"⚠️ Segment {i}: batch {bidx} group starts at {seg_indices[0]}, "
                f"not {i} — keeping raw chunk"
            )
            new_paths.append(chunk)
            i += 1
            continue

        if layout_id == "full_avatar":
            composited = _composite_full_avatar_batch_block(
                avatar_batch_path=group["path"],
                layout_geom=layout_geom,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                chunk_folder=chunk_folder,
                tag=f"b{bidx}",
            )
            if composited:
                new_paths.append(composited)
                batch_count += 1
            else:
                logging.warning(
                    f"⚠️ Batch {bidx}: full_avatar composite failed — keeping raw chunks"
                )
                for seg_idx in seg_indices:
                    if seg_idx < len(chunk_paths):
                        new_paths.append(chunk_paths[seg_idx])
                        fail_count += 1
            i += len(seg_indices)
            continue

        # Collect this batch's segment chunks (must be contiguous run i..i+len-1)
        block_chunks = []
        ok = True
        for offset, seg_idx in enumerate(seg_indices):
            expected = i + offset
            if seg_idx != expected:
                logging.warning(
                    f"⚠️ Batch {bidx}: non-contiguous seg_indices {seg_indices} "
                    f"at offset {offset} (expected {expected}) — keeping raw chunks"
                )
                ok = False
                break
            c = chunk_paths[seg_idx] if seg_idx < len(chunk_paths) else None
            if not c or not os.path.exists(c):
                logging.error(f"❌ Batch {bidx}: segment {seg_idx} chunk missing")
                ok = False
                break
            block_chunks.append(c)

        if not ok or not block_chunks:
            for seg_idx in seg_indices:
                if seg_idx < len(chunk_paths):
                    new_paths.append(chunk_paths[seg_idx])
                    fail_count += 1
            i += len(seg_indices)
            continue

        content_block = _concat_chunks(block_chunks, chunk_folder, tag=f"b{bidx}")
        if not content_block:
            logging.error(f"❌ Batch {bidx}: content block concat failed — keeping raw chunks")
            new_paths.extend(block_chunks)
            fail_count += 1
            i += len(seg_indices)
            continue

        composited = _composite_batch_block(
            content_block     = content_block,
            avatar_batch_path = group["path"],
            layout_geom       = layout_geom,
            canvas_w          = canvas_w,
            canvas_h          = canvas_h,
            chunk_folder      = chunk_folder,
            tag               = f"b{bidx}",
        )

        if content_block not in block_chunks and os.path.exists(content_block):
            try:
                os.remove(content_block)
            except Exception:
                pass

        if composited:
            new_paths.append(composited)
            batch_count += 1
        else:
            logging.warning(f"⚠️ Batch {bidx}: composite failed — keeping raw chunks")
            new_paths.extend(block_chunks)
            fail_count += 1

        i += len(seg_indices)

    logging.info("")
    logging.info("=" * 55)
    logging.info("🎭 BATCH COMPOSITE SUMMARY (CLIP VIDEO)")
    logging.info(f"   batches composited : {batch_count}")
    logging.info(f"   clips (centered)   : {clip_count}")
    logging.info(f"   failures           : {fail_count}")
    logging.info("=" * 55)

    return new_paths

# ════════════════════════════════════════════════════════════════════════
# NO-CLIP VIDEOS
# ════════════════════════════════════════════════════════════════════════

def _concat_avatar_batches(avatar_segment_map, chunk_folder):
    """
    Stream-copy concat of avatar batches — mirrors zoom2's concat_avatars.
    """
    batches = {}
    for _, info in avatar_segment_map.items():
        b = info["batch_index"]
        if b not in batches:
            batches[b] = info["path"]

    ordered_paths = [batches[b] for b in sorted(batches)]
    if not ordered_paths:
        logging.error("❌ avatar_segment_map produced no batch paths")
        return None

    avatar_track = os.path.join(
        chunk_folder, f"avatar_track_{int(time.time() * 1000)}.mp4"
    )
    concat_txt = os.path.join(
        chunk_folder, f"avatar_concat_{int(time.time() * 1000)}.txt"
    )

    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in ordered_paths:
            f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")

    logging.info(
        f"📦 Concat {len(ordered_paths)} batch(es): "
        f"{[os.path.basename(p) for p in ordered_paths]}"
    )

    t0 = time.time()
    r = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        avatar_track,
    ], capture_output=True, text=True)

    if r.returncode != 0 or not os.path.exists(avatar_track):
        logging.error(f"❌ Avatar batch concat failed:\n{r.stderr[-800:]}")
        return None

    dur = _probe_media_duration(avatar_track)
    mb  = os.path.getsize(avatar_track) / (1024 * 1024)
    logging.info(
        f"✅ Avatar track ready | {dur:.3f}s | {mb:.1f}MB | "
        f"{time.time() - t0:.1f}s"
    )
    return avatar_track, concat_txt

def _composite_avatar_over_full_video(
    content_video,
    avatar_segment_map,
    layout_config,
    canvas_w,
    canvas_h,
    chunk_folder,
):
    """
    NO-CLIP VIDEOS — avatar is the master clock.

    Avatar batches are concatenated into a single avatar_track. The
    content_video is fitted to EXACTLY avatar_track's duration:
      - content shorter than avatar → FREEZE LAST FRAME for the remainder
      - content longer  than avatar → TRIM to avatar duration
      - equal                       → re-encode to exact length

    Visual composite uses content_fitted + avatar_track (avatar PTS
    untouched, no setpts). Final audio is avatar_track's own audio only
    (stream copy) — guarantees lip sync regardless of content length.
    """
    layout_id = layout_config.get("layout_id", "")
    if layout_id == "full_content":
        logging.info("full_content — skipping avatar composite")
        return None

    if not avatar_segment_map:
        logging.error("❌ avatar_segment_map is empty")
        return None

    layout_geom_early = _extract_layout_geometry(layout_config, canvas_w, canvas_h)

    if layout_id == "full_avatar":
        logging.info(
            "⚡ full_avatar — avatar-only export (skipped content download/composite)"
        )
        concat_result = _concat_avatar_batches(avatar_segment_map, chunk_folder)
        if not concat_result:
            return None
        avatar_track, concat_txt = concat_result
        avatar_normalized = _normalize_avatar_batch_input(
            avatar_track, chunk_folder, "full"
        )
        for p in (avatar_track, concat_txt):
            try:
                os.remove(p)
            except Exception:
                pass
        if not avatar_normalized:
            return None
        out = _composite_full_avatar_from_normalized_avatar(
            avatar_normalized,
            canvas_w,
            canvas_h,
            chunk_folder,
            "full",
            layout_geom=layout_geom_early,
        )
        try:
            os.remove(avatar_normalized)
        except Exception:
            pass
        return out

    layout_geom = layout_geom_early

    av_x, av_y = layout_geom["av_x"], layout_geom["av_y"]
    av_w, av_h = layout_geom["av_w"], layout_geom["av_h"]
    ct_x, ct_y = layout_geom["ct_x"], layout_geom["ct_y"]
    ct_w, ct_h = layout_geom["ct_w"], layout_geom["ct_h"]

    content_dur = _probe_media_duration(content_video)

    logging.info("🎭 mode=direct (no clips, avatar-master)")

    concat_result = _concat_avatar_batches(avatar_segment_map, chunk_folder)
    if not concat_result:
        return None
    avatar_track, concat_txt = concat_result
    av_track_dur = _probe_media_duration(avatar_track)

    if av_track_dur <= 0:
        logging.error("❌ avatar_track has no duration")
        for p in (avatar_track, concat_txt):
            try:
                os.remove(p)
            except Exception:
                pass
        return None

    logging.info(
        f"⏱️  avatar_file={av_track_dur:.3f}s | content={content_dur:.3f}s"
    )

    ts = int(time.time() * 1000)

    # ────────────────────────────────────────────────────────────────────
    # PASS 1 — fit content_video to EXACTLY av_track_dur.
    # Avatar is master (own audio = lip sync truth). Content is capped/
    # padded to match — never the reverse.
    #   - content shorter than avatar → FREEZE LAST FRAME for remainder
    #   - content longer  than avatar → TRIM to avatar duration
    #   - equal                       → re-encode to exact length
    # Isolated single-input call — no shared filtergraph with avatar.
    # ────────────────────────────────────────────────────────────────────
    content_fitted = os.path.join(chunk_folder, f"fullfit_{ts}.mp4")

    if content_dur < av_track_dur - 0.01:
        pad_dur = av_track_dur - content_dur
        cmd_fit = [
            "ffmpeg", "-y",
            "-i", content_video,
            "-vf", f"tpad=stop_mode=clone:stop_duration={pad_dur:.4f}",
            "-t", f"{av_track_dur:.4f}",
            "-c:v", "h264_nvenc", "-preset", "p4",
            "-pix_fmt", "yuv420p",
            "-an",
            content_fitted,
        ]
        fit_mode = "freeze_last_frame"
    elif content_dur > av_track_dur + 0.01:
        cmd_fit = [
            "ffmpeg", "-y",
            "-i", content_video,
            "-t", f"{av_track_dur:.4f}",
            "-c:v", "h264_nvenc", "-preset", "p4",
            "-pix_fmt", "yuv420p",
            "-an",
            content_fitted,
        ]
        fit_mode = "trim"
    else:
        cmd_fit = [
            "ffmpeg", "-y",
            "-i", content_video,
            "-t", f"{av_track_dur:.4f}",
            "-c:v", "h264_nvenc", "-preset", "p4",
            "-pix_fmt", "yuv420p",
            "-an",
            content_fitted,
        ]
        fit_mode = "exact"

    r_fit = subprocess.run(cmd_fit, capture_output=True, text=True)
    if r_fit.returncode != 0 or not os.path.exists(content_fitted):
        logging.error(f"❌ content fit ({fit_mode}) failed:\n{r_fit.stderr[-600:]}")
        for p in (avatar_track, concat_txt):
            try:
                os.remove(p)
            except Exception:
                pass
        return None

    fitted_dur = _probe_media_duration(content_fitted)
    logging.info(
        f"   content fit ({fit_mode}) | in={content_dur:.3f}s → "
        f"out={fitted_dur:.3f}s (target={av_track_dur:.3f}s)"
    )

    # ────────────────────────────────────────────────────────────────────
    # Visual composite — content_fitted + avatar_track
    # ────────────────────────────────────────────────────────────────────
    bg_type  = layout_geom["bg_type"]
    bg_color = layout_geom["bg_color"]
    bg_media = None

    if bg_type == "image":
        bg_media = _fetch_bg_media(layout_geom["bg_url"], ".jpg", chunk_folder)
    elif bg_type == "bgvideo":
        bg_media = _fetch_bg_media(layout_geom["bg_url"], ".mp4", chunk_folder)

    logging.info(f"🎨 BG type={bg_type}")

    # input 0 : content_fitted (video only, duration == avatar duration)
    # input 1 : avatar track (always)
    # input 2+: optional bg media (image/bgvideo only)
    bg_input_idx = 2
    parts        = []
    current      = "base"

    if bg_type == "blurred":
        parts.append(
            f"[0:v]scale={canvas_w//8}:{canvas_h//8}:flags=bilinear,"
            f"scale={canvas_w}:{canvas_h}:flags=bilinear,"
            f"setsar=1,format=yuv420p[base]"
        )
    elif bg_type == "color":
        parts.append(
            f"color=c=#{bg_color}:s={canvas_w}x{canvas_h}:r=30:d={av_track_dur:.4f},"
            f"format=yuv420p[base]"
        )
    elif bg_type == "image" and bg_media:
        parts.append(
            f"[{bg_input_idx}:v]{_thc_bg_scale(layout_geom, canvas_w, canvas_h)},"
            f"setsar=1,format=yuv420p[base]"
        )
    elif bg_type == "bgvideo" and bg_media:
        parts.append(
            f"[{bg_input_idx}:v]{_thc_bg_scale(layout_geom, canvas_w, canvas_h)},"
            f"trim=duration={av_track_dur:.4f},"
            f"setsar=1,format=yuv420p[base]"
        )
    else:
        parts.append(
            f"color=c=#0a0a0a:s={canvas_w}x{canvas_h}:r=30:d={av_track_dur:.4f},"
            f"format=yuv420p[base]"
        )

    # Designed canvas (style_version 2) — stills appended after the bg input
    fx = _thc_fx(
        layout_geom, canvas_w, canvas_h, "narration", chunk_folder,
        2 + (1 if bg_type in ("image", "bgvideo") and bg_media else 0),
    )
    if fx:
        current = fx.after_base(parts, current)

    # Content layer
    if layout_geom["content_layer"] and layout_id != "full_avatar":
        parts.append(
            _ffmpeg_content_box_filter(
                "0:v", "ct", ct_w, ct_h, layout_geom.get("content_fit", "cover")
            )
        )
        ct_label = fx.mask_layer(parts, "content", "ct") if fx else "ct"
        parts.append(
            f"[{current}][{ct_label}]overlay=x={ct_x}:y={ct_y}:format=yuv420[after_ct]"
        )
        current = "after_ct"
        if not fx and layout_geom["ct_border_w"] > 0:
            db = _drawbox(
                ct_x, ct_y, ct_w, ct_h,
                layout_geom["ct_border_w"],
                layout_geom["ct_border_color"],
                layout_geom["ct_border_radius"],
            )
            parts.append(f"[{current}]{db}[after_ct_border]")
            current = "after_ct_border"

    # Avatar overlay — NO setpts, PTS passed through as-is
    sw, sh, ox, oy = av_w, av_h, av_x, av_y
    parts.append(
        _ffmpeg_avatar_box_filter("1:v", "av", sw, sh)
    )
    av_label = fx.mask_layer(parts, "avatar", "av") if fx else "av"
    parts.append(
        f"[{current}][{av_label}]overlay=x={ox}:y={oy}:format=yuv420[after_av]"
    )
    current = "after_av"

    if not fx and layout_geom["av_border_w"] > 0 and layout_id not in ("full_content",):
        db = _drawbox(
            av_x, av_y, av_w, av_h,
            layout_geom["av_border_w"],
            layout_geom["av_border_color"],
            layout_geom["av_border_radius"],
        )
        parts.append(f"[{current}]{db}[after_av_border]")
        current = "after_av_border"

    if fx:
        current = fx.finish(parts, current)

    filter_complex = ";".join(parts)
    final_label    = current

    logging.info(
        f"🔧 filter_complex ({len(filter_complex)} chars):\n{filter_complex}"
    )

    # ── inputs — 0 = content_fitted, 1 = avatar_track ──────────────────────
    inputs = [
        "-hwaccel", "cuda", "-i", content_fitted,
        "-hwaccel", "cuda", "-i", avatar_track,
    ]
    if bg_type in ("image", "bgvideo") and bg_media:
        if bg_type == "bgvideo":
            inputs += ["-stream_loop", "-1"]
        inputs += ["-hwaccel", "cuda", "-i", bg_media]
    if fx:
        inputs += fx.input_args()

    visual_path = os.path.join(chunk_folder, f"composited_visual_{ts}.mp4")

    cmd_visual = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map",       f"[{final_label}]",
        "-an",
        "-c:v",       "h264_nvenc",
        "-preset",    "p4",
        "-rc",        "constqp",
        "-qp",        "18",
        "-profile:v", "high",
        "-pix_fmt",   "yuv420p",
        visual_path,
    ]

    logging.info("🚀 NVENC visual composite (avatar-master, no audio)…")
    t0      = time.time()
    result  = subprocess.run(cmd_visual, capture_output=True, text=True)
    elapsed = time.time() - t0

    try:
        os.remove(content_fitted)
    except Exception:
        pass
    if bg_media and os.path.exists(bg_media):
        try:
            os.remove(bg_media)
        except Exception:
            pass

    if result.returncode != 0 or not os.path.exists(visual_path):
        logging.error(
            f"❌ Visual composite failed ({elapsed:.1f}s):\n{result.stderr[-1500:]}"
        )
        for p in (avatar_track, concat_txt):
            try:
                os.remove(p)
            except Exception:
                pass
        return None

    # ────────────────────────────────────────────────────────────────────
    # PASS 2 — mux avatar_track's own audio in, stream copy, no re-encode.
    # Avatar's audio is the ONLY audio → guaranteed lip sync.
    # ────────────────────────────────────────────────────────────────────
    output_path = os.path.join(chunk_folder, f"composited_{int(time.time())}.mp4")

    cmd_mux = [
        "ffmpeg", "-y",
        "-i", visual_path,
        "-i", avatar_track,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]

    r_mux = subprocess.run(cmd_mux, capture_output=True, text=True)

    for p in (visual_path, avatar_track, concat_txt):
        try:
            os.remove(p)
        except Exception:
            pass

    if r_mux.returncode != 0 or not os.path.exists(output_path):
        logging.error(f"❌ Audio mux failed:\n{r_mux.stderr[-800:]}")
        return None

    out_dur = _probe_media_duration(output_path)
    mb      = os.path.getsize(output_path) / (1024 * 1024)

    logging.info("")
    logging.info("=" * 55)
    logging.info("🎉 COMPOSITE DONE (AVATAR-MASTER)")
    logging.info(f"   output  : {output_path}")
    logging.info(f"   size    : {mb:.1f} MB")
    logging.info(f"   time    : {elapsed:.1f}s")
    logging.info(f"   content : {content_dur:.3f}s (fitted: {fitted_dur:.3f}s, mode={fit_mode})")
    logging.info(f"   avatar  : {av_track_dur:.3f}s")
    logging.info(f"   output  : {out_dur:.3f}s")
    logging.info(f"   audio   : avatar (copy, untouched) — lip sync guaranteed")
    logging.info("=" * 55)

    return output_path


def _normalize_audio_only(chunk_path, chunk_folder, tag, apply_loudnorm=False):
    """
    Re-encode audio to 44100Hz/stereo/aac, stream-copy video.

    When apply_loudnorm is True, also run EBU R128 loudnorm so DreamFace
    avatar narration and hot clip-source audio land at the same level.
    """
    if not chunk_path or not os.path.exists(chunk_path):
        return None
    out_path = os.path.join(chunk_folder, f"normaud_{tag}_{int(time.time()*1000)}.mp4")
    has_audio = _clip_has_audio(chunk_path)
    cmd = ["ffmpeg", "-y", "-i", chunk_path, "-c:v", "copy"]
    if has_audio:
        if apply_loudnorm:
            cmd += [
                "-af",
                (
                    f"loudnorm=I={PIPELINE_AUDIO_TARGET_LUFS}:TP=-1.5:LRA=11"
                ),
            ]
        cmd += [
            "-ar", "44100", "-ac", "2",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
        ]
    cmd += ["-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_path):
        logging.error(f"❌ Audio normalize failed ({tag}):\n{r.stderr[-600:]}")
        return None
    return out_path

def _prepare_chunk_for_concat(src, dst, loudness_normalize=False):
    """
    Remux chunk to PTS=0 + uniform audio before final concat.
    Prevents 40-min timeline inflation and A/V drift at segment joins.

    When loudness_normalize is True (talking-heads + clip videos), each chunk
    is brought to PIPELINE_AUDIO_TARGET_LUFS so avatar commentary and clip
    audio feel consistent at segment transitions.
    """
    if not src or not os.path.exists(src):
        return None

    has_audio = _clip_has_audio(src)
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-map", "0:v:0",
    ]
    if has_audio:
        cmd += ["-map", "0:a:0"]
    cmd += ["-c:v", "copy"]
    if has_audio:
        if loudness_normalize:
            cmd += [
                "-af",
                (
                    f"loudnorm=I={PIPELINE_AUDIO_TARGET_LUFS}:TP=-1.5:LRA=11"
                ),
            ]
        cmd += [
            "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "192k",
        ]
    cmd += [
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        "-movflags", "+faststart",
        dst,
    ]
    label = (
        f"chunk-prep+loudnorm {os.path.basename(src)}"
        if loudness_normalize and has_audio
        else f"chunk-prep {os.path.basename(src)}"
    )
    r = _ffmpeg_run(
        cmd,
        timeout=FFMPEG_CHUNK_PREP_TIMEOUT,
        label=label,
    )
    if loudness_normalize and has_audio and (
        not r or r.returncode != 0 or not os.path.exists(dst)
    ):
        logging.warning(
            f"⚠️ Loudnorm prep failed for {os.path.basename(src)} — "
            "falling back to format-only prep"
        )
        return _prepare_chunk_for_concat(src, dst, loudness_normalize=False)

    if not r or r.returncode != 0 or not os.path.exists(dst):
        if r:
            logging.error(f"❌ Chunk prep failed:\n{r.stderr[-800:]}")
        return None
    return dst


def _loudnorm_measure(src):
    """Pass 1 of EBU R128 loudnorm — returns the measured values (or None)."""
    r = _ffmpeg_run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", src, "-map", "0:a:0",
            "-af", f"loudnorm=I={PIPELINE_AUDIO_TARGET_LUFS}:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        timeout=max(FFMPEG_CHUNK_PREP_TIMEOUT, 600),
        label=f"loudnorm measure {os.path.basename(src)}",
    )
    if not r or r.returncode != 0:
        return None
    txt = r.stderr or ""
    i = txt.rfind("{")
    j = txt.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return None


def _loudness_match_chunk(src, dst):
    """
    Two-pass loudnorm (linear = one steady gain, no pumping) on ONE chunk.
    Silent / muted chunks are left untouched. Never accepted if any audio
    would be lost: the result must be at least as long as the source audio.
    """
    from helper.banner_backend import _decoded_audio_seconds

    if not src or not os.path.exists(src) or not _clip_has_audio(src):
        return src
    m = _loudnorm_measure(src)
    try:
        measured_i = float(m["input_i"]) if m else float("-inf")
    except (TypeError, ValueError, KeyError):
        measured_i = float("-inf")
    if not m or not (measured_i > -60.0):
        return src
    af = (
        f"loudnorm=I={PIPELINE_AUDIO_TARGET_LUFS}:TP=-1.5:LRA=11:"
        f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
        f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
        f"offset={m['target_offset']}:linear=true,aresample=48000"
    )
    r = _ffmpeg_run(
        [
            "ffmpeg", "-y", "-i", src, "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy", "-af", af,
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
            "-movflags", "+faststart", dst,
        ],
        timeout=max(FFMPEG_CHUNK_PREP_TIMEOUT, 600),
        label=f"loudnorm apply {os.path.basename(src)}",
    )
    if not r or r.returncode != 0 or not os.path.exists(dst):
        logging.warning(f"⚠️ Loudness match failed for {os.path.basename(src)} — keeping original audio")
        return src
    a_in = _decoded_audio_seconds(src)
    a_out = _decoded_audio_seconds(dst)
    if a_out + 0.0005 < a_in:
        logging.warning(
            f"⚠️ Loudness match would shorten audio ({a_in:.4f}s → {a_out:.4f}s) "
            f"for {os.path.basename(src)} — keeping original"
        )
        return src
    return dst


def _loudness_match_chunks(chunk_paths, chunk_folder):
    """Level every chunk to PIPELINE_AUDIO_TARGET_LUFS in parallel (before banners)."""
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()

    def _one(item):
        idx, p = item
        if not p:
            return p
        dst = os.path.join(chunk_folder, f"leveled_{idx}_{int(time.time() * 1000)}.mp4")
        return _loudness_match_chunk(p, dst)

    workers = max(1, int(os.getenv("LOUDNESS_WORKERS", "6")))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_one, enumerate(chunk_paths)))
    changed = sum(1 for a, b in zip(chunk_paths, out) if a != b)
    logging.info(
        f"🔊 Loudness matched {changed}/{len(chunk_paths)} chunk(s) to "
        f"{PIPELINE_AUDIO_TARGET_LUFS} LUFS (two-pass, steady gain) in {time.time() - t0:.1f}s"
    )
    return out


def _encode_narration_to_audio_length(
    video_path,
    segment_audio_path,
    duration_needed,
    chunk_path,
    canvas_w,
    canvas_h,
    video_codec,
    preset_val,
    mute_source=False,
    nvenc_semaphore=None,
    segment_index=None,
    ffmpeg_timeout=FFMPEG_SEGMENT_TIMEOUT,
):
    """
    Narration rule: output duration == narration audio duration.
    Video loops/trims to fit; audio is narration; PTS zeroed.
    """
    if mute_source:
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", video_path,
            "-i", segment_audio_path,
            "-filter_complex",
            (
                f"[0:v]trim=duration={duration_needed:.4f},"
                f"setpts=PTS-STARTPTS,fps=30,"
                f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                f"crop={canvas_w}:{canvas_h},format=yuv420p[v]"
            ),
            "-map", "[v]",
            "-map", "1:a:0",
            "-c:v", video_codec, "-preset", preset_val, *_segment_rate_args(video_codec),
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-r", "30", "-vsync", "cfr",
            "-shortest",
            "-avoid_negative_ts", "make_zero",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            chunk_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", segment_audio_path,
            "-filter_complex",
            (
                f"[0:v]trim=duration={duration_needed:.4f},"
                f"setpts=PTS-STARTPTS,fps=30,"
                f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                f"crop={canvas_w}:{canvas_h},format=yuv420p[v]"
            ),
            "-map", "[v]",
            "-map", "1:a:0",
            "-c:v", video_codec, "-preset", preset_val, *_segment_rate_args(video_codec),
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-r", "30", "-vsync", "cfr",
            "-shortest",
            "-avoid_negative_ts", "make_zero",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            chunk_path,
        ]

    label = (
        f"seg {segment_index} narration-mux"
        if segment_index is not None
        else "narration-mux"
    )
    uses_nvenc = "nvenc" in (video_codec or "")
    r = _ffmpeg_run(
        cmd,
        timeout=ffmpeg_timeout,
        label=label,
        nvenc_semaphore=nvenc_semaphore,
        uses_nvenc=uses_nvenc,
    )
    if not r or r.returncode != 0:
        if r:
            logging.error(f"❌ Narration encode failed:\n{r.stderr[-800:]}")
        return None
    return chunk_path


def _encode_clip_segment(
    src,
    chunk_path,
    canvas_w,
    canvas_h,
    video_codec,
    preset_val,
    mute=False,
    nvenc_semaphore=None,
    segment_index=None,
    ffmpeg_timeout=FFMPEG_SEGMENT_TIMEOUT,
):
    """
    CLIP rule: play external clip out fully; A/V stay together.
    """
    encode_src = src
    repaired_path = None

    def _run_encode(source_path):
        clip_dur = _probe_stream_duration(source_path, "video") or _probe_media_duration(
            source_path
        )
        if clip_dur <= 0:
            logging.error(
                "❌ CLIP segment: no duration | seg=%s | src=%s",
                segment_index,
                source_path,
            )
            return None, 0.0, False

        has_audio = _clip_has_audio(source_path) and not mute

        vf = (
            f"fps=30,setpts=PTS-STARTPTS,"
            f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease,"
            f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
        )

        cmd = [
            "ffmpeg", "-y",
            "-fflags", "+genpts+discardcorrupt",
            "-i", source_path,
            "-vf", vf,
            "-t", f"{clip_dur:.4f}",
            "-r", "30", "-vsync", "cfr",
            "-c:v", video_codec, "-preset", preset_val, *_segment_rate_args(video_codec),
            "-pix_fmt", "yuv420p",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            chunk_path,
        ]

        if has_audio:
            cmd += [
                "-map", "0:v:0", "-map", "0:a:0",
                "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "192k",
            ]
        else:
            cmd += ["-map", "0:v:0", "-an"]

        label = (
            f"seg {segment_index} clip-encode"
            if segment_index is not None
            else "clip-encode"
        )
        uses_nvenc = "nvenc" in (video_codec or "")
        r = _ffmpeg_run(
            cmd,
            timeout=ffmpeg_timeout,
            label=label,
            nvenc_semaphore=nvenc_semaphore,
            uses_nvenc=uses_nvenc,
        )
        if not r or r.returncode != 0:
            if r:
                logging.error(
                    "❌ CLIP encode failed | seg=%s | src=%s\n%s",
                    segment_index,
                    source_path,
                    r.stderr[-800:],
                )
            return None, clip_dur, has_audio
        return r, clip_dur, has_audio

    logging.info(
        "🎬 CLIP encode | seg=%s | src=%s",
        segment_index,
        os.path.basename(src),
    )

    r, clip_dur, has_audio = _run_encode(encode_src)
    if r is None:
        repaired_path = os.path.join(
            os.path.dirname(chunk_path),
            f"clip_repair_{segment_index}_{uuid.uuid4().hex[:8]}.mp4",
        )
        if _repair_clip_source_for_encode(src, repaired_path):
            encode_src = repaired_path
            r, clip_dur, has_audio = _run_encode(encode_src)

    if repaired_path and os.path.exists(repaired_path):
        try:
            os.remove(repaired_path)
        except OSError:
            pass

    if r is None:
        return None

    logging.info(f"✅ CLIP encoded | dur={clip_dur:.3f}s | audio={has_audio}")
    return chunk_path




def create_video(
    phase_media,
    full_audio,
    mode,
    avatar,
    background_video,
    background,
    user_id,
    section_type,
    orientation,
    storyboard,
    kenburns,
    srt,
    avatar_active,
    caption_settings,
    captions_enabled,
    transition_config=None,
    script=None,
    clip_data_list=None,
    image_data_list=None,
    segment_audio_paths=None,
    narration_text=None,
    characters=None,
    vid_segments=None,
    asset_folder=None,
    output_dir=OUTPUT,
    layout_config=None,
    avatar_segment_map=None,
    intro_teaser_config=None,
    narration_images_config=None,
    transition_banner_config=None,
):
    """
    Main video creation function with support for smooth transitions
    between segments. Includes comprehensive validation and failsafe
    mechanisms.

    Talking heads composite strategy:
      - NO CLIPS  → post-assembly batch concat + overlay (zoom2.py)
      - WITH CLIPS → batch-level avatar composite BEFORE concat,
                      then final chunk prep loudness-matches every unit
                      (avatar batchblocks + clipcomposited segments) to
                      PIPELINE_AUDIO_TARGET_LUFS before copy-concat

    layout_config + avatar_segment_map never touch encode_video_segment workers.
    """
    global CURRENT_WIDTH, CURRENT_HEIGHT, CURRENT_ORIENTATION
    global CURRENT_STORYBOARD_ENABLED, CURRENT_KENBURNS_ENABLED

    from helper.teaser_backend import normalize_teaser_config
    from helper.banner_backend import (
        normalize_transition_banner,
        apply_banner_boundary_transitions,
    )
    teaser_cfg = normalize_teaser_config(intro_teaser_config or {})
    narration_images_cfg = narration_images_config or {}
    banner_state = normalize_transition_banner(transition_banner_config or {})

    logging.info(
        f"📺 Intro teaser active | text={teaser_cfg.get('chyronText')!r} | "
        f"layout={teaser_cfg.get('layoutMode')} | pos=({teaser_cfg.get('xPct')},{teaser_cfg.get('yPct')})"
    )
    logging.info(f"🖼 Narration image mode={narration_images_cfg.get('mode', 'default')}")
    logging.info(
        f"🎬 Transition banner | enabled={banner_state.get('enabled')} | "
        f"preset={banner_state.get('preset_id')} | "
        f"transition={banner_state.get('config', {}).get('transitionType')}"
    )


    print("NEW VALUES", orientation, storyboard, kenburns)
    print("Transition Config", transition_config)
    print(
        "Talking Heads mode:",
        layout_config is not None and avatar_segment_map is not None,
    )

    CURRENT_ORIENTATION = orientation.lower().strip()
    if layout_config and layout_config.get("canvas"):
        canvas_w, canvas_h = _resolve_canvas(CURRENT_ORIENTATION, layout_config)
    elif CURRENT_ORIENTATION == "portrait":
        canvas_w, canvas_h = 1080, 1920
    else:
        canvas_w, canvas_h = 1920, 1080

    CURRENT_WIDTH  = canvas_w
    CURRENT_HEIGHT = canvas_h

    CURRENT_STORYBOARD_ENABLED = bool(storyboard)
    CURRENT_KENBURNS_ENABLED   = bool(kenburns)

    kenburns_enabled = bool(kenburns)
    from global_render_scheduler import get_scheduler

    scheduler = get_scheduler()
    if not scheduler.initialized:
        scheduler.init()
    gpu_lock = scheduler.gpu_semaphore()
    gpu_semaphore = gpu_lock
    nvenc_semaphore = gpu_lock
    storyboard_semaphore = scheduler.storyboard_semaphore()
    sched_status = scheduler.status()
    logging.info(
        f"🌐 Global scheduler | GPU slots={sched_status['gpu_slots']} | "
        f"storyboard slots={sched_status.get('storyboard_slots', STORYBOARD_MAX_CONCURRENT)} | "
        f"segment workers={sched_status['segment_workers']} | "
        f"kenburns={'on' if kenburns_enabled else 'off'}"
    )

    os.makedirs(output_dir, exist_ok=True)
    script_context = None

    if mode in ["create", "commentary", "original"]:
        timestamp   = time.strftime("%Y%m%d_%H%M%S")
        section_dir = os.path.join(TEMP_DIR, f"{timestamp}")
        os.makedirs(section_dir, exist_ok=True)
    else:
        if "-" in asset_folder:
            intro_timestamp, body_timestamp = asset_folder.split("-")
            section_dir = os.path.join(
                TEMP_DIR,
                intro_timestamp if section_type == "intro" else body_timestamp,
            )
        else:
            section_dir = os.path.join(TEMP_DIR, asset_folder)

    output_path = os.path.join(
        output_dir,
        f"{section_type}_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
    )

    if mode == "original":
        media_data     = phase_media["phase_media"]
        script_context = None

        vid_segments = assign_timeline_phases(
            vid_segments,
            segment_audio_paths,
            media_data,
            intro_sec=phase_media["intro_sec"],
            finale_sec=phase_media["finale_sec"],
        )

        if any(s.get("imagefx_required") for s in vid_segments):
            from helper.imagefx_styles import SPACE_STYLE_KEY, generate_space_visual_bible

            uses_space = any(
                (s.get("ai_prompt") or "").strip().lower() == SPACE_STYLE_KEY
                for s in vid_segments
                if s.get("imagefx_required")
            )
            full_script = (script or narration_text or "").strip()
            if uses_space and full_script:
                logging.info("🎨 Building space visual bible for ImageFX...")
                script_context = generate_space_visual_bible(full_script, client)
                logging.info(f"📖 Space visual bible ready ({len(script_context)} chars)")

    manager = None
    try:
        media_pool = []

        if mode in ["create", "commentary", "original"]:
            logging.info(f"📦 Preparing media for {section_type}...")

            video_results = download_video_clips_in_parallel(
                clip_data_list, section_dir, user_id
            )

            logging.info(
                f"✅ Completed {len(video_results)} video clips successfully."
            )

            skip_narration_media = _dreamface_skip_narration_media_pool(
                layout_config, avatar_segment_map
            )
            if skip_narration_media:
                logging.info(
                    "⚡ full_avatar Talking Heads — tagged clips use ffprobe only; "
                    "skipping narration image pool"
                )

            logging.info("🔍 CHECKPOINT 1: Starting video processing...")

            for idx, result in enumerate(video_results):
                try:
                    video_path = result["path"]
                    logging.info(
                        f"📹 Processing video {idx+1}/{len(video_results)}: "
                        f"{os.path.basename(video_path)}"
                    )

                    if not os.path.exists(video_path):
                        logging.error(f"❌ Video file missing: {video_path}")
                        continue

                    use_fast_clip = bool(result.get("tag")) or skip_narration_media
                    if use_fast_clip:
                        if not _quick_video_file_ok(video_path):
                            logging.error(
                                f"❌ Video ffprobe check failed: {video_path}"
                            )
                            continue
                        if not _verify_video_decodable(video_path):
                            logging.error(
                                f"❌ Video decode check failed (re-download or re-cut Grok clip): "
                                f"{video_path}"
                            )
                            continue
                        if not _append_tagged_clip_to_media_pool(
                            media_pool, result, video_path
                        ):
                            logging.error(
                                f"❌ Could not register clip (no duration): {video_path}"
                            )
                            continue
                        logging.info(
                            f"✅ Registered clip {idx+1}/{len(video_results)} "
                            f"(fast, no CLIP embed)"
                        )
                        continue

                    file_size = os.path.getsize(video_path)
                    if file_size < 10000:
                        logging.error(
                            f"❌ Video file too small: {file_size} bytes"
                        )
                        continue

                    if not validate_video_file(video_path, max_retries=1):
                        logging.error(
                            f"❌ Video validation failed: {video_path}"
                        )
                        continue

                    clip      = VideoFileClip(video_path)
                    rep_frame = extract_representative_frame(clip)
                    embedding = encode_image_from_pil(rep_frame)

                    media_pool.append({
                        "type":      "video",
                        "path":      video_path,
                        "character": result["character"],
                        "embedding": embedding,
                        "mute":      result["mute"],
                        "duration":  clip.duration,
                        "tag":       result.get("tag"),
                    })

                    clip.close()
                    if hasattr(clip, "reader") and clip.reader:
                        clip.reader.close()
                    del clip

                    logging.info(
                        f"✅ Processed video {idx+1}/{len(video_results)}"
                    )

                except Exception as e:
                    logging.error(f"❌ Error processing video {idx+1}: {e}")
                    logging.error(traceback.format_exc())

            logging.info(
                f"✅ All {len(video_results)} videos processed successfully"
            )

            if skip_narration_media:
                logging.info("⚡ Skipped image download/processing (full_avatar)")
                image_paths = {}
            else:
                logging.info("🖼️  Processing images...")
                image_paths = download_images_in_parallel(
                    image_data_list, section_dir
                )

            image_count = 0
            for character, img_list in image_paths.items():
                for img_path in img_list:
                    try:
                        if not validate_image_file(img_path):
                            logging.error(
                                f"❌ Image validation failed: {img_path}"
                            )
                            continue

                        img = Image.open(img_path).convert("RGB")
                        media_pool.append({
                            "type":      "image",
                            "path":      img_path,
                            "character": character,
                            "embedding": encode_image_from_pil(img),
                            "duration":  DEFAULT_DURATION,
                            "used":      False,
                        })
                        image_count += 1
                    except Exception as e:
                        logging.error(f"❌ Image error: {img_path} -> {e}")

            logging.info(f"✅ Processed {image_count} images")
            logging.info(f"📊 Total media pool: {len(media_pool)} items")

            corrupt_count = 0
            for media in media_pool:
                if media["type"] == "video":
                    if (
                        not os.path.exists(media["path"])
                        or os.path.getsize(media["path"]) < 10000
                    ):
                        corrupt_count += 1
                        logging.warning(
                            f"⚠️ Corrupt video in pool: {media['path']}"
                        )
                elif media["type"] == "image":
                    if (
                        not os.path.exists(media["path"])
                        or os.path.getsize(media["path"]) < 1000
                    ):
                        corrupt_count += 1
                        logging.warning(
                            f"⚠️ Corrupt image in pool: {media['path']}"
                        )

            if corrupt_count > 0:
                logging.warning(
                    f"⚠️ Found {corrupt_count} potentially corrupt "
                    f"media files in pool"
                )
            else:
                logging.info(
                    "✅ All media pool files passed basic integrity check"
                )

            save_metadata(media_pool, section_dir)

        else:
            logging.info(
                f"♻️ Regenerating media for {section_type}..."
            )
            media_pool = load_metadata(section_dir)

        logging.info("🧱 Encoding segments in parallel...")
        chunk_folder = os.path.join(section_dir, "chunks")
        os.makedirs(chunk_folder, exist_ok=True)

        manager            = Manager()
        used_media_dict    = manager.dict()
        media_lock         = manager.Lock()
        whisk_shared_state = manager.dict()

        if mode == "original":
            needs_character = any(
                seg.get("whisk_required") for seg in vid_segments
            )

            if needs_character:
                character_id = next(
                    (
                        seg.get("character_id")
                        for seg in vid_segments
                        if seg.get("character_id")
                    ),
                    None,
                )

                if not character_id:
                    raise ValueError(
                        "whisk_required set but no character_id found"
                    )

                char_context = initialize_whisk_character(
                    character_id=character_id,
                    db=get_db(),
                    cookie=None,
                )

                if not char_context:
                    raise ValueError(
                        "Failed to initialize Whisk character"
                    )

                whisk_shared_state.update(char_context)
                logging.info(
                    f"🎭 Whisk character initialized: "
                    f"{char_context['character_name']}"
                )
            else:
                logging.info(
                    "ℹ️  No Whisk character required for this video"
                )

        from gpu_kenburns_simple import precompute_segment_effects
        effect_assignments, _video_seed = precompute_segment_effects(
            vid_segments=vid_segments,
        )

        if transition_config is None:
            transition_config = {
                "enabled": False,
                "types":   ["zoom", "translation"],
                "max_pct": 0.15,
                "max_sec": 0.6,
            }

        transition_config["kenburns_mode"] = bool(kenburns)
        if kenburns:
            transition_config["segment_effects"] = [
                a["primary_effect"] for a in effect_assignments
            ]

        whoosh_config = {
            "enabled":     True,
            "volume":      0.6,
            "transitions": [
                "slideleft", "slideright", "slideup",
                "wipeleft",  "wiperight",  "wipeup",
            ],
            "mp3_path": "sounds/whoosh.mp3",
        }

        transitions_enabled = transition_config.get("enabled", False)
        max_pct             = float(transition_config.get("max_pct", 0.15))
        max_sec             = float(transition_config.get("max_sec", 0.6))
        min_frames          = 6
        freeze_fps          = 30.0

        def _get_audio_dur(path):
            if path and os.path.exists(path):
                try:
                    c = AudioFileClip(path)
                    d = c.duration
                    c.close()
                    return d
                except Exception:
                    pass
            return 0.0

        audio_durations = [
            _get_audio_dur(p) for p in segment_audio_paths
        ]

        head_freeze_per_segment = []
        for i in range(len(vid_segments)):
            if not transitions_enabled or i == 0:
                head_freeze_per_segment.append(0.0)
            else:
                dur_a = audio_durations[i - 1]
                dur_b = audio_durations[i]
                if dur_a < 1.0 or dur_b < 1.0:
                    head_freeze_per_segment.append(0.0)
                else:
                    raw    = min(dur_a * max_pct, dur_b * max_pct, max_sec)
                    frames = max(min_frames, round(raw * freeze_fps))
                    T      = frames / freeze_fps
                    head_freeze_per_segment.append(T)

        logging.info(
            f"🧊 Head freeze durations: "
            f"{[round(t, 3) for t in head_freeze_per_segment]}"
        )

        talking_heads_mode = bool(layout_config and avatar_segment_map)
        has_clips          = _has_clip_segments(vid_segments) if talking_heads_mode else False

        segment_info_list = [
            (
                segment,
                media_pool,
                segment_audio_paths[i],
                section_dir,
                chunk_folder,
                i,
                used_media_dict,
                media_lock,
                mode,
                avatar,
                background_video,
                vid_segments,
                script_context,
                orientation,
                storyboard,
                kenburns,
                avatar_active,
                gpu_semaphore,
                nvenc_semaphore,
                effect_assignments[i],
                whisk_shared_state,
                head_freeze_per_segment[i],
                talking_heads_mode,
                teaser_cfg,
                narration_images_cfg,
                section_type,
                storyboard_semaphore,
                layout_config if talking_heads_mode else None,
            )
            for i, segment in enumerate(vid_segments)
        ]



        if talking_heads_mode:
            mode_str = (
                "batch-level before concat (clip video)"
                if has_clips
                else "post-assembly batch concat (no clips / zoom2)"
            )
            logging.info(
                f"🎭 Talking Heads active | "
                f"layout={layout_config.get('layout_id')} | "
                f"segments={len(avatar_segment_map)} | {mode_str}"
            )

        total_seg = len(segment_info_list)
        job_label = f"{section_type}:{os.path.basename(output_path)}"
        logging.info(
            f"🧱 Global pool encoding {total_seg} segment(s) for {job_label}..."
        )
        chunk_paths = scheduler.encode_segments(
            segment_info_list,
            job_label=job_label,
        )

        logging.info("🔍 Validating segment completeness...")

        validated_chunks  = [None] * len(chunk_paths)
        missing_segments  = []
        fallback_segments = []

        for i, path in enumerate(chunk_paths):
            if path and os.path.exists(path) and os.path.getsize(path) > 10000:
                validated_chunks[i] = path
                if (
                    "fallback"  in os.path.basename(path) or
                    "emergency" in os.path.basename(path)
                ):
                    fallback_segments.append(i)
                    logging.warning(
                        f"⚠️ Segment {i} is a FALLBACK placeholder"
                    )
                else:
                    logging.debug(
                        f"✅ Segment {i} validated: "
                        f"{os.path.basename(path)}"
                    )
            else:
                missing_segments.append(i)
                logging.error(f"❌ Segment {i} is MISSING or invalid")

        total_segments      = len(vid_segments)
        successful_segments = sum(
            1 for i, p in enumerate(validated_chunks)
            if p and i not in fallback_segments
        )

        logging.info(f"")
        logging.info(f"{'='*60}")
        logging.info(f"📊 SEGMENT ENCODING SUMMARY")
        logging.info(f"{'='*60}")
        logging.info(f"   Total segments:          {total_segments}")
        logging.info(
            f"   ✅ Successfully encoded:   {successful_segments} "
            f"({successful_segments/total_segments*100:.1f}%)"
        )
        logging.info(
            f"   ⚠️  Fallback placeholders:  {len(fallback_segments)} "
            f"({len(fallback_segments)/total_segments*100:.1f}%)"
        )
        logging.info(
            f"   ❌ Missing/Failed:         {len(missing_segments)} "
            f"({len(missing_segments)/total_segments*100:.1f}%)"
        )
        if fallback_segments:
            logging.warning(
                f"   Fallback segment indices: {fallback_segments}"
            )
        if missing_segments:
            logging.error(
                f"   Missing segment indices: {missing_segments}"
            )
        logging.info(f"{'='*60}")
        logging.info(f"")

        if missing_segments:
            logging.critical(
                "🚨 CRITICAL: Some segments are completely missing!"
            )
            logging.warning("🔧 Creating emergency placeholders...")

            for seg_idx in missing_segments:
                segment    = vid_segments[seg_idx]
                audio_path = segment_audio_paths[seg_idx]

                duration = 5.0
                if audio_path and os.path.exists(audio_path):
                    try:
                        audio_clip = AudioFileClip(audio_path)
                        duration   = audio_clip.duration
                        audio_clip.close()
                        del audio_clip
                    except Exception:
                        words    = len(segment.get("text", "").split())
                        duration = max(3.0, words * 0.35)
                else:
                    words    = len(segment.get("text", "").split())
                    duration = max(3.0, words * 0.35)

                try:
                    emergency_path = create_full_segment_fallback(
                        duration=duration,
                        canvas_w=canvas_w,
                        canvas_h=canvas_h,
                        segment_index=seg_idx,
                        audio_path=audio_path,
                        chunk_folder=chunk_folder,
                        reason="Emergency - segment completely missing",
                    )

                    if emergency_path and os.path.exists(emergency_path):
                        validated_chunks[seg_idx] = emergency_path
                        fallback_segments.append(seg_idx)
                        logging.info(
                            f"✅ Emergency placeholder inserted "
                            f"for segment {seg_idx}"
                        )
                    else:
                        logging.critical(
                            f"💥 FATAL: Cannot create emergency "
                            f"placeholder for segment {seg_idx}"
                        )
                        validated_chunks[seg_idx] = None

                except Exception as e:
                    logging.critical(
                        f"💥 Emergency placeholder creation failed "
                        f"for segment {seg_idx}: {e}"
                    )
                    logging.error(traceback.format_exc())
                    validated_chunks[seg_idx] = None

        chunk_paths = validated_chunks

        if not chunk_paths or None in chunk_paths:
            none_count = (
                chunk_paths.count(None) if chunk_paths else total_segments
            )
            error_msg = (
                f"Video generation FAILED: "
                f"{none_count}/{total_segments} segments could not be created"
            )
            logging.critical(f"💥 {error_msg}")

            send_telegram_notification(
                f"🚨 VIDEO GENERATION FAILED\n\n{error_msg}\n\n"
                f"Missing segments: {missing_segments}\n"
                f"Fallback segments: {fallback_segments}",
                "Critical Failure",
                "https://claude.ai/error",
                user_id or "Unknown",
            )

            raise RuntimeError(error_msg)

        logging.info(f"✅ Proceeding with {len(chunk_paths)} segments")
        if fallback_segments:
            logging.warning(
                f"⚠️  Including {len(fallback_segments)} BLACK "
                f"PLACEHOLDER segments (will need manual editing)"
            )
            logging.warning(
                f"   Black placeholder segment indices: {fallback_segments}"
            )

        # -------------------------------------------------
        # CLIP VIDEOS: batch-level avatar composite BEFORE concat
        # NO-CLIP VIDEOS: skip here — composite post-assembly (zoom2)
        # -------------------------------------------------
        if talking_heads_mode and has_clips:
            logging.info("🎭 Clip video → batch-level avatar composite")
            try:
                chunk_paths = _apply_batch_avatar_composite(
                    chunk_paths        = chunk_paths,
                    avatar_segment_map = avatar_segment_map,
                    vid_segments       = vid_segments,
                    layout_config      = layout_config,
                    canvas_w           = canvas_w,
                    canvas_h           = canvas_h,
                    chunk_folder       = chunk_folder,
                )
            except Exception as e:
                logging.error(
                    f"❌ Batch composite error: {e}", exc_info=True
                )
                logging.warning("⚠️ Continuing with content-only chunks")

        # -------------------------------------------------
        # LOUDNESS — level each avatar / clip chunk on its own BEFORE banners
        # (one steady gain per chunk; levelling the joined file afterwards
        # made the start of narration pump after long muted clips)
        # -------------------------------------------------
        pre_leveled = False
        if talking_heads_mode and has_clips:
            try:
                chunk_paths = _loudness_match_chunks(chunk_paths, chunk_folder)
                pre_leveled = True
            except Exception as e:
                logging.error(f"❌ Loudness matching failed — continuing: {e}", exc_info=True)

        # -------------------------------------------------
        # TRANSITION BANNER — narration ↔ CLIP boundaries
        # (chunk prep runs AFTER banner so banner_merge_* is included)
        # -------------------------------------------------
        if banner_state.get("enabled"):
            batch_groups = None
            if talking_heads_mode and has_clips and avatar_segment_map:
                batch_groups = _group_batches_from_map(avatar_segment_map)
            static_root = os.getenv("STATIC_ROOT") or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "static"
            )
            try:
                chunk_paths = apply_banner_boundary_transitions(
                    chunk_paths=chunk_paths,
                    vid_segments=vid_segments,
                    banner_state=banner_state,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    chunk_folder=chunk_folder,
                    talking_heads_mode=talking_heads_mode,
                    has_clips=has_clips,
                    batch_groups=batch_groups,
                    static_root=static_root,
                )
            except Exception as e:
                logging.error(
                    f"❌ Transition banner failed — continuing without banners: {e}",
                    exc_info=True,
                )

        # CLIP ↔ CLIP transitions removed for now — successive clips are joined
        # with hard cuts. (helper/clip_segment_transitions.py is kept for later.)

        # -------------------------------------------------
        # PREP ALL FINAL CONCAT UNITS — PTS=0 before copy-concat
        # Includes clipcomposited_*, banner_merge_*, normaud_*, etc.
        # Fixes successive CLIP desync at chunk seams.
        # -------------------------------------------------
        match_avatar_clip_loudness = talking_heads_mode and has_clips and not pre_leveled
        if match_avatar_clip_loudness:
            logging.info(
                f"🔊 Final chunk prep will loudness-match avatar + clip audio "
                f"to {PIPELINE_AUDIO_TARGET_LUFS} LUFS"
            )
        logging.info("🧩 Preparing all chunks for final concat (post-banner)...")
        final_prepared = []
        prep_total = len(chunk_paths)
        for idx, p in enumerate(chunk_paths):
            if not p:
                final_prepared.append(p)
                continue
            if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == prep_total:
                logging.info(f"🧩 Final chunk prep {idx + 1}/{prep_total}...")
            ready = os.path.join(
                chunk_folder, f"ready_final_{idx}_{int(time.time() * 1000)}.mp4"
            )
            fixed = _prepare_chunk_for_concat(
                p, ready, loudness_normalize=match_avatar_clip_loudness
            )
            final_prepared.append(fixed or p)
        chunk_paths = final_prepared
        logging.info(f"✅ Final chunk prep complete | {len(chunk_paths)} chunks")

        # -------------------------------------------------
        # FINAL CONCAT
        # -------------------------------------------------
        logging.info("📦 Final assembly via FFmpeg concat...")
        concat_txt_path = os.path.join(chunk_folder, "concat.txt")

        with open(concat_txt_path, "w", encoding="utf-8") as f:
            for path in chunk_paths:
                f.write(
                    f"file '{os.path.abspath(path).replace(os.sep, '/')}'\n"
                )

        if transition_config.get("enabled"):
            logging.info(
                "🎬 Transitions enabled — applying before final assembly..."
            )

            temp_video_no_audio = os.path.join(
                chunk_folder, "temp_video_no_audio.mp4"
            )

            apply_transitions_and_concat(
                chunk_paths,
                temp_video_no_audio,
                transition_config,
                whoosh_config,
            )

            if mode == "original" and full_audio:
                logging.info(
                    "🎵 Original mode — overlaying full audio track..."
                )
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", temp_video_no_audio,
                    "-i", full_audio,
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    "-movflags", "+faststart",
                    "-y", output_path,
                ]
                try:
                    subprocess.run(
                        ffmpeg_cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    logging.info(
                        "✅ Audio overlay completed successfully"
                    )
                except subprocess.CalledProcessError as e:
                    logging.error(
                        f"❌ FFmpeg audio overlay failed: {e.stderr}"
                    )
                    raise
                finally:
                    if os.path.exists(temp_video_no_audio):
                        os.remove(temp_video_no_audio)
            else:
                logging.info(
                    "🎵 Commentary/create mode — "
                    "audio baked in, finalizing..."
                )
                os.rename(temp_video_no_audio, output_path)

        else:
            logging.info("🎬 Standard concat (no transitions) — copy first")
            copy_cmd = [
                "ffmpeg", "-y",
                "-fflags", "+genpts",
                "-avoid_negative_ts", "make_zero",
                "-f", "concat", "-safe", "0",
                "-i", concat_txt_path,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path,
            ]
            try:
                subprocess.run(
                    copy_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logging.info("✅ Copy concat completed successfully")
            except subprocess.CalledProcessError as e:
                logging.warning(
                    f"⚠️ Copy concat failed — fallback NVENC re-encode:\n"
                    f"{e.stderr[-500:]}"
                )
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-fflags", "+genpts",
                    "-avoid_negative_ts", "make_zero",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_txt_path,
                    "-c:v", "h264_nvenc", "-preset", "p3", "-tune", "hq",
                    "-rc", "vbr", "-b:v", "8M", "-maxrate", "12M",
                    "-bufsize", "24M",
                    "-profile:v", "high", "-pix_fmt", "yuv420p",
                    "-colorspace",      "bt709",
                    "-color_primaries", "bt709",
                    "-color_trc",       "bt709",
                    "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "192k",
                    "-movflags", "+faststart",
                    output_path,
                ]
                subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logging.info(
                    "✅ Re-encode concatenation completed successfully"
                )

        # -------------------------------------------------
        # NO-CLIP VIDEOS: post-assembly batch concat composite (zoom2)
        # -------------------------------------------------
        if talking_heads_mode and not has_clips:
            logging.info(
                "🎭 No-clip video → post-assembly batch concat composite"
            )
            try:
                composited = _composite_avatar_over_full_video(
                    content_video      = output_path,
                    avatar_segment_map = avatar_segment_map,
                    layout_config      = layout_config,
                    canvas_w           = canvas_w,
                    canvas_h           = canvas_h,
                    chunk_folder       = chunk_folder,
                )
                if composited and os.path.exists(composited):
                    os.replace(composited, output_path)
                    new_size = os.path.getsize(output_path) / (1024 * 1024)
                    logging.info(
                        f"✅ Talking Heads composite complete → "
                        f"{output_path} ({new_size:.1f}MB)"
                    )
                else:
                    logging.warning(
                        "⚠️ Composite returned None — keeping content-only video"
                    )
            except Exception as e:
                logging.error(
                    f"❌ Talking Heads composite error: {e}", exc_info=True
                )
                logging.warning("⚠️ Continuing without avatar composite")

        # -------------------------------------------------
        # BACKGROUND MUSIC
        # -------------------------------------------------
        if background:
            logging.info("🎵 Adding background music...")
            try:
                add_background_music(output_path, background)
                logging.info("✅ Background music added successfully")
            except Exception as e:
                logging.error(
                    f"❌ Background music addition failed: {e}"
                )
                logging.warning(
                    "⚠️  Continuing without background music"
                )
        else:
            logging.info("ℹ️  No background music specified")

        # -------------------------------------------------
        # CAPTIONS / SUBTITLES
        # -------------------------------------------------
        if captions_enabled and srt and os.path.exists(srt):
            logging.info("📝 Burning subtitles into final video...")
            subbed_output_path = output_path.replace(".mp4", "_subbed.mp4")
            try:
                success = burn_srt_to_video(
                    input_video=output_path,
                    srt_file=srt,
                    output_video=subbed_output_path,
                    caption_settings=caption_settings,
                )
                if success:
                    try:
                        os.remove(output_path)
                        os.rename(subbed_output_path, output_path)
                        logging.info(
                            f"✅ Subtitles burned → {output_path}"
                        )
                    except Exception as e:
                        logging.error(
                            f"❌ Failed to replace original video: {e}"
                        )
                        if os.path.exists(subbed_output_path):
                            output_path = subbed_output_path
                else:
                    logging.warning(
                        "⚠️  Subtitle burning failed, "
                        "continuing without subtitles"
                    )
            except Exception as e:
                logging.error(f"❌ Subtitle burning crashed: {e}")
                logging.warning("⚠️  Continuing without subtitles")
        else:
            if captions_enabled:
                logging.warning(
                    "⚠️  Captions enabled but no SRT file provided"
                )

        cleanup_files = ["output.srt", "temp_audio.mp3"]
        for f in cleanup_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    logging.debug(f"🧹 Cleaned up: {f}")
                except Exception as e:
                    logging.warning(f"⚠️  Could not remove {f}: {e}")

        if os.path.exists(output_path):
            final_size = os.path.getsize(output_path)
            if final_size < 100000:
                logging.error(
                    f"⚠️  Final video suspiciously small: "
                    f"{final_size} bytes"
                )
            else:
                logging.info(
                    f"✅ Final video size: "
                    f"{final_size / (1024*1024):.2f} MB"
                )

        logging.info(f"")
        logging.info(f"{'='*60}")
        logging.info(f"🎉 VIDEO GENERATION COMPLETE")
        logging.info(f"{'='*60}")
        logging.info(f"   Output: {output_path}")
        logging.info(f"   Total segments:    {total_segments}")
        logging.info(f"   Successful:        {successful_segments}")
        logging.info(f"   Fallbacks:         {len(fallback_segments)}")
        if fallback_segments:
            logging.info(f"   ⚠️  BLACK SEGMENTS: {fallback_segments}")
            logging.info(
                f"   ⚠️  These segments need manual editing!"
            )
        logging.info(f"{'='*60}")
        logging.info(f"")

        return (
            output_path,
            (
                timestamp
                if mode in ["create", "commentary", "original"]
                else asset_folder
            ),
            lambda: print("✅ Done!"),
        )

    except Exception as e:
        full_trace = traceback.format_exc()
        logging.error(f"🚨 Video generation error:\n{full_trace}")

        send_telegram_notification(
            f"🚨 VIDEO GENERATION FAILED\n\n"
            f"Error: {str(e)}\n\n"
            f"Full trace:\n{full_trace[:500]}",
            "Failure",
            "https://claude.ai/error",
            user_id or "Unknown",
        )

        return None, None, None

    finally:
        if manager is not None:
            try:
                manager.shutdown()
            except Exception as exc:
                logging.debug(f"Manager shutdown: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.info("🧹 Final cleanup completed")