from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, send_file, session, redirect, url_for, abort, Response
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from bson.json_util import dumps
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import asyncio
import os

import env_loader  # noqa: F401 — loads backend/.env before other imports

import time
from datetime import datetime,timedelta
from threading import Thread
import tempfile
import openai
import yt_dlp
import requests
from bs4 import BeautifulSoup
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips, ImageClip
from moviepy.video.fx.resize import resize
from gtts import gTTS
from googleapiclient.discovery import build
from mongodb import mongo_instance
from news_scrape import start_news_fetcher, check_for_updates
from flickrapi import FlickrAPI  # Ensure FlickrAPI is installed
from helper.script_generator import generate_full_script
from helper.sections import extract_main_character_keywords_and_visuals
from helper.video_scraper import fetch_related_videos
from helper.image_scraper import fetch_related_images
from helper.emotion_detector import detect_emotion
from helper.regenerate import full_pipeline
from helper.process_full_video import process_full_video,process_regenerated_video
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess  # Needed for final GPU-accelerated encoding
import uuid
import copy
import gc  # for memory cleanup
import multiprocessing
import shutil
from segment_pool_worker import get_create_video


def create_video(*args, **kwargs):
    return get_create_video()(*args, **kwargs)


from test2 import generate_audio_segments
from helper.transcript_generator import full_processing_pipeline
from tqdm import tqdm  
import threading
from pathlib import Path
import subprocess
import logging
import json
from helper.telegram_notification_engine import send_telegram_notification,send_telegram_error
from video_job_controller import (
    enqueue_video_job,
    enqueue_script_job,
    start_video_worker,
    get_queue_status,
    move_queued_video_job,
    get_admin_pipeline_snapshot,
)
from global_render_scheduler import get_scheduler
from helper.pipeline_board import build_pipeline_board, board_age_cutoff, board_mongo_cutoff_filter
from flask_cors import CORS
import multiprocessing
from daemon_launcher import ensure_cookie_daemon_running

# .env already loaded via env_loader at top of this file


# Initialize Flask app
app = Flask(__name__)

if multiprocessing.current_process().name == "MainProcess":
    ensure_cookie_daemon_running()
    start_video_worker()

from flask_cors import CORS
from speech_routes import speechify_bp
app.register_blueprint(speechify_bp)


from cartesia_routes import cartesia_bp
app.register_blueprint(cartesia_bp)

# Upload Folder
UPLOAD_FOLDER = "./static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY=os.getenv("SERPER_API")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
FLICKR_API_KEY = os.getenv("FLICKR_API_KEY")
FLICKR_SECRET = os.getenv("FLICKR_SECRET")
SECRET_KEY=os.getenv('SECRET_KEY')
JWT_SECRET_KEY=os.getenv('JWT_SECRET_KEY')
FRONTEND_URL=os.getenv("FRONTEND_URL")

# CORS — browser blocks API calls if the page origin is not listed here.
_cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://www.alchemymix.com",
    "https://alchemymix.com",
]
if FRONTEND_URL:
    _cors_origins.append(FRONTEND_URL.rstrip("/"))
CORS(app, origins=list(dict.fromkeys(o for o in _cors_origins if o)))


# JWT Configuration
app.config['SECRET_KEY'] =SECRET_KEY
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=10)  # Token expiration time
# app.py or config
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB


# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



# Initialize OpenAI Client (Optimize calls)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# JWT Initialization
jwt = JWTManager(app)
bcrypt = Bcrypt(app)


# Initialize MongoDB
db = mongo_instance.get_db()


def admin_required(fn):
    """JWT + MongoDB status=admin."""

    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        email = get_jwt_identity()
        user = db.users.find_one({"email": email})
        if not user or user.get("status") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return fn(*args, **kwargs)

    return wrapper


def _is_admin(email: str | None = None) -> bool:
    email = email or get_jwt_identity()
    user = db.users.find_one({"email": email}, {"status": 1})
    return bool(user and user.get("status") == "admin")


def _dreamface_job_lookup(job_id: str, user_id: str | None = None, projection: dict | None = None):
    """Owner-scoped unless caller is admin."""
    user_id = user_id or get_jwt_identity()
    query: dict = {"_id": ObjectId(job_id)}
    if not _is_admin(user_id):
        query["user_id"] = user_id
    if projection is not None:
        return db.dreamface_jobs.find_one(query, projection)
    return db.dreamface_jobs.find_one(query)


def _tail_log_file(max_lines: int = 150) -> list[str]:
    """Return last N lines from PIPELINE_LOG_FILE if configured."""
    log_path = os.getenv("PIPELINE_LOG_FILE", "").strip()
    if not log_path or not os.path.isfile(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return [ln.rstrip("\n\r") for ln in lines[-max_lines:]]
    except Exception as exc:
        logging.debug(f"Log tail failed: {exc}")
        return []


def _user_display_map(emails: set[str]) -> dict[str, str]:
    if not emails:
        return {}
    users = db.users.find({"email": {"$in": list(emails)}}, {"email": 1, "firstName": 1, "lastName": 1})
    out = {}
    for u in users:
        email = u.get("email")
        name = f"{u.get('firstName', '')} {u.get('lastName', '')}".strip()
        out[email] = name or email
    return out


def _build_admin_overview(*, include_logs: bool = False) -> dict:
    """Full pipeline state for admin dashboard."""
    base = get_queue_status()
    now = datetime.utcnow()
    cutoff = board_age_cutoff(now)
    age_filter = board_mongo_cutoff_filter(cutoff)

    df_projection = {
        "create_video_payload": 0,
        "audio_payload": 0,
        "batches.segments_data": 0,
    }
    dreamface_jobs = list(
        db.dreamface_jobs.find(age_filter, df_projection)
        .sort("created_at", -1)
        .limit(100)
    )
    for job in dreamface_jobs:
        job["_id"] = str(job["_id"])
        status = job.get("status")
        started = job.get("processing_started_at")
        stuck_sec = 0
        if status == "processing" and started:
            stuck_sec = (now - started).total_seconds()
        job["recovery"] = {
            "can_retry": status in ("processing", "failed", "completed")
            and bool(job.get("layout_config"))
            and _dreamface_batches_ready(job.get("batches")),
            "can_reset": status == "processing",
            "is_likely_stuck": status == "processing"
            and stuck_sec >= _dreamface_stuck_minutes() * 60,
            "processing_seconds": int(stuck_sec),
        }

    video_drafts = list(
        db.video_data.find(
            {
                "status": {"$in": ["processing", "awaiting_dreamface", "failed"]},
                **age_filter,
            },
            {"data": 0},
        ).sort("_id", -1).limit(100)
    )
    for draft in video_drafts:
        draft["_id"] = str(draft["_id"])
        if draft.get("dreamface_job_id"):
            draft["dreamface_job_id"] = str(draft["dreamface_job_id"])

    emails = set()
    for j in dreamface_jobs:
        if j.get("user_id"):
            emails.add(j["user_id"])
    for d in video_drafts:
        if d.get("user_id"):
            emails.add(d["user_id"])
    for j in base.get("queued_video_jobs") or []:
        if j.get("user_id"):
            emails.add(j["user_id"])
    for j in base.get("running_video_jobs") or []:
        if j.get("user_id"):
            emails.add(j["user_id"])

    user_names = _user_display_map(emails)

    recent_completions = list(
        db.completed_videos.find(
            age_filter,
            {"title": 1, "user_id": 1, "created_at": 1, "video_path": 1},
        )
        .sort("created_at", -1)
        .limit(20)
    )
    for vid in recent_completions:
        vid["_id"] = str(vid["_id"])

    board = build_pipeline_board(
        base=base,
        dreamface_jobs=dreamface_jobs,
        video_drafts=video_drafts,
        user_names=user_names,
        job_events=base.get("job_events") or [],
        recent_completions=recent_completions,
    )

    return {
        **base,
        "dreamface_jobs": dreamface_jobs,
        "video_drafts": video_drafts,
        "user_names": user_names,
        "board": board,
        "recent_completions": recent_completions,
        "server_logs": _tail_log_file(200) if include_logs else [],
        "log_file_configured": bool(os.getenv("PIPELINE_LOG_FILE", "").strip()),
        "generated_at": now.isoformat() + "Z",
    }


@app.route("/api/pipeline/admin/status", methods=["GET"])
@admin_required
def pipeline_admin_status():
    """Live pipeline metrics for admin dashboard."""
    include_logs = request.args.get("logs", "").lower() in ("1", "true", "yes")
    return jsonify(_build_admin_overview(include_logs=include_logs))


@app.route("/api/pipeline/admin/recover", methods=["POST"])
@admin_required
def pipeline_admin_recover():
    """
    Force recovery: clear leaked global slots, recreate worker pool,
    reset GPU/storyboard semaphores.
    """
    data = request.get_json(silent=True) or {}
    reason = (data.get("reason") or "").strip() or f"admin UI | {get_jwt_identity()}"
    sched = get_scheduler()
    before = get_queue_status()
    sched.force_recovery(f"manual admin recover | {reason}")
    after = get_queue_status()
    logging.warning(f"🛠️ Admin pipeline recover by {get_jwt_identity()} | {reason}")
    return jsonify({
        "ok": True,
        "action": "force_recovery",
        "reason": reason,
        "before": before,
        "after": after,
    })


@app.route("/api/pipeline/admin/recreate-pool", methods=["POST"])
@admin_required
def pipeline_admin_recreate_pool():
    """Kill and respawn segment worker processes (pool recycle)."""
    data = request.get_json(silent=True) or {}
    reason = (data.get("reason") or "").strip() or f"admin UI | {get_jwt_identity()}"
    sched = get_scheduler()
    before = get_queue_status()
    recreated = sched.admin_recreate_pool(f"manual admin recreate | {reason}")
    after = get_queue_status()
    logging.warning(f"🛠️ Admin pool recreate by {get_jwt_identity()} | {reason}")
    return jsonify({
        "ok": True,
        "action": "recreate_pool",
        "recreated": recreated,
        "reason": reason,
        "before": before,
        "after": after,
    })


@app.route("/api/pipeline/admin/reset-semaphores", methods=["POST"])
@admin_required
def pipeline_admin_reset_semaphores():
    """Reset GPU + storyboard IPC semaphores without full slot clear."""
    data = request.get_json(silent=True) or {}
    reason = (data.get("reason") or "").strip() or f"admin UI | {get_jwt_identity()}"
    sched = get_scheduler()
    before = get_queue_status()
    sched.admin_reset_semaphores(f"manual admin reset | {reason}")
    after = get_queue_status()
    logging.warning(f"🛠️ Admin semaphore reset by {get_jwt_identity()} | {reason}")
    return jsonify({
        "ok": True,
        "action": "reset_semaphores",
        "reason": reason,
        "before": before,
        "after": after,
    })


@app.route("/api/pipeline/admin/queue/move", methods=["POST"])
@admin_required
def pipeline_admin_queue_move():
    """Move a queued video job up or down (not running jobs)."""
    data = request.get_json(silent=True) or {}
    queue_id = (data.get("queue_id") or "").strip()
    direction = (data.get("direction") or "").strip().lower()
    if not queue_id or direction not in ("up", "down"):
        return jsonify({"error": "queue_id and direction (up|down) required"}), 400
    moved = move_queued_video_job(queue_id, direction)
    if not moved:
        return jsonify({"error": "Job not in queue or already at boundary"}), 400
    logging.info(
        f"🛠️ Admin queue move {direction} | id={queue_id} by {get_jwt_identity()}"
    )
    return jsonify({"ok": True, "overview": _build_admin_overview()})


@app.route("/api/pipeline/admin/logs", methods=["GET"])
@admin_required
def pipeline_admin_logs():
    lines = min(500, max(20, int(request.args.get("lines", 150))))
    return jsonify({
        "lines": _tail_log_file(lines),
        "log_file_configured": bool(os.getenv("PIPELINE_LOG_FILE", "").strip()),
        "path": os.getenv("PIPELINE_LOG_FILE", ""),
    })


@app.route("/api/pipeline/admin/dreamface/<job_id>/retry", methods=["POST"])
@admin_required
def pipeline_admin_dreamface_retry(job_id):
    data = request.get_json(silent=True) or {}
    data["force"] = True
    job = db.dreamface_jobs.find_one({"_id": ObjectId(job_id)})
    if not job:
        return jsonify({"error": "Job not found"}), 404
    user_id = job.get("user_id")
    if not user_id:
        return jsonify({"error": "Job has no user_id"}), 400
    return _dreamface_retry_or_rerender(job_id, data, owner_user_id=user_id)


@app.route("/api/pipeline/admin/dreamface/<job_id>/reset", methods=["POST"])
@admin_required
def pipeline_admin_dreamface_reset(job_id):
    job = db.dreamface_jobs.find_one({"_id": ObjectId(job_id)})
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "processing":
        return jsonify({"error": f"Reset only for processing jobs (current: {job.get('status')})"}), 400
    db.dreamface_jobs.update_one(
        {"_id": ObjectId(job_id)},
        {
            "$set": {
                "status": "awaiting_dreamface",
                "error": "Reset by admin — ready to edit or retry",
            },
            "$unset": {"processing_started_at": ""},
        },
    )
    vid_data_id = job.get("vid_data_id")
    if vid_data_id:
        db.video_data.update_one(
            {"_id": ObjectId(vid_data_id)},
            {"$set": {"status": "awaiting_dreamface"}},
        )
    logging.info(f"🛠️ Admin DreamFace reset | job={job_id} by {get_jwt_identity()}")
    return jsonify({"ok": True, "overview": _build_admin_overview()})


# Legacy — redirect to admin-only flow (no unauthenticated recover)
@app.route("/api/pipeline/health", methods=["GET"])
@jwt_required()
def pipeline_health_legacy():
    return jsonify({
        "deprecated": True,
        "use": "/api/pipeline/admin/status (admin only)",
    }), 410


# ✅ Route: Fetch Latest News
@app.route('/api/news', methods=['GET'])
def get_news():
    articles = list(db.news.find().sort("publish_date", -1).limit(300))

    for article in articles:
        article["_id"] = str(article["_id"])
        if isinstance(article["publish_date"], datetime):
            article["publish_date"] = article["publish_date"].isoformat()

    return jsonify({"articles": articles, "last_updated": datetime.now().isoformat(), "count": len(articles)})


@app.route('/api/register', methods=['POST'])
def register():
    email = request.json.get('email', None)
    firstName = request.json.get('firstName', None)
    lastName=request.json.get('lastName',None)
    password = request.json.get('password', None)
    status=request.json.get('status',None)

    if not email or not password or not status or not firstName or not lastName:
        return jsonify({"error": "Email, name, status and password  are required"}), 400
    
    
    userExists=db.users.find_one({"email":email})
    if userExists:
       return jsonify({"error":"User with email already exists"}),400
    
    # Hash the password using bcrypt
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user_data={
        "firstName":firstName,
        "lastName":lastName,
        "email":email,
        "status":status,
        "voice":"alloy",
        "password":hashed_password
    }
    # Store the user in the fake "database"
    db.users.insert_one(user_data)

    return jsonify({"message": "User registered successfully"}), 201


@app.route('/api/login', methods=['POST'])
def login():
    email = request.json.get('email', None)
    password = request.json.get('password', None)

    if not email or not password:
        return jsonify({"msg": "Username and password are required"}), 400

    user = db.users.find_one({"email":email})
    if user and bcrypt.check_password_hash(user['password'], password):
        access_token = create_access_token(identity=email)
        return jsonify({
            "token": access_token,
            "email": email,
            "status": user.get("status", "user"),
            "firstName": user.get("firstName", ""),
            "lastName": user.get("lastName", ""),
        }), 200
    else:
        return jsonify({"msg": "Invalid username or password"}), 401


@app.route('/api/news/<id>',methods=['GET'])
@jwt_required()
def single_news(id):
    db=mongo_instance.get_db()
    news=db.news.find_one({'_id':ObjectId(id)})
    if not news:
        return not_found()
    res=dumps(news)
    return res


# ✅ Route: Get Video Data by ID
@app.route('/api/video_data/<id>', methods=['GET'])
@jwt_required()
def get_video_data(id):
    try:
        oid = ObjectId(id)
    except Exception:
        return jsonify({"error": "Invalid video draft id", "id": id}), 400

    video_data = db.video_data.find_one({'_id': oid})
    if not video_data:
        logging.warning("video_data GET 404 — no document for id=%s", id)
        return jsonify({"error": "Video draft not found", "id": id}), 404

    video_data["_id"] = str(video_data["_id"])
    return jsonify(video_data)


def _compose_placeholder_doc(pre_id, current_user, *, niche=None, orientation=None, two_sections=False):
    """Immediate stub so /compose/:id can load while the script queue runs."""
    empty_seg = [{"character": "", "text": ""}]
    sections = [
        {
            "section": "intro",
            "section_script": "",
            "section_videos": [],
            "section_images": [],
            "section_characters": [],
            "segment": list(empty_seg),
            "clips": [],
            "imageData": [],
        },
    ]
    if two_sections:
        sections.append({
            "section": "body",
            "section_script": "",
            "section_videos": [],
            "section_images": [],
            "section_characters": [],
            "segment": list(empty_seg),
            "clips": [],
            "imageData": [],
        })
    return {
        "_id": pre_id,
        "status": "composing",
        "title": "Composing your script…",
        "script": "",
        "creator": current_user,
        "user_id": current_user,
        "created_at": datetime.utcnow(),
        "niche": niche,
        "orientation": orientation,
        "sections": sections,
    }


def generate_narration(script):


    # Generate unique filename for the narration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    narration_filename = f"narration_{timestamp}.mp3"
    narration_path = os.path.join(UPLOAD_FOLDER, narration_filename)

    # Step 2: Generating Audio Narration with Progress Tracking
    print("\n🎙️ Step 2: Generating Narration...")

    def save_audio():
        """ Function to generate and save the narration using gTTS. """
        tts = gTTS(script, lang="en", slow=False)
        tts.save(narration_path)

    # Use threading to allow progress tracking while generating narration
    thread = threading.Thread(target=save_audio)
    thread.start()

    # Display progress bar while narration is being generated
    with tqdm(total=100, desc="🔊 Saving Narration", unit="%", position=0, leave=True) as pbar:
        while thread.is_alive():
            pbar.update(10 if pbar.n < 90 else 100 - pbar.n)  # Gradually fill the bar
            time.sleep(0.5)  # Small delay for smooth progress
        pbar.update(100 - pbar.n)  # Ensure progress bar reaches 100%

    thread.join()  # Ensure thread completes

    print(f"\n✅ Narration successfully saved: {narration_path}")

    return narration_path, f"http://localhost:5000/static/{narration_filename}"



def _run_generate_video(current_user, data, pre_id):
    """
    All the heavy work that used to run inside the generate_video route.
    Now runs in the script queue -- never blocks Waitress or the GPU queue.
    Telegram notification fires from here on completion.
    """
    try:
        description     = data.get("description")
        duration        = data.get("duration")
        niche           = data.get("niche", None)
        optional_prompt = data.get("optionalPrompt")
        emotion         = data.get("emotion") or "dramatic"
 
        scripts     = generate_full_script(description, emotion, duration, optional_prompt)
        full_script = scripts["fullScript"]
 
        sections = extract_main_character_keywords_and_visuals(
            scripts["intro"], scripts["body"], niche
        )
 
        intro_yt_keywords = [
            (char, kw.strip('"'))
            for char, kws in sections["intro"]["yt_keywords"].items()
            for kw in kws
        ]
        body_yt_keywords = [
            (char, kw.strip('"'))
            for char, kws in sections["body"]["yt_keywords"].items()
            for kw in kws
        ]
 
        intro_video_urls = fetch_related_videos(intro_yt_keywords)
        body_video_urls  = fetch_related_videos(body_yt_keywords)
        intro_image_urls = fetch_related_images(
            [sections["intro"]["main_characters"]]
            if isinstance(sections["intro"]["main_characters"], str)
            else sections["intro"]["main_characters"]
        )
        body_image_urls = fetch_related_images(
            [sections["body"]["main_characters"]]
            if isinstance(sections["body"]["main_characters"], str)
            else sections["body"]["main_characters"]
        )
 
        video_data = {
            "title":   scripts["title"],
            "script":  full_script,
            "creator": current_user,
            "sections": [
                {
                    "section":            "intro",
                    "section_script":     scripts["intro"],
                    "section_videos":     intro_video_urls,
                    "section_images":     intro_image_urls,
                    "section_characters": sections["intro"]["main_characters"],
                    "segment":            sections["intro"]["segments"],
                },
                {
                    "section":            "body",
                    "section_script":     scripts["body"],
                    "section_videos":     body_video_urls,
                    "section_images":     body_image_urls,
                    "section_characters": sections["body"]["main_characters"],
                    "segment":            sections["body"]["segments"],
                },
            ],
            "status":     "draft",
            "created_at": datetime.utcnow(),
            "user_id":    current_user,
            "niche":      niche,
        }

        db.video_data.update_one({"_id": pre_id}, {"$set": video_data})
        saved_id = str(pre_id)

        send_telegram_notification(
            "Your Script Composition is Complete",
            "View Composition",
            f"{FRONTEND_URL}/compose/{saved_id}",
            current_user,
        )
 
    except Exception as e:
        send_telegram_error(f"❌ Script composition failed: {e}", current_user)
        raise
 
 
@app.route("/api/generate-video", methods=["POST"])
@jwt_required()
def generate_video():
    data         = request.json
    current_user = get_jwt_identity()
 
    if not data.get("duration"):
        return jsonify({"status": 400, "error": "Enter Script Duration"}), 400
    if not data.get("description"):
        return jsonify({"status": 400, "error": "Prompt is required"}), 400
 
    pre_id = ObjectId()
    db.video_data.insert_one(
        _compose_placeholder_doc(
            pre_id,
            current_user,
            niche=data.get("niche"),
            orientation=data.get("orientation"),
            two_sections=True,
        )
    )
    enqueue_script_job(current_user, _run_generate_video, current_user, data, pre_id)
    return jsonify({
        "_id": str(pre_id),
        "msg": "✍️ Script composition started. You will be notified when ready.",
    }), 202


# @app.route("/api/generate-video", methods=["POST"])
# @jwt_required()
# def generate_video():
    
#     try:
#         # Step 1: Get input data from request
#         data = request.json
#         current_user = get_jwt_identity()
#         description =data.get("description")
#         duaration=data.get('duration')
#         niche=data.get('niche',None)
#         print("This is my niche",niche)
#         optional_prompt=data.get('optionalPrompt')
#         emotion='neutral'
#         if not duaration:
#             return jsonify({"status": 400, "error": "Enter Script Duration" + request.url}), 400
#         if  not description:
#             return jsonify({"status":400,"error": "Prompt is required"}), 400
#         if not data.get("emotion"):
#             emotion = 'dramatic'
#         else:
#             emotion=data.get("emotion")

#         # Step 2: Generate script, narration, and captions
#         scripts = generate_full_script(description, emotion,duaration,optional_prompt)
#         full_script = scripts["fullScript"]
        
#         # narration_result = generate_narration(full_script)
#         # if not narration_result or len(narration_result) != 2:
#         #     return jsonify({"error": "Narration generation failed"}), 500
#         # narration_path, narration_url = narration_result

#         # Step 3: Extract sections
#         sections = extract_main_character_keywords_and_visuals(scripts['intro'], scripts["body"],niche)
#         # Step 4: Fetch related videos and images
#         intro_characters = sections['intro']['main_characters']
#         body_characters = sections['body']['main_characters']
#         intro_yt = sections['intro']['yt_keywords']
#         intro_yt_keywords= [(character, keyword.strip('"')) for character, keywords in intro_yt.items() for keyword in keywords]
#         body_yt = sections['body']['yt_keywords']
#         body_yt_keywords= [(character, keyword.strip('"')) for character, keywords in body_yt.items() for keyword in keywords]


#         intro_video_urls = fetch_related_videos(intro_yt_keywords)  
#         body_video_urls = fetch_related_videos(body_yt_keywords)  
        
#         intro_image_urls = fetch_related_images([intro_characters] if isinstance(intro_characters, str) else intro_characters)
#         body_image_urls = fetch_related_images([body_characters] if isinstance(body_characters, str) else body_characters)
#         intro_segments=sections['intro']['segments']
#         body_segemnets=sections['body']['segments']
#         # Step 5: Save video data to MongoDB
#         video_data = {
#             "title": scripts['title'],
#             "script": full_script,
#             "creator": current_user,
#             "sections": [
#                 {
#                     "section": "intro",
#                     "section_script": scripts["intro"],
#                     "section_videos": intro_video_urls,
#                     "section_images": intro_image_urls,
#                     "section_characters": intro_characters,
#                     "segment":intro_segments,
#                 },
#                 {
#                     "section": "body",
#                     "section_script":scripts["body"],
#                     "section_videos": body_video_urls,
#                     "section_images": body_image_urls,
#                     "section_characters": body_characters,
#                     "segment":body_segemnets,
#                 }
#             ],
#             "status":"draft",
#             "created_at": datetime.utcnow(),
#             "user_id":current_user,
#             "niche":niche
            
#         }

#         try:
#             inserted_data = db.video_data.insert_one(video_data)
#         except Exception as db_error:
#             return jsonify({"error": f"Database insertion failed: {str(db_error)}"}), 500

#         # Step 6: Retrieve and return stored data
#         saved_video = db.video_data.find_one({"_id": inserted_data.inserted_id})
#         if not saved_video:
#             return jsonify({"error": "Video data retrieval failed"}), 500

#         saved_video["_id"] = str(saved_video["_id"])
#         message='Your Script Composition is Complete'
#         button_text="View Composition"
#         id=saved_video["_id"] 
#         button_url=f"{FRONTEND_URL}/compose/{id}"
#         email=current_user
#         send_telegram_notification(message,button_text,button_url,email)
#         return jsonify(saved_video)

#     except Exception as e:
#         send_telegram_error(
#             f"🎉 ❌ Script composition failed: {e}",
#             current_user
#         )
#         return jsonify({"error": str(e)}), 500


def _run_compose_transcript(current_user, data, pre_id):
    try:
        script          = data.get("script")
        niche           = data.get("niche")
        orientation     = data.get("orientation")
        split_mode      = data.get("splitMode", None)
        commentary_mode = data.get("commentaryMode", "paraphrase")
        is_commentary_original = niche == "commentary" and commentary_mode == "original"

        scripts = asyncio.run(
            full_processing_pipeline(
                script, niche, intro_ratio=3,
                commentary_mode=commentary_mode if niche == "commentary" else None,
            )
        )

        full_script   = scripts.get("full_script", "")
        intro_text    = scripts.get("intro", "")
        body_text     = scripts.get("body", "")
        combined_text = (
            f"{intro_text.strip()}\n\n{body_text.strip()}"
            if body_text and body_text.strip()
            else intro_text.strip()
        )

        sections = extract_main_character_keywords_and_visuals(
            intro=combined_text, body=None,
            split_mode=split_mode, niche=niche,
            commentary_mode=commentary_mode if niche == "commentary" else None,
        )

        intro_characters      = sections["intro"]["main_characters"]
        intro_character_names = list(intro_characters.keys())
        intro_segments        = sections["intro"]["segments"]

        if is_commentary_original:
            intro_video_urls = []
            intro_image_urls = []
        else:
            intro_yt_keywords = [
                (char, kw.strip('"'))
                for char, kws in sections["intro"]["yt_keywords"].items()
                for kw in kws
            ]
            intro_video_urls = fetch_related_videos(intro_yt_keywords)
            intro_image_urls = fetch_related_images(
                [intro_characters] if isinstance(intro_characters, str) else intro_characters,
                niche, used_urls_global=set(),
            )

        video_data = {
            "title":          scripts.get("title", ""),
            "script":         full_script,
            "creator":        current_user,
            "sections": [
                {
                    "section":            "intro",
                    "section_script":     combined_text,
                    "section_videos":     intro_video_urls,
                    "section_images":     intro_image_urls,
                    "section_characters": intro_character_names,
                    "segment":            intro_segments,
                }
            ],
            "status":         "draft",
            "niche":          niche,
            "orientation":    orientation,
            "commentaryMode": commentary_mode if niche == "commentary" else None,
        }

        db.video_data.update_one({"_id": pre_id}, {"$set": video_data})

        send_telegram_notification(
            "Your Script Composition is Complete",
            "View Composition",
            f"{FRONTEND_URL}/compose/{str(pre_id)}",
            current_user,
        )

    except Exception as e:
        send_telegram_error(f"❌ Transcript composition failed: {e}", current_user)
        raise

@app.route("/api/compose-transcript-video", methods=["POST"])
@jwt_required()
def compose_transcript_video():
    data         = request.json
    current_user = get_jwt_identity()

    if not data.get("script") or not data.get("niche"):
        return jsonify({"error": "script and niche are required"}), 400

    pre_id = ObjectId()
    db.video_data.insert_one(
        _compose_placeholder_doc(
            pre_id,
            current_user,
            niche=data.get("niche"),
            orientation=data.get("orientation"),
            two_sections=False,
        )
    )
    enqueue_script_job(current_user, _run_compose_transcript, current_user, data, pre_id)
    return jsonify({"_id": str(pre_id)}), 202


def _normalize_import_segment(seg, default_character):
    text = (seg.get("text") or "").strip()
    character = (seg.get("character") or default_character or "").strip()
    tag = (seg.get("tag") or "").strip()
    out = {"character": character, "text": text}
    if text.startswith("[CLIP:") and text.endswith("]") and not tag:
        tag = text[len("[CLIP:") : -1].strip()
    if tag:
        out["tag"] = tag
        if not (text.startswith("[CLIP:") and text.endswith("]")):
            out["text"] = f"[CLIP:{tag}]"
    return out


def _section_script_from_segments(segments):
    return " ".join(
        (s.get("text") or "").strip()
        for s in (segments or [])
        if (s.get("text") or "").strip()
    )


def _sanitize_composition_import_for_runtime(composition):
    """Strip LLM-only fields (refinement, bridgeHints, followsClip) before persist."""
    if not composition or not isinstance(composition, dict):
        return composition
    meta = composition.get("meta") or {}
    keep_meta = (
        "batchVideo",
        "commentaryId",
        "workingTitle",
        "commentaryUrl",
        "protocol",
        "generatedAt",
        "sourceNote",
        "compositionSection",
    )
    runtime_meta = {k: meta[k] for k in keep_meta if meta.get(k) is not None}

    script = composition.get("script")
    runtime_script = None
    if isinstance(script, dict):
        runtime_script = {}
        if script.get("intro") is not None:
            runtime_script["intro"] = script["intro"]
        if script.get("outro") is not None:
            runtime_script["outro"] = script["outro"]
        bridges = []
        for b in script.get("bridges") or []:
            if not isinstance(b, dict):
                continue
            bridges.append({"afterTag": b.get("afterTag"), "text": b.get("text")})
        runtime_script["bridges"] = bridges

    sections = composition.get("sections") or {}
    intro = sections.get("intro")

    out = {
        "schemaVersion": composition.get("schemaVersion"),
        "character": composition.get("character"),
        "sources": composition.get("sources"),
        "sections": {"intro": intro} if intro is not None else {},
    }
    if runtime_meta:
        out["meta"] = runtime_meta
    if runtime_script is not None:
        out["script"] = runtime_script
    eh = composition.get("editHints")
    if isinstance(eh, dict):
        eh = dict(eh)
        eh.pop("transcriptFile", None)
        out["editHints"] = eh
    return out


def _build_video_data_from_composition_import(composition, *, niche, orientation, commentary_mode):
    composition = _sanitize_composition_import_for_runtime(composition)
    if composition.get("schemaVersion") != 1:
        raise ValueError("Unsupported schemaVersion (expected 1)")
    character = (composition.get("character") or "").strip()
    if not character:
        raise ValueError("character is required")

    sources = composition.get("sources") or []
    if not sources:
        raise ValueError("sources[] is required")

    source_tags = set()
    body_videos = []
    for src in sources:
        url = (src.get("url") or "").strip()
        if not url.startswith("http"):
            raise ValueError(f"Invalid source URL: {url!r}")
        vidclips = []
        for clip in src.get("vidclips") or []:
            tag = (clip.get("tag") or "").strip()
            if not tag:
                raise ValueError("Each vidclip requires a tag")
            if tag in source_tags:
                raise ValueError(f"Duplicate clip tag: {tag}")
            source_tags.add(tag)
            vidclips.append(
                {
                    "startTime": clip.get("startTime"),
                    "endTime": clip.get("endTime"),
                    "mute": clip.get("mute", True),
                    "tag": tag,
                }
            )
        if not vidclips:
            raise ValueError("Each source needs at least one vidclip")
        body_videos.append(
            {
                "url": url,
                "title": (src.get("title") or "").strip() or "Source",
                "vidclips": vidclips,
            }
        )

    sections_in = composition.get("sections") or {}
    intro_in = sections_in.get("intro") or {}
    if not isinstance(intro_in.get("segment"), list):
        raise ValueError("sections.intro.segment is required (full episode on Intro tab)")

    intro_segments = [
        _normalize_import_segment(s, character)
        for s in intro_in["segment"]
    ]
    for seg in intro_segments:
        text = (seg.get("text") or "").strip()
        if text.startswith("[CLIP:") and text.endswith("]"):
            tag = seg.get("tag") or text[len("[CLIP:") : -1].strip()
            if tag not in source_tags:
                raise ValueError(f"Intro timeline tag {tag!r} has no vidclip in sources")

    intro_clips = [{"characterName": character, "videos": body_videos}]
    intro_section_videos = [{"Character": character, "videos": body_videos}]

    built_sections = [
        {
            "section": "intro",
            "section_characters": [character],
            "segment": intro_segments,
            "section_script": _section_script_from_segments(intro_segments),
            "section_videos": intro_section_videos,
            "section_images": [],
            "clips": intro_clips,
            "imageData": [],
        },
    ]

    meta = composition.get("meta") or {}
    title = (meta.get("workingTitle") or meta.get("protocol") or "").strip()
    if not title and body_videos:
        title = body_videos[0].get("title") or "Composition import"

    script_blob = composition.get("script") or {}
    doc = {
        "title": title,
        "script": json.dumps(script_blob, ensure_ascii=False),
        "sections": built_sections,
        "status": "draft",
        "niche": niche,
        "orientation": orientation,
        "commentaryMode": commentary_mode if niche == "commentary" else None,
        "compositionImportMeta": {
            k: meta.get(k)
            for k in (
                "commentaryUrl",
                "protocol",
                "sourceNote",
                "generatedAt",
                "compositionSection",
                "commentaryId",
                "batchVideo",
                "workingTitle",
            )
            if meta.get(k) is not None
        },
    }
    if composition.get("editHints"):
        doc["editHints"] = composition["editHints"]
    return doc


@app.route("/api/compose-composition-import", methods=["POST"])
@jwt_required()
def compose_composition_import():
    data = request.get_json(silent=True) or {}
    current_user = get_jwt_identity()
    niche = data.get("niche")
    if not niche:
        return jsonify({"error": "niche is required"}), 400

    composition = data.get("composition")
    if composition is None:
        raw = (data.get("script") or data.get("compositionJson") or "").strip()
        if raw:
            try:
                composition = json.loads(raw)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON in script field"}), 400
    if not composition:
        return jsonify({"error": "composition object or JSON script is required"}), 400

    commentary_mode = data.get("commentaryMode", "paraphrase")
    try:
        video_payload = _build_video_data_from_composition_import(
            composition,
            niche=niche,
            orientation=data.get("orientation"),
            commentary_mode=commentary_mode,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    pre_id = ObjectId()
    doc = {
        "_id": pre_id,
        "creator": current_user,
        "user_id": current_user,
        "created_at": datetime.utcnow(),
        **video_payload,
    }
    db.video_data.insert_one(doc)
    return jsonify({"_id": str(pre_id), "title": doc["title"]}), 201


# @app.route("/api/compose-transcript-video", methods=["POST"])
# @jwt_required()
# def compose_transcript_video():
#     try:
#         data = request.json
#         script = data.get("script")
#         niche = data.get("niche")
#         orientation = data.get("orientation")
#         split_mode = data.get("splitMode", None)
#         commentary_mode = data.get("commentaryMode", "paraphrase")

#         is_commentary_original = niche == "commentary" and commentary_mode == "original"

#         print(f"We are in {niche} in split mode {split_mode}, commentary mode {commentary_mode}")
#         current_user = get_jwt_identity()

#         scripts = asyncio.run(
#             full_processing_pipeline(
#                 script,
#                 niche,
#                 intro_ratio=3,
#                 commentary_mode=commentary_mode if niche == "commentary" else None,
#             )
#         )
#         full_script = scripts.get("full_script", "")

#         intro_text = scripts.get("intro", "")
#         body_text = scripts.get("body", "")

#         if body_text and body_text.strip():
#             combined_text = f"{intro_text.strip()}\n\n{body_text.strip()}"
#         else:
#             combined_text = intro_text.strip()

#         print(combined_text)

#         sections = extract_main_character_keywords_and_visuals(
#             intro=combined_text,
#             body=None,
#             split_mode=split_mode,
#             niche=niche,
#             commentary_mode=commentary_mode if niche == "commentary" else None,
#         )

#         intro_characters = sections['intro']['main_characters']
#         intro_character_names = list(intro_characters.keys())
#         intro_segments = sections['intro']['segments']

#         if is_commentary_original:
#             intro_video_urls = []
#             intro_image_urls = []
#         else:
#             intro_yt = sections['intro']['yt_keywords']
#             intro_yt_keywords = [
#                 (character, keyword.strip('"'))
#                 for character, keywords in intro_yt.items()
#                 for keyword in keywords
#             ]
#             intro_video_urls = fetch_related_videos(intro_yt_keywords)
#             used_image_urls = set()
#             intro_image_urls = fetch_related_images(
#                 [intro_characters] if isinstance(intro_characters, str) else intro_characters,
#                 niche,
#                 used_urls_global=used_image_urls,
#             )

#         video_data = {
#             "title": scripts.get('title', ''),
#             "script": full_script,
#             "creator": current_user,
#             "sections": [
#                 {
#                     "section": "intro",
#                     "section_script": combined_text,
#                     "section_videos": intro_video_urls,
#                     "section_images": intro_image_urls,
#                     "section_characters": intro_character_names,
#                     "segment": intro_segments,
#                 }
#             ],
#             "status": "draft",
#             "niche": niche,
#             "orientation": orientation,
#             "commentaryMode": commentary_mode if niche == "commentary" else None,
#         }

#         try:
#             inserted_data = db.video_data.insert_one(video_data)
#         except Exception as db_error:
#             return jsonify({"error": f"Database insertion failed: {str(db_error)}"}), 500

#         saved_video = db.video_data.find_one({"_id": inserted_data.inserted_id})
#         if not saved_video:
#             return jsonify({"error": "Video data retrieval failed"}), 500

#         saved_video["_id"] = str(saved_video["_id"])

#         message = 'Your Script Composition is Complete'
#         button_text = "View Composition"
#         button_url = f"{FRONTEND_URL}/compose/{saved_video['_id']}"
#         send_telegram_notification(message, button_text, button_url, current_user)

#         return jsonify(saved_video)

#     except Exception as e:
#         send_telegram_error(
#             f"🎉 ❌ Script composition failed: {e}",
#             get_jwt_identity()
#         )
#         return jsonify({"error": str(e)}), 500


@app.route("/api/update_sections", methods=["PUT"])
@jwt_required()
def update_section():
    print("✅ Update Route was hit!")
    try:
        data = request.json
        video_id = data.get("video_id")  # Video document ID
        updated_sections = data.get("section_data")  # Full array of sections (intro, body, etc.)
        edit_hints = data.get("editHints")
        print(video_id, updated_sections)
        if not video_id:
            return jsonify({"error": "Missing video_id"}), 400
        if updated_sections is None and edit_hints is None:
            return jsonify({"error": "Missing section_data or editHints"}), 400

        set_fields = {}
        if updated_sections is not None:
            set_fields["sections"] = updated_sections
        if edit_hints is not None:
            set_fields["editHints"] = edit_hints

        update_result = db.video_data.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": set_fields}
        )

        if update_result.matched_count == 0:
            print("No matching video found")
            return jsonify({"error": "No matching video found"}), 404

        return jsonify({"message": f"Sections updated successfully {updated_sections}"}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": f"Update failed: {str(e)}"}), 500



def merge_videos(video_paths, output_name,output_dir="finsal_outputs"):
    """Merge videos quickly using FFmpeg's concat demuxer without re-encoding."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / output_name
    
    # Create temporary list file
    list_file = Path(output_dir) / "concat_list.txt"
    with open(list_file, "w") as f:
        for path in video_paths:
            f.write(f"file '{Path(path).absolute()}'\n")

    # FFmpeg command for fastest possible merge
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",  # Stream copy (no re-encoding)
        "-y",  # Overwrite without asking
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        list_file.unlink()  # Cleanup temp file
        return str(output_path)
    except subprocess.CalledProcessError as e:
        logging.error(f"Merge failed: {e.stderr.decode()}")
        return None


def _run_regenerate_video(current_user, vid_data):
    """
    Script regeneration + video enqueue — runs off Waitress (script queue).
    """
    try:
        regenerate_data = vid_data.get("data")
        asset_folder = vid_data.get("asset_folder")

        data = asyncio.run(full_pipeline(regenerate_data))
        data["avatarImage"] = regenerate_data.get("avatarImage")
        data["backgroundMusic"] = regenerate_data.get("backgroundMusic")
        data["backgroundVideo"] = regenerate_data.get("backgroundVideo")
        data["channelId"] = regenerate_data.get("channelId")
        data["channelLogo"] = regenerate_data.get("channelLogo")
        data["niche"] = regenerate_data.get("niche")
        data["voice"] = regenerate_data.get("voice")
        data["introTeaser"] = regenerate_data.get("introTeaser")
        data["narrationImages"] = regenerate_data.get("narrationImages")
        data["transitionBanner"] = regenerate_data.get("transitionBanner")

        if not data.get("sections") or not data.get("title"):
            send_telegram_error(
                "❌ Video regeneration failed: incomplete script data",
                current_user,
            )
            return

        enqueue_video_job(
            current_user,
            process_regenerated_video,
            current_user,
            data,
            asset_folder,
        )

        send_telegram_notification(
            "Your video regeneration has started",
            "View progress",
            f"{FRONTEND_URL}/drafts",
            current_user,
        )
    except Exception as exc:
        send_telegram_error(f"❌ Video regeneration failed: {exc}", current_user)
        raise


@app.route("/api/regenerate-video", methods=["POST"])
@jwt_required()
def regenerate_video():
    try:
        vid_data = request.json
        current_user = get_jwt_identity()

        user = db.users.find_one({"email": current_user})
        if not user:
            return jsonify({"msg": "No such user found"}), 404

        regenerate_data = vid_data.get("data")
        if not regenerate_data:
            return jsonify({"error": "Missing regeneration data"}), 400

        if not vid_data.get("asset_folder"):
            return jsonify({"error": "Missing asset_folder for regeneration"}), 400

        enqueue_script_job(current_user, _run_regenerate_video, current_user, vid_data)
        return jsonify({
            "msg": "🎬 Video regeneration started. You will be notified when it’s ready.",
        }), 202

    except Exception as e:
        logging.exception("Failed during regenerate_video API call")
        return jsonify({"error": str(e)}), 500


    
   
    
@app.route("/api/video-drafts", methods=['GET'])
@jwt_required()
def get_drafts():
 try:
    query = {}
    mine = request.args.get("mine", "").lower() in ("1", "true", "yes")
    if mine:
        current_user = get_jwt_identity()
        query["$or"] = [
            {"user_id": current_user},
            {"creator": current_user},
        ]
    drafts = list(db.video_data.find(query).sort("_id", -1))
    if not drafts:
        if mine:
            return jsonify([]), 200
        return jsonify({"error": "No draft videos found"}), 404

    def _latest_at(draft):
        created = draft.get("created_at")
        if isinstance(created, datetime):
            return created
        oid = draft.get("_id")
        try:
            if not isinstance(oid, ObjectId):
                oid = ObjectId(oid)
            return oid.generation_time.replace(tzinfo=None)
        except Exception:
            return datetime.min

    drafts.sort(key=_latest_at, reverse=True)
    for draft in drafts:
        draft["_id"] = str(draft["_id"])

    return jsonify(drafts)
 except Exception as e:
      return jsonify({"error": "Internal server Error during drafts"}), 500  # Handle empty result


@app.route("/api/finalize-video", methods=["POST"])
@jwt_required()
def finalize_video():
    data = request.json
    current_user = get_jwt_identity()
    asset_folder =data.get("asset_folder",None)

    # 🔍 Look up user
    user = db.users.find_one({"email": current_user})
    if not user:
        return jsonify({"msg": "No such user found"}), 404

    # ✅ Extract & validate video data ID
    try:
        vid_data_id = ObjectId(data['vidDataId']) if isinstance(data['vidDataId'], str) else data['vidDataId']
    except Exception:
        return jsonify({"msg": "Invalid video data ID format"}), 400

    # 🟡 Update DB to reflect job has started
    db.video_data.update_one({"_id": vid_data_id}, {"$set": {"status": "processing"}})

    # 🧠 Determine voice (default = alloy)
    voice = data.get("voice", "alloy")

    # 🧠 Determine voice (default = alloy)
    background = data.get("backgroundMusic",None)

    print(data)

    # 🚀 Enqueue the full background video processing job with positional args
    enqueue_video_job(
        current_user,             # used for tracking per-user job count
        process_full_video,       # the function to call
        current_user,             # arg 1: user_id (passed to process_full_video)
        data,                     # arg 2: video data payload
        str(vid_data_id),
        asset_folder           # arg 4: stringified ID to avoid bson errors
    )

    # ✅ Respond immediately
    return jsonify({"msg": "🎬 Video is being processed. You will be notified when it’s ready."}), 202



@app.route("/api/completed-videos", methods=["GET"])
@jwt_required()
def completed_videos():
    current_user = get_jwt_identity()  # e.g. "techwebdev3@gmail.com"

    completed_videos = list(
        db.completed_videos.find({"user_id": current_user})
    )

    if not completed_videos:
        return jsonify({"error": "No completed videos found"}), 404

    # Convert ObjectId to string for JSON serialization
    for video in completed_videos:
        video["_id"] = str(video["_id"])

    return jsonify(completed_videos), 200



import json
from datetime import datetime

COOKIE_DIR = "cookies"
COOKIE_FILE = os.path.join(COOKIE_DIR, "imagefx_cookies.json")  # Changed to JSON
TMP_FILE = os.path.join(COOKIE_DIR, "imagefx_cookies.tmp")


@app.route("/api/imagefx/cookies", methods=["POST"])
@jwt_required()
def update_imagefx_cookies():
    """
    Save ImageFX cookies (supports 1-3 cookies).
    
    Request body:
    - Single cookie: {"cookies": "cookie_string"}
    - Multiple cookies: {"cookies": ["cookie1", "cookie2", "cookie3"]}
    """
    current_user = get_jwt_identity()

    data = request.get_json(silent=True)
    if not data or "cookies" not in data:
        return jsonify({"error": "Missing 'cookies' in request body"}), 400

    cookies_input = data["cookies"]
    
    # Normalize input to list
    if isinstance(cookies_input, str):
        # Single cookie string (backward compatible)
        cookies = [cookies_input.strip()]
    elif isinstance(cookies_input, list):
        # List of cookies
        cookies = [c.strip() for c in cookies_input if isinstance(c, str) and c.strip()]
    else:
        return jsonify({"error": "Invalid 'cookies' format. Must be string or array"}), 400

    if not cookies:
        return jsonify({"error": "No valid cookies provided"}), 400

    # Limit to 3 cookies
    if len(cookies) > 3:
        return jsonify({"error": "Maximum 3 cookies allowed"}), 400

    # Ensure directory exists
    os.makedirs(COOKIE_DIR, exist_ok=True)

    try:
        # Prepare data to save
        data_to_save = {
            "cookies": cookies,
            "updated_at": datetime.utcnow().isoformat(),
            "updated_by": current_user,
            "count": len(cookies)
        }
        
        # Atomic write: write to temp file then replace
        with open(TMP_FILE, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)

        os.replace(TMP_FILE, COOKIE_FILE)

        return jsonify({
            "message": "ImageFX cookies updated successfully",
            "cookie_count": len(cookies),
            "updated_by": current_user,
            "file": "imagefx_cookies.json"
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to update cookies",
            "details": str(e)
        }), 500


@app.route("/api/completed-videos/cleanup", methods=["DELETE"])
# @jwt_required()
def cleanup_completed_videos():
    # Calculate cutoff: 10 days ago from now (UTC)
    ten_days_ago = datetime.utcnow() - timedelta(days=10)

    # Delete all videos older than 10 days
    result = db.completed_videos.delete_many({
        "created_at": {"$lt": ten_days_ago}
    })

    return jsonify({
        "message": "Old completed videos deleted successfully",
        "deleted_count": result.deleted_count,
        "cutoff_date": ten_days_ago.isoformat() + "Z"
    }), 200

@app.route("/api/users", methods=["GET"])
@jwt_required()
def get_users():
    users = list(db.users.find({}))  # Convert cursor to list
    # print(users)

    if not users:  # Correct way to check if list is empty
        return jsonify({"error": "No completed videos found"}), 404  # Handle empty result

    # Convert ObjectId to string for JSON serialization
    for user in users:
        user["_id"] = str(user["_id"])

    return jsonify(users)
 

@app.route("/api/get-user", methods=['GET'])
@jwt_required()
def get_user():
    print("✅ Route was hit!")
    current_user = get_jwt_identity()
    print(current_user)
    
    user=db.users.find_one({"email":current_user})
    if not user:
        return jsonify({"msg":"No such user found"}),404
    user["_id"] = str(user["_id"])
    
    return jsonify(user)
     
@app.route("/api/change-voice", methods=['POST'])
@jwt_required()
def chnage_voice():
    current_user = get_jwt_identity()
    data=request.json
    voice=data['voice']
    print(data,voice)
  
    user = db.users.update_one( {"email": current_user},{"$set": {"voice": voice}})
    
    return jsonify({"msg":"Successful"}),200


@app.route("/api/change-background", methods=['POST'])
@jwt_required()
def change_background():
    current_user = get_jwt_identity()
    data=request.json
    music=data['background']
    print(data,music)
    user = db.users.update_one( {"email": current_user},{"$set": {"background": music}})
    print(user)
    return jsonify({"msg":"Successful"}),200


@app.route('/api/fetch_images', methods=['POST'])
@jwt_required()
def fetch_images():
    # Get parameters from the request
    data=request.json
    keyword = data.get('keyword', '')
    n = data.get('n', 100)
    page=data.get('page',1)
    # print(data)
    # Validate parameters
    if not keyword:
        return jsonify({"error": "Keyword parameter is required"}), 400
    
    try:
        n = int(n)
    except ValueError:
        return jsonify({"error": "Parameter 'n' must be an integer"}), 400
    
    # Call the Serper API
    try:
        url = "https://google.serper.dev/images"
        payload = json.dumps({
            "q": keyword,
            "num": n,
            "page": page,
        })
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        data = json.loads(response.text)
        
        # Extract image URLs from response and return just the array of URLs
        images = data.get('images', [])
        urls = []
        for image in images[:n]:
            image_url = image.get('imageUrl')
            if image_url:
                urls.append(image_url)
        
        return jsonify(urls)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500   
@app.route('/api/trigger-news-update', methods=['POST'])
@jwt_required()
def trigger_news_update():
    # Start news update in a separate thread to not block the response
    thread = Thread(target=check_for_updates)
    thread.daemon = True
    thread.start()
    return jsonify({"message": "News update triggered"}), 200



@app.route('/api/update-news-compose', methods=['POST'])
@jwt_required()
def update_news_compose():
    data = request.json
    id = ObjectId(data['newsId'])  # Convert to ObjectId
    news_update = db.news.find_one_and_update({"_id": id}, {"$set": {"composed": True}})
    return jsonify({"message": "News update triggered"}), 200
   
OUTPUT_FOLDER = 'outputs'
    
# ✅ Serve Video Files (long downloads — use WAITRESS_THREADS=32+ in .env)
@app.route('/videos/<path:filename>')
def serve_video(filename):
    full_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.isfile(full_path):
        print(f"❌ File not found: {full_path}")
        return abort(404)
    resp = send_from_directory(OUTPUT_FOLDER, filename, conditional=True)
    resp.headers["Cache-Control"] = "public, max-age=3600"
    resp.headers["Accept-Ranges"] = "bytes"
    return resp

VOICE_FOLDER='voices'
@app.route('/voice/<path:filename>')
def serve_voice(filename):
    """Serve a video file from the output folder."""
    return send_from_directory(VOICE_FOLDER, filename)

BACKGROUND_FOLDER='background_music'
@app.route('/background/<path:filename>')
def serve_background(filename):
    """Serve a video file from the output folder."""
    return send_from_directory(BACKGROUND_FOLDER, filename)


BG_FOLDER='bg_static/music'
@app.route('/bg-music/<path:filename>')
def serve_background_static(filename):
    """Serve a video file from the output folder."""
    return send_from_directory(BG_FOLDER, filename)



# Error handlers for JWT-related issues
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({
        'status': 401,
        'sub_status': 'token_expired',
        'message': 'The token has expired'
    }), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({
        'status': 401,
        'sub_status': 'invalid_token',
        'message': 'Signature verification failed'
    }), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({
        'status': 401,
        'sub_status': 'missing_token',
        'message': 'Authorization header is missing'
    }), 401


# ✅ Handle 404 Errors
@app.errorhandler(404)
def not_found(error=None):
    return jsonify({"status": 404, "message": "Not Found " + request.url}), 404


from werkzeug.utils import secure_filename
# Make sure this directory exists
CHANNEL_FOLDER = 'channel_media'
os.makedirs(CHANNEL_FOLDER, exist_ok=True)

@app.route("/api/create-channel", methods=["POST", "OPTIONS"])
@jwt_required(optional=True)
def create_channel():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        user_email = get_jwt_identity()
        if not user_email:
            return jsonify({"msg": "Unauthorized"}), 401

        user = db.users.find_one({"email": user_email})
        if not user:
            return jsonify({"msg": "User not found"}), 404

        user_id    = str(user["_id"])
        timestamp  = int(time.time())

        channel_name = request.form.get("channelName", "").strip()
        niche        = request.form.get("niche", "").strip()

        if not channel_name or not niche:
            return jsonify({"msg": "Missing channel name or niche"}), 400

        channel_data = {
            "userId":      user_id,
            "channelName": channel_name,
            "niche":       niche,
            "createdAt":   datetime.utcnow(),
        }

        logo = request.files.get("logo")
        if logo and logo.filename:
            filename = secure_filename(f"{user_id}_logo_{timestamp}_{logo.filename}")
            logo.save(os.path.join(CHANNEL_FOLDER, filename))
            channel_data["logo"] = filename

        if niche == "commentary":
            bg_image = request.files.get("backgroundImage")
            if bg_image and bg_image.filename:
                filename = secure_filename(f"{user_id}_bgimg_{timestamp}_{bg_image.filename}")
                bg_image.save(os.path.join(CHANNEL_FOLDER, filename))
                channel_data["backgroundImage"] = filename

            bg_video = request.files.get("backgroundVideo")
            if bg_video and bg_video.filename:
                filename = secure_filename(f"{user_id}_bgvid_{timestamp}_{bg_video.filename}")
                bg_video.save(os.path.join(CHANNEL_FOLDER, filename))
                channel_data["backgroundVideo"] = filename

        result = db.channels.insert_one(channel_data)
        channel_data["_id"] = str(result.inserted_id)

        return jsonify({"msg": "Channel created successfully", "data": channel_data}), 201

    except Exception as e:
        print("❌ create_channel error:", e)
        return jsonify({"msg": "Server error", "error": str(e)}), 500

from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

@app.route("/api/get-channels", methods=["GET"])
@jwt_required()
def get_user_channels():
    current_user_email = get_jwt_identity()

    user = db.users.find_one({"email": current_user_email})
    if not user:
        return jsonify({"msg": "User not found"}), 404

    user_id_obj = user["_id"]              # ObjectId
    user_id_str = str(user_id_obj)         # string

    # Matches BOTH:
    # - userId: "user_id_str"
    # - userId: ["user_id_str", ...]
    # And also covers if userId was stored as ObjectId (string vs ObjectId mix)
    query = {
        "$or": [
            {"userId": user_id_str},
            {"userId": user_id_obj},
            {"userId": {"$in": [user_id_str]}},
            {"userId": {"$in": [user_id_obj]}},
        ]
    }

    user_channels = list(db.channels.find(query))

    for ch in user_channels:
        ch["_id"] = str(ch["_id"])
        # optional: normalize userId for frontend
        if "userId" in ch and isinstance(ch["userId"], ObjectId):
            ch["userId"] = str(ch["userId"])

    return jsonify({"channels": user_channels}), 200

@app.route("/api/update-channel/<channel_id>", methods=["PUT", "OPTIONS"])
@jwt_required(optional=True)
def update_channel(channel_id):
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        user_email = get_jwt_identity()
        if not user_email:
            return jsonify({"msg": "Unauthorized"}), 401

        user = db.users.find_one({"email": user_email})
        if not user:
            return jsonify({"msg": "User not found"}), 404

        user_id   = str(user["_id"])
        timestamp = int(time.time())

        channel = db.channels.find_one({
            "_id":    ObjectId(channel_id),
            "userId": user_id,
        })
        if not channel:
            return jsonify({"msg": "Channel not found or access denied"}), 404

        updates = {}

        channel_name = request.form.get("channelName", "").strip()
        niche        = request.form.get("niche", "").strip()
        if channel_name:
            updates["channelName"] = channel_name
        if niche:
            updates["niche"] = niche

        logo = request.files.get("logo")
        if logo and logo.filename:
            filename = secure_filename(f"{user_id}_logo_{timestamp}_{logo.filename}")
            logo.save(os.path.join(CHANNEL_FOLDER, filename))
            updates["logo"] = filename

        effective_niche = niche or channel.get("niche", "")
        if effective_niche == "commentary":
            bg_image = request.files.get("backgroundImage")
            if bg_image and bg_image.filename:
                filename = secure_filename(f"{user_id}_bgimg_{timestamp}_{bg_image.filename}")
                bg_image.save(os.path.join(CHANNEL_FOLDER, filename))
                updates["backgroundImage"] = filename

            bg_video = request.files.get("backgroundVideo")
            if bg_video and bg_video.filename:
                filename = secure_filename(f"{user_id}_bgvid_{timestamp}_{bg_video.filename}")
                bg_video.save(os.path.join(CHANNEL_FOLDER, filename))
                updates["backgroundVideo"] = filename

        if updates:
            updates["updatedAt"] = datetime.utcnow()
            db.channels.update_one({"_id": ObjectId(channel_id)}, {"$set": updates})

        return jsonify({"msg": "Channel updated successfully", "updated": updates}), 200

    except Exception as e:
        print("❌ update_channel error:", e)
        return jsonify({"msg": "Server error", "error": str(e)}), 500

def update_channel(channel_id):
    user_email = get_jwt_identity()
    print("in update")

    # Find the user
    user = db.users.find_one({"email": user_email})
    if not user:
        return jsonify({"msg": "User not found"}), 404

    user_id = str(user["_id"])
    timestamp = int(time.time())

    # Validate channel exists and belongs to user
    channel = db.channels.find_one({"_id": ObjectId(channel_id), "userId": user_id})
    if not channel:
        return jsonify({"msg": "Channel not found or access denied"}), 404

    updates = {}
    channel_name = request.form.get("channelName")
    niche = request.form.get("niche")

    if channel_name:
        updates["channelName"] = channel_name
    if niche:
        updates["niche"] = niche

    # Handle logo replacement
    if 'logo' in request.files:
        logo = request.files['logo']
        if logo.filename:
            logo_filename = secure_filename(f"{user_id}_logo_{timestamp}_{logo.filename}")
            logo_path = os.path.join(CHANNEL_FOLDER, logo_filename)
            logo.save(logo_path)
            updates['logo'] = logo_filename

    # Handle commentary files if applicable
    if niche == "commentary":
        if 'backgroundImage' in request.files:
            img = request.files['backgroundImage']
            if img.filename:
                img_filename = secure_filename(f"{user_id}_bgimg_{timestamp}_{img.filename}")
                img_path = os.path.join(CHANNEL_FOLDER, img_filename)
                img.save(img_path)
                updates['backgroundImage'] = img_filename

        if 'backgroundVideo' in request.files:
            video = request.files['backgroundVideo']
            if video.filename:
                video_filename = secure_filename(f"{user_id}_bgvid_{timestamp}_{video.filename}")
                video_path = os.path.join(CHANNEL_FOLDER, video_filename)
                video.save(video_path)
                updates['backgroundVideo'] = video_filename

    # Perform update
    if updates:
        updates["updatedAt"] = datetime.utcnow()
        db.channels.update_one({"_id": ObjectId(channel_id)}, {"$set": updates})

    return jsonify({"msg": "✅ Channel updated successfully", "updated": updates}), 200

@app.route("/api/delete-channel/<channel_id>", methods=["DELETE"])
@jwt_required()
def delete_channel(channel_id):
    try:
        user_email = get_jwt_identity()
        user = db.users.find_one({"email": user_email})
        if not user:
            return jsonify({"msg": "User not found"}), 404

        user_id = str(user["_id"])

        # Verify ownership before deleting
        channel = db.channels.find_one({
            "_id": ObjectId(channel_id),
            "userId": user_id
        })
        if not channel:
            return jsonify({"msg": "Channel not found or access denied"}), 404

        # Optionally: clean up media files from disk
        for field in ["logo", "backgroundImage", "backgroundVideo"]:
            filename = channel.get(field)
            if filename:
                filepath = os.path.join(CHANNEL_FOLDER, filename)
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        print(f"⚠️ Could not delete file {filepath}: {e}")

        db.channels.delete_one({"_id": ObjectId(channel_id)})

        return jsonify({"msg": "✅ Channel deleted successfully"}), 200

    except Exception as e:
        print("❌ Delete channel failed:", e)
        return jsonify({"msg": "Delete error", "error": str(e)}), 500

@app.route('/channel_media/<path:filename>')
def serve_channel_media(filename):
    return send_from_directory('channel_media', filename)


@app.route("/api/channel/update-audio", methods=["POST"])
@jwt_required()
def update_channel_audio():
    try:
        user_email = get_jwt_identity()
        user = db.users.find_one({"email": user_email})
        if not user:
            return jsonify({"msg": "User not found"}), 404

        data       = request.get_json()
        channel_id = data.get("channelId")
        voice_id   = data.get("voice")       # now expects a Speechify voice ID string
        music      = data.get("music")

        if not channel_id:
            return jsonify({"msg": "channelId is required"}), 400

        if voice_id and music:
            return jsonify({"msg": "Only one of 'voice' or 'music' can be updated at a time"}), 400

        update_field = {}
        if voice_id:
            update_field["voice"] = voice_id          # store the ID e.g. "en-US-andrew"
        elif music:
            update_field["background_music"] = music
        else:
            return jsonify({"msg": "Either 'voice' or 'music' must be provided"}), 400

        result = db.channels.update_one(
            {"_id": ObjectId(channel_id), "userId": str(user["_id"])},
            {"$set": update_field}
        )

        if result.modified_count == 0:
            return jsonify({"msg": "Channel not found or no update applied"}), 404

        return jsonify({"msg": "Channel updated successfully", "voice": voice_id}), 200

    except Exception as e:
        print("Update failed:", e)
        return jsonify({"msg": "Update error", "error": str(e)}), 500



from werkzeug.utils import secure_filename
import uuid

EDIT_FOLDER = Path(os.getenv("EDIT_FOLDER", "edit_uploads"))
EDIT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'mp4', 'mov', 'avi', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload overlay media (images/videos)"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Generate unique filename
    original_ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex[:16]}.{original_ext}"
    filepath = EDIT_FOLDER / unique_filename
    
    try:
        file.save(str(filepath))
        
        # Build absolute URL using request context
        public_base = os.getenv("PUBLIC_BASE_URL")
        if public_base:
            base = _public_base_url()
        else:
            # Use the current request's host
            base = request.host_url.rstrip('/')
        
        public_url = f"{base}/api/uploads/{unique_filename}"
        
        return jsonify({
            "url": public_url,
            "filename": unique_filename,
            "path": str(filepath)
        }), 200
    
    except Exception as e:
        app.logger.exception("Upload failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/uploads/<filename>", methods=["GET"])
def serve_upload(filename):
    """Serve uploaded overlay files"""
    return send_from_directory(EDIT_FOLDER, filename)

# app.py
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from editor import apply_edits, OUTPUT_FOLDER

DREAM_FOLDER='dream'

@app.route('/dream_uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(DREAM_FOLDER, filename)

@app.route("/api/process_video", methods=["POST"])
def process_video():
    data = request.get_json(force=True, silent=False)
    print(data)
    video_input = data.get("videoFilename")
    overlays = data.get("overlays", [])
    
    # Parse vignette config
    vignette_data = data.get("vignette")
    if isinstance(vignette_data, dict):
        vignette_config = vignette_data
    elif isinstance(vignette_data, (int, float)):
        strength = float(vignette_data)
        vignette_config = {
            "enabled": strength > 0,
            "strength": strength,
            "centerX": 0.5,
            "centerY": 0.5,
            "innerRadius": 0.4,
            "outerRadius": 1.0,
            "color": [0, 0, 0],
            "maxAlpha": 0.99 * strength
        }
    else:
        vignette_config = {"enabled": False}
    
    noise = float(data.get("noise", 0) or 0)
    ken_burns_global = data.get("kenBurns")
    
    if not video_input:
        return jsonify({"error": "Missing videoFilename"}), 400
    
    # Handle URL or filename
    if video_input.startswith("http://") or video_input.startswith("https://"):
        from urllib.parse import urlparse
        parsed = urlparse(video_input)
        video_filename = os.path.basename(parsed.path)
    else:
        video_filename = video_input
    
    base_path = OUTPUT_FOLDER / video_filename
    if not base_path.exists():
        return jsonify({"error": f"Base video not found: {video_filename}"}), 404
    
    # Sanitize overlays
    clean_overlays = []
    for ov in overlays:
        if not ov:
            continue
        src = ov.get("url") or ov.get("path")
        if not src:
            continue
        
        s = max(0.0, float(ov.get("start", 0)))
        e = max(s, float(ov.get("end", s)))
        if e <= s:
            continue
        
        t = "video" if (ov.get("type") or "image").lower() == "video" else "image"
        x = int(ov.get("x", 0))
        y = int(ov.get("y", 0))
        opacity = float(ov.get("opacity", 1.0))
        scale = ov.get("scale")
        scale = float(scale) if scale not in (None, "", False) else None
        
        ken = ov.get("kenBurns")
        
        item = {
            "url": src,
            "start": s,
            "end": e,
            "type": t,
            "x": 0,
            "y": 0,
            "fitScreen": bool(ov.get("fitScreen", True)),
            "opacity": opacity,
            "scale": scale,
        }
        if ken is not None:
            item["kenBurns"] = ken
        
        clean_overlays.append(item)
    
    # Check if there's ANYTHING to process
    has_overlays = len(clean_overlays) > 0
    has_vignette = vignette_config.get("enabled", False)
    has_noise = noise > 0
    if not has_overlays and not has_vignette and not has_noise:
        return jsonify({
            "error": "Nothing to process. Please add overlays, enable vignette, or noise."
        }), 400
    
    app.logger.info(f"Processing video: overlays={len(clean_overlays)}, vignette={has_vignette}, noise={has_noise}")
    
    try:
        out_path, public_url = apply_edits(
            video_path=str(base_path),
            overlays=clean_overlays,
            replace_audio=bool(data.get("replaceAudio", False)),
            vignette=vignette_config,
            noise=noise,
            ken_burns_default=ken_burns_global,
            output_name_hint=video_filename,
            target_fps=None
        )
    except Exception as e:
        app.logger.exception("Processing failed")
        return jsonify({"error": str(e)}), 500
    
    response_data = {
        "processedVideoUrl": public_url or f"/api/outputs/{os.path.basename(out_path)}",
        "filename": os.path.basename(out_path)
    }
    
    return jsonify(response_data), 200




# ============================================================================
# CHARACTER STUDIO API ROUTES (FIXED IMAGE SERVING)
# ============================================================================

import os
import json
import uuid
import time
import shutil
import subprocess
import re
from pathlib import Path
from datetime import datetime
from flask import request, jsonify, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename

# ── Configuration ─────────────────────────────────────────────────────────────
CHARACTER_FOLDER = Path('character_uploads')
CHARACTER_FOLDER.mkdir(parents=True, exist_ok=True)

WHISK_COOKIE_FILE = "cookies/imagefx_cookies.json"
ALLOWED_CHARACTER_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ── Helper Functions ──────────────────────────────────────────────────────────

def read_whisk_cookie():
    """
    Read the active cookie from the rotation file.
    Supports both JSON and TXT formats.
    """
    try:
        # Try JSON format first
        json_path = "cookies/imagefx_cookies.json"
        txt_path = "cookies/imagefx_cookies.txt"
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cookies = data.get('cookies', [])
                return cookies[0] if cookies else None
        
        # Fallback to TXT format (rotator file)
        elif os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                cookie = f.read().strip()
                return cookie if cookie else None
        
        else:
            app.logger.error("No cookie file found (JSON or TXT)")
            return None
            
    except Exception as e:
        app.logger.error(f"Failed to read cookie: {e}")
        return None


def find_whisk_cmd():
    """Find whisk executable across platforms."""
    found = shutil.which("whisk")
    if found:
        return found
    
    appdata = os.environ.get("APPDATA", "")
    userprofile = os.environ.get("USERPROFILE", "")
    
    candidates = [
        os.path.join(appdata, "npm", "whisk.cmd"),
        os.path.join(appdata, "npm", "whisk"),
        os.path.join(userprofile, ".bun", "bin", "whisk"),
        os.path.join(userprofile, ".bun", "bin", "whisk.exe"),
        "/usr/local/bin/whisk",
        os.path.expanduser("~/.bun/bin/whisk"),
        os.path.expanduser("~/.npm-global/bin/whisk"),
    ]
    
    for p in candidates:
        if os.path.isfile(p):
            return p
    
    return "whisk"


def find_imagefx_cmd():
    """Find imagefx executable across platforms."""
    found = shutil.which("imagefx")
    if found:
        return found
    
    appdata = os.environ.get("APPDATA", "")
    userprofile = os.environ.get("USERPROFILE", "")
    
    candidates = [
        os.path.join(appdata, "npm", "imagefx.cmd"),
        os.path.join(appdata, "npm", "imagefx.ps1"),
        os.path.join(appdata, "npm", "imagefx"),
        os.path.join(userprofile, ".bun", "bin", "imagefx"),
        "/usr/local/bin/imagefx",
        os.path.expanduser("~/.npm-global/bin/imagefx"),
    ]
    
    for p in candidates:
        if os.path.isfile(p):
            return p
    
    return "imagefx"


def extract_whisk_media_id(stdout):
    """Extract media ID from whisk CLI output."""
    m = re.search(r'\[\+\]\s+ID:\s+(\S+)', stdout)
    if m and len(m.group(1)) >= 20:
        return m.group(1)
    
    m = re.search(r'(CAM[a-zA-Z0-9+/=_\-]{20,})', stdout)
    return m.group(1) if m else None


def whisk_generate_image(prompt, out_dir, cookie):
    """Generate image via Whisk CLI. Returns (media_id, file_path)."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        whisk_cmd = find_whisk_cmd()
        use_shell = os.name == "nt" and whisk_cmd.lower().endswith(".cmd")
        
        cmd = [
            whisk_cmd, "generate",
            "--prompt", prompt,
            "--aspect", "LANDSCAPE",
            "--dir", out_dir,
            "--cookie", cookie,
        ]
        
        known_files = set(Path(out_dir).glob("*"))
        
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=180, shell=use_shell,
        )
        
        stdout = result.stdout or ""
        
        if result.returncode != 0:
            app.logger.error(f"Whisk generate failed: {stdout[:500]}")
            return "", ""
        
        media_id = extract_whisk_media_id(stdout)
        time.sleep(0.5)
        new_files = [f for f in Path(out_dir).glob("*") if f not in known_files]
        file_path = str(new_files[0]) if new_files else ""
        
        return media_id or "", file_path
        
    except Exception as e:
        app.logger.error(f"Whisk generate exception: {e}")
        return "", ""


def imagefx_generate_image(prompt, out_dir, cookie):
    """Generate image via ImageFX CLI. Returns file_path."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        imagefx_cmd = find_imagefx_cmd()
        
        if imagefx_cmd.endswith('.ps1'):
            cmd = [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", imagefx_cmd, "generate",
                "--prompt", prompt, "--size", "LANDSCAPE",
                "--dir", out_dir, "--cookie", cookie,
            ]
        else:
            use_shell = os.name == "nt" and imagefx_cmd.lower().endswith(".cmd")
            cmd = [
                imagefx_cmd, "generate",
                "--prompt", prompt, "--size", "LANDSCAPE",
                "--dir", out_dir, "--cookie", cookie,
            ]
        
        known_files = set(Path(out_dir).glob("*"))
        
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=180,
            shell=use_shell if not imagefx_cmd.endswith('.ps1') else False,
        )
        
        if result.returncode != 0:
            app.logger.error(f"ImageFX generate failed: {result.stdout[:500]}")
            return ""
        
        time.sleep(0.5)
        new_files = [f for f in Path(out_dir).glob("*") if f not in known_files]
        return str(new_files[0]) if new_files else ""
        
    except Exception as e:
        app.logger.error(f"ImageFX generate exception: {e}")
        return ""


def whisk_upload_to_cloud(image_path, cookie):
    """Upload image to Whisk cloud and return media_id."""
    try:
        temp_dir = CHARACTER_FOLDER / "temp_upload"
        temp_dir.mkdir(exist_ok=True)
        
        descriptor = "Character base image for consistent storytelling"
        media_id, _ = whisk_generate_image(descriptor, str(temp_dir), cookie)
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return media_id or ""
    except Exception as e:
        app.logger.error(f"Whisk upload failed: {e}")
        return ""


def whisk_create_project(name):
    """Create a dedicated Whisk project for this character."""
    return f"proj_{uuid.uuid4().hex[:12]}"


def whisk_refine_test(media_id, test_prompt, out_dir, cookie):
    """Run a refine operation for testing. Returns file_path."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        whisk_cmd = find_whisk_cmd()
        use_shell = os.name == "nt" and whisk_cmd.lower().endswith(".cmd")
        
        cmd = [
            whisk_cmd, "refine", media_id,
            "--prompt", test_prompt,
            "--dir", out_dir,
            "--cookie", cookie,
        ]
        
        known_files = set(Path(out_dir).glob("*"))
        
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=180, shell=use_shell,
        )
        
        if result.returncode != 0:
            app.logger.error(f"Whisk refine failed: {result.stdout[:500]}")
            return ""
        
        time.sleep(0.5)
        new_files = [f for f in Path(out_dir).glob("*") if f not in known_files]
        return str(new_files[0]) if new_files else ""
        
    except Exception as e:
        app.logger.error(f"Whisk refine exception: {e}")
        return ""


def generate_test_refinements(character_descriptor, style_lock, media_id, character_folder_name, cookie):
    """Auto-generate 3 test refinements. Returns dict with test image filenames."""
    test_dir = CHARACTER_FOLDER / f"{character_folder_name}_tests"
    test_dir.mkdir(exist_ok=True)
    
    # Clear old tests if they exist
    for old_test in test_dir.glob("*"):
        old_test.unlink()
    
    full_descriptor = f"{character_descriptor}. Style: {style_lock}"
    
    tests = {
        "sitting": f"{full_descriptor} — now sitting down. Keep style identical. Only change: seated pose.",
        "pointing": f"{full_descriptor} — now pointing at camera. Keep style identical. Only change: right arm pointing forward.",
        "holding": f"{full_descriptor} — now holding an object. Keep style identical. Only change: add object in hand.",
    }
    
    results = {}
    for test_name, test_prompt in tests.items():
        file_path = whisk_refine_test(media_id, test_prompt, str(test_dir), cookie)
        if file_path:
            results[test_name] = Path(file_path).name
        else:
            results[test_name] = None
    
    return results


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/api/characters/generate", methods=["POST"])
@jwt_required()
def generate_character_variations():
    """
    Generate 4 character variations via whisk or imagefx.
    Request: {name, character_descriptor, style_lock, generator: "whisk"|"imagefx"}
    """
    try:
        current_user = get_jwt_identity()
        data = request.json
        
        name = data.get("name", "").strip()
        character_descriptor = data.get("character_descriptor", "").strip()
        style_lock = data.get("style_lock", "").strip()
        generator = data.get("generator", "whisk").lower()
        
        if not name or not character_descriptor or not style_lock:
            return jsonify({"error": "Name, character_descriptor, and style_lock are required"}), 400
        
        if generator not in ["whisk", "imagefx"]:
            return jsonify({"error": "Generator must be 'whisk' or 'imagefx'"}), 400
        
        cookie = read_whisk_cookie()
        if not cookie:
            return jsonify({"error": "No active cookie available"}), 500
        
        full_prompt = f"{character_descriptor}. Style: {style_lock}. LANDSCAPE aspect ratio."
        
        temp_id = uuid.uuid4().hex[:12]
        temp_dir = CHARACTER_FOLDER / f"temp_{temp_id}"
        temp_dir.mkdir(exist_ok=True)
        
        variations = []
        
        for i in range(4):
            if generator == "whisk":
                media_id, file_path = whisk_generate_image(full_prompt, str(temp_dir), cookie)
                if not file_path:
                    continue
                
                # ✅ CORRECT: Use temp folder in URL
                filename = Path(file_path).name
                variations.append({
                    "variation_id": f"{temp_id}_var_{i}",
                    "preview_url": f"/character_media/temp_{temp_id}/{filename}",
                    "media_id": media_id,
                    "generator": "whisk"
                })
            else:
                file_path = imagefx_generate_image(full_prompt, str(temp_dir), cookie)
                if not file_path:
                    continue
                
                filename = Path(file_path).name
                variations.append({
                    "variation_id": f"{temp_id}_var_{i}",
                    "preview_url": f"/character_media/temp_{temp_id}/{filename}",
                    "media_id": None,
                    "generator": "imagefx"
                })
        
        if not variations:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return jsonify({"error": "Failed to generate any variations"}), 500
        
        return jsonify({
            "variations": variations,
            "temp_id": temp_id,
            "name": name,
            "character_descriptor": character_descriptor,
            "style_lock": style_lock,
            "status": "pending_selection"
        }), 200
        
    except Exception as e:
        app.logger.exception("Generate variations failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/characters/save-and-test", methods=["POST"])
@jwt_required()
def save_and_test_character():
    """
    Save selected variation and run test refinements.
    Request: {name, character_descriptor, style_lock, selected_variation_id, temp_id, media_id, generator}
    """
    try:
        current_user = get_jwt_identity()
        data = request.json
        
        name = data.get("name", "").strip()
        character_descriptor = data.get("character_descriptor", "").strip()
        style_lock = data.get("style_lock", "").strip()
        selected_var_id = data.get("selected_variation_id")
        temp_id = data.get("temp_id")
        
        if not all([name, character_descriptor, style_lock, selected_var_id, temp_id]):
            return jsonify({"error": "Missing required fields"}), 400
        
        cookie = read_whisk_cookie()
        if not cookie:
            return jsonify({"error": "No active cookie available"}), 500
        
        temp_dir = CHARACTER_FOLDER / f"temp_{temp_id}"
        if not temp_dir.exists():
            return jsonify({"error": "Temporary files not found"}), 404
        
        # ✅ Generate clean character folder name
        character_folder_name = f"char_{uuid.uuid4().hex[:12]}"
        character_dir = CHARACTER_FOLDER / character_folder_name
        character_dir.mkdir(exist_ok=True)
        
        temp_files = sorted(temp_dir.glob("*"))
        var_index = int(selected_var_id.split("_")[-1])
        
        if var_index >= len(temp_files):
            return jsonify({"error": "Selected variation not found"}), 404
        
        selected_file = temp_files[var_index]
        base_filename = f"{character_folder_name}_base{selected_file.suffix}"
        base_path = character_dir / base_filename
        shutil.copy(selected_file, base_path)
        
        media_id_from_selection = data.get("media_id")
        generator = data.get("generator", "whisk")
        
        if generator == "imagefx" or not media_id_from_selection:
            media_id = whisk_upload_to_cloud(str(base_path), cookie)
        else:
            media_id = media_id_from_selection
        
        if not media_id:
            if character_dir.exists():
                shutil.rmtree(character_dir)
            return jsonify({"error": "Failed to upload to Whisk cloud"}), 500
        
        project_id = whisk_create_project(name)
        
        test_images = generate_test_refinements(
            character_descriptor, style_lock, media_id, character_folder_name, cookie
        )
        
        character_data = {
            "user_id": current_user,
            "name": name,
            "character_descriptor": character_descriptor,
            "style_lock": style_lock,
            "whisk_project_id": project_id,
            "media_id": media_id,
            "base_image": base_filename,
            "character_folder": character_folder_name,  # ✅ STORE FOLDER NAME
            "test_images": test_images,
            "status": "awaiting_approval",
            "generator": generator,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "used_count": 0,
        }
        
        result = db.characters.insert_one(character_data)
        character_data["_id"] = str(result.inserted_id)
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # ✅ CORRECT URL GENERATION
        test_urls = {
            k: f"/character_media/{character_folder_name}_tests/{v}" 
            for k, v in test_images.items() if v
        }
        
        return jsonify({
            "character_id": character_data["_id"],
            "base_image_url": f"/character_media/{character_folder_name}/{base_filename}",
            "test_images": test_urls,
            "status": "awaiting_approval"
        }), 201
        
    except Exception as e:
        app.logger.exception("Save and test failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/characters/upload", methods=["POST"])
@jwt_required()
def upload_character():
    """
    Upload external character image (DALL-E/Midjourney).
    Form data: name, character_descriptor, style_lock, image file
    """
    try:
        current_user = get_jwt_identity()
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        name = request.form.get('name', '').strip()
        character_descriptor = request.form.get('character_descriptor', '').strip()
        style_lock = request.form.get('style_lock', '').strip()
        
        if not name or not character_descriptor or not style_lock:
            return jsonify({"error": "Name, character_descriptor, and style_lock are required"}), 400
        
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        ext = image_file.filename.rsplit('.', 1)[1].lower()
        if ext not in ALLOWED_CHARACTER_EXTENSIONS:
            return jsonify({"error": f"File type not allowed"}), 400
        
        cookie = read_whisk_cookie()
        if not cookie:
            return jsonify({"error": "No active cookie available"}), 500
        
        # ✅ Generate clean character folder name
        character_folder_name = f"char_{uuid.uuid4().hex[:12]}"
        character_dir = CHARACTER_FOLDER / character_folder_name
        character_dir.mkdir(exist_ok=True)
        
        base_filename = f"{character_folder_name}_base.{ext}"
        base_path = character_dir / base_filename
        image_file.save(str(base_path))
        
        media_id = whisk_upload_to_cloud(str(base_path), cookie)
        
        if not media_id:
            if character_dir.exists():
                shutil.rmtree(character_dir)
            return jsonify({"error": "Failed to upload to Whisk cloud"}), 500
        
        project_id = whisk_create_project(name)
        
        test_images = generate_test_refinements(
            character_descriptor, style_lock, media_id, character_folder_name, cookie
        )
        
        character_data = {
            "user_id": current_user,
            "name": name,
            "character_descriptor": character_descriptor,
            "style_lock": style_lock,
            "whisk_project_id": project_id,
            "media_id": media_id,
            "base_image": base_filename,
            "character_folder": character_folder_name,  # ✅ STORE FOLDER NAME
            "test_images": test_images,
            "status": "awaiting_approval",
            "generator": "user_upload",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "used_count": 0,
        }
        
        result = db.characters.insert_one(character_data)
        character_data["_id"] = str(result.inserted_id)
        
        # ✅ CORRECT URL GENERATION
        test_urls = {
            k: f"/character_media/{character_folder_name}_tests/{v}" 
            for k, v in test_images.items() if v
        }
        
        return jsonify({
            "character_id": character_data["_id"],
            "base_image_url": f"/character_media/{character_folder_name}/{base_filename}",
            "test_images": test_urls,
            "status": "awaiting_approval"
        }), 201
        
    except Exception as e:
        app.logger.exception("Upload character failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/characters/<character_id>/approve", methods=["POST"])
@jwt_required()
def approve_character(character_id):
    """Approve character for production use."""
    try:
        current_user = get_jwt_identity()
        
        character = db.characters.find_one({
            "_id": ObjectId(character_id),
            "user_id": current_user
        })
        
        if not character:
            return jsonify({"error": "Character not found"}), 404
        
        if character.get("status") == "approved":
            return jsonify({"message": "Character already approved"}), 200
        
        db.characters.update_one(
            {"_id": ObjectId(character_id)},
            {
                "$set": {
                    "status": "approved",
                    "approved_at": datetime.utcnow()
                }
            }
        )
        
        return jsonify({"message": "Character approved successfully"}), 200
        
    except Exception as e:
        app.logger.exception("Approve character failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/characters/<character_id>/reject", methods=["DELETE"])
@jwt_required()
def reject_character(character_id):
    """Reject and delete character."""
    try:
        current_user = get_jwt_identity()
        
        character = db.characters.find_one({
            "_id": ObjectId(character_id),
            "user_id": current_user
        })
        
        if not character:
            return jsonify({"error": "Character not found"}), 404
        
        # ✅ Use stored folder name
        char_folder = character.get("character_folder")
        if char_folder:
            char_dir = CHARACTER_FOLDER / char_folder
            if char_dir.exists():
                shutil.rmtree(char_dir)
            
            test_dir = CHARACTER_FOLDER / f"{char_folder}_tests"
            if test_dir.exists():
                shutil.rmtree(test_dir)
        
        db.characters.delete_one({"_id": ObjectId(character_id)})
        
        return jsonify({"message": "Character rejected and deleted"}), 200
        
    except Exception as e:
        app.logger.exception("Reject character failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/characters/<character_id>", methods=["PUT"])
@jwt_required()
def edit_character(character_id):
    """
    Edit character descriptor and/or style_lock.
    Triggers re-testing. Character goes back to awaiting_approval.
    """
    try:
        current_user = get_jwt_identity()
        data = request.json
        
        character = db.characters.find_one({
            "_id": ObjectId(character_id),
            "user_id": current_user
        })
        
        if not character:
            return jsonify({"error": "Character not found"}), 404
        
        updates = {}
        old_data = {}
        
        if "name" in data and data["name"].strip():
            old_data["name"] = character.get("name")
            updates["name"] = data["name"].strip()
        
        if "character_descriptor" in data and data["character_descriptor"].strip():
            old_data["character_descriptor"] = character.get("character_descriptor")
            updates["character_descriptor"] = data["character_descriptor"].strip()
        
        if "style_lock" in data and data["style_lock"].strip():
            old_data["style_lock"] = character.get("style_lock")
            updates["style_lock"] = data["style_lock"].strip()
        
        if not updates:
            return jsonify({"error": "No fields to update"}), 400
        
        needs_retest = "character_descriptor" in updates or "style_lock" in updates
        
        if needs_retest:
            cookie = read_whisk_cookie()
            if not cookie:
                return jsonify({"error": "No active cookie available"}), 500
            
            new_descriptor = updates.get("character_descriptor", character["character_descriptor"])
            new_style = updates.get("style_lock", character["style_lock"])
            media_id = character["media_id"]
            char_folder = character.get("character_folder")
            
            if not char_folder:
                return jsonify({"error": "Character folder not found"}), 500
            
            test_images = generate_test_refinements(
                new_descriptor, new_style, media_id, char_folder, cookie
            )
            
            updates["test_images"] = test_images
            updates["status"] = "awaiting_approval"
        
        updates["updated_at"] = datetime.utcnow()
        
        if old_data:
            old_data["edited_at"] = datetime.utcnow()
            db.characters.update_one(
                {"_id": ObjectId(character_id)},
                {"$push": {"edit_history": old_data}}
            )
        
        db.characters.update_one(
            {"_id": ObjectId(character_id)},
            {"$set": updates}
        )
        
        updated_character = db.characters.find_one({"_id": ObjectId(character_id)})
        updated_character["_id"] = str(updated_character["_id"])
        
        if needs_retest:
            char_folder = updated_character.get("character_folder")
            test_urls = {
                k: f"/character_media/{char_folder}_tests/{v}" 
                for k, v in test_images.items() if v
            }
            updated_character["test_image_urls"] = test_urls
        
        return jsonify({
            "message": "Character updated successfully" + (" and re-tested" if needs_retest else ""),
            "character": updated_character
        }), 200
        
    except Exception as e:
        app.logger.exception("Edit character failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/characters", methods=["GET"])
@jwt_required()
def get_user_characters():
    """Get all approved characters for current user."""
    try:
        current_user = get_jwt_identity()
        
        characters = list(db.characters.find({
            "user_id": current_user,
            "status": "approved"
        }).sort("created_at", -1))
        
        for char in characters:
            char["_id"] = str(char["_id"])
            char_folder = char.get("character_folder")  # ✅ USE STORED FOLDER NAME
            
            if char.get("base_image") and char_folder:
                char["image_url"] = f"/character_media/{char_folder}/{char['base_image']}"
        
        return jsonify({"characters": characters}), 200
        
    except Exception as e:
        app.logger.exception("Get characters failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/characters/<character_id>", methods=["GET"])
@jwt_required()
def get_character_details(character_id):
    """Get full character details including test images."""
    try:
        current_user = get_jwt_identity()
        
        character = db.characters.find_one({
            "_id": ObjectId(character_id),
            "user_id": current_user
        })
        
        if not character:
            return jsonify({"error": "Character not found"}), 404
        
        character["_id"] = str(character["_id"])
        char_folder = character.get("character_folder")  # ✅ USE STORED FOLDER NAME
        
        if character.get("base_image") and char_folder:
            character["base_image_url"] = f"/character_media/{char_folder}/{character['base_image']}"
        
        if character.get("test_images") and char_folder:
            test_urls = {}
            for test_name, filename in character["test_images"].items():
                if filename:
                    test_urls[test_name] = f"/character_media/{char_folder}_tests/{filename}"
            character["test_image_urls"] = test_urls
        
        return jsonify(character), 200
        
    except Exception as e:
        app.logger.exception("Get character details failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/characters/<character_id>", methods=["DELETE"])
@jwt_required()
def delete_character(character_id):
    """Delete character (approved or not)."""
    try:
        current_user = get_jwt_identity()
        
        character = db.characters.find_one({
            "_id": ObjectId(character_id),
            "user_id": current_user
        })
        
        if not character:
            return jsonify({"error": "Character not found"}), 404
        
        # ✅ Use stored folder name
        char_folder = character.get("character_folder")
        if char_folder:
            char_dir = CHARACTER_FOLDER / char_folder
            if char_dir.exists():
                shutil.rmtree(char_dir)
            
            test_dir = CHARACTER_FOLDER / f"{char_folder}_tests"
            if test_dir.exists():
                shutil.rmtree(test_dir)
        
        db.characters.delete_one({"_id": ObjectId(character_id)})
        
        return jsonify({"message": "Character deleted successfully"}), 200
        
    except Exception as e:
        app.logger.exception("Delete character failed")
        return jsonify({"error": str(e)}), 500


# ── Serve Character Media ─────────────────────────────────────────────────────

@app.route('/character_media/<path:filepath>')
def serve_character_media(filepath):
    """
    Serve character images and test refinements.
    Supports paths like:
      - temp_abc123/image.png (during generation)
      - char_abc123/char_abc123_base.png (saved character)
      - char_abc123_tests/sitting.png (test refinements)
    """
    try:
        full_path = CHARACTER_FOLDER / filepath
        
        if not full_path.exists():
            app.logger.error(f"File not found: {full_path}")
            return jsonify({"error": "File not found"}), 404
        
        if not full_path.is_file():
            app.logger.error(f"Path is not a file: {full_path}")
            return jsonify({"error": "Invalid file path"}), 400
        
        directory = str(full_path.parent)
        filename = full_path.name
        
        return send_from_directory(directory, filename)
        
    except Exception as e:
        app.logger.exception(f"Serve character media failed: {e}")
        return jsonify({"error": "Failed to serve file"}), 500


# ── Debug Route (Optional - Remove in Production) ────────────────────────────

@app.route('/api/debug/character-files/<character_id>', methods=['GET'])
@jwt_required()
def debug_character_files(character_id):
    """Debug route to check what files exist for a character."""
    try:
        character = db.characters.find_one({"_id": ObjectId(character_id)})
        if not character:
            return jsonify({"error": "Character not found"}), 404
        
        char_folder = character.get("character_folder")
        if not char_folder:
            return jsonify({"error": "No character_folder in database"}), 500
        
        char_dir = CHARACTER_FOLDER / char_folder
        test_dir = CHARACTER_FOLDER / f"{char_folder}_tests"
        
        char_files = [f.name for f in char_dir.glob("*")] if char_dir.exists() else []
        test_files = [f.name for f in test_dir.glob("*")] if test_dir.exists() else []
        
        return jsonify({
            "character_folder": char_folder,
            "character_dir": str(char_dir),
            "character_dir_exists": char_dir.exists(),
            "character_files": char_files,
            "test_dir": str(test_dir),
            "test_dir_exists": test_dir.exists(),
            "test_files": test_files,
            "db_data": {
                "base_image": character.get("base_image"),
                "test_images": character.get("test_images"),
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500






# ============================================================
# DREAMFACE ROUTES
# ============================================================

DREAMFACE_AUDIO_FOLDER = os.getenv("UPLOAD_FOLDER", "static/audio")


@app.route("/api/dreamface/audio/<path:filename>", methods=["GET"])
def serve_dreamface_audio(filename):
    """
    Serve batch MP3 for download.
    Uses <path:filename> to allow subdirectory in URL
    e.g. /api/dreamface/audio/batches/dreamface_batch_0_xxx.mp3
    """
    # Sanitize — prevent path traversal
    if ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400

    if not filename.endswith(".mp3"):
        return jsonify({"error": "Invalid file type"}), 400

    audio_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "static", "audio"
    )
    file_path = os.path.join(audio_dir, filename)

    logging.info(f"📥 Audio download: {filename}")
    logging.info(f"   Path: {file_path}")
    logging.info(f"   Exists: {os.path.exists(file_path)}")

    if not os.path.exists(file_path):
        logging.error(f"❌ Not found: {file_path}")
        return jsonify({"error": "File not found"}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=os.path.basename(filename),
        mimetype="audio/mpeg",
    )

@app.route("/api/dreamface/jobs", methods=["GET"])
@jwt_required()
def get_dreamface_jobs():
    """Get all DreamFace jobs for current user."""
    current_user = get_jwt_identity()

    try:
        jobs = list(
            db.dreamface_jobs.find(
                {"user_id": current_user},
                # Exclude heavy fields — not needed for job list
                {"create_video_payload": 0, "audio_payload": 0},
            ).sort("created_at", -1)
        )
    except Exception as exc:
        logging.error("get_dreamface_jobs MongoDB error: %s", exc, exc_info=True)
        return jsonify(
            {
                "error": "Database temporarily unavailable — try again in a few seconds.",
                "jobs": [],
            }
        ), 503

    for job in jobs:
        job["_id"] = str(job["_id"])
        status = job.get("status")
        started = job.get("processing_started_at")
        stuck_sec = 0
        if status == "processing" and started:
            stuck_sec = (datetime.utcnow() - started).total_seconds()
        job["recovery"] = {
            "can_retry": status in ("processing", "failed", "completed")
            and bool(job.get("layout_config"))
            and _dreamface_batches_ready(job.get("batches")),
            "can_reset": status == "processing",
            "is_likely_stuck": status == "processing"
            and stuck_sec >= _dreamface_stuck_minutes() * 60,
        }

    return jsonify({"jobs": jobs}), 200


@app.route("/api/dreamface/jobs/<job_id>", methods=["GET"])
@jwt_required()
def get_dreamface_job(job_id):
    """Get single DreamFace job with full batch details and download links."""
    current_user = get_jwt_identity()

    job = _dreamface_job_lookup(
        job_id,
        current_user,
        {"create_video_payload": 0, "audio_payload": 0},
    )

    if not job:
        return jsonify({"error": "Job not found"}), 404

    job["_id"] = str(job["_id"])
    status = job.get("status")
    started = job.get("processing_started_at")
    stuck_min = _dreamface_stuck_minutes()
    stuck_sec = 0
    if status == "processing" and started:
        stuck_sec = (datetime.utcnow() - started).total_seconds()
    job["recovery"] = {
        "can_retry": status in ("processing", "failed", "completed")
        and bool(job.get("layout_config"))
        and _dreamface_batches_ready(job.get("batches")),
        "can_reset": status == "processing",
        "stuck_minutes": stuck_min,
        "processing_seconds": int(stuck_sec),
        "is_likely_stuck": status == "processing" and stuck_sec >= stuck_min * 60,
    }
    if _is_admin(current_user):
        job["admin_view"] = True
    return jsonify(job), 200


def _normalize_avatar_link_map(avatar_links):
    """Map batch_index → trimmed URL; coerce indices to int."""
    link_map = {}
    for item in avatar_links or []:
        if not isinstance(item, dict):
            continue
        if "batch_index" not in item or "url" not in item:
            continue
        try:
            idx = int(item["batch_index"])
        except (TypeError, ValueError):
            continue
        url = str(item["url"] or "").strip()
        if url:
            link_map[idx] = url
    return link_map


def _apply_avatar_links_to_batches(batches, avatar_links, *, job_id=None):
    """
    Overwrite every batch's avatar_url from the submitted payload.
    Raises ValueError when any batch is missing a link.
    """
    link_map = _normalize_avatar_link_map(avatar_links)
    updated = copy.deepcopy(batches or [])

    for batch in updated:
        idx = batch.get("batch_index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            raise ValueError(f"Batch has invalid batch_index: {idx!r}")

        if idx not in link_map:
            raise ValueError(f"Missing avatar link for batch {idx}")

        new_url = link_map[idx]
        old_url = (batch.get("avatar_url") or "").strip()
        if old_url and old_url != new_url:
            def _short(u):
                return u if len(u) <= 80 else f"{u[:77]}..."
            logging.info(
                f"🔄 Job {job_id}: batch {idx} avatar_url updated "
                f"({_short(old_url)} → {_short(new_url)})"
            )
        batch["avatar_url"] = new_url
        batch.pop("avatar_local_path", None)

    return updated


def _dreamface_batches_ready(batches) -> bool:
    """True when every batch has a non-empty avatar_url."""
    for batch in batches or []:
        if not (batch.get("avatar_url") or "").strip():
            return False
    return bool(batches)


def _dreamface_stuck_minutes() -> int:
    return max(1, int(os.getenv("DREAMFACE_STUCK_MINUTES", "5")))


def _enqueue_dreamface_resume(job_id, user_id, job, batches, layout_config):
    """Set job + vid_data to processing and queue resume_dreamface_pipeline."""
    from helper.process_full_video import resume_dreamface_pipeline

    now = datetime.utcnow()
    db.dreamface_jobs.update_one(
        {"_id": ObjectId(job_id)},
        {
            "$set": {
                "status": "processing",
                "error": None,
                "layout_config": layout_config,
                "batches": batches,
                "processing_started_at": now,
            },
            "$inc": {"retry_count": 1},
        },
    )
    vid_data_id = job.get("vid_data_id")
    if vid_data_id:
        db.video_data.update_one(
            {"_id": ObjectId(vid_data_id)},
            {"$set": {"status": "processing", "dreamface_job_id": str(job_id)}},
        )
    title = (job.get("title") or f"Talking Heads {str(job_id)[-8:]}")[:120]
    enqueue_video_job(
        user_id,
        resume_dreamface_pipeline,
        job_id,
        user_id,
        job_label=title,
        job_type="dreamface",
    )
    logging.info(f"▶️ DreamFace job {job_id} queued for render (retry)")


@app.route("/api/dreamface/submit-links", methods=["POST"])
@jwt_required()
def submit_dreamface_links():
    """
    User submits avatar links + layout_config back.
    Validates all batches have links then enqueues pipeline resumption.
    """
    current_user  = get_jwt_identity()
    data          = request.json

    job_id        = data.get("job_id")
    avatar_links  = data.get("avatar_links", [])
    layout_config = data.get("layout_config", None)   # ← from TalkingHeadsStudio

    if not job_id or not avatar_links:
        return jsonify({"error": "job_id and avatar_links are required"}), 400

    if not layout_config:
        return jsonify({"error": "layout_config is required"}), 400

    job = db.dreamface_jobs.find_one({
        "_id":     ObjectId(job_id),
        "user_id": current_user,
    })

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["status"] != "awaiting_dreamface":
        return jsonify({
            "error": f"Job is not awaiting links (status: {job['status']})"
        }), 400

    try:
        batches = _apply_avatar_links_to_batches(
            job["batches"], avatar_links, job_id=job_id,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    logging.info(
        f"✅ Job {job_id}: layout_config saved "
        f"[layout_id={layout_config.get('layout_id', 'unknown')}]"
    )

    _enqueue_dreamface_resume(job_id, current_user, job, batches, layout_config)

    return jsonify({
        "msg":       "🎬 Avatar links received — video generation resuming.",
        "job_id":    job_id,
        "layout_id": layout_config.get("layout_id", "unknown"),
    }), 202


def _is_clip_segment_text(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("[CLIP:") and t.endswith("]")


def _validate_dreamface_segment_update(original_segments, new_segments):
    """Narration text is frozen; clip segments may be added at end or removed."""
    if not isinstance(original_segments, list) or not isinstance(new_segments, list):
        raise ValueError("Invalid segment data")

    orig_n = len(original_segments)
    new_n = len(new_segments)

    if new_n < orig_n:
        removed = orig_n - new_n
        for i in range(orig_n - removed, orig_n):
            if not _is_clip_segment_text(original_segments[i].get("text", "")):
                raise ValueError(
                    f"Cannot remove narration segment {i + 1} — narration is already generated"
                )

    for i in range(min(orig_n, new_n)):
        orig_text = (original_segments[i].get("text") or "").strip()
        new_text = (new_segments[i].get("text") or "").strip()
        if _is_clip_segment_text(orig_text):
            if not _is_clip_segment_text(new_text):
                raise ValueError(f"Segment {i + 1} must remain a clip segment [CLIP:…]")
        elif orig_text != new_text:
            raise ValueError(
                f"Segment {i + 1} narration text cannot be changed after DreamFace audio was generated"
            )

    if new_n > orig_n:
        for i in range(orig_n, new_n):
            if not _is_clip_segment_text(new_segments[i].get("text", "")):
                raise ValueError(
                    "New segments must be clip-only [CLIP:…] — narration cannot be added in DreamFace"
                )


def _append_silent_audio_paths(existing_paths, extra_count, work_dir):
    """Append short silent WAV stubs for new clip-only segments (clip audio is muxed at render)."""
    os.makedirs(work_dir, exist_ok=True)
    paths = list(existing_paths or [])
    for n in range(extra_count):
        out = os.path.join(work_dir, f"silent_clip_{uuid.uuid4().hex[:8]}.wav")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-t", "0.25",
                out,
            ],
            capture_output=True,
            check=True,
        )
        paths.append(out)
    return paths


@app.route("/api/dreamface/jobs/<job_id>/composition", methods=["GET"])
@jwt_required()
def get_dreamface_job_composition(job_id):
    """Load editable composition snapshot for a DreamFace job."""
    current_user = get_jwt_identity()
    job = _dreamface_job_lookup(job_id, current_user)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    payload = job.get("create_video_payload") or {}
    vid_data_id = job.get("vid_data_id")
    intro_vd = None
    if vid_data_id:
        vd = db.video_data.find_one({"_id": ObjectId(vid_data_id)})
        if vd:
            for sec in vd.get("sections") or []:
                if sec.get("section") == "intro":
                    intro_vd = sec
                    break

    segments = payload.get("vid_segments") or []
    section = {
        "section": "intro",
        "segment": segments,
        "clips": payload.get("clip_data_list") or [],
        "imageData": payload.get("image_data_list") or [],
        "section_characters": payload.get("characters") or [],
        "section_videos": (intro_vd or {}).get("section_videos") or [],
        "section_images": (intro_vd or {}).get("section_images") or [],
    }

    return jsonify({
        "job_id": str(job["_id"]),
        "vid_data_id": str(vid_data_id) if vid_data_id else None,
        "status": job.get("status"),
        "can_edit": job.get("status") in ("awaiting_dreamface", "completed", "failed", "processing"),
        "can_rerender": job.get("status") in ("completed", "failed")
            and bool(job.get("layout_config"))
            and all(b.get("avatar_url") for b in (job.get("batches") or [])),
        "section": section,
        "narration_images": payload.get("narration_images") or {"mode": "default"},
        "intro_teaser": payload.get("intro_teaser"),
        "transition_banner": payload.get("transition_banner"),
    }), 200


@app.route("/api/dreamface/jobs/<job_id>/composition", methods=["PUT"])
@jwt_required()
def update_dreamface_job_composition(job_id):
    """Save clips, images, tags, and narration-image overrides without re-TTS."""
    current_user = get_jwt_identity()
    data = request.json or {}

    job = db.dreamface_jobs.find_one(
        {"_id": ObjectId(job_id), "user_id": current_user},
    )
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.get("status") not in ("awaiting_dreamface", "completed", "failed", "processing"):
        return jsonify({"error": f"Cannot edit composition in status: {job.get('status')}"}), 400

    payload = dict(job.get("create_video_payload") or {})
    audio_payload = dict(job.get("audio_payload") or {})
    original_segments = payload.get("vid_segments") or []

    section_in = data.get("section") or {}
    new_segments = section_in.get("segment")
    if new_segments is None:
        return jsonify({"error": "section.segment is required"}), 400

    try:
        _validate_dreamface_segment_update(original_segments, new_segments)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    orig_n = len(original_segments)
    new_n = len(new_segments)
    if new_n > orig_n:
        work_dir = os.path.join(
            tempfile.gettempdir(), "dreamface_comp", str(job_id)[:8]
        )
        try:
            audio_payload["segment_audio_paths"] = _append_silent_audio_paths(
                audio_payload.get("segment_audio_paths"),
                new_n - orig_n,
                work_dir,
            )
        except subprocess.CalledProcessError as exc:
            logging.error(f"Silent audio append failed: {exc}")
            return jsonify({"error": "Failed to extend audio track for new clip segments"}), 500

    payload["vid_segments"] = new_segments
    payload["clip_data_list"] = section_in.get("clips") or payload.get("clip_data_list") or []
    payload["image_data_list"] = section_in.get("imageData") or payload.get("image_data_list") or []
    payload["characters"] = section_in.get("section_characters") or payload.get("characters") or []

    if "narration_images" in data:
        payload["narration_images"] = data["narration_images"]
    if "intro_teaser" in data:
        payload["intro_teaser"] = data["intro_teaser"]
    if "transition_banner" in data:
        payload["transition_banner"] = data["transition_banner"]

    db.dreamface_jobs.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {
            "create_video_payload": payload,
            "audio_payload": audio_payload,
            "composition_updated_at": datetime.utcnow(),
        }},
    )

    vid_data_id = job.get("vid_data_id")
    if vid_data_id:
        vd = db.video_data.find_one({"_id": ObjectId(vid_data_id)})
        if vd and isinstance(vd.get("sections"), list):
            sections = []
            for sec in vd["sections"]:
                if sec.get("section") == "intro":
                    updated = dict(sec)
                    updated["segment"] = new_segments
                    updated["clips"] = payload["clip_data_list"]
                    updated["imageData"] = payload["image_data_list"]
                    updated["section_characters"] = payload["characters"]
                    if section_in.get("section_videos") is not None:
                        updated["section_videos"] = section_in["section_videos"]
                    if section_in.get("section_images") is not None:
                        updated["section_images"] = section_in["section_images"]
                    sections.append(updated)
                else:
                    sections.append(sec)
            db.video_data.update_one(
                {"_id": ObjectId(vid_data_id)},
                {"$set": {"sections": sections}},
            )

    return jsonify({"msg": "Composition saved", "job_id": job_id}), 200


@app.route("/api/dreamface/jobs/<job_id>/re-render", methods=["POST"])
@jwt_required()
def rerender_dreamface_job(job_id):
    """Re-run video render using frozen audio + updated composition."""
    data = request.json or {}
    data.setdefault("force", True)
    return _dreamface_retry_or_rerender(job_id, data)


@app.route("/api/dreamface/jobs/<job_id>/retry", methods=["POST"])
@jwt_required()
def retry_dreamface_job(job_id):
    """
    Re-queue a stuck or failed DreamFace render using saved audio + avatar links.
    Use force=true to retry while status is still 'processing'.
    """
    return _dreamface_retry_or_rerender(job_id, request.json or {})


def _dreamface_retry_or_rerender(job_id, data, owner_user_id=None):
    force = bool(data.get("force"))
    run_as = owner_user_id or get_jwt_identity()

    query = {"_id": ObjectId(job_id)}
    if owner_user_id is None:
        query["user_id"] = run_as

    job = db.dreamface_jobs.find_one(query)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    status = job.get("status")
    if status == "awaiting_dreamface":
        return jsonify({
            "error": "Job is still awaiting avatar links — use Generate Video in the studio.",
        }), 400

    if status not in ("processing", "failed", "completed"):
        return jsonify({"error": f"Cannot retry job with status: {status}"}), 400

    if status == "processing" and not force:
        started = job.get("processing_started_at")
        stuck_limit = _dreamface_stuck_minutes() * 60
        if started and (datetime.utcnow() - started).total_seconds() < stuck_limit:
            elapsed = int((datetime.utcnow() - started).total_seconds())
            return jsonify({
                "error": (
                    f"Job started {elapsed}s ago — wait {_dreamface_stuck_minutes()} min "
                    "or send force=true to retry now."
                ),
                "processing_seconds": elapsed,
                "stuck_minutes": _dreamface_stuck_minutes(),
            }), 409

    layout_config = data.get("layout_config") or job.get("layout_config")
    if not layout_config:
        return jsonify({"error": "layout_config missing — submit avatar links first"}), 400

    avatar_links = data.get("avatar_links")
    batches = job.get("batches") or []
    if avatar_links:
        try:
            batches = _apply_avatar_links_to_batches(
                batches, avatar_links, job_id=job_id,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    elif not _dreamface_batches_ready(batches):
        return jsonify({
            "error": "avatar_links missing on job — paste links in the studio and retry.",
        }), 400

    _enqueue_dreamface_resume(job_id, run_as, job, batches, layout_config)

    payload = {
        "msg": "Video generation re-queued",
        "job_id": job_id,
        "status": "processing",
    }
    if owner_user_id is not None:
        payload["overview"] = _build_admin_overview()
    return jsonify(payload), 202


@app.route("/api/dreamface/jobs/<job_id>/reset", methods=["POST"])
@jwt_required()
def reset_dreamface_job(job_id):
    """
    Unstick a job stuck in 'processing' so the user can edit avatar links again.
    Does not start rendering — use /retry after links are confirmed.
    """
    current_user = get_jwt_identity()
    data = request.json or {}
    new_status = (data.get("status") or "awaiting_dreamface").strip()

    if new_status not in ("awaiting_dreamface", "failed"):
        return jsonify({
            "error": "status must be awaiting_dreamface or failed",
        }), 400

    job = db.dreamface_jobs.find_one(
        {"_id": ObjectId(job_id), "user_id": current_user},
    )
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.get("status") != "processing":
        return jsonify({
            "error": f"Reset is only for stuck processing jobs (current: {job.get('status')})",
        }), 400

    db.dreamface_jobs.update_one(
        {"_id": ObjectId(job_id)},
        {
            "$set": {
                "status": new_status,
                "error": data.get("error") or "Reset by user — ready to edit or retry",
            },
            "$unset": {"processing_started_at": ""},
        },
    )
    vid_data_id = job.get("vid_data_id")
    if vid_data_id:
        vid_data_status = (
            "awaiting_dreamface"
            if new_status == "awaiting_dreamface"
            else "failed"
        )
        db.video_data.update_one(
            {"_id": ObjectId(vid_data_id)},
            {"$set": {"status": vid_data_status}},
        )

    logging.info(f"🔄 DreamFace job {job_id} reset → {new_status} by {current_user}")
    return jsonify({
        "msg": f"Job reset to {new_status}",
        "job_id": job_id,
        "status": new_status,
    }), 200


@app.route("/api/dreamface/resolve-preview", methods=["POST"])
@jwt_required()
def resolve_dreamface_preview():
    """
    Resolve a DreamFace share link to a direct MP4 stream URL for studio preview.
    Direct .mp4 links are returned unchanged.
    """
    data = request.json or {}
    raw_url = (data.get("url") or "").strip()
    if not raw_url:
        return jsonify({"error": "url is required"}), 400

    from helper.avatar_extractor import (
        is_dreamface_share_link,
        resolve_dreamface_video_url,
    )
    from helper.grok_imagine_extractor import (
        is_grok_imagine_share_link,
        resolve_grok_imagine_video_url,
    )

    if is_dreamface_share_link(raw_url):
        resolved = resolve_dreamface_video_url(raw_url)
        if not resolved:
            return jsonify({"error": "Could not resolve DreamFace share link"}), 422
        return jsonify({
            "source": "dreamface",
            "preview_id": resolved["preview_id"],
            "stream_url": resolved["stream_url"],
            "poster_url": resolved.get("poster_url") or "",
            "work_name": resolved.get("work_name") or "",
        }), 200

    if is_grok_imagine_share_link(raw_url):
        resolved = resolve_grok_imagine_video_url(raw_url)
        if not resolved:
            return jsonify({
                "error": "Could not resolve Grok share link (private or removed?)",
            }), 422
        return jsonify({
            "source": "grok_imagine",
            "preview_id": resolved["post_id"],
            "stream_url": resolved["stream_url"],
            "poster_url": resolved.get("poster_url") or "",
            "work_name": resolved.get("title") or "",
            "duration": float(resolved.get("duration") or 0),
        }), 200

    if raw_url.lower().startswith("http"):
        return jsonify({
            "source": "direct",
            "stream_url": raw_url,
            "preview_id": None,
            "poster_url": "",
            "work_name": "",
        }), 200

    return jsonify({"error": "Unsupported URL"}), 400


def _public_https_media_url(url: str) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return False
    return True


def _resolve_share_stream(raw_url: str) -> str | None:
    """Turn a Grok or DreamFace share link into a direct HTTPS media URL."""
    from helper.avatar_extractor import (
        is_dreamface_share_link,
        resolve_dreamface_video_url,
    )
    from helper.grok_imagine_extractor import (
        is_grok_imagine_share_link,
        resolve_grok_imagine_video_url,
    )

    if is_grok_imagine_share_link(raw_url):
        resolved = resolve_grok_imagine_video_url(raw_url)
    elif is_dreamface_share_link(raw_url):
        resolved = resolve_dreamface_video_url(raw_url)
    else:
        return None
    stream = (resolved or {}).get("stream_url") or ""
    if not _public_https_media_url(stream):
        return None
    return stream


@app.route("/api/media/clip-stream.mp4", methods=["GET", "HEAD"])
def clip_stream_preview():
    """
    Stream a resolved Grok / DreamFace clip so the browser player can seek.
    Only share links are accepted — the upstream URL is resolved on the server.
    """
    import urllib.request
    from urllib.error import HTTPError, URLError

    raw_url = (request.args.get("url") or "").strip()
    if not raw_url:
        return jsonify({"error": "url is required"}), 400

    stream_url = _resolve_share_stream(raw_url)
    if not stream_url:
        return jsonify({"error": "Could not resolve that share link for preview"}), 422

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Referer": "https://grok.com/imagine" if "grok.com" in raw_url.lower() else "https://www.dreamfaceapp.com/",
    }
    range_header = request.headers.get("Range")
    if range_header:
        headers["Range"] = range_header

    try:
        upstream = urllib.request.urlopen(
            urllib.request.Request(stream_url, headers=headers),
            timeout=60,
        )
    except HTTPError as exc:
        logging.warning(f"Clip preview upstream HTTP {exc.code} for share link")
        return jsonify({"error": "Preview source rejected the request"}), 502
    except URLError as exc:
        logging.warning(f"Clip preview upstream failed: {exc}")
        return jsonify({"error": "Preview source unreachable"}), 502

    status = getattr(upstream, "status", 200) or 200
    content_type = upstream.headers.get("Content-Type") or "video/mp4"
    passthrough = {
        name: upstream.headers.get(name)
        for name in ("Content-Length", "Content-Range", "Accept-Ranges")
        if upstream.headers.get(name)
    }

    def generate():
        try:
            while True:
                chunk = upstream.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            upstream.close()

    if request.method == "HEAD":
        upstream.close()
        resp = Response(status=status, mimetype=content_type)
    else:
        resp = Response(generate(), status=status, mimetype=content_type)

    for name, value in passthrough.items():
        resp.headers[name] = value
    resp.headers.setdefault("Accept-Ranges", "bytes")
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp


def _public_base_url(default="http://localhost:5000"):
    """PUBLIC_BASE_URL with a scheme ('api.example.com' → 'https://api.example.com')."""
    base = (os.getenv("PUBLIC_BASE_URL") or default or "").strip().rstrip("/")
    if base.startswith("//"):
        base = "https:" + base
    elif base and "://" not in base:
        base = "https://" + base
    return base


@app.route("/api/dreamface/upload-background", methods=["POST"])
@jwt_required()
def upload_dreamface_background():
    """Upload image or video background for Talking Heads Studio."""
    user = get_jwt_identity()

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file"}), 400

    media_type = (request.form.get("type") or "image").lower().strip()
    ext = os.path.splitext(file.filename)[1].lower()

    allowed_image = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    allowed_video = {".mp4", ".webm", ".mov", ".mkv"}
    allowed = allowed_image if media_type == "image" else allowed_video

    if ext not in allowed:
        return jsonify({"error": f"Unsupported {media_type} type"}), 400

    upload_dir = os.getenv(
        "DREAMFACE_BG_UPLOAD_DIR",
        os.path.join("static", "uploads", "dreamface", "backgrounds"),
    )
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = secure_filename(file.filename)
    filename = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = os.path.join(upload_dir, filename)
    file.save(save_path)

    base = _public_base_url()
    url = f"{base}/{upload_dir.replace(os.sep, '/')}/{filename}"

    logging.info(f"✅ DreamFace background uploaded | user={user} | {filename}")
    return jsonify({"url": url}), 200


from helper import composition_templates as composition_templates_helper


def _talking_heads_template_row(doc: dict) -> dict:
    payload = doc.get("payload") or {}
    return {
        "_id": doc["_id"],
        "name": doc.get("name"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "layout_config": payload.get("layout_config"),
        "is_default": doc.get("is_default", False),
        "channel_id": doc.get("channel_id"),
    }


@app.route("/api/composition/templates", methods=["GET"])
@jwt_required()
def list_composition_templates():
    """List saved composition templates (transition banners, talking heads, etc.)."""
    user_id = get_jwt_identity()
    template_type = request.args.get("type")
    channel_id = request.args.get("channel_id")

    composition_templates_helper.migrate_legacy_talking_heads(db, user_id)
    templates = composition_templates_helper.list_templates(
        db,
        user_id,
        template_type,
        channel_id if channel_id is not None else None,
    )
    return jsonify({"templates": templates}), 200


@app.route("/api/composition/templates/default", methods=["GET"])
@jwt_required()
def get_composition_template_default():
    """Return the default template for a type (optionally scoped to a channel)."""
    user_id = get_jwt_identity()
    template_type = request.args.get("type")
    channel_id = request.args.get("channel_id")

    if not template_type:
        return jsonify({"error": "type query param is required"}), 400

    composition_templates_helper.migrate_legacy_talking_heads(db, user_id)
    doc = composition_templates_helper.get_default_template(
        db,
        user_id,
        template_type,
        channel_id or None,
    )
    return jsonify({"template": doc}), 200


@app.route("/api/composition/templates", methods=["POST"])
@jwt_required()
def save_composition_template():
    """Create or update a composition template in the user's account."""
    user_id = get_jwt_identity()
    data = request.json or {}

    template_type = data.get("type") or data.get("template_type")
    name = (data.get("name") or "").strip()
    payload = data.get("payload")
    channel_id = data.get("channel_id")
    template_id = data.get("template_id")
    set_as_default = bool(data.get("set_as_default"))

    if payload is None and template_type == "talking_heads" and data.get("layout_config"):
        payload = {"layout_config": data.get("layout_config")}
    if payload is None and template_type == "transition_banner" and data.get("config"):
        payload = {
            "presetId": data.get("presetId", "custom"),
            "config": data.get("config"),
            "enabled": data.get("enabled", True),
        }
    if payload is None and template_type == "intro_teaser" and data.get("config"):
        payload = {
            "presetId": data.get("presetId", "custom"),
            "config": data.get("config"),
        }

    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "payload is required"}), 400

    try:
        saved_id, status = composition_templates_helper.save_template(
            db,
            user_id,
            template_type,
            name,
            payload,
            template_id=template_id,
            channel_id=channel_id,
            set_as_default=set_as_default,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404

    code = 200 if status == "updated" else 201
    msg = "Template updated" if status == "updated" else "Template saved"
    return jsonify({"msg": msg, "template_id": saved_id, "status": status}), code


@app.route("/api/composition/templates/<template_id>", methods=["DELETE"])
@jwt_required()
def delete_composition_template(template_id):
    """Delete a composition template."""
    user_id = get_jwt_identity()

    if not composition_templates_helper.delete_template(db, user_id, template_id):
        return jsonify({"error": "Template not found"}), 404

    return jsonify({"msg": "Template deleted"}), 200


@app.route("/api/dreamface/templates", methods=["GET"])
@jwt_required()
def list_dreamface_templates():
    """List saved Talking Heads layout templates for the current user."""
    user_id = get_jwt_identity()
    composition_templates_helper.migrate_legacy_talking_heads(db, user_id)
    templates = composition_templates_helper.list_templates(db, user_id, "talking_heads")
    return jsonify({"templates": [_talking_heads_template_row(t) for t in templates]}), 200


@app.route("/api/dreamface/templates", methods=["POST"])
@jwt_required()
def save_dreamface_template():
    """Save a Talking Heads layout template to the user's account."""
    user_id = get_jwt_identity()
    data = request.json or {}

    name = (data.get("name") or "").strip()
    layout_config = data.get("layout_config")

    if not name:
        return jsonify({"error": "Template name is required"}), 400
    if not layout_config or not isinstance(layout_config, dict):
        return jsonify({"error": "layout_config is required"}), 400

    try:
        saved_id, status = composition_templates_helper.save_template(
            db,
            user_id,
            "talking_heads",
            name,
            {"layout_config": layout_config},
            template_id=data.get("template_id"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404

    code = 200 if status == "updated" else 201
    msg = "Template updated" if status == "updated" else "Template saved"
    return jsonify({"msg": msg, "template_id": saved_id}), code


@app.route("/api/dreamface/templates/<template_id>", methods=["DELETE"])
@jwt_required()
def delete_dreamface_template(template_id):
    """Delete a saved Talking Heads template."""
    user_id = get_jwt_identity()

    if not composition_templates_helper.delete_template(db, user_id, template_id):
        return jsonify({"error": "Template not found"}), 404

    return jsonify({"msg": "Template deleted"}), 200



import os
import uuid
from werkzeug.utils import secure_filename
from flask_jwt_extended import jwt_required, get_jwt_identity

UPLOAD_DIR = os.getenv("COMPOSITION_UPLOAD_DIR", "static/uploads/composition")
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

@app.route("/api/upload_composition_image", methods=["POST"])
@jwt_required()
def upload_composition_image():
    user = get_jwt_identity()

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": "Unsupported image type"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    safe_name = secure_filename(file.filename)
    filename = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    base = _public_base_url()
    url = f"{base}/{UPLOAD_DIR.replace(os.sep, '/')}/{filename}"

    return jsonify({"url": url}), 200