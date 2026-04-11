from flask import Flask, request, jsonify, send_from_directory, send_file,session,redirect, url_for,abort
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from bson.json_util import dumps
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import asyncio
import os
import time
from datetime import datetime,timedelta
from threading import Thread
import tempfile
from dotenv import load_dotenv
import openai
import yt_dlp
import requests
from bs4 import BeautifulSoup
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips, ImageClip
from moviepy.video.fx.resize import resize
from gtts import gTTS
import whisper
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
import gc  # for memory cleanup
import multiprocessing
import shutil
from tests import create_video
from test2 import generate_audio_segments
from helper.transcript_generator import full_processing_pipeline
from tqdm import tqdm  
import threading
from pathlib import Path
import subprocess
import logging
import json
from helper.telegram_notification_engine import send_telegram_notification,send_telegram_error
from video_job_controller import enqueue_video_job,start_video_worker
from flask_cors import CORS
import multiprocessing
from daemon_launcher import ensure_cookie_daemon_running
# Load environment variables
load_dotenv()



# Initialize Flask app
app = Flask(__name__)

if multiprocessing.current_process().name == "MainProcess":
    ensure_cookie_daemon_running()

from flask_cors import CORS
from speech_routes import speechify_bp
app.register_blueprint(speechify_bp)


from cartesia_routes import cartesia_bp
app.register_blueprint(cartesia_bp)

# Add this after creating your Flask app
CORS(app, origins=["http://localhost:3000", "https://www.alchemymix.com"])

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

# Load Whisper Model Globally
whisper_model = whisper.load_model("small")

# JWT Initialization
jwt = JWTManager(app)
bcrypt = Bcrypt(app)


# Initialize MongoDB
db = mongo_instance.get_db()

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
        # Create JWT token
        access_token = create_access_token(identity=email)
        return jsonify({"token":access_token}), 200
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
    
    video_data = db.video_data.find_one({'_id': ObjectId(id)})
    if not video_data:
        return not_found()
    
    video_data["_id"] = str(video_data["_id"])
    return jsonify(video_data)

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



# ✅ Generate Captions using Whisper
def generate_captions(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result["text"]


@app.route("/api/generate-video", methods=["POST"])
@jwt_required()
def generate_video():
    
    try:
        # Step 1: Get input data from request
        data = request.json
        current_user = get_jwt_identity()
        description =data.get("description")
        duaration=data.get('duration')
        niche=data.get('niche',None)
        print("This is my niche",niche)
        optional_prompt=data.get('optionalPrompt')
        emotion='neutral'
        if not duaration:
            return jsonify({"status": 400, "error": "Enter Script Duration" + request.url}), 400
        if  not description:
            return jsonify({"status":400,"error": "Prompt is required"}), 400
        if not data.get("emotion"):
            emotion = 'dramatic'
        else:
            emotion=data.get("emotion")

        # Step 2: Generate script, narration, and captions
        scripts = generate_full_script(description, emotion,duaration,optional_prompt)
        full_script = scripts["fullScript"]
        
        # narration_result = generate_narration(full_script)
        # if not narration_result or len(narration_result) != 2:
        #     return jsonify({"error": "Narration generation failed"}), 500
        # narration_path, narration_url = narration_result

        # captions = generate_captions(narration_path)

        # Step 3: Extract sections
        sections = extract_main_character_keywords_and_visuals(scripts['intro'], scripts["body"],niche)
        # Step 4: Fetch related videos and images
        intro_characters = sections['intro']['main_characters']
        body_characters = sections['body']['main_characters']
        intro_yt = sections['intro']['yt_keywords']
        intro_yt_keywords= [(character, keyword.strip('"')) for character, keywords in intro_yt.items() for keyword in keywords]
        body_yt = sections['body']['yt_keywords']
        body_yt_keywords= [(character, keyword.strip('"')) for character, keywords in body_yt.items() for keyword in keywords]


        intro_video_urls = fetch_related_videos(intro_yt_keywords)  
        body_video_urls = fetch_related_videos(body_yt_keywords)  
        
        intro_image_urls = fetch_related_images([intro_characters] if isinstance(intro_characters, str) else intro_characters)
        body_image_urls = fetch_related_images([body_characters] if isinstance(body_characters, str) else body_characters)
        intro_segments=sections['intro']['segments']
        body_segemnets=sections['body']['segments']
        # Step 5: Save video data to MongoDB
        video_data = {
            "title": scripts['title'],
            "script": full_script,
            "creator": current_user,
            "sections": [
                {
                    "section": "intro",
                    "section_script": scripts["intro"],
                    "section_videos": intro_video_urls,
                    "section_images": intro_image_urls,
                    "section_characters": intro_characters,
                    "segment":intro_segments,
                },
                {
                    "section": "body",
                    "section_script":scripts["body"],
                    "section_videos": body_video_urls,
                    "section_images": body_image_urls,
                    "section_characters": body_characters,
                    "segment":body_segemnets,
                }
            ],
            "status":"draft",
            "created_at": datetime.utcnow(),
            "user_id":current_user,
            "niche":niche
            
        }

        try:
            inserted_data = db.video_data.insert_one(video_data)
        except Exception as db_error:
            return jsonify({"error": f"Database insertion failed: {str(db_error)}"}), 500

        # Step 6: Retrieve and return stored data
        saved_video = db.video_data.find_one({"_id": inserted_data.inserted_id})
        if not saved_video:
            return jsonify({"error": "Video data retrieval failed"}), 500

        saved_video["_id"] = str(saved_video["_id"])
        message='Your Script Composition is Complete'
        button_text="View Composition"
        id=saved_video["_id"] 
        button_url=f"{FRONTEND_URL}/compose/{id}"
        email=current_user
        send_telegram_notification(message,button_text,button_url,email)
        return jsonify(saved_video)

    except Exception as e:
        send_telegram_error(
            f"🎉 ❌ Script composition failed: {e}",
            current_user
        )
        return jsonify({"error": str(e)}), 500


@app.route("/api/compose-transcript-video", methods=["POST"])
@jwt_required()
def compose_transcript_video():
    try:
        # Step 1: Get input data from request
        data = request.json
        script = data.get("script")
        niche = data.get("niche")
        orientation=data.get("orientation")
        split_mode=data.get("splitMode",None)
        print(f"We are in {niche} in split mode {split_mode}")
        current_user = get_jwt_identity()

        # Step 2: Generate script sections
        scripts = asyncio.run(full_processing_pipeline(script, niche, intro_ratio=3))
        full_script = scripts.get("full_script", "")

        # Step 3: Merge intro+body if body exists, else just use intro
        intro_text = scripts.get("intro", "")
        body_text = scripts.get("body", "")

        if body_text and body_text.strip():
            combined_text = f"{intro_text.strip()}\n\n{body_text.strip()}"
        else:
            combined_text = intro_text.strip()
        
        print("{combined_text}")

        # Step 4: Extract characters, keywords, and segments (all in one 'intro')
        sections = extract_main_character_keywords_and_visuals(
    intro=combined_text, 
    body=None, 
    split_mode=split_mode, 
    niche=niche
)

        intro_characters = sections['intro']['main_characters']
        intro_character_names = list(intro_characters.keys())

        intro_yt = sections['intro']['yt_keywords']
        intro_yt_keywords = [
            (character, keyword.strip('"'))
            for character, keywords in intro_yt.items()
            for keyword in keywords
        ]

        # Step 5: Fetch media
        intro_video_urls = fetch_related_videos(intro_yt_keywords)
        used_image_urls = set()
        intro_image_urls = fetch_related_images(
            [intro_characters] if isinstance(intro_characters, str) else intro_characters,
            niche,
            used_urls_global=used_image_urls
        )
        intro_segments = sections['intro']['segments']

        # Step 6: Save video data to MongoDB
        video_data = {
            "title": scripts.get('title', ''),
            "script": full_script,
            "creator": current_user,
            "sections": [
                {
                    "section": "intro",
                    "section_script": combined_text,
                    "section_videos": intro_video_urls,
                    "section_images": intro_image_urls,
                    "section_characters": intro_character_names,
                    "segment": intro_segments,
                }
            ],
            "status": "draft",
            "niche": niche,
            "orientation":orientation
        }

        try:
            inserted_data = db.video_data.insert_one(video_data)
        except Exception as db_error:
            return jsonify({"error": f"Database insertion failed: {str(db_error)}"}), 500

        # Step 7: Retrieve and return stored data
        saved_video = db.video_data.find_one({"_id": inserted_data.inserted_id})
        if not saved_video:
            return jsonify({"error": "Video data retrieval failed"}), 500

        saved_video["_id"] = str(saved_video["_id"])

        # Notify
        message = 'Your Script Composition is Complete'
        button_text = "View Composition"
        button_url = f"{FRONTEND_URL}/compose/{saved_video['_id']}"
        send_telegram_notification(message, button_text, button_url, current_user)

        return jsonify(saved_video)

    except Exception as e:
        send_telegram_error(
            f"🎉 ❌ Script composition failed: {e}",
            get_jwt_identity()
        )
        return jsonify({"error": str(e)}), 500



@app.route("/api/update_sections", methods=["PUT"])
@jwt_required()
def update_section():
    print("✅ Update Route was hit!")
    try:
        data = request.json
        video_id = data.get("video_id")  # Video document ID
        updated_sections = data.get("section_data")  # Full array of sections (intro, body, etc.)
        print(video_id, updated_sections)
        if not video_id or not updated_sections:
            print("Missing video_id or sections_data")
            return jsonify({"error": "Missing video_id or sections_data"}), 400

        # 🔥 Replace the entire "sections" field in the document
        update_result = db.video_data.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": {"sections": updated_sections}}
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


@app.route("/api/regenerate-video", methods=["POST"])
@jwt_required()
def regenerate_video():
    try:
        vid_data = request.json
        current_user = get_jwt_identity()

        # 🔐 Get user info and voice
        user = db.users.find_one({"email": current_user})
            # 🧠 Determine voice (default = alloy)
        background = user.get("background")
        if not user:
            return jsonify({"msg": "No such user found"}), 404
        user["_id"] = str(user["_id"])
        voice = user.get("voice", "alloy")

        # 🧠 Regeneration input
        regenerate_data = vid_data.get("data")
        if not regenerate_data:
            return jsonify({"error": "Missing regeneration data"}), 400
        

        asset_folder = vid_data.get("asset_folder")
        if not asset_folder:
            return jsonify({"error": "Missing asset_folder for regeneration"}), 400

        # 🧬 Run the full script regeneration pipeline
        data = asyncio.run(full_pipeline(regenerate_data))
        data["avatarImage"] = regenerate_data.get("avatarImage")
        data["backgroundMusic"] = regenerate_data.get("backgroundMusic")
        data["backgroundVideo"] = regenerate_data.get("backgroundVideo")
        data["channelId"] = regenerate_data.get("channelId")
        data["channelLogo"] = regenerate_data.get("channelLogo")
        data["niche"] = regenerate_data.get("niche")
        data["voice"] = regenerate_data.get("voice")


        # 🛡️ Validate regenerated data
        if not data.get("sections") or not data.get("title"):
            return jsonify({"error": "Incomplete regenerated script data"}), 500

        # ✅ Enqueue the video creation job
        enqueue_video_job(
            current_user,               # job key
            process_regenerated_video,  # function to call
            current_user,               # user_id
            data,                       # regenerated script data
            # voice,                      # voice model
            # background,
            asset_folder                # media folder
        )

        return jsonify({
            "msg": "🎬 Video is being regenerated. You will be notified when it’s ready.",
            "viral_title": data.get("title")  # Optional: expose this for UI use
        }), 202

    except Exception as e:
        logging.exception("Failed during regenerate_video API call")
        return jsonify({"error": str(e)}), 500


    
   
    
@app.route("/api/video-drafts", methods=['GET'])
@jwt_required()
def get_drafts():
 try:
    drafts = list(db.video_data.find({}))
    # print(drafts)
    if not drafts:  # Correct way to check if list is empty
        return jsonify({"error": "No draft videos found"}), 404  # Handle empty result
    # Convert ObjectId to string for JSON serialization
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
    
# ✅ Serve Video Files
@app.route('/videos/<path:filename>')
def serve_video(filename):
    full_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.isfile(full_path):
        print(f"❌ File not found: {full_path}")
        return abort(404)
    return send_from_directory(OUTPUT_FOLDER, filename, conditional=True)

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
            base = public_base.rstrip('/')
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
    
    # Parse captions config
    captions_data = data.get("captions")
    if isinstance(captions_data, dict) and captions_data.get('enabled', False):
        captions_config = {
            "enabled": True,
            "model": captions_data.get('model', 'base'),
            "saveSrt": captions_data.get('saveSrt', True),
            "style": captions_data.get('style', {})
        }
    else:
        captions_config = {"enabled": False}
    
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
            "x": x,
            "y": y,
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
    has_captions = captions_config.get("enabled", False)
    
    if not has_overlays and not has_vignette and not has_noise and not has_captions:
        return jsonify({
            "error": "Nothing to process. Please add overlays, enable vignette, noise, or captions."
        }), 400
    
    app.logger.info(f"Processing video: overlays={len(clean_overlays)}, vignette={has_vignette}, noise={has_noise}, captions={has_captions}")
    
    try:
        out_path, public_url = apply_edits(
            video_path=str(base_path),
            overlays=clean_overlays,
            replace_audio=bool(data.get("replaceAudio", False)),
            vignette=vignette_config,
            noise=noise,
            ken_burns_default=ken_burns_global,
            captions=captions_config,  # ✅ CAPTIONS CONFIG
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
    
    # Include SRT file if generated
    if has_captions and captions_config.get('saveSrt', False):
        srt_filename = f"{Path(video_filename).stem}.srt"
        srt_path = OUTPUT_FOLDER / srt_filename
        if srt_path.exists():
            response_data["srtUrl"] = f"/api/outputs/{srt_filename}"
            response_data["srtFilename"] = srt_filename
    
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