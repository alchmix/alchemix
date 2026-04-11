"""
speechify_routes.py
────────────────────────────────────────────────────────────────
Speechify API integration for your Flask app.

HOW TO REGISTER (add these 2 lines to your app.py):
    from speechify_routes import speechify_bp
    app.register_blueprint(speechify_bp)

ADD TO YOUR .env:
    SPEECHIFY_API_KEY=your_key_here

ENDPOINTS:
    GET    /api/speechify/voices                        – list all voices
    GET    /api/speechify/voices/<id>/preview           – stream preview audio
    POST   /api/speechify/voices/clone                  – clone a voice
    DELETE /api/speechify/voices/<id>                   – delete a cloned voice
    POST   /api/speechify/tts                           – generate speech
    POST   /api/speechify/channels/<id>/assign-voice    – save voice to channel in MongoDB
    GET    /api/speechify/channels/<id>/voice           – get channel's assigned voice
"""

import os
import json
import requests
from flask import Blueprint, request, jsonify, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
from bson.objectid import ObjectId
from dotenv import load_dotenv

from mongodb import mongo_instance
db = mongo_instance.get_db()

load_dotenv()

speechify_bp = Blueprint("speechify", __name__, url_prefix="/api/speechify")

SPEECHIFY_BASE   = "https://api.sws.speechify.com/v1"
SPEECHIFY_KEY    = os.getenv("SPEECHIFY_API_KEY")
REQUEST_TIMEOUT  = 30   # seconds for regular calls
CLONE_TIMEOUT    = 60   # seconds for voice cloning (upload can be slow)


def _auth_headers():
    """Bearer auth header. Raises if key is missing."""
    if not SPEECHIFY_KEY:
        raise EnvironmentError("SPEECHIFY_API_KEY is not set in your .env")
    return {"Authorization": f"Bearer {SPEECHIFY_KEY}"}


def _json_headers():
    return {**_auth_headers(), "Content-Type": "application/json"}


# def _get_db():
#     """Reuse your existing mongo_instance — avoids a second connection."""
#     from mongodb import mongo_instance
#     return mongo_instance.get_db()


# ────────────────────────────────────────────────────────────────
# 1. LIST ALL VOICES
#    Returns Speechify's shared library + any voices you've cloned.
#    Each voice includes: id, display_name, gender, locale,
#                         type (shared|personal), preview_audio, tags
# ────────────────────────────────────────────────────────────────
@speechify_bp.route("/voices", methods=["GET"])
@jwt_required()
def list_voices():
    try:
        resp = requests.get(
            f"{SPEECHIFY_BASE}/voices",
            headers=_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        return jsonify(resp.json()), resp.status_code

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Speechify unreachable: {str(e)}"}), 502


# ────────────────────────────────────────────────────────────────
# 2. PREVIEW A VOICE
#    Streams the voice's sample audio back to the caller.
#    For shared voices, Speechify returns the URL via preview_audio
#    on the list endpoint — so this route is mainly for cloned voices
#    (type=personal) which require an authenticated fetch.
# ────────────────────────────────────────────────────────────────
@speechify_bp.route("/voices/<voice_id>/preview", methods=["GET"])
@jwt_required()
def preview_voice(voice_id):
    try:
        resp = requests.get(
            f"{SPEECHIFY_BASE}/voices/{voice_id}/sample",
            headers=_auth_headers(),
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )

        if resp.status_code != 200:
            return jsonify({"error": "Preview not available for this voice"}), resp.status_code

        content_type = resp.headers.get("Content-Type", "audio/mpeg")
        return Response(
            resp.content,
            status=200,
            mimetype=content_type,
            headers={
                "Content-Disposition": "inline",
                "Accept-Ranges": "bytes",
            },
        )

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Speechify unreachable: {str(e)}"}), 502


# ────────────────────────────────────────────────────────────────
# 3. CLONE A VOICE
#    Accepts a multipart/form-data request.
#
#    Required fields:
#       name               string   — display name for the cloned voice
#       gender             string   — "male" | "female" | "notSpecified"
#       consent_full_name  string   — full name of the person consenting
#       consent_email      string   — email of the person consenting
#       sample             file     — audio file, 10–30 sec, under 5 MB
#                                     (mp3, wav, ogg, webm all accepted)
#
#    Optional fields:
#       locale             string   — e.g. "en-US" (default: en-US)
#       avatar             file     — profile image for the voice
#
#    Returns the created voice object including its id.
# ────────────────────────────────────────────────────────────────
@speechify_bp.route("/voices/clone", methods=["POST"])
@jwt_required()
def clone_voice():
    try:
        # ── Validate required form fields ──
        name               = request.form.get("name", "").strip()
        gender             = request.form.get("gender", "").strip()
        locale             = request.form.get("locale", "en-US").strip()
        consent_full_name  = request.form.get("consent_full_name", "").strip()
        consent_email      = request.form.get("consent_email", "").strip()

        missing = [f for f, v in [
            ("name", name), ("gender", gender),
            ("consent_full_name", consent_full_name),
            ("consent_email", consent_email),
        ] if not v]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        if gender not in ("male", "female", "notSpecified"):
            return jsonify({"error": "gender must be 'male', 'female', or 'notSpecified'"}), 400

        if "sample" not in request.files:
            return jsonify({"error": "Audio sample file is required (field: 'sample')"}), 400

        sample = request.files["sample"]
        if sample.content_length and sample.content_length > 5 * 1024 * 1024:
            return jsonify({"error": "Sample file exceeds 5 MB limit"}), 400

        # ── Build multipart payload ──
        consent_json = json.dumps({
            "fullName": consent_full_name,
            "email":    consent_email,
        })

        files = {
            "sample": (sample.filename, sample.read(), sample.content_type or "audio/mpeg"),
        }
        if "avatar" in request.files:
            av = request.files["avatar"]
            if av.filename:
                files["avatar"] = (av.filename, av.read(), av.content_type or "image/jpeg")

        data = {
            "name":    name,
            "gender":  gender,
            "locale":  locale,
            "consent": consent_json,
        }

        # ── Call Speechify (no Content-Type header — requests sets the boundary) ──
        resp = requests.post(
            f"{SPEECHIFY_BASE}/voices",
            headers=_auth_headers(),
            data=data,
            files=files,
            timeout=CLONE_TIMEOUT,
        )

        return jsonify(resp.json()), resp.status_code

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Speechify unreachable: {str(e)}"}), 502


# ────────────────────────────────────────────────────────────────
# 4. DELETE A CLONED VOICE
#    Only works on type=personal voices you created.
#    Also removes the voice assignment from any of your channels.
# ────────────────────────────────────────────────────────────────
@speechify_bp.route("/voices/<voice_id>", methods=["DELETE"])
@jwt_required()
def delete_voice(voice_id):
    try:
        resp = requests.delete(
            f"{SPEECHIFY_BASE}/voices/{voice_id}",
            headers=_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code in (200, 204):
            # Clean up any channel that had this voice assigned
            db.channels.update_many(
                {"speechify_voice_id": voice_id},
                {"$unset": {
                    "speechify_voice_id":   "",
                    "speechify_voice_name": "",
                }}
            )
            return jsonify({"deleted": True, "voice_id": voice_id}), 200

        return jsonify(resp.json()), resp.status_code

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Speechify unreachable: {str(e)}"}), 502


# ────────────────────────────────────────────────────────────────
# 5. GENERATE SPEECH (TTS)
#    Converts text to audio and streams the bytes back.
#
#    JSON body:
#       input         string   — text to speak (max 2000 chars)
#       voice_id      string   — voice id from /voices
#       audio_format  string   — "mp3" | "wav" | "ogg" | "aac"  (default: mp3)
#
#    Returns raw audio bytes with the appropriate Content-Type.
#    Save the response bytes directly to a file on your end.
# ────────────────────────────────────────────────────────────────
@speechify_bp.route("/tts", methods=["POST"])
@jwt_required()
def generate_speech():
    try:
        body         = request.get_json(force=True) or {}
        input_text   = (body.get("input") or "").strip()
        voice_id     = (body.get("voice_id") or "").strip()
        audio_format = body.get("audio_format", "mp3").strip().lower()

        if not input_text:
            return jsonify({"error": "'input' text is required"}), 400
        if len(input_text) > 2000:
            return jsonify({"error": f"Input is {len(input_text)} chars — max is 2000"}), 400
        if not voice_id:
            return jsonify({"error": "'voice_id' is required"}), 400
        if audio_format not in ("mp3", "wav", "ogg", "aac"):
            return jsonify({"error": "audio_format must be mp3, wav, ogg, or aac"}), 400

        resp = requests.post(
            f"{SPEECHIFY_BASE}/audio/speech",
            headers=_json_headers(),
            json={
                "input":        input_text,
                "voice_id":     voice_id,
                "audio_format": audio_format,
            },
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            try:
                return jsonify(resp.json()), resp.status_code
            except Exception:
                return jsonify({"error": resp.text}), resp.status_code

        # ── Speechify returns JSON with base64-encoded audio_data ──
        data = resp.json()
        audio_b64 = data.get("audio_data")
        if not audio_b64:
            return jsonify({"error": "No audio_data in Speechify response"}), 500

        import base64
        audio_bytes = base64.b64decode(audio_b64)

        mime_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "aac": "audio/aac",
        }

        return Response(
            audio_bytes,
            status=200,
            mimetype=mime_map[audio_format],
            headers={"Content-Disposition": "inline"},
        )

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Speechify unreachable: {str(e)}"}), 502

# ────────────────────────────────────────────────────────────────
# 6. ASSIGN VOICE TO CHANNEL  (MongoDB)
#    Saves the chosen Speechify voice_id onto your channel document.
#    Your finalize-video pipeline can then read
#    channel["speechify_voice_id"] instead of hardcoding a voice.
#
#    JSON body:
#       voice_id      string   — Speechify voice id
#       voice_name    string   — human-readable name (for display)
# ────────────────────────────────────────────────────────────────
@speechify_bp.route("/channels/<channel_id>/assign-voice", methods=["POST"])
@jwt_required()
def assign_voice_to_channel(channel_id):
    try:
        current_user = get_jwt_identity()


        user = db.users.find_one({"email": current_user})
        if not user:
            return jsonify({"error": "User not found"}), 404

        body       = request.get_json(force=True) or {}
        voice_id   = (body.get("voice_id") or "").strip()
        voice_name = (body.get("voice_name") or "").strip()

        if not voice_id:
            return jsonify({"error": "'voice_id' is required"}), 400

        # Verify channel belongs to the current user
        channel = db.channels.find_one({
            "_id":    ObjectId(channel_id),
            "userId": str(user["_id"]),
        })
        if not channel:
            return jsonify({"error": "Channel not found or access denied"}), 404

        db.channels.update_one(
            {"_id": ObjectId(channel_id)},
            {"$set": {
                "speechify_voice_id":   voice_id,
                "speechify_voice_name": voice_name,
            }}
        )

        return jsonify({
            "message":    "Voice assigned successfully",
            "channel_id": channel_id,
            "voice_id":   voice_id,
            "voice_name": voice_name,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────────
# 7. GET CHANNEL'S ASSIGNED VOICE
#    Handy for pre-populating the UI or for the video pipeline
#    to know which voice to use without re-fetching the whole channel.
# ────────────────────────────────────────────────────────────────
@speechify_bp.route("/channels/<channel_id>/voice", methods=["GET"])
@jwt_required()
def get_channel_voice(channel_id):
    try:
        channel = db.channels.find_one({"_id": ObjectId(channel_id)})
        if not channel:
            return jsonify({"error": "Channel not found"}), 404

        return jsonify({
            "channel_id":   channel_id,
            "voice_id":     channel.get("speechify_voice_id"),
            "voice_name":   channel.get("speechify_voice_name"),
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────────
# HEALTH CHECK — quick sanity test, no JWT needed
# ────────────────────────────────────────────────────────────────
@speechify_bp.route("/health", methods=["GET"])
def health():
    key_set = bool(SPEECHIFY_KEY)
    status  = "ok" if key_set else "missing_api_key"
    return jsonify({
        "status":              status,
        "api_key_configured":  key_set,
    }), 200 if key_set else 500