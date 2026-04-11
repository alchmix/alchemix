"""
cartesia_routes.py
──────────────────────────────────────────────────────────────
Cartesia API integration for Flask.

HOW TO REGISTER (add to app.py):
    from cartesia_routes import cartesia_bp
    app.register_blueprint(cartesia_bp)

ENDPOINTS:
    GET  /api/cartesia/voices          – list all Cartesia voices
    GET  /api/cartesia/voices/<id>     – get single voice
    POST /api/cartesia/voices/clone    – clone a voice from audio clip
    POST /api/cartesia/tts/preview     – generate a short TTS preview
    POST /api/cartesia/verify-pin      – verify premium access PIN
"""

import os
import json
import requests
from flask import Blueprint, request, jsonify, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
from bson.objectid import ObjectId
from dotenv import load_dotenv
import logging
from mongodb import mongo_instance
db = mongo_instance.get_db()

load_dotenv()

cartesia_bp = Blueprint("cartesia", __name__, url_prefix="/api/cartesia")

CARTESIA_BASE    = "https://api.cartesia.ai"
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
CARTESIA_VERSION = "2025-04-16"
REQUEST_TIMEOUT  = 30
CLONE_TIMEOUT    = 60

# Simple PIN for premium Cartesia access
# Change this in .env whenever needed
CARTESIA_ACCESS_PIN = os.getenv("CARTESIA_ACCESS_PIN", "1234")


def _headers():
    if not CARTESIA_API_KEY:
        raise EnvironmentError("CARTESIA_API_KEY is not set in your .env")
    return {
        "Authorization":    f"Bearer {CARTESIA_API_KEY}",
        "Cartesia-Version": CARTESIA_VERSION,
        "Content-Type":     "application/json",
    }


# ──────────────────────────────────────────────────────────────
# 1. VERIFY PIN — simple premium gate
#    POST { pin: "1234" }
#    Returns { valid: true } or { valid: false }
# ──────────────────────────────────────────────────────────────
@cartesia_bp.route("/verify-pin", methods=["POST"])
@jwt_required()
def verify_pin():
    body = request.get_json(force=True) or {}
    pin  = str(body.get("pin", "")).strip()

    if not pin:
        return jsonify({"error": "PIN is required"}), 400

    if pin == CARTESIA_ACCESS_PIN:
        return jsonify({"valid": True}), 200

    return jsonify({"valid": False}), 200


# ──────────────────────────────────────────────────────────────
# 2. LIST VOICES
#    Docs: GET /voices
#    Query params forwarded: limit, q, is_owner, gender, language
# ──────────────────────────────────────────────────────────────
@cartesia_bp.route("/voices", methods=["GET"])
@jwt_required()
def list_voices():
    try:
        all_voices   = []
        starting_after = None

        while True:
            params = {
                "expand[]": "preview_file_url",
                "limit":    100,  # max per page
            }
            if starting_after:
                params["starting_after"] = starting_after

            # Forward any frontend filters
            for key in ("q", "is_owner", "gender", "language"):
                val = request.args.get(key)
                if val is not None:
                    params[key] = val

            resp = requests.get(
                f"{CARTESIA_BASE}/voices",
                headers=_headers(),
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if not resp.ok:
                return jsonify(resp.json()), resp.status_code

            data       = resp.json()
            voices     = data.get("data", [])
            has_more   = data.get("has_more", False)

            all_voices.extend(voices)

            if not has_more or not voices:
                break

            # Use last voice ID as cursor for next page
            starting_after = voices[-1]["id"]

        logging.info(f"Cartesia: fetched {len(all_voices)} total voices")
        return jsonify({"data": all_voices, "has_more": False}), 200

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Cartesia unreachable: {str(e)}"}), 502


@cartesia_bp.route("/voices/preview-audio", methods=["GET"])
@jwt_required()
def proxy_preview_audio():
    """
    Proxies Cartesia preview audio files.
    The preview_file_url requires Authorization header which
    browsers cannot send for audio src / WaveSurfer URLs.
    
    Query param:
        url  string  — the preview_file_url from Cartesia
    """
    try:
        preview_url = request.args.get("url", "").strip()

        if not preview_url:
            return jsonify({"error": "url param is required"}), 400

        # Only allow Cartesia domains for security
        if "cartesia.ai" not in preview_url:
            return jsonify({"error": "Invalid preview URL"}), 400

        resp = requests.get(
            preview_url,
            headers={
                "Authorization":    f"Bearer {CARTESIA_API_KEY}",
                "Cartesia-Version": CARTESIA_VERSION,
            },
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )

        if not resp.ok:
            return jsonify({"error": f"Preview fetch failed: {resp.status_code}"}), resp.status_code

        content_type = resp.headers.get("Content-Type", "audio/mpeg")

        return Response(
            resp.content,
            status=200,
            mimetype=content_type,
            headers={"Content-Disposition": "inline"},
        )

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Cartesia unreachable: {str(e)}"}), 502

# ──────────────────────────────────────────────────────────────
# 3. GET SINGLE VOICE
#    Docs: GET /voices/{id}
# ──────────────────────────────────────────────────────────────
@cartesia_bp.route("/voices/<voice_id>", methods=["GET"])
@jwt_required()
def get_voice(voice_id):
    try:
        resp = requests.get(
            f"{CARTESIA_BASE}/voices/{voice_id}",
            headers=_headers(),
            params={"expand[]": "preview_file_url"},
            timeout=REQUEST_TIMEOUT,
        )
        return jsonify(resp.json()), resp.status_code

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Cartesia unreachable: {str(e)}"}), 502


# ──────────────────────────────────────────────────────────────
# 4. CLONE A VOICE
#    Docs: POST /voices/clone (multipart/form-data)
#    Required form fields:
#       clip     file    — audio clip ~5 seconds
#       name     string  — name for the cloned voice
#    Optional:
#       description  string
#       language     string  (default: en)
#       base_voice_id string
# ──────────────────────────────────────────────────────────────
@cartesia_bp.route("/voices/clone", methods=["POST"])
@jwt_required()
def clone_voice():
    try:
        name        = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()
        language    = request.form.get("language", "en").strip()
        base_voice  = request.form.get("base_voice_id", "").strip()

        if not name:
            return jsonify({"error": "name is required"}), 400
        if "clip" not in request.files:
            return jsonify({"error": "clip audio file is required"}), 400

        clip = request.files["clip"]

        # Build headers without Content-Type so requests sets the boundary
        headers = {
            "Authorization":    f"Bearer {CARTESIA_API_KEY}",
            "Cartesia-Version": CARTESIA_VERSION,
        }

        data = {"name": name, "language": language}
        if description:
            data["description"] = description
        if base_voice:
            data["base_voice_id"] = base_voice

        files = {
            "clip": (clip.filename, clip.read(), clip.content_type or "audio/mpeg"),
        }

        resp = requests.post(
            f"{CARTESIA_BASE}/voices/clone",
            headers=headers,
            data=data,
            files=files,
            timeout=CLONE_TIMEOUT,
        )

        return jsonify(resp.json()), resp.status_code

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Cartesia unreachable: {str(e)}"}), 502


# ──────────────────────────────────────────────────────────────
# 5. TTS PREVIEW
#    Generates a short audio preview for a voice via SSE endpoint
#    Returns raw audio bytes as audio/wav
#
#    JSON body:
#       voice_id  string  — Cartesia voice ID
#       text      string  — text to speak (optional, uses default)
# ──────────────────────────────────────────────────────────────
@cartesia_bp.route("/tts/preview", methods=["POST"])
@jwt_required()
def tts_preview():
    try:
        body     = request.get_json(force=True) or {}
        voice_id = (body.get("voice_id") or "").strip()
        text     = (body.get("text") or "").strip()

        if not voice_id:
            return jsonify({"error": "voice_id is required"}), 400

        if not text:
            text = (
                "This is a preview of how this voice sounds. "
                "Natural, clear, and expressive — ready for your content."
            )

        import base64

        payload = {
            "model_id":   "sonic-3",
            "transcript": text,
            "voice":      {"mode": "id", "id": voice_id},
            "output_format": {
                "container":   "raw",
                "encoding":    "pcm_s16le",
                "sample_rate": 44100,
            },
        }

        resp = requests.post(
            f"{CARTESIA_BASE}/tts/sse",
            headers=_headers(),
            json=payload,
            stream=True,
            timeout=60,
        )

        if not resp.ok:
            return jsonify({"error": f"Cartesia TTS error: {resp.status_code}"}), resp.status_code

        # Collect raw PCM from SSE stream
        audio_chunks = []
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                event = json.loads(data_str)
                if event.get("type") == "chunk" and event.get("data"):
                    audio_chunks.append(base64.b64decode(event["data"]))
            except json.JSONDecodeError:
                continue

        if not audio_chunks:
            return jsonify({"error": "No audio received from Cartesia"}), 500

        raw_pcm = b"".join(audio_chunks)

        # Wrap raw PCM s16le in a WAV container so browser can play it
        import struct
        sample_rate  = 44100
        channels     = 1
        bits         = 16
        byte_rate    = sample_rate * channels * bits // 8
        block_align  = channels * bits // 8
        data_size    = len(raw_pcm)
        wav_header   = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE",
            b"fmt ", 16, 1, channels,
            sample_rate, byte_rate, block_align, bits,
            b"data", data_size,
        )
        wav_bytes = wav_header + raw_pcm

        return Response(
            wav_bytes,
            status=200,
            mimetype="audio/wav",
            headers={"Content-Disposition": "inline"},
        )

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Cartesia unreachable: {str(e)}"}), 502


# ──────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────
@cartesia_bp.route("/health", methods=["GET"])
def health():
    key_set = bool(CARTESIA_API_KEY)
    return jsonify({
        "status":             "ok" if key_set else "missing_api_key",
        "api_key_configured": key_set,
    }), 200 if key_set else 500