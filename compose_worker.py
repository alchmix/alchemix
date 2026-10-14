import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU — compose must not touch GPU

import time
import logging
from datetime import datetime
import os

from dotenv import load_dotenv
load_dotenv()

from job_utils import claim_oldest, load, done, failed
from mongodb import mongo_instance
from helper.script_generator import generate_full_script
from helper.sections import extract_main_character_keywords_and_visuals
from helper.video_scraper import fetch_related_videos
from helper.image_scraper import fetch_related_images
from helper.telegram_notification_engine import send_telegram_notification, send_telegram_error

logging.basicConfig(level=logging.INFO)

log = logging.getLogger("compose_worker")

db = mongo_instance.get_db()
FRONTEND_URL = os.getenv("FRONTEND_URL", "")


def run_generate_video(job):
    data = job["data"]
    user = job["user"]

    description = data.get("description")
    duration = data.get("duration")
    niche = data.get("niche")
    optional_prompt = data.get("optionalPrompt")
    emotion = data.get("emotion") or "dramatic"

    scripts = generate_full_script(description, emotion, duration, optional_prompt)
    full_script = scripts["fullScript"]

    sections = extract_main_character_keywords_and_visuals(
        scripts["intro"], scripts["body"], niche
    )

    intro_characters = sections["intro"]["main_characters"]
    body_characters = sections["body"]["main_characters"]

    intro_yt_keywords = [
        (c, kw.strip('"'))
        for c, kws in sections["intro"]["yt_keywords"].items()
        for kw in kws
    ]
    body_yt_keywords = [
        (c, kw.strip('"'))
        for c, kws in sections["body"]["yt_keywords"].items()
        for kw in kws
    ]

    intro_video_urls = fetch_related_videos(intro_yt_keywords)
    body_video_urls = fetch_related_videos(body_yt_keywords)
    intro_image_urls = fetch_related_images(
        [intro_characters] if isinstance(intro_characters, str) else intro_characters
    )
    body_image_urls = fetch_related_images(
        [body_characters] if isinstance(body_characters, str) else body_characters
    )

    video_data = {
        "title": scripts["title"],
        "script": full_script,
        "creator": user,
        "sections": [
            {
                "section": "intro",
                "section_script": scripts["intro"],
                "section_videos": intro_video_urls,
                "section_images": intro_image_urls,
                "section_characters": intro_characters,
                "segment": sections["intro"]["segments"],
            },
            {
                "section": "body",
                "section_script": scripts["body"],
                "section_videos": body_video_urls,
                "section_images": body_image_urls,
                "section_characters": body_characters,
                "segment": sections["body"]["segments"],
            },
        ],
        "status": "draft",
        "created_at": datetime.utcnow(),
        "user_id": user,
        "niche": niche,
    }

    inserted = db.video_data.insert_one(video_data)
    saved_id = str(inserted.inserted_id)

    send_telegram_notification(
        "Your Script Composition is Complete",
        "View Composition",
        f"{FRONTEND_URL}/compose/{saved_id}",
        user,
    )


if __name__ == "__main__":
    log.info("Compose worker started")
    while True:
        path = claim_oldest("generate")
        if not path:
            time.sleep(2)
            continue

        job = load(path)
        try:
            log.info("generate job %s user=%s", job["job_id"], job["user"])
            run_generate_video(job)
            done(path)
        except Exception as e:
            log.exception("generate job failed")
            failed(path, e)
            send_telegram_error(f"❌ Script composition failed: {e}", job["user"])