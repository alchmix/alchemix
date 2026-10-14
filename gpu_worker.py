import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import time
import logging
from bson.objectid import ObjectId

from dotenv import load_dotenv
load_dotenv()

from job_utils import claim_oldest, load, done, failed
from mongodb import mongo_instance
from helper.process_full_video import process_full_video
from helper.telegram_notification_engine import send_telegram_error

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gpu_worker")

db = mongo_instance.get_db()


def run_finalize(job):
    data = job["data"]
    user = job["user"]
    vid_data_id = job["extra"]["vid_data_id"]
    vid_oid = ObjectId(vid_data_id)

    db.video_data.update_one({"_id": vid_oid}, {"$set": {"status": "processing"}})

    process_full_video(
        user,
        data,
        vid_data_id,
        data.get("asset_folder"),
    )


if __name__ == "__main__":
    log.info("GPU worker started")
    while True:
        path = claim_oldest("video")
        if not path:
            time.sleep(2)
            continue

        job = load(path)
        try:
            log.info("finalize job %s user=%s", job["job_id"], job["user"])
            run_finalize(job)
            done(path)
        except Exception as e:
            log.exception("finalize job failed")
            failed(path, e)
            send_telegram_error(f"❌ Video processing failed: {e}", job["user"])
            vid = job.get("extra", {}).get("vid_data_id")
            if vid:
                db.video_data.update_one(
                    {"_id": ObjectId(vid)},
                    {"$set": {"status": "failed"}},
                )