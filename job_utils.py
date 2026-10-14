import json
import uuid
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent
PENDING = BASE / "job_payloads" / "pending"
FAILED = BASE / "job_payloads" / "failed"

PENDING.mkdir(parents=True, exist_ok=True)
FAILED.mkdir(parents=True, exist_ok=True)


def enqueue(job_type, user, data, extra=None):
    """API calls this — writes one JSON file for a worker to pick up."""
    job_id = uuid.uuid4().hex
    payload = {
        "job_id": job_id,
        "type": job_type,       # "generate" or "video"
        "user": user,
        "data": data,
        "extra": extra or {},
        "created_at": datetime.utcnow().isoformat(),
    }
    path = PENDING / f"{job_type}_{job_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return job_id

def claim_oldest_any(job_types):
    """Worker gets oldest pending file across multiple job types."""
    files = []
    for job_type in job_types:
        files.extend(PENDING.glob(f"{job_type}_*.json"))
    if not files:
        return None
    return min(files, key=lambda p: p.stat().st_mtime)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def done(path):
    path.unlink(missing_ok=True)


def failed(path, error):
    try:
        payload = load(path)
        payload["error"] = str(error)
        payload["failed_at"] = datetime.utcnow().isoformat()
        dest = FAILED / path.name
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    finally:
        path.unlink(missing_ok=True)