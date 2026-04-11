import threading
import queue
import logging
import traceback
import time
import psutil
import torch
from collections import defaultdict

# === CONFIGURATION ===
CONCURRENT_WORKERS = 1  # Exactly 4 jobs run in parallel
MAX_JOBS_PER_USER = 10
JOB_QUEUE_TIMEOUT = 5

# === GLOBAL STATE ===
video_job_queue = queue.Queue()
job_semaphore = threading.Semaphore(CONCURRENT_WORKERS)  # Limits concurrent jobs to exactly 2
active_users = defaultdict(int)
job_lock = threading.Lock()

# === JOB EXECUTOR ===
def _run_job(user_id, func, args, kwargs):
    try:
        logging.info(f"🚀 Starting video job for user {user_id}")
        func(*args, **kwargs)
        logging.info(f"✅ Finished video job for user {user_id}")
    except Exception as e:
        logging.error(f"❌ Error in video job for user {user_id}: {e}")
        logging.error(traceback.format_exc())
    finally:
        with job_lock:
            active_users[user_id] -= 1
            if active_users[user_id] <= 0:
                del active_users[user_id]
        # Release the semaphore to allow another job to start
        job_semaphore.release()

# === WORKER FUNCTION ===
def process_job(user_id, func, args, kwargs):
    # Acquire semaphore - this will block if 3 jobs are already running
    job_semaphore.acquire()
    
    # Start the job in a new thread
    threading.Thread(target=_run_job, args=(user_id, func, args, kwargs), daemon=True).start()

# === BACKGROUND WORKER THREAD ===
def video_job_worker():
    while True:
        try:
            # Get job from queue
            user_id, func, args, kwargs = video_job_queue.get(timeout=JOB_QUEUE_TIMEOUT)
            
            # Check user job limit
            with job_lock:
                if active_users[user_id] >= MAX_JOBS_PER_USER:
                    # Re-queue the job if user is at their limit
                    video_job_queue.put((user_id, func, args, kwargs))
                    time.sleep(0.5)
                    continue
                
                # Increment user job count
                active_users[user_id] += 1
            
            # Process the job (will wait if semaphore count is 0)
            threading.Thread(target=process_job, args=(user_id, func, args, kwargs), daemon=True).start()
            
        except queue.Empty:
            continue

# === PUBLIC API FOR QUEUEING ===
def enqueue_video_job(user_id, func, *args, **kwargs):
    video_job_queue.put((user_id, func, args, kwargs))
    logging.info(f"📥 Job enqueued for {user_id}")

# === START THREAD ===
def start_video_worker():
    threading.Thread(target=video_job_worker, daemon=True).start()
    logging.info(f"🔄 Video worker started with {CONCURRENT_WORKERS} parallel job limit")