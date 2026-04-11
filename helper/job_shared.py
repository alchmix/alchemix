# helper/job_shared.py
from multiprocessing import Manager, Semaphore

# ⚠️ Wrap everything in a function so it's not initialized at import time
def init_shared_state():
    manager = Manager()
    return {
        "job_queue": manager.Queue(),
        "job_semaphore": Semaphore(3),
        "active_jobs": manager.dict(),
        "active_users": manager.dict(),
        "heartbeat_dict": manager.dict()
    }
