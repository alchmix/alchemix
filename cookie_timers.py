import json
import time
import os
import logging
import sys

COOKIE_SOURCE_FILE = "cookies/imagefx_cookies.json"
ACTIVE_COOKIE_FILE = "cookies/imagefxcookies.txt"
FORCE_ROTATE_FILE = "cookies/force_rotate.signal"
ROTATION_INTERVAL = 7 * 60
PID_FILE = "cookies/cookie_daemon.pid"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CookieDaemon] %(message)s",
    handlers=[
        logging.FileHandler("cookies/cookie_daemon.log"),
        logging.StreamHandler()
    ]
)

def load_cookies() -> list:
    try:
        with open(COOKIE_SOURCE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cookies = data.get("cookies", [])
        logging.info(f"Loaded {len(cookies)} cookies from source file")
        return cookies
    except Exception as e:
        logging.error(f"Failed to load cookies: {e}")
        return []

def write_active_cookie(cookie: str):
    """Write atomically so workers never read a partial file."""
    os.makedirs("cookies", exist_ok=True)
    tmp_path = ACTIVE_COOKIE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(cookie)
    os.replace(tmp_path, ACTIVE_COOKIE_FILE)
    logging.info(f"Active cookie written (length={len(cookie)})")

def write_pid():
    os.makedirs("cookies", exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

def clear_pid():
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

def wait_or_force_rotate(interval: int) -> bool:
    """
    Waits for interval seconds but checks every 2 seconds
    for a force rotate signal file.
    Returns True if force rotated, False if normal timeout.
    """
    elapsed = 0
    while elapsed < interval:
        time.sleep(2)
        elapsed += 2
        if os.path.exists(FORCE_ROTATE_FILE):
            os.remove(FORCE_ROTATE_FILE)
            logging.warning("⚡ Force rotate signal received")
            return True
    return False

def run_daemon():
    os.makedirs("cookies", exist_ok=True)
    write_pid()
    logging.info(f"Daemon started. PID={os.getpid()}")

    index = 0

    try:
        while True:
            # Reload cookies every cycle in case file was updated
            cookies = load_cookies()
            if not cookies:
                logging.error("No cookies available. Retrying in 30s...")
                time.sleep(30)
                continue

            index = index % len(cookies)
            active = cookies[index]
            write_active_cookie(active)
            logging.info(f"Rotated to cookie index {index}/{len(cookies)}")

            # Wait for next rotation or force rotate signal
            force_rotated = wait_or_force_rotate(ROTATION_INTERVAL)
            if force_rotated:
                logging.info("Force rotating to next cookie...")

            index += 1

    except KeyboardInterrupt:
        logging.info("Daemon shutting down.")
    finally:
        clear_pid()

if __name__ == "__main__":
    run_daemon()