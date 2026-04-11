import os
import time
import logging

ACTIVE_COOKIE_FILE = "cookies/imagefxcookies.txt"
FORCE_ROTATE_FILE = "cookies/force_rotate.signal"


def read_active_cookie() -> str | None:
    """
    Workers call this to get the current active cookie.
    Just reads a file. No timer, no threads, no imports.
    """
    try:
        if not os.path.exists(ACTIVE_COOKIE_FILE):
            logging.error("❌ imagefxcookies.txt not found. Is the daemon running?")
            return None

        with open(ACTIVE_COOKIE_FILE, "r", encoding="utf-8") as f:
            cookie = f.read().strip()

        if not cookie:
            logging.error("❌ imagefxcookies.txt is empty")
            return None

        return cookie

    except Exception as e:
        logging.error(f"❌ Failed to read active cookie: {e}")
        return None


def signal_force_rotate(timeout: float = 15.0, poll_interval: float = 0.5) -> bool:
    """
    Worker calls this on 401.
    Creates a signal file, then blocks until the daemon writes a new cookie
    or the timeout is reached.
    Returns True if a new cookie was detected, False on timeout.
    """
    old_cookie = read_active_cookie()

    try:
        os.makedirs("cookies", exist_ok=True)
        with open(FORCE_ROTATE_FILE, "w") as f:
            f.write("1")
        logging.warning("⚡ Force rotate signal sent to daemon")
    except Exception as e:
        logging.error(f"❌ Failed to write force rotate signal: {e}")
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        new_cookie = read_active_cookie()
        if new_cookie and new_cookie != old_cookie:
            logging.info("✅ Cookie rotated successfully by daemon.")
            return True

    logging.error(f"❌ Daemon did not rotate cookie within {timeout}s.")
    return False