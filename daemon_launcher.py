def ensure_cookie_daemon_running():
    # All imports inside function — never runs at module level in workers
    import os
    import sys
    import time
    import logging
    import subprocess

    pid_file = "cookies/cookie_daemon.pid"
    active_cookie_file = "cookies/imagefxcookies.txt"

    # Check if already running via PID file
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            try:
                pid = int(f.read().strip())
            except ValueError:
                logging.warning("Invalid PID file. Restarting daemon.")
                os.remove(pid_file)
                pid = None

        if pid:
            try:
                os.kill(pid, 0)  # just checks if process exists, no signal sent
                logging.info(f"✅ Cookie daemon already running (PID={pid})")
                return
            except OSError:
                logging.warning("Stale PID file found. Restarting daemon.")
                os.remove(pid_file)

    # Start daemon as completely separate process
    subprocess.Popen(
        [sys.executable, "cookie_timers.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True
    )
    logging.info("🚀 Cookie daemon process started")

    # Wait for first cookie to be written before continuing
    for _ in range(10):
        time.sleep(1)
        if os.path.exists(active_cookie_file):
            logging.info("✅ Cookie daemon ready — imagefxcookies.txt found")
            return

    logging.error("❌ Cookie daemon failed to start within 10 seconds")