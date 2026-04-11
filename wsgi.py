from threading import Thread
import multiprocessing
from waitress import serve
from app import app  # Your Flask app
from video_job_controller import enqueue_video_job, start_video_worker
from news_scrape import start_news_fetcher, check_for_updates

def start_background_services():
    # Optional: Start background news fetcher in a thread
    # news_thread = Thread(target=start_news_fetcher)
    # news_thread.daemon = True
    # news_thread.start()

    # Start multiprocessing-safe video worker
    multiprocessing.freeze_support()
    start_video_worker()

if __name__ == "__main__":
    start_background_services()

    # 🛠️ Critical fix: bind to 0.0.0.0 to allow external access (not just localhost)
    serve(
    app,
    host="0.0.0.0",
    port=5000,
    threads=32,
    _quiet=False,
    max_request_body_size=1073741824  # 1 GB
)

