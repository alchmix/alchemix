import os
import time
import shutil
import logging
from datetime import datetime, timedelta
import schedule
from threading import Thread
from mongodb import mongo_instance  # Assuming you're using the same mongodb connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration
TEMP_VIDEO_DATA_STORAGE = "temp_video_data_storage"  # Path to temp storage folder
OUTPUTS_FOLDER = "outputs"  # Path to outputs folder
MAX_AGE_DAYS = 7  # Files older than this will be deleted

def cleanup_temp_files():
    """Delete files in temp_video_data_storage older than MAX_AGE_DAYS"""
    now = time.time()
    max_age_seconds = MAX_AGE_DAYS * 24 * 60 * 60  # Convert days to seconds
    
    if not os.path.exists(TEMP_VIDEO_DATA_STORAGE):
        logging.warning(f"Temp directory {TEMP_VIDEO_DATA_STORAGE} does not exist")
        return
    
    deleted_count = 0
    error_count = 0
    
    logging.info(f"Starting cleanup of files older than {MAX_AGE_DAYS} days in {TEMP_VIDEO_DATA_STORAGE}")
    
    for item in os.listdir(TEMP_VIDEO_DATA_STORAGE):
        item_path = os.path.join(TEMP_VIDEO_DATA_STORAGE, item)
        
        # Get the last modification time
        try:
            item_age = now - os.path.getmtime(item_path)
            
            # Check if the file/folder is older than MAX_AGE_DAYS
            if item_age > max_age_seconds:
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    deleted_count += 1
                    logging.info(f"Deleted: {item_path}")
                except Exception as e:
                    logging.error(f"Error deleting {item_path}: {str(e)}")
                    error_count += 1
        except Exception as e:
            logging.error(f"Error checking age of {item_path}: {str(e)}")
            error_count += 1
    
    logging.info(f"Temp files cleanup complete: {deleted_count} items deleted, {error_count} errors")

def cleanup_output_videos():
    """Delete video files in outputs folder older than MAX_AGE_DAYS"""
    now = time.time()
    max_age_seconds = MAX_AGE_DAYS * 24 * 60 * 60
    
    if not os.path.exists(OUTPUTS_FOLDER):
        logging.warning(f"Outputs directory {OUTPUTS_FOLDER} does not exist")
        return
    
    deleted_count = 0
    error_count = 0
    
    logging.info(f"Starting cleanup of videos older than {MAX_AGE_DAYS} days in {OUTPUTS_FOLDER}")
    
    for item in os.listdir(OUTPUTS_FOLDER):
        item_path = os.path.join(OUTPUTS_FOLDER, item)
        
        # Skip if not a file
        if not os.path.isfile(item_path):
            continue
            
        # Check if it's a video file (basic check for common video extensions)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        if not any(item.lower().endswith(ext) for ext in video_extensions):
            continue
        
        try:
            item_age = now - os.path.getmtime(item_path)
            
            if item_age > max_age_seconds:
                try:
                    os.remove(item_path)
                    deleted_count += 1
                    logging.info(f"Deleted: {item_path}")
                except Exception as e:
                    logging.error(f"Error deleting {item_path}: {str(e)}")
                    error_count += 1
        except Exception as e:
            logging.error(f"Error checking age of {item_path}: {str(e)}")
            error_count += 1
    
    logging.info(f"Output videos cleanup complete: {deleted_count} videos deleted, {error_count} errors")

def cleanup_completed_videos_db():
    """Delete documents in completed_videos collection older than MAX_AGE_DAYS"""
    try:
        db = mongo_instance.get_db()
        collection = db.completed_videos
        
        # Calculate the cutoff date (3 days ago)
        cutoff_date = datetime.now() - timedelta(days=MAX_AGE_DAYS)
        
        # Delete documents older than cutoff date
        # Assuming there's a 'created_at' or similar timestamp field
        # Adjust the field name as needed for your database schema
        result = collection.delete_many({"created_at": {"$lt": cutoff_date}})
        
        logging.info(f"Database cleanup complete: {result.deleted_count} documents deleted from completed_videos")
    except Exception as e:
        logging.error(f"Error cleaning up database: {str(e)}")

def run_cleanup():
    """Run all cleanup functions"""
    logging.info("Starting scheduled cleanup task")
    cleanup_temp_files()
    cleanup_output_videos()
    cleanup_completed_videos_db()
    logging.info("Scheduled cleanup task completed")

def start_cleanup_scheduler():
    """Start the cleanup scheduler in a background thread"""
    def run_scheduler():
        # Run once at startup
        run_cleanup()
        
        # Schedule to run daily at midnight
        schedule.every().day.at("00:00").do(run_cleanup)
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logging.error(f"Scheduler error: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying after an error
    
    # Start scheduler in a separate thread
    Thread(target=run_scheduler, daemon=True).start()
    logging.info("🧹 Cleanup scheduler started")

if __name__ == "__main__":
    logging.info("Starting cleanup service")
    start_cleanup_scheduler()  # 🔁 This runs cleanup now + schedules daily tasks

    # Keep the script alive
    while True:
        time.sleep(60)

    
    # Uncomment to run as a continuous service with daily scheduling
    # start_cleanup_scheduler()