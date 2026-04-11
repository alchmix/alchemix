import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ✅ Setup Selenium WebDriver with Human-like Configurations (Headless Mode)
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # ✅ Run in background (new method for stability)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent detection
    chrome_options.add_argument("--disable-gpu")  # Helps with headless mode stability
    chrome_options.add_argument("--window-size=1920,1080")  # Avoid viewport issues
    chrome_options.add_argument("--disable-extensions")  # No extensions
    chrome_options.add_argument("--disable-popup-blocking")  # No popups
    chrome_options.add_argument("--disable-notifications")  # No notifications
    chrome_options.add_argument("--no-sandbox")  # Bypass sandbox issues
    chrome_options.add_argument("--disable-dev-shm-usage")  # Fix shared memory issue
    chrome_options.add_argument("--incognito")  # Use incognito mode
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    # ✅ Automatically download and install the correct ChromeDriver
    service = Service(ChromeDriverManager().install())

    # ✅ Initialize the WebDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


# ✅ Random delays to mimic human behavior
def human_delay():
    time.sleep(random.uniform(1.5, 4.5))


# ✅ Function to fetch related YouTube videos
def fetch_related_videos(keywords, max_results=3):
    driver = setup_driver()
    video_results = []

    try:
        for keyword in keywords:
            try:
                print(f"\n🔍 Searching for videos: {keyword}")
                
                # Navigate to YouTube
                driver.get("https://www.youtube.com")
                human_delay()

                # Find the search box and enter the keyword
                search_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "search_query"))
                )
                search_box.clear()

                # Simulate typing the keyword (mimic human typing)
                for char in keyword:
                    search_box.send_keys(char)
                    time.sleep(random.uniform(0.1, 0.3))  # Random typing speed

                search_box.send_keys(Keys.RETURN)
                human_delay()

                # Wait for results to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "contents"))
                )
                human_delay()

                # ✅ Scroll to load more results (mimicking user behavior)
                for _ in range(2):  # Scroll down twice to load more videos
                    driver.execute_script("window.scrollBy(0, 1000);")
                    time.sleep(random.uniform(1.0, 2.0))

                # ✅ Extract video links
                video_elements = driver.find_elements(By.CSS_SELECTOR, "a#video-title")
                count = 0

                for element in video_elements:
                    if count >= max_results:
                        break

                    video_url = element.get_attribute("href")
                    if video_url and "watch?v=" in video_url:
                        video_results.append(video_url)
                        count += 1

                print(f"✅ Found {count} videos for '{keyword}'")

            except Exception as e:
                print(f"❌ Error processing keyword '{keyword}': {str(e)}")
            finally:
                # Add delay between searches
                human_delay()

    finally:
        # ✅ Close the browser after execution
        driver.quit()

    return video_results


# # ✅ Example Usage (Runs in Background)
# keywords = ["Elon Musk latest news", "AI in 2025"]
# max_results = 3
# videos = fetch_related_videos(keywords, max_results)

# print("\n🎥 Fetched Video Links:")
# for vid in videos:
#     print(vid)
