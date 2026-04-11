import requests
from bs4 import BeautifulSoup
import re
import json
import time
import random

# Configure browser-like headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.youtube.com/',
    'DNT': '1'
}

# Create a session to persist cookies
session = requests.Session()

# Configure random delays to mimic human behavior
def human_delay():
    time.sleep(random.uniform(1.5, 4.5))

# Function to fetch related videos
def fetch_related_videos(keywords, max_results=3):
    video_results = []

    for keyword in keywords:
        try:
            # Construct the search URL
            search_query = "+".join(keyword.split())
            url = f"https://www.youtube.com/results?search_query={search_query}"
            
            # Send request with headers
            response = session.get(url, headers=HEADERS)
            response.raise_for_status()
            
            # Add random delay before parsing
            human_delay()

            # Parse content
            soup = BeautifulSoup(response.text, 'lxml')

            # Find script tag safely
            script = soup.find("script", text=re.compile('var ytInitialData ='))
            if not script:
                print(f"No initialization data found for keyword: {keyword}")
                continue

            # Add random processing delay
            human_delay()

            # Extract JSON with better error handling
            json_text = re.search(r'var ytInitialData\s*=\s*({.*?})\s*;', script.string, re.DOTALL)
            if not json_text:
                print(f"JSON data pattern not found for keyword: {keyword}")
                continue
                
            json_data = json.loads(json_text.group(1))

            # Navigate JSON with safety checks
            def safe_get(data, *keys):
                for key in keys:
                    try:
                        data = data[key]
                    except (KeyError, TypeError):
                        return None
                return data

            contents = safe_get(json_data, "contents", "twoColumnSearchResultsRenderer", 
                               "primaryContents", "sectionListRenderer", "contents")
            
            if not contents:
                print(f"No video contents found for keyword: {keyword}")
                continue

            # Collect video URLs
            count = 0
            for content in contents:
                items = safe_get(content, "itemSectionRenderer", "contents")
                if not items:
                    continue
                    
                for item in items:
                    video = safe_get(item, "videoRenderer")
                    if not video:
                        continue
                        
                    video_id = safe_get(video, "videoId")
                    if video_id:
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        video_results.append(video_url)
                        count += 1
                        if count >= max_results:
                            break
                
                if count >= max_results:
                    break

        except Exception as e:
            print(f"Error processing keyword '{keyword}': {str(e)}")
        
        # Add delay between different keyword searches
        human_delay()

    return video_results

# # Example usage
# keywords = [""]
# max_results = 3
# videos = fetch_related_videos(keywords, max_results)
# print(videos)