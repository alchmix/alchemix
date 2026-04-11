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

def fetch_videos_for_term(keyword, max_results, pbar):
    """
    Fetch related YouTube video titles and URLs for a given keyword.
    """
    videos = []
    
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
            return videos

        # Extract JSON with better error handling
        json_text = re.search(r'var ytInitialData\s*=\s*({.*?})\s*;', script.string, re.DOTALL)
        if not json_text:
            print(f"JSON data pattern not found for keyword: {keyword}")
            return videos

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
            return videos

        # Collect video titles and URLs
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
                title = safe_get(video, "title", "runs", 0, "text")

                if video_id and title:
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    videos.append({"title": title, "url": video_url})
                    count += 1
                    pbar.update(1)  # Update progress bar
                    if count >= max_results:
                        break

            if count >= max_results:
                break

    except Exception as e:
        print(f"Error processing keyword '{keyword}': {str(e)}")

    # Add delay between different keyword searches
    human_delay()
    return videos


# def fetch_related_videos(keywords, max_results=3):
#     """
#     Fetches 'max_results' YouTube video titles and links per search term.

#     Args:
#         keywords (list): A list of search terms.
#         max_results (int): Number of videos per search term.

#     Returns:
#         list: A list of dictionaries, each containing a keyword and its related video titles and URLs.
#     """
#     from tqdm import tqdm  # Import tqdm here to avoid issues in headless environments

#     # Ensure keywords is a list
#     if isinstance(keywords, str):
#         keywords = [keywords]

#     # Total progress bar length
#     total_tasks = len(keywords) * max_results
#     pbar = tqdm(total=total_tasks, desc="Fetching Videos", unit="vid")

#     results = []
#     for keyword in keywords:
#         videos = fetch_videos_for_term(keyword, max_results, pbar)
#         results.append({"character": keyword, "videos": videos})

#     pbar.close()
#     return results


def fetch_related_videos(character_keywords, max_results=3):
    """
    Fetches 'max_results' YouTube video titles and links per search term, grouped by character.

    Args:
        character_keywords (list of tuples): A list where each element is (character, keyword).
        max_results (int): Number of videos per search term.

    Returns:
        list: A list of dictionaries where each contains "Character" and their related "videos".
    """
    from tqdm import tqdm  # Import tqdm here to avoid issues in headless environments

    # Dictionary to temporarily store grouped results
    temp_character_video_data = {}

    # Get the total progress bar length
    total_tasks = len(character_keywords) * max_results
    pbar = tqdm(total=total_tasks, desc="Fetching Videos", unit="vid")

    for character, keyword in character_keywords:
        videos = fetch_videos_for_term(keyword, max_results, pbar)  # Fetch videos for this keyword

        if character not in temp_character_video_data:
            temp_character_video_data[character] = []  # Initialize list if not present

        # Append video title and URL as an object
        temp_character_video_data[character].extend([
            {"title": video["title"], "url": video["url"]} for video in videos
        ])

    pbar.close()

    # Convert dictionary to list of dictionaries with explicit "Character" key
    character_video_data = [{"Character": char, "videos": vids} for char, vids in temp_character_video_data.items()]

    return character_video_data




# # Example Usage
# search_terms ={
#     "President Trump": [
#         '"President Trump Speech"',
#         '"Donald Trump latest interview"',
#         '"Trump news today"'
#     ],
#     "Elon Musk": [
#         '"Elon Musk Tesla announcement"',
#         '"Elon Musk Twitter news"',
#         '"SpaceX new launch"'
#     ]
# }
# character_keyword_list = [(character, keyword.strip('"')) for character, keywords in search_terms.items() for keyword in keywords]
# video_data = fetch_related_videos(character_keyword_list, max_results=3)
# print(video_data)  # Structured output
