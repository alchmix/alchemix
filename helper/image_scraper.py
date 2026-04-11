# import requests
# import json
# from tqdm import tqdm
# from dotenv import load_dotenv
# import os

# # Load API credentials
# load_dotenv()

# SERPER_API = os.getenv("SERPER_API")
# def fetch_images_for_term(term, n, pbar):
#     """Fetch images for a single search term using Serper API."""
#     urls = []
#     try:
#         url = "https://google.serper.dev/images"
#         payload = json.dumps({
#             "q": term,
#             "num": 50  # No limit as requested
#         })
#         headers = {
#             'X-API-KEY':SERPER_API,
#             'Content-Type': 'application/json'
#         }
        
#         response = requests.request("POST", url, headers=headers, data=payload)
#         data = json.loads(response.text)
        
#         # Extract image URLs from response
#         images = data.get('images', [])
#         for image in images[:n]:
#             # Using 'link' as the URL source based on your data sample
#             url = image.get('imageUrl')
#             if url:
#                 urls.append(url)
#                 pbar.update(1)  # Update progress bar
    
#     except Exception as e:
#         print(f"Error fetching images for '{term}': {e}")
    
#     return urls

# # def fetch_related_images(search_terms, n=100):
# #     """
# #     Fetches 'n' images per search term and organizes them in a structured array.
    
# #     Args:
# #         search_terms (list): A list of character names or topics.
# #         n (int): Number of images per search term.
    
# #     Returns:
# #         list: A list of dictionaries, each containing a character and their image URLs.
# #     """
# #     # Ensure search_terms is a list
# #     if isinstance(search_terms, str):
# #         search_terms = [search_terms]
    
# #     # Total progress bar length
# #     total_tasks = len(search_terms) * n
# #     pbar = tqdm(total=total_tasks, desc="Fetching Images", unit="img")
    
# #     results = []
# #     for term in search_terms:
# #         image_urls = fetch_images_for_term(term, n, pbar)
# #         results.append({"character": term, "images": image_urls})
    
# #     pbar.close()
# #     return results


# def fetch_related_images(term_frequency_dict, images_per_occurrence=10, max_images_per_character=60):
#     """
#     Fetches images per character/concept based on frequency, capped at max_images_per_character.
    
#     Args:
#         term_frequency_dict (dict): {term: frequency}
#         images_per_occurrence (int): Number of images per mention
#         max_images_per_character (int): Hard limit of images per character
    
#     Returns:
#         list: [{"character": name, "images": [urls...]}]
#     """
#     if isinstance(term_frequency_dict, list):
#         term_frequency_dict = {term: 1 for term in term_frequency_dict}

#     total_tasks = sum(min(freq * images_per_occurrence, max_images_per_character)
#                       for freq in term_frequency_dict.values())
#     pbar = tqdm(total=total_tasks, desc="Fetching Images", unit="img")

#     results = []
#     for term, freq in term_frequency_dict.items():
#         total_images = min(freq * images_per_occurrence, max_images_per_character)
#         image_urls = fetch_images_for_term(term, total_images, pbar)
#         results.append({"character": term, "images": image_urls})

#     pbar.close()
#     return results



# # # Example Usage
# # search_terms = ["Elon Musk", "Tesla Gigafactory"]
# # image_data = fetch_related_images(search_terms)

# # print(image_data)  # Prints a structured list of characters and their related images
import requests
import json
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re

# Load API key from .env
load_dotenv()
SERPER_API = os.getenv("SERPER_API")

def fetch_images_for_term(term, total_needed, pbar, used_urls_global, max_pages=5):
    """
    Fetches unique images for a term across multiple Serper pages, avoiding duplicates.
    """
    urls = []
    seen_this_term = set()
    page = 1

    try:
        while len(urls) < total_needed and page <= max_pages:
            url = "https://google.serper.dev/images"
            payload = json.dumps({
                "q": term,
                "num": 50,
                "page": page
            })
            headers = {
                'X-API-KEY': SERPER_API,
                'Content-Type': 'application/json'
            }

            response = requests.post(url, headers=headers, data=payload)
            data = response.json()

            images = data.get("images", [])
            if not images:
                break  # No more results

            for image in images:
                image_url = image.get("imageUrl")
                if (
                    image_url and
                    re.match(r'^https?://', image_url) and
                    image_url not in seen_this_term and
                    image_url not in used_urls_global
                ):
                    urls.append(image_url)
                    seen_this_term.add(image_url)
                    used_urls_global.add(image_url)
                    pbar.update(1)
                    if len(urls) >= total_needed:
                        break

            page += 1

    except Exception as e:
        print(f"❌ Error fetching images for '{term}': {e}")

    return urls

def fetch_related_images(
    term_frequency_dict,
    niche,
    images_per_occurrence=10,
    max_images_per_character=60,
    used_urls_global=None
):
    """
    Fetch images per term based on frequency, enriched with HD/stylistic keywords.
    """
    if isinstance(term_frequency_dict, list):
        term_frequency_dict = {term: 1 for term in term_frequency_dict}

    if used_urls_global is None:
        used_urls_global = set()

    total_expected = sum(
        min(freq * images_per_occurrence, max_images_per_character)
        for freq in term_frequency_dict.values()
    )
    pbar = tqdm(total=total_expected, desc="Fetching Unique Images", unit="img")

    results = []

    for term, freq in term_frequency_dict.items():
        total_images = min(freq * images_per_occurrence, max_images_per_character)

        # Add enhancements based on niche
        if niche == "philosophy":
            enriched_term = f"{term} symbolic oil painting dramatic lighting site:pinterest.com OR site:artstation.com"
        else:
            enriched_term = f"{term} HD"

        image_urls = fetch_images_for_term(enriched_term, total_images, pbar, used_urls_global)
        results.append({
            "character": term,
            "images": image_urls
        })

    pbar.close()
    return results
