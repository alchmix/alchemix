import os
import concurrent.futures
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple
from backend.tests import download_image,process_face_embeddings,setup_face_recognition_cache
MAX_WORKERS = 5  # Adjust this based on system capabilities

def process_reference_image(img_url: str, temp_dir: str, face_cache: Dict) -> Tuple[str, List]:
    """
    Downloads an image, extracts face embeddings, and returns the results.

    Args:
        img_url (str): URL of the image to process.
        temp_dir (str): Directory to temporarily store the image.
        face_cache (Dict): Cache for previously processed face embeddings.

    Returns:
        Tuple[str, List]: File path of the saved image and its face embeddings.
    """
    img_path = os.path.join(temp_dir, f"ref_{len(face_cache)}.jpg")
    
    if download_image(img_url, img_path):  # Assuming download_image() exists
        return img_path, process_face_embeddings(img_path, face_cache)  # Assuming process_face_embeddings() exists
    
    return None, []

def process_reference_images(image_urls: List[str], face_cache: Dict) -> Dict[str, List]:
    """
    Processes multiple reference images concurrently.

    Args:
        image_urls (List[str]): List of image URLs to process.
        face_cache (Dict): Cache for storing face embeddings.

    Returns:
        Dict[str, List]: Dictionary mapping image paths to their face embeddings.
    """
    reference_faces = {}

    with TemporaryDirectory() as temp_dir:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(process_reference_image, url, temp_dir, face_cache): url for url in image_urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                img_path, encodings = future.result()
                if img_path and encodings:
                    reference_faces[img_path] = encodings

    return reference_faces
image_urls = [
            "https://live.staticflickr.com/2396/2071470089_c2ef9708a7_o.jpg",
            "https://live.staticflickr.com/86/233344084_497d8dcf49_o.jpg",
            "https://live.staticflickr.com/4009/4174971224_f99cb78787_o.jpg",
            "https://live.staticflickr.com/93/254826594_92ab57a73b_o.jpg",
            "https://live.staticflickr.com/83/254824777_118addd4f3_o.jpg",
            "https://live.staticflickr.com/4076/4922094906_2457775618_o.jpg",
            "https://live.staticflickr.com/4142/4923003130_851551ba3d_o.jpg",
            "https://live.staticflickr.com/8530/8675141209_cabf762d47_o.jpg",
            "https://live.staticflickr.com/5456/8864144937_abd6c42430_o.jpg",
            "https://farm9.staticflickr.com/8118/8602828910_651164e654_b.jpg",
            "https://live.staticflickr.com/65535/52164477528_87c236d4d8_o.jpg",
            "https://live.staticflickr.com/65535/52164477488_e7ef91b530_o.jpg",
            "https://live.staticflickr.com/2421/4093711193_b1b1be2490_o.jpg",
            "https://live.staticflickr.com/3254/2971063270_45e60361b1_o.jpg",
            "https://live.staticflickr.com/8013/7496767378_4883e7acff_o.jpg",
            "https://live.staticflickr.com/8430/7722871112_213d58924a_o.jpg",
            "https://live.staticflickr.com/8285/7722866286_dbf94e33a5_o.jpg",
            "https://live.staticflickr.com/8429/7722868136_def002be22_o.jpg",
            "https://live.staticflickr.com/7124/7722870126_b3a768f6da_o.jpg",
            "https://live.staticflickr.com/8289/7722869156_d58fe251e1_o.jpg"
        ]
face_cache = setup_face_recognition_cache()  # Load cache
reference_faces = process_reference_images(image_urls, face_cache)

print(reference_faces) 