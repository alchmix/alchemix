import os
import openai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def detect_emotion(text):
    """
    Detects the **strongest primary emotion** conveyed by a given news title.
    
    Returns:
        str: A **single-word** response representing the **best-fit** emotion.
    
    Possible Emotions: "shock", "drama", "humor", "anger", "excitement", "awkwardness", "sadness", etc.
    """

    # Refined AI prompt for **maximum accuracy**
    emotion_analysis_prompt = f"""
    Give the **dominant emotion** conveyed by the following news title for youtube engagement:
    
    **"{text}"**
    
    **Rules:**
    - Provide ONLY **one word** as the response.
    - Choose from strong emotions like: "shock", "drama", "humor", "anger", "excitement", "awkwardness", "sadness".
    - If uncertain, return the closest fitting **single emotion**.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing emotions from text. Your responses must be precise and limited to ONE word only."},
            {"role": "user", "content": emotion_analysis_prompt}
        ]
    )

    # Extract and clean the response
    detected_emotion = response.choices[0].message.content.strip().lower()

    # Ensure only **one-word responses**
    detected_emotion = detected_emotion.split()[0] if " " in detected_emotion else detected_emotion

    return detected_emotion

# result=detect_emotion('The man was very very angry')

# print(result)


