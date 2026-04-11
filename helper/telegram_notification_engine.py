import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Your Telegram bot token
BOT_TOKEN = os.getenv('BOT_TOKEN')
# Your group chat ID (obtained from getUpdates)
CHAT_ID = os.getenv('CHAT_ID')

def send_telegram_notification(message_text, button_text, button_url,userEmail):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    payload = {
        'chat_id': CHAT_ID,
        'text':f"{userEmail}: {message_text}",
        'reply_markup': {
            'inline_keyboard': [[{
                'text': button_text,
                'url': button_url
            }]]
        }
    }
    
    response = requests.post(url, json=payload)  # Note: using json=payload instead of data=payload
    
    if response.status_code != 200:
        print(f"Error sending message: {response.status_code}")
        print(response.text)
    else:
        print("Message sent successfully")
    
    return response.json()


def send_telegram_error(message_text,userEmail):
     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
     payload = {
        'chat_id': CHAT_ID,
        'text':f"{message_text} for {userEmail}",
    }
    
     response = requests.post(url, json=payload)  # Note: using json=payload instead of data=payload
    
     if response.status_code != 200:
        print(f"Error sending message: {response.status_code}")
        print(response.text)
     else:
        print("Message sent successfully")
    
     return response.json()
    

# # Message info
# message_text = "Your composition is Done"
# button_text = "View Composition"
# button_url = "https://animista.net/"
# userEmail='techwebdev3@gmail.com'

# # Send the message with button
# result = send_telegram_notification(message_text, button_text, button_url,userEmail)
# print(result)