import requests
from bs4 import BeautifulSoup
import feedparser
from newspaper import Article
from datetime import datetime, timedelta
import time
import schedule
from threading import Thread
import logging
import re
import os
from mongodb import mongo_instance
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Telegram configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")  # Replace with your actual bot token
CHANNEL_ID =os.getenv("CHANNEL_ID")  # Replace with your channel ID
CAPTION_LIMIT = 500  # Telegram caption limit
FRONTEND_URL=os.getenv("FRONTEND_URL")

# Initialize bot


# 🔹 Focused News Sources (US Celebrity & Politics)
SOURCES = {
    "TMZ": {
        "rss": "https://www.tmz.com/rss.xml",
        "parser": "tmz",
        "default_category": "celebrity"
    },
    "E! News": {
        "rss": "https://www.eonline.com/syndication/feeds/rssfeeds/topstories.xml",
        "parser": "enews",
        "default_category": "celebrity"
    },
    "Fox News Politics": {
        "rss": "https://moxie.foxnews.com/google-publisher/politics.xml",
        "parser": "foxnews",
        "default_category": "politics"
    },
    "New York Times Politics": {
        "rss": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
        "parser": "nyt_politics",
        "default_category": "politics"
    }
}

# Keywords for better category detection
CATEGORY_KEYWORDS = {
    "celebrity": [
        "actor", "actress", "singer", "rapper", "star", "celebrity", "celeb", 
        "hollywood", "entertainment", "movie star", "tv star", "famous", 
        "red carpet", "award show", "Grammy", "Oscar", "Emmy", "Golden Globe",
        "dating", "relationship", "married", "divorce", "breakup"
    ],
    "politics": [
        "president", "senator", "congress", "representative", "election", 
        "campaign", "vote", "voter", "ballot", "democrat", "republican", 
        "policy", "legislation", "bill", "law", "government", "administration",
        "supreme court", "justice", "constitution", "house", "senate"
    ]
}

def load_processed_urls():
    """Load processed URLs from database for duplicate checking"""
    db = mongo_instance.get_db()
    return set(article["url"] for article in db.news.find({}, {"url": 1}))

def clean_old_articles():
    """Remove articles older than 2 days from the database"""
    db = mongo_instance.get_db()
    seven_days_ago = datetime.now() - timedelta(days=7)
    result = db.news.delete_many({"created_at": {"$lt": seven_days_ago}})
    logging.info(f"🧹 Removed {result.deleted_count} articles older than 2 days")

def save_new_articles(articles):
    """Save new articles to database with proper date handling"""
    if not articles:
        return
    
    db = mongo_instance.get_db()
    
    # Convert publish_date strings to datetime objects
    for article in articles:
        if isinstance(article["publish_date"], str):
            try:
                article["publish_date"] = datetime.fromisoformat(article["publish_date"])
            except:
                # If unable to parse the date string, use current time
                article["publish_date"] = datetime.now()
    
    # Insert only new articles
    try:
        result = db.news.insert_many(articles, ordered=False)
        logging.info(f"💾 Saved {len(result.inserted_ids)} new articles")
        
        # Print category distribution
        categories = {}
        for article in articles:
            cat = article.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            logging.info(f"  - {cat.capitalize()}: {count} articles")
            
        # Return new articles for notification
        return articles
    except Exception as e:
        logging.error(f"⚠️ Database insert error: {str(e)}")
        return []

def determine_category(article_data, source_name):
    """
    Determine whether an article is about celebrity news or politics
    Uses source default category, title, and content analysis
    """
    default_category = SOURCES[source_name]["default_category"]
    
    # Combine title and excerpt for analysis
    text_to_analyze = (article_data.get("title", "") + " " + 
                      article_data.get("excerpt", "")).lower()
    
    # Count keyword matches
    category_scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, text_to_analyze)
            score += len(matches)
        category_scores[category] = score
    
    # Determine category based on keyword scores
    if category_scores["celebrity"] > category_scores["politics"]:
        return "celebrity"
    elif category_scores["politics"] > category_scores["celebrity"]:
        return "politics"
    else:
        # If tied or no matches, use source default
        return default_category

def fetch_full_article(url, source_name):
    """Improved article fetching with better error handling and categorization"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Extract main image if available
        image_url = None
        if article.top_image:
            image_url = article.top_image
        
        article_data = {
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": article.publish_date or datetime.now(),
            "source": source_name,
            "url": url,
            "excerpt": article.text[:500] + "..." if len(article.text) > 500 else article.text,
            "created_at": datetime.now(),
            "composed":False,
            "image_url": image_url
        }
        
        # Determine and add category
        article_data["category"] = determine_category(article_data, source_name)
        
        # Format the text as paragraphs for better display
        paragraphs = [p.strip() for p in article.text.split('\n') if p.strip()]
        article_data["paragraphs"] = paragraphs
        
        return article_data
    except Exception as e:
        logging.error(f"⚠️ Error fetching {url}: {str(e)}")
        return None

def check_for_updates():
    """Main update checker with improved logging"""
    logging.info(f"⏰ Update check at {datetime.now().isoformat()}")
    processed_urls = load_processed_urls()
    new_articles = []
    
    # Clean old articles first
    clean_old_articles()
    
    for source_name, config in SOURCES.items():
        try:
            feed = feedparser.parse(config["rss"])
            logging.info(f"🔎 Checking {source_name} ({len(feed.entries)} entries)")
            
            for entry in feed.entries:
                if entry.link not in processed_urls:
                    # Check if the entry is recent (within the last 2 days)
                    entry_date = None
                    if hasattr(entry, 'published_parsed'):
                        entry_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed'):
                        entry_date = datetime(*entry.updated_parsed[:6])
                    
                    # Skip if the entry is older than 2 days
                    if entry_date and (datetime.now() - entry_date).days > 2:
                        continue
                        
                    article = fetch_full_article(entry.link, source_name)
                    if article:
                        new_articles.append(article)
                        processed_urls.add(entry.link)
            
        except Exception as e:
            logging.error(f"🚨 Error processing {source_name}: {str(e)}")
    
    # Save articles and get the saved ones for notifications
    saved_articles = save_new_articles(new_articles)
    
    # Send notifications for new articles
    if saved_articles:
        send_news_notifications(saved_articles)

def send_news_notifications(articles):
    """Send notifications for new articles to Telegram channel using HTTP requests"""
    for article in articles:
        try:
            # Create message content
            message = f"*{article['title']}*\n"
            message += f"_{article['source']} | {article['publish_date'].strftime('%Y-%m-%d %H:%M')}_\n\n"
            
            # Add a short excerpt instead of full text
            message += article['excerpt']
            # Get the database ID of the article
            news_db_id = str(article['_id'])  # MongoDB ObjectId needs to be converted to string
            
            # Add source link as button
            button_text = "Compose Article"
            button_url = f"{FRONTEND_URL}/news-script/{news_db_id}"
            
            # Check if the article has an image
            if article.get('image_url'):
                # Send photo with caption
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
                payload = {
                    'chat_id': CHANNEL_ID,
                    'photo': article['image_url'],
                    'caption': message[:CAPTION_LIMIT],
                    'parse_mode': 'Markdown',
                    'reply_markup': {
                        'inline_keyboard': [[{
                            'text': button_text,
                            'url': button_url
                        }]]
                    }
                }
            else:
                # Send text message
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                payload = {
                    'chat_id': CHANNEL_ID,
                    'text': message,
                    'parse_mode': 'Markdown',
                    'reply_markup': {
                        'inline_keyboard': [[{
                            'text': button_text,
                            'url': button_url
                        }]]
                    }
                }
            
            # Send the request
            response = requests.post(url, json=payload)
            
            if response.status_code != 200:
                logging.error(f"Error sending notification: {response.status_code}")
                logging.error(response.text)
            else:
                logging.info(f"Notification sent successfully for article: {article['title']}")
                
            # Add a small delay to avoid flooding
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error sending notification for article '{article['title']}': {str(e)}")
def get_articles_by_category(category, limit=20):
    """Retrieve articles by category"""
    db = mongo_instance.get_db()
    return list(db.news.find({"category": category}).sort("publish_date", -1).limit(limit))

def start_news_fetcher():
    """Background scheduler with error recovery"""
    def run_scheduler():
        # Initial run
        check_for_updates()
        
        # Schedule periodic updates - every 60 minutes to avoid bombarding servers
        schedule.every(300).minutes.do(check_for_updates)
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logging.error(f"🔴 Scheduler error: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying
    
    # Start scheduler in a separate thread
    Thread(target=run_scheduler, daemon=True).start()
    logging.info("🚀 News fetcher started")

# if __name__ == "__main__":
#     logging.info("📰 Starting News Scraper Service")
#     start_news_fetcher()
    
#     # Keep the main thread alive
#     try:
#         while True:
#             time.sleep(7200)
#     except KeyboardInterrupt:
#         logging.info("Service stopped by user")