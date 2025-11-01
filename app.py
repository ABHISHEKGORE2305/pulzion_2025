import os
import re
from collections import Counter
import nltk
from flask import Flask, jsonify, request
from googleapiclient.discovery import build
from dotenv import load_dotenv
from flask_cors import CORS

# --- 1. Setup and Config ---

# Load environment variables (our API key) from the .env file
load_dotenv()
API_KEY = "AIzaSyBe_JsNMzh94KhA_MUGMfFgs2VfWGAfu8g"

# Initialize the NLTK library for keyword analysis
# We only need to do this once
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
# --- 2. API Service ---

# Create the YouTube API service object
# This is what we'll use to make our calls
youtube_service = build('youtube', 'v3', developerKey=API_KEY)


# --- 3. Test Route ---

# This is a simple route to make sure our server is running
@app.route('/')
def home():
    return "Hello, Hackathon! Our server is running."

# --- 3. Helper Functions (The Analysis Core) ---

def analyze_categories(video_items):
    """Counts the occurrences of each video category ID."""
    category_counter = Counter()
    for item in video_items:
        # 'categoryId' is in the 'snippet' part of the video item
        category_id = item['snippet'].get('categoryId')
        if category_id:
            category_counter[category_id] += 1
    
    # We'll get category *names* in a later step
    # For now, just return the counts by ID
    return category_counter

def analyze_keywords(video_items):
    """Extracts and counts common keywords from video titles, filtering stopwords."""
    all_titles = " ".join([item['snippet']['title'] for item in video_items])
    
    # Clean the text: keep only letters and spaces, and make lowercase
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', all_titles).lower()
    
    # Split into words and filter out stopwords
    words = [word for word in cleaned_text.split() if word not in STOP_WORDS and len(word) > 2]
    
    # Return the 15 most common keywords
    return Counter(words).most_common(15)

def analyze_upload_vs_popularity(video_items):
    """Analyzes time since upload vs popularity for scatter/bubble chart visualization."""
    from datetime import datetime, timezone
    
    data_points = []
    
    for item in video_items:
        snippet = item.get('snippet', {})
        statistics = item.get('statistics', {})
        
        published_at = snippet.get('publishedAt')
        views_str = statistics.get('viewCount')
        
        # Calculate engagement rate
        likes_str = statistics.get('likeCount')
        comments_str = statistics.get('commentCount')
        
        try:
            views = int(views_str) if views_str is not None else 0
        except ValueError:
            views = 0
        
        try:
            likes = int(likes_str) if likes_str is not None else 0
        except ValueError:
            likes = 0
        
        try:
            comments = int(comments_str) if comments_str is not None else 0
        except ValueError:
            comments = 0
        
        engagement_rate = round(((likes + comments) / views) * 100, 2) if views > 0 else 0.0
        
        # Calculate days since upload
        if published_at:
            try:
                # Handle different date formats from YouTube API
                if published_at.endswith('Z'):
                    upload_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                else:
                    upload_date = datetime.strptime(published_at.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                
                days_since_upload = (datetime.now(timezone.utc) - upload_date.replace(tzinfo=timezone.utc)).days
                
                if days_since_upload is not None and days_since_upload >= 0:
                    video_title = snippet.get('title', 'Unknown Video')
                    video_id = item.get('id', '')
                    data_points.append({
                        "x": days_since_upload,
                        "y": views,
                        "engagement_rate": engagement_rate,  # Store for later normalization
                        "title": video_title,
                        "video_id": video_id
                    })
            except Exception as e:
                # Skip videos with invalid date formats
                continue
    
    # Set uniform bubble size for all data points
    for point in data_points:
        point["r"] = 8  # Uniform size for all bubbles (8 pixels radius)
    
    return data_points

# --- 4. Main API Endpoint ---

@app.route('/get_trending_data')
def get_trending_data():
    """
    Fetches trending videos for a given country and returns analyzed data.
    """
    # Get the 'country' code from the request (e.g., /get_trending_data?country=US)
    country_code = request.args.get('country', 'US') # Default to 'US'
    keyword = request.args.get('keyword', '').strip().lower() # Get keyword filter
    
    try:
        # --- API Integration (30%) ---
        # This is the actual call to the YouTube API
        # Fetch top 25 for main display
        api_request = youtube_service.videos().list(
            part="snippet,statistics", # Request video details and view counts
            chart="mostPopular",      # Get the "trending" chart
            regionCode=country_code,  # Set the country
            maxResults=25             # Get the top 25 videos
        )
        api_response = api_request.execute()
        
        video_items = api_response.get("items", [])
        
        # If keyword is provided, fetch more videos (top 100) for "Also trending" section
        extended_video_items = []
        if keyword:
            extended_request = youtube_service.videos().list(
                part="snippet,statistics",
                chart="mostPopular",
                regionCode=country_code,
                maxResults=100  # Get top 100 for extended search
            )
            extended_response = extended_request.execute()
            extended_video_items = extended_response.get("items", [])
        
        # --- Data Analysis (50%) ---
        category_analysis = analyze_categories(video_items)
        keyword_analysis = analyze_keywords(video_items)
        upload_vs_popularity = analyze_upload_vs_popularity(video_items)
        
        # We also need a simple list of videos for the dashboard
        video_dashboard_list = []
        for item in video_items:
            # Extract statistics safely
            views_str = item['statistics'].get('viewCount')
            likes_str = item['statistics'].get('likeCount')
            comments_str = item['statistics'].get('commentCount')

            try:
                views = int(views_str) if views_str is not None else 0
            except ValueError:
                views = 0

            try:
                likes = int(likes_str) if likes_str is not None else 0
            except ValueError:
                likes = 0

            try:
                comments = int(comments_str) if comments_str is not None else 0
            except ValueError:
                comments = 0

            engagement_rate = round(((likes + comments) / views) * 100, 2) if views > 0 else 0.0

            # Get category ID from snippet (ensure it's a string for consistency)
            category_id = str(item['snippet'].get('categoryId', ''))
            
            # Get higher quality thumbnail (try maxres, then high, fallback to default)
            thumbnails = item['snippet'].get('thumbnails', {})
            thumbnail_url = thumbnails.get('maxres', {}).get('url') or \
                           thumbnails.get('high', {}).get('url') or \
                           thumbnails.get('medium', {}).get('url') or \
                           thumbnails.get('default', {}).get('url', '')

            # Get description from snippet
            description = item['snippet'].get('description', '')
            
            video_dashboard_list.append({
                "video_id": item['id'],
                "title": item['snippet']['title'],
                "thumbnail": thumbnail_url,
                "views": views,
                "likes": likes,
                "comment_count": comments,
                "like_count": likes,
                "engagement_rate": engagement_rate,
                "category_id": category_id,
                "description": description
            })

        # Helper function to check if video contains keyword
        def video_contains_keyword(item, keyword):
            if not keyword:
                return True
            title = item['snippet'].get('title', '').lower()
            description = item['snippet'].get('description', '').lower()
            return keyword in title or keyword in description
        
        # Filter main videos by keyword if provided
        if keyword:
            video_dashboard_list = [v for v in video_dashboard_list 
                                  if keyword in v['title'].lower()]
        
        # Process extended videos for "Also trending" section
        also_trending_list = []
        if keyword and extended_video_items:
            for item in extended_video_items:
                # Skip videos already in main list
                item_id = item['id']
                if any(v['video_id'] == item_id for v in video_dashboard_list):
                    continue
                
                # Check if video contains keyword
                if not video_contains_keyword(item, keyword):
                    continue
                
                # Extract statistics
                views_str = item['statistics'].get('viewCount')
                likes_str = item['statistics'].get('likeCount')
                comments_str = item['statistics'].get('commentCount')

                try:
                    views = int(views_str) if views_str is not None else 0
                except ValueError:
                    views = 0

                try:
                    likes = int(likes_str) if likes_str is not None else 0
                except ValueError:
                    likes = 0

                try:
                    comments = int(comments_str) if comments_str is not None else 0
                except ValueError:
                    comments = 0

                engagement_rate = round(((likes + comments) / views) * 100, 2) if views > 0 else 0.0

                # Get category ID (ensure it's a string for consistency)
                category_id = str(item['snippet'].get('categoryId', ''))
                
                # Get thumbnail
                thumbnails = item['snippet'].get('thumbnails', {})
                thumbnail_url = thumbnails.get('maxres', {}).get('url') or \
                               thumbnails.get('high', {}).get('url') or \
                               thumbnails.get('medium', {}).get('url') or \
                               thumbnails.get('default', {}).get('url', '')

                # Get description
                description = item['snippet'].get('description', '')
                
                also_trending_list.append({
                    "video_id": item_id,
                    "title": item['snippet']['title'],
                    "thumbnail": thumbnail_url,
                    "views": views,
                    "likes": likes,
                    "comment_count": comments,
                    "like_count": likes,
                    "engagement_rate": engagement_rate,
                    "category_id": category_id,
                    "description": description
                })

        # Return everything in a structured JSON format
        return jsonify({
            "success": True,
            "country": country_code,
            "keyword": keyword,
            "videos": video_dashboard_list,
            "also_trending": also_trending_list if keyword else [],
            "category_analysis": category_analysis,
            "keyword_analysis": keyword_analysis,
            "upload_vs_popularity": upload_vs_popularity
        })

    except Exception as e:
        # Handle errors (like an invalid API key or bad country code)
        print(f"An error occurred: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# --- 4. Main Server Execution ---


# This makes the server run when we execute 'python app.py'
if __name__ == '__main__':
    app.run(debug=True)