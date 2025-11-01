import os
import re
from collections import Counter
import nltk
from flask import Flask, jsonify, request
from googleapiclient.discovery import build
from dotenv import load_dotenv
from flask_cors import CORS

from creator_suggestions import suggest_content


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
                    data_points.append({
                        "x": days_since_upload,
                        "y": views,
                        "engagement_rate": engagement_rate  # Store for later normalization
                    })
            except Exception as e:
                # Skip videos with invalid date formats
                continue
    
    # Normalize bubble sizes based on engagement rates for better visualization
    if data_points:
        engagement_rates = [point["engagement_rate"] for point in data_points]
        min_engagement = min(engagement_rates) if engagement_rates else 0
        max_engagement = max(engagement_rates) if engagement_rates else 1
        
        # Normalize to a reasonable bubble size range (4 to 12 pixels)
        for point in data_points:
            if max_engagement > min_engagement:
                # Normalize engagement rate to 0-1, then scale to 4-12
                normalized = (point["engagement_rate"] - min_engagement) / (max_engagement - min_engagement)
                point["r"] = 4 + (normalized * 8)  # Range from 4 to 12 pixels
            else:
                point["r"] = 6  # Default size if all engagement rates are the same
            
            # Keep engagement_rate for tooltip display (Chart.js ignores extra properties)
            # Note: engagement_rate is kept for frontend tooltip use
    
    return data_points

# --- 4. Main API Endpoint ---

@app.route('/get_trending_data')
def get_trending_data():
    """
    Fetches trending videos for a given country and returns analyzed data.
    """
    # Get the 'country' code from the request (e.g., /get_trending_data?country=US)
    country_code = request.args.get('country', 'US') # Default to 'US'
    
    try:
        # --- API Integration (30%) ---
        # This is the actual call to the YouTube API
        api_request = youtube_service.videos().list(
            part="snippet,statistics", # Request video details and view counts
            chart="mostPopular",      # Get the "trending" chart
            regionCode=country_code,  # Set the country
            maxResults=25             # Get the top 25 videos
        )
        api_response = api_request.execute()
        
        video_items = api_response.get("items", [])
        
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

            video_dashboard_list.append({
                "video_id": item['id'],
                "title": item['snippet']['title'],
                "thumbnail": item['snippet']['thumbnails']['default']['url'],
                "views": item['statistics'].get('viewCount', 'N/A'),
                "engagement_rate": engagement_rate
            })

        # Return everything in a structured JSON format
        return jsonify({
            "success": True,
            "country": country_code,
            "videos": video_dashboard_list,
            "category_analysis": category_analysis,
            "keyword_analysis": keyword_analysis,
            "upload_vs_popularity": upload_vs_popularity
        })

    except Exception as e:
        # Handle errors (like an invalid API key or bad country code)
        print(f"An error occurred: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
    
@app.route('/get_creator_suggestions')
def get_creator_suggestions():
    """
    Runs the advanced clustering and insight generator from creator_suggestions.py
    and returns the results as JSON.
    """
    region = request.args.get('country', 'US')
    max_results = int(request.args.get('max_results', 50))

    try:
        insights, df = suggest_content(region=region, max_results=max_results)

        # Convert DataFrame to dictionary for JSON
        videos_data = df.to_dict(orient='records')

        return jsonify({
            "success": True,
            "region": region,
            "insights": insights,
            "video_data": videos_data
        })

    except Exception as e:
        print(f"Error generating creator suggestions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# --- 4. Main Server Execution ---


# This makes the server run when we execute 'python app.py'
if __name__ == '__main__':
    app.run(debug=True)