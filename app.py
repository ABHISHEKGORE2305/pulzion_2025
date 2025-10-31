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
        
        # We also need a simple list of videos for the dashboard
        video_dashboard_list = []
        for item in video_items:
            video_dashboard_list.append({
                "video_id": item['id'],
                "title": item['snippet']['title'],
                "thumbnail": item['snippet']['thumbnails']['default']['url'],
                "views": item['statistics'].get('viewCount', 'N/A')
            })

        # Return everything in a structured JSON format
        return jsonify({
            "success": True,
            "country": country_code,
            "videos": video_dashboard_list,
            "category_analysis": category_analysis,
            "keyword_analysis": keyword_analysis
        })

    except Exception as e:
        # Handle errors (like an invalid API key or bad country code)
        print(f"An error occurred: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# --- 4. Main Server Execution ---


# This makes the server run when we execute 'python app.py'
if __name__ == '__main__':
    app.run(debug=True)