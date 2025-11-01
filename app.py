import os
import re
from collections import Counter
import nltk
from flask import Flask, jsonify, request
from googleapiclient.discovery import build
from dotenv import load_dotenv
from flask_cors import CORS

from creator_suggestions import suggest_content
from creator_coach_ai import fetch_trending_videos, analyze_trends_with_gemini


# --- 1. Setup and Config ---

# Load environment variables (our API key) from the .env file
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not YOUTUBE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("⚠️ API keys not found in environment variables.")

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
youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)


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

def analyze_upload_times(video_items):
    """Analyzes which hour of the day (0-23) trending videos were uploaded and their average view counts."""
    from datetime import datetime, timezone
    from collections import defaultdict
    
    # Dictionary to store hour -> [views, count]
    hour_data = defaultdict(lambda: {"total_views": 0, "count": 0, "total_engagement": 0.0})
    
    for item in video_items:
        snippet = item.get('snippet', {})
        statistics = item.get('statistics', {})
        
        published_at = snippet.get('publishedAt')
        views_str = statistics.get('viewCount')
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
        
        if published_at:
            try:
                # Parse the upload date and time
                if published_at.endswith('Z'):
                    upload_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                else:
                    upload_date = datetime.strptime(published_at.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                
                # Extract the hour (0-23)
                upload_hour = upload_date.hour
                
                # Add to the hour's data
                hour_data[upload_hour]["total_views"] += views
                hour_data[upload_hour]["count"] += 1
                hour_data[upload_hour]["total_engagement"] += engagement_rate
            except Exception as e:
                # Skip videos with invalid date formats
                continue
    
    # Calculate average views per hour and create data points for line chart
    upload_time_data = []
    for hour in range(24):
        if hour in hour_data:
            avg_views = hour_data[hour]["total_views"] / hour_data[hour]["count"]
            avg_engagement = hour_data[hour]["total_engagement"] / hour_data[hour]["count"]
            video_count = hour_data[hour]["count"]
        else:
            avg_views = 0
            avg_engagement = 0
            video_count = 0
        
        # Format hour for display (12-hour format with AM/PM)
        hour_label = f"{hour % 12 if hour % 12 != 0 else 12}{'AM' if hour < 12 else 'PM'}"
        
        upload_time_data.append({
            "hour": hour,
            "hour_label": hour_label,
            "average_views": round(avg_views, 0),
            "average_engagement": round(avg_engagement, 2),
            "video_count": video_count
        })
    
    return upload_time_data

def generate_upload_recommendations(video_items, upload_time_data, category_analysis):
    """Uses ML/statistical analysis to recommend best upload times and categories."""
    
    # Find the best upload hour based on average views (only consider hours with videos)
    hours_with_videos = [h for h in upload_time_data if h["video_count"] > 0]
    if not hours_with_videos:
        # Fallback if no videos found
        best_hour = 12
        best_hour_label = "12PM"
        best_hour_views = 0
    else:
        best_hour_data = max(hours_with_videos, key=lambda x: x["average_views"])
        best_hour = best_hour_data["hour"]
        best_hour_label = best_hour_data["hour_label"]
        best_hour_views = best_hour_data["average_views"]
    
    # Find top 3 hours for views (only those with videos)
    top_hours = sorted(hours_with_videos if hours_with_videos else upload_time_data, 
                      key=lambda x: x["average_views"], reverse=True)[:3]
    
    # Find best category based on view count and frequency
    if category_analysis:
        # Get category IDs and their counts
        category_items = list(category_analysis.items())
        
        # Analyze videos in each category to find average views
        category_views = {}
        for item in video_items:
            category_id = str(item.get('snippet', {}).get('categoryId', ''))
            views_str = item.get('statistics', {}).get('viewCount', '0')
            try:
                views = int(views_str) if views_str else 0
            except ValueError:
                views = 0
            
            if category_id:
                if category_id not in category_views:
                    category_views[category_id] = {"total_views": 0, "count": 0}
                category_views[category_id]["total_views"] += views
                category_views[category_id]["count"] += 1
        
        # Calculate average views per category
        category_performance = []
        for cat_id, cat_count in category_analysis.items():
            if cat_id in category_views and category_views[cat_id]["count"] > 0:
                avg_views = category_views[cat_id]["total_views"] / category_views[cat_id]["count"]
                category_performance.append({
                    "category_id": cat_id,
                    "video_count": cat_count,
                    "average_views": avg_views,
                    "performance_score": avg_views * cat_count  # Weight by frequency
                })
        
        if category_performance:
            best_category = max(category_performance, key=lambda x: x["performance_score"])
            best_category_id = best_category["category_id"]
        else:
            best_category_id = None
    else:
        best_category_id = None
    
    # Category name mapping (partial - will be completed in frontend)
    category_map = {
        "1": "Film & Animation",
        "2": "Autos & Vehicles",
        "10": "Music",
        "15": "Pets & Animals",
        "17": "Sports",
        "19": "Travel & Events",
        "20": "Gaming",
        "22": "People & Blogs",
        "23": "Comedy",
        "24": "Entertainment",
        "25": "News & Politics",
        "26": "Howto & Style",
        "27": "Education",
        "28": "Science & Technology"
    }
    
    best_category_name = category_map.get(best_category_id, f"Category {best_category_id}") if best_category_id else "Various Categories"
    
    # Generate insights text
    insights = {
        "best_upload_hour": best_hour,
        "best_upload_hour_label": best_hour_label,
        "best_upload_hour_views": int(best_hour_views),
        "top_hours": [
            {
                "hour": h["hour"],
                "hour_label": h["hour_label"],
                "average_views": int(h["average_views"])
            }
            for h in top_hours
        ],
        "best_category_id": best_category_id,
        "best_category_name": best_category_name,
        "recommendation_text": f"Based on trending video analytics, upload your content at {best_hour_label} for maximum reach. Videos uploaded at this time show an average of {int(best_hour_views):,} views. The most successful category in trending videos is {best_category_name}."
    }
    
    return insights

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
            maxResults=100             # Get the top 25 videos
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
        upload_times_analysis = analyze_upload_times(video_items)
        upload_recommendations = generate_upload_recommendations(video_items, upload_times_analysis, category_analysis)
        
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
            "upload_vs_popularity": upload_vs_popularity,
            "upload_times_analysis": upload_times_analysis,
            "upload_recommendations": upload_recommendations
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



@app.route('/get_creator_coach')
def get_creator_coach():
    """
    Runs the Creator Coach AI analysis for a given country and genre.
    Returns Gemini insights based on trending YouTube data.
    """
    country = request.args.get('country', 'US').upper()
    genre = request.args.get('genre', None)
    
    try:
        # Fetch trending videos (from your Creator Coach module)
        videos_df = fetch_trending_videos(region=country, genre=genre, max_results=20)
        
        # If no videos found, return message
        if videos_df.empty:
            return jsonify({
                "success": False,
                "message": f"No trending videos found for {country} (genre: {genre})"
            }), 404

        # Run Gemini analysis (cleaned text output)
        insights = analyze_trends_with_gemini(videos_df, country=country, genre=genre)

        # Return insights as JSON
        return jsonify({
            "success": True,
            "country": country,
            "genre": genre,
            "insights": insights
        })

    except Exception as e:
        print(f"Error running Creator Coach AI: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# --- 4. Main Server Execution ---


# This makes the server run when we execute 'python app.py'
if __name__ == '__main__':
    app.run(debug=True)