"""
Creator Coach AI — Gemini 2.5 Flash Edition 🚀
----------------------------------------------
Fetches trending YouTube videos, analyzes them with Gemini,
and returns data-driven advice PLUS actionable referral links.

Dependencies:
    pip install google-api-python-client google-generativeai python-dotenv pandas
"""

import os
import json
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
import google.generativeai as genai
import re

# ----------------------------
# 1. Setup API Keys
# ----------------------------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ----------------------------
# 2. Fetch trending videos
# ----------------------------
def fetch_trending_videos(region="US", genre="music", max_results=20):
    """
    Fetch trending YouTube videos by region and optionally by genre/category.
    
    Args:
        region (str): ISO 3166-1 alpha-2 country code (e.g., 'US', 'IN', 'JP')
        genre (str or int): Optional YouTube video category ID (e.g., '10' for Music, '20' for Gaming)
        max_results (int): Number of results to fetch (default 20)
    """
    request_params = {
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": region,
        "maxResults": max_results
    }

    if genre:
        request_params["videoCategoryId"] = str(genre)

    response = youtube.videos().list(**request_params).execute()

    videos = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        videos.append({
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channelTitle": snippet.get("channelTitle"),
            "categoryId": snippet.get("categoryId"),
            "publishedAt": snippet.get("publishedAt"),
            "viewCount": int(stats.get("viewCount", 0)),
            "likeCount": int(stats.get("likeCount", 0)),
            "commentCount": int(stats.get("commentCount", 0)),
            "tags": snippet.get("tags", [])
        })

    return pd.DataFrame(videos)


# ----------------------------
# 3. Build prompt (Refined)
# ----------------------------
def build_prompt(videos_df: pd.DataFrame, country: str, genre: str = None) -> str:
    """Create a structured, human-readable prompt for Gemini."""
    sample_rows = videos_df.head(10)
    video_text = "\n".join(
        [
            f"- {row['title']} (tags: {', '.join(row['tags']) if isinstance(row['tags'], list) else 'N/A'}, "
            f"views: {row['viewCount']:,})"
            for _, row in sample_rows.iterrows()
        ]
    )

    genre_text = f" within the {genre} category" if genre else ""

    prompt = f"""
You are Creator Coach AI, a YouTube mentor that delivers insights in a friendly,
structured, and visually appealing format (no code blocks, no markdown symbols).

Analyze these trending YouTube videos from {country}{genre_text}:
{video_text}

🎯 Your task:
1. Summarize what’s trending right now (mention 3–5 key themes or categories, specific to this genre if applicable).
2. Highlight what successful creators in this genre are doing right.
3. Identify common mistakes or struggles creators face in this genre and how to fix them.
4. Provide 5 clear, actionable recommendations for growth (content, SEO, thumbnails, engagement) — tailored to this genre.
5. End with one short motivational line, as if you’re a coach inspiring the creator.

Format your report exactly as follows (no markdown syntax like **, ##, or _):

🎬 Creator Coach Report — {country}{genre_text}

🔥 What’s Trending
- point 1
- point 2
- point 3

💡 What Top Creators Are Doing Right
- insight 1
- insight 2

⚠️ Common Struggles (and Fixes)
Problem: ...
Solution: ...

🧭 Actionable Recommendations
1. tip 1
2. tip 2
3. tip 3
4. tip 4
5. tip 5

💪 Coach’s Motivation
Short, inspiring line that ends the report.

Keep it conversational, practical, and professional.
"""
    return prompt.strip()


# ----------------------------
# 4. Clean Gemini output
# ----------------------------
def clean_gemini_output(text: str) -> str:
    """Remove Markdown, stray symbols, and formatting artifacts."""
    text = re.sub(r"[#*_`]+", "", text)  # Remove markdown symbols
    text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse triple newlines
    text = re.sub(r" {2,}", " ", text)  # Collapse spaces
    text = text.replace("\\n", "\n").replace("\\", "").strip()
    return text


# ----------------------------
# 5. Analyze via Gemini
# ----------------------------
def analyze_trends_with_gemini(videos_df: pd.DataFrame, country: str, genre: str = None):
    """Send video data to Gemini API and return a cleaned, plain-text report."""
    prompt = build_prompt(videos_df, country, genre)
    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        cleaned_text = clean_gemini_output(raw_text)
        return cleaned_text
    except Exception as e:
        return f"⚠️ Error analyzing with Gemini: {str(e)}"


# ----------------------------
# 6. Main Runner
# ----------------------------
if __name__ == "__main__":
    print("🌍 Welcome to Creator Coach AI!")
    
    country = input("Enter country code (e.g., US, IN, JP): ").strip().upper()
    genre = input("Enter YouTube genre/category ID (or press Enter for all): ").strip()
    genre = genre if genre else None

    print(f"\n📡 Fetching trending YouTube videos for {country}{' - Genre ID: ' + genre if genre else ''}...")
    videos_df = fetch_trending_videos(region=country, genre=genre, max_results=20)
    print(f"✅ Fetched {len(videos_df)} videos.")

    insights = analyze_trends_with_gemini(videos_df, country=country, genre=genre)

    print("\n🎬 === Creator Coach AI (Gemini 2.5 Flash) ===\n")
    print(insights)
