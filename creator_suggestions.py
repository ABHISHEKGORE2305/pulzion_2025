"""
creator_suggestions.py
Generates creative YouTube content suggestions by analyzing trending videos
and enhancing them using Gemini (Google DeepMind) LLM.
"""

import os
import re
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
from dotenv import load_dotenv
import google.generativeai as genai

# -----------------------
# 1. Setup
# -----------------------
load_dotenv()

YOUTUBE_API_KEY = "AIzaSyBe_JsNMzh94KhA_MUGMfFgs2VfWGAfu8g"
GEMINI_API_KEY = "AIzaSyCuApQORQ6Ka12xNGnr9or0yunxytaFAPY"

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------
# 2. Fetch Trending Videos
# -----------------------
def fetch_trending_videos(region="US", max_results=50):
    """Fetch trending YouTube videos from a specific region."""
    response = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode=region,
        maxResults=max_results
    ).execute()

    videos = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        videos.append({
            "title": snippet.get("title"),
            "categoryId": snippet.get("categoryId"),
            "publishedAt": snippet.get("publishedAt"),
            "viewCount": int(stats.get("viewCount", 0)),
            "likeCount": int(stats.get("likeCount", 0)),
            "commentCount": int(stats.get("commentCount", 0))
        })
    return pd.DataFrame(videos)

# -----------------------
# 3. Feature Engineering
# -----------------------
def preprocess_text(text):
    """Clean and normalize titles."""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text))
    return text.lower().strip()

def compute_engagement_metrics(df):
    df["like_ratio"] = df["likeCount"] / (df["viewCount"] + 1)
    df["comment_ratio"] = df["commentCount"] / (df["viewCount"] + 1)
    df["engagement_score"] = df["like_ratio"] + df["comment_ratio"]
    return df

# -----------------------
# 4. Clustering
# -----------------------
def cluster_titles(df, num_clusters=5):
    df["clean_title"] = df["title"].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(df["clean_title"])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    df["cluster"] = kmeans.labels_

    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for i in range(num_clusters):
        center = kmeans.cluster_centers_[i]
        top_terms = [terms[idx] for idx in center.argsort()[::-1][:5]]
        cluster_keywords[i] = top_terms

    return df, cluster_keywords

# -----------------------
# 5. Cluster Analysis
# -----------------------
def analyze_clusters(df, cluster_keywords):
    grouped = df.groupby("cluster").agg({
        "viewCount": "mean",
        "like_ratio": "mean",
        "comment_ratio": "mean",
        "engagement_score": "mean"
    }).reset_index()

    top_cluster = grouped.sort_values("engagement_score", ascending=False).iloc[0]
    cluster_id = int(top_cluster["cluster"])
    keywords = ", ".join(cluster_keywords[cluster_id])
    avg_engagement = round(top_cluster["engagement_score"] * 100, 2)

    return cluster_id, cluster_keywords, avg_engagement

# -----------------------
# 6. Sentiment Analysis
# -----------------------
def add_sentiment(df):
    df["sentiment"] = df["title"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

# -----------------------
# 7. Generate Ideas with Gemini
# -----------------------
def generate_ai_ideas_with_gemini(top_keywords, sample_titles, avg_engagement, sentiment):
    """Generate YouTube video ideas using Gemini LLM."""
    prompt = f"""
You are a YouTube content strategist who helps creators go viral.

Use the analytics below to generate 5 creative video ideas.

Each idea must include:
1. A **Title**
2. A 1-line **Description** (what to cover)
3. A **Hook** (first 10 seconds)
4. A short **Thumbnail Text** (max 5 words)

---
Trending keywords: {', '.join(top_keywords)}
Average engagement rate: {avg_engagement:.2f}%
Average sentiment polarity: {sentiment:.2f}

Here are sample trending titles for reference:
{chr(10).join(['- ' + t for t in sample_titles])}

Be creative and specific. Return in a clean numbered list format like:

1. "Surviving the Halloween Apocalypse"
   - Cover: trending spooky challenges, short horror skits.
   - Hook: “What if your favorite YouTubers vanished on Halloween night?”
   - Thumbnail: “It Actually Happened... 😱”
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

# -----------------------
# 8. Full Suggestion Pipeline
# -----------------------
def suggest_content(region="US", max_results=50):
    print("📊 Fetching trending videos...")
    df = fetch_trending_videos(region, max_results)
    df = compute_engagement_metrics(df)
    df = add_sentiment(df)

    print("🤖 Clustering trending topics...")
    df, cluster_keywords = cluster_titles(df, num_clusters=5)

    print("🔍 Analyzing cluster performance...")
    top_cluster_id, cluster_keywords, avg_engagement = analyze_clusters(df, cluster_keywords)

    top_keywords = cluster_keywords[top_cluster_id]
    sentiment = df["sentiment"].mean()
    sample_titles = df[df["cluster"] == top_cluster_id]["title"].head(5).tolist()

    print("💡 Generating AI-powered content ideas with Gemini...")
    ai_output = generate_ai_ideas_with_gemini(top_keywords, sample_titles, avg_engagement, sentiment)

    print("\n✅ Suggested YouTube Ideas:\n")
    print(ai_output)

    return ai_output, df

# -----------------------
# 9. Run if main
# -----------------------
if __name__ == "__main__":
    suggest_content(region="US", max_results=50)
