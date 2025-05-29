import streamlit as st
import requests
import re
import nltk
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from rapidfuzz import fuzz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Initialize VADER
_vader_analyzer = SentimentIntensityAnalyzer()

# Private method for VADER sentiment analysis
def _analyze_sentiment_vader(reviews):
    results = []
    for review in reviews:
        if review and len(review) > 20:
            cleaned = preprocess(review)
            scores = _vader_analyzer.polarity_scores(cleaned)
            # Only positive or negative
            sentiment = "positive" if scores['compound'] >= 0 else "negative"
            results.append((review, sentiment, scores['compound']))
    return results

# NLTK setup
nltk.download("stopwords")
nltk.download("wordnet")

# Preprocessing function
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Remove near duplicates
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # remove excessive whitespace
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.strip()

def remove_near_duplicates(reviews, threshold=95):
    seen = []
    for review in reviews:
        if not review or len(review.strip()) <= 20:
            continue
        norm_review = normalize_text(review)
        if all(fuzz.token_set_ratio(norm_review, normalize_text(s)) < threshold for s in seen):
            seen.append(review.strip())
    return seen

# Fetch reviews using movie ID
def get_reviews(movie_id):
    headers = {
        "Authorization": f"Bearer {TMDB_API_KEY}",
        "accept": "application/json"
    }
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [r["content"] for r in data.get("results", [])]
    except Exception as e:
        st.error(f"Error fetching reviews: {e}")
        return []

# Search movies by name with caching
@st.cache_data(ttl=3600)
def search_movies_by_name(movie_name):
    headers = {
        "Authorization": f"Bearer {TMDB_API_KEY}",
        "accept": "application/json"
    }
    url = f"https://api.themoviedb.org/3/search/movie?query={movie_name}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        return []

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="wide")

# Sidebar: Sentiment analyzer selection
with st.sidebar:
    model_choice = st.selectbox("Select Sentiment Analyzer", ("VADER", "ML (in progress)"))

# Dynamic title
st.title(f"🎬 Movie Review Sentiment Analyzer ({model_choice})")

if not TMDB_API_KEY:
    st.error("❌ TMDB_API_KEY not set in environment variables.")
    st.stop()

search_mode = st.radio("Search by:", ["Movie ID", "Movie Name"])

selected_movie_id = None

# Search by ID
if search_mode == "Movie ID":
    movie_id_input = st.text_input("Enter TMDB Movie ID")
    if movie_id_input:
        selected_movie_id = movie_id_input

# Search by name
elif search_mode == "Movie Name":
    movie_name_input = st.text_input("Enter Movie Name")
    if movie_name_input:
        results = search_movies_by_name(movie_name_input)
        if results:
            movie_options = {
                f"{r['title']} ({r.get('release_date', 'N/A')[:4]})": r["id"]
                for r in results
            }
            selected_movie_title = st.selectbox("Select a movie:", list(movie_options.keys()))
            selected_movie_id = movie_options[selected_movie_title]
        else:
            st.warning("No movies found.")

# Place the checkbox before the fetch button
show_reviews = st.checkbox("Show individual reviews")

# Review Fetch + Sentiment Analysis
if selected_movie_id and st.button("Fetch & Analyze Sentiment"):
    reviews = get_reviews(selected_movie_id)
    if not reviews:
        st.info("No reviews found for this movie.")
    else:
        unique_reviews = remove_near_duplicates(reviews, threshold=95)

        if model_choice == "VADER":
            results = _analyze_sentiment_vader(unique_reviews)
            if results:
                avg_compound = sum(score for _, _, score in results) / len(results)
                overall_pct = (avg_compound + 1) / 2 * 100  # Convert from [-1, 1] to [0, 100]
                st.subheader("🧠 Overall Sentiment (VADER)")
                st.metric(label="Overall Sentiment Score", value=f"{overall_pct:.1f}% Positive")

                if show_reviews:
                    st.subheader("🗂 Review Breakdown")
                    for idx, (review, sentiment, score) in enumerate(results, 1):
                        st.markdown(f"**{idx}. Sentiment:** `{sentiment}` &nbsp;&nbsp; | &nbsp;&nbsp; **Score:** `{score:.3f}`")
                        st.write(review)
                        st.markdown("---")
                
            else:
                st.info("No valid reviews after preprocessing.")

        elif model_choice == "ML (in progress)":
            st.subheader("🧪 Sentiment Analysis (ML Model)")
            st.info("This feature is currently under development.")