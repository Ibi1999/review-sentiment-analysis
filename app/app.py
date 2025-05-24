import streamlit as st
import requests
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from rapidfuzz import fuzz

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

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
st.title("🎬 Movie Review Preprocessor")

if not TMDB_API_KEY:
    st.error("❌ TMDB_API_KEY not set in environment variables.")
    st.stop()

search_mode = st.radio("Search by:", ["Movie ID", "Movie Name"])

selected_movie_id = None

# 🎯 Search by ID
if search_mode == "Movie ID":
    movie_id_input = st.text_input("Enter TMDB Movie ID")
    if movie_id_input:
        selected_movie_id = movie_id_input

# 🔍 Search by name
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

# ✅ Unified fetch and clean button
if selected_movie_id and st.button("Fetch & Clean Reviews"):
    reviews = get_reviews(selected_movie_id)
    if reviews:
        unique_reviews = remove_near_duplicates(reviews, threshold=95)
        cleaned_reviews = [preprocess(r) for r in reviews if r and len(r) > 20]
        st.subheader("🧼 Cleaned Reviews")
        for idx, r in enumerate(cleaned_reviews, 1):
            st.markdown(f"**{idx}.** {r}")
    else:
        st.info("No reviews found for this movie.")
