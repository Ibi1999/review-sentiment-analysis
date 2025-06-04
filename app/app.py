import streamlit as st
import requests
import re
import nltk
import os
import torch
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from rapidfuzz import fuzz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🤖 Machine Learning - Sentiment Model App")
    with st.expander("❓ How to use", expanded=True):
        st.markdown("""
        **Step 1**: Search for a movie by name or TMDB ID.  
        **Step 2**: Click **Fetch & Analyze Sentiment** to retrieve and analyze reviews.  
        **Step 3**: View results in tabs for each model with sentiment scores and visual breakdowns.
        """)

    with st.expander("📘 Model Descriptions", expanded=False):
        st.markdown("""
        - **🧠 RoBERTa**: Deep learning transformer-based model trained for sentiment classification.
        - **🧮 TF-IDF**: Machine learning pipeline using text vectorization and a classifier.
        - **🧾 VADER**: Lexicon and rule-based sentiment analysis tool specialized for social media text.
        """)

# Load environment
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Session state initialization
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None
if "movie_options" not in st.session_state:
    st.session_state.movie_options = {}
if "search_done" not in st.session_state:
    st.session_state.search_done = False

# NLTK setup
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
for resource in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# Preprocessing functions
def preprocess_vader(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def preprocess_roberta(text):
    text = re.sub(r"<.*?>", " ", text)
    return text.lower()

# VADER sentiment analyzer
_vader_analyzer = SentimentIntensityAnalyzer()

def _analyze_sentiment_vader(reviews):
    results = []
    for review in reviews:
        if review and len(review) > 20:
            cleaned = preprocess_vader(review)
            scores = _vader_analyzer.polarity_scores(cleaned)
            standardized_score = (scores['compound'] + 1) / 2  # 0 to 1 scale

            if standardized_score >= 0.55:
                sentiment = "positive"
            elif standardized_score <= 0.45:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            results.append((review, sentiment, standardized_score))
    return results



# RoBERTa model loader
@st.cache_resource
def load_roberta_model():
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'roberta_sentiment'))
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

def roberta_predict(texts, model, tokenizer):
    model.eval()
    cleaned = [preprocess_roberta(r) for r in texts]
    inputs = tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        scores = probs[:, 1].cpu().numpy()  # positive class prob

    sentiments = []
    for i, score in enumerate(scores):
        if score >= 0.55:
            sentiment = "positive"
        elif score <= 0.45:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        sentiments.append((texts[i], sentiment, score))
    return sentiments


# TF-IDF model loader
@st.cache_resource
def load_tfidf_model():
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'TF-IDF_trained_model.pkl'))
    pipeline = joblib.load(MODEL_PATH)
    return pipeline

def tfidf_predict(texts, pipeline):
    preprocessed = [preprocess_vader(text) for text in texts]
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(preprocessed)
        scores = probs[:, 1]
    else:
        scores = [1.0 if p == 1 else 0.0 for p in pipeline.predict(preprocessed)]

    sentiments = []
    for score in scores:
        if score >= 0.55:
            sentiments.append("positive")
        elif score <= 0.45:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")

    return list(zip(texts, sentiments, scores))


# Dedupe helper
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
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

# API: Fetch reviews
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

# API: Search movie
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

# Plotting functions
def plot_donut_chart(results, title, key):
    pos_count = sum(1 for r in results if r[1] == "positive")
    neg_count = sum(1 for r in results if r[1] == "negative")
    neu_count = sum(1 for r in results if r[1] == "neutral")
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[pos_count, neu_count, neg_count],
        hole=.6,
        marker_colors=['#2ecc40', '#f1c40f', '#e74c3c']  # Green, Yellow, Red
    )])
    
    fig.update_layout(
        title_text=title,
        margin=dict(t=40, b=0, l=0, r=0),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1)
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)


def plot_horizontal_bar(roberta_results, tfidf_results, vader_results, key):
    model_names = ["RoBERTa", "TF-IDF", "VADER"]
    counts = [
        sum(1 for r in roberta_results if r[1] == "positive"),
        sum(1 for r in tfidf_results if r[1] == "positive"),
        sum(1 for r in vader_results if r[1] == "positive"),
    ]
    fig = go.Figure(go.Bar(
        x=counts,
        y=model_names,
        orientation='h',
        marker=dict(color=['#ff69b4', '#3498db', '#f39c12']),  # Pink, Blue, Orange
    ))
    fig.update_layout(
        title="Positive Review Count by Model",
        xaxis_title="Count",
        yaxis_title="Model",
        margin=dict(t=40, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def display_reviews(results, sentiment_filter=None):
    filtered = [r for r in results if sentiment_filter is None or r[1] == sentiment_filter]
    for idx, (review, sentiment, score) in enumerate(filtered, 1):
        if sentiment == "positive":
            color = "#2ecc40"  # Green
        elif sentiment == "negative":
            color = "#e74c3c"  # Red
        elif sentiment == "neutral":
            color = "#f1c40f"  # Yellow
        else:
            color = "#000000"  # Default black just in case

        st.markdown("---")
        st.markdown(
            f"<span style='color:{color}; font-weight:bold;'>Sentiment:</span> {sentiment.capitalize()}  |  "
            f"<span style='color:{color}; font-weight:bold;'>Score:</span> {score:.3f}",
            unsafe_allow_html=True
        )
        st.markdown(f"**Review #{idx}:**")
        st.markdown(review)


# -------- Main App --------
st.title("🎬 Movie Review Sentiment Analyzer")

if not TMDB_API_KEY:
    st.error("❌ TMDB_API_KEY not set in environment variables.")
    st.stop()

# Search mode selection
search_mode = st.radio("Search by:", ["Movie Name", "Movie ID"], index=0)

if search_mode == "Movie Name":
    movie_name_input = st.text_input("Enter Movie Name")

    if st.button("Find Movie"):
        if movie_name_input.strip() == "":
            st.warning("Please enter a movie name.")
            st.session_state.movie_options = {}
            st.session_state.search_done = False
            st.session_state.selected_movie_id = None
        else:
            results = search_movies_by_name(movie_name_input.strip())
            if results:
                st.session_state.movie_options = {
                    f"{r['title']} ({r.get('release_date', 'N/A')[:4]})": r["id"] for r in results
                }
                st.session_state.search_done = True
                st.session_state.selected_movie_id = None  # reset selected movie
            else:
                st.warning("No movies found.")
                st.session_state.movie_options = {}
                st.session_state.search_done = False
                st.session_state.selected_movie_id = None

    # Show selectbox only after search is done and results exist
    if st.session_state.search_done and st.session_state.movie_options:
        selected_movie_title = st.selectbox(
            "Select a movie:",
            options=list(st.session_state.movie_options.keys()),
            index=0,
            key="selected_movie_title"
        )
        if selected_movie_title:
            st.session_state.selected_movie_id = st.session_state.movie_options[selected_movie_title]
        else:
            st.session_state.selected_movie_id = None
            st.info("Please select a movie from the dropdown.")
    elif st.session_state.search_done and not st.session_state.movie_options:
        st.info("Please find a movie first.")

elif search_mode == "Movie ID":
    movie_id_input = st.text_input("Enter TMDB Movie ID")
    if movie_id_input.strip():
        st.session_state.selected_movie_id = movie_id_input.strip()
    else:
        st.session_state.selected_movie_id = None
#fetch and analyse reviews
if st.session_state.selected_movie_id:
    if st.button("Fetch & Analyze Sentiment"):
        reviews = get_reviews(st.session_state.selected_movie_id)
        if not reviews:
            st.info("No reviews found for this movie.")
                    # Clear previous analysis results to avoid showing stale data
            for key in ['roberta_results', 'tfidf_results', 'vader_results']:
                if key in st.session_state:
                    del st.session_state[key]
        else:
            st.success(f"Fetched {len(reviews)} reviews from TMDB API.")
            unique_reviews = remove_near_duplicates(reviews, threshold=95)

            vader_results = _analyze_sentiment_vader(unique_reviews)
            tfidf_pipeline = load_tfidf_model()
            tfidf_results = tfidf_predict(unique_reviews, tfidf_pipeline)
            roberta_model, roberta_tokenizer = load_roberta_model()
            roberta_results = roberta_predict(unique_reviews, roberta_model, roberta_tokenizer)

            st.session_state['roberta_results'] = roberta_results
            st.session_state['tfidf_results'] = tfidf_results
            st.session_state['vader_results'] = vader_results

    def roberta_pos_changed():
        if st.session_state["roberta_pos"]:
            st.session_state["roberta_neg"] = False
            st.session_state["roberta_neu"] = False

    def roberta_neg_changed():
        if st.session_state["roberta_neg"]:
            st.session_state["roberta_pos"] = False
            st.session_state["roberta_neu"] = False

    def roberta_neu_changed():
        if st.session_state["roberta_neu"]:
            st.session_state["roberta_pos"] = False
            st.session_state["roberta_neg"] = False


    def tfidf_pos_changed():
        if st.session_state["tfidf_pos"]:
            st.session_state["tfidf_neg"] = False
            st.session_state["tfidf_neu"] = False

    def tfidf_neg_changed():
        if st.session_state["tfidf_neg"]:
            st.session_state["tfidf_pos"] = False
            st.session_state["tfidf_neu"] = False

    def tfidf_neu_changed():
        if st.session_state["tfidf_neu"]:
            st.session_state["tfidf_pos"] = False
            st.session_state["tfidf_neg"] = False


    def vader_pos_changed():
        if st.session_state["vader_pos"]:
            st.session_state["vader_neg"] = False
            st.session_state["vader_neu"] = False

    def vader_neg_changed():
        if st.session_state["vader_neg"]:
            st.session_state["vader_pos"] = False
            st.session_state["vader_neu"] = False

    def vader_neu_changed():
        if st.session_state["vader_neu"]:
            st.session_state["vader_pos"] = False
            st.session_state["vader_neg"] = False


    if 'roberta_results' in st.session_state and 'tfidf_results' in st.session_state and 'vader_results' in st.session_state:
        roberta_results = st.session_state['roberta_results']
        tfidf_results = st.session_state['tfidf_results']
        vader_results = st.session_state['vader_results']

        st.subheader("🔍 Review Breakdown by Model")
        tab_roberta, tab_tfidf, tab_vader = st.tabs(["🧠 RoBERTa ML", "🧮 TF-IDF ML", "🧾 VADER"])

        with tab_roberta:
            col1, col2 = st.columns(2)
            with col1:
                plot_donut_chart(roberta_results, "RoBERTa Sentiment Distribution", key="donut_roberta")

                pos_checked = st.checkbox(
                    "Only show positive",
                    value=False,
                    key="roberta_pos",
                    on_change=roberta_pos_changed
                )
                neg_checked = st.checkbox(
                    "Only show negative",
                    value=False,
                    key="roberta_neg",
                    on_change=roberta_neg_changed
                )
                neutral_checked = st.checkbox(
                    "Only show neutral",
                    value=False,
                    key="roberta_neu",
                    on_change=roberta_neu_changed
                )

            with col2:
                plot_horizontal_bar(roberta_results, tfidf_results, vader_results, key="bar_roberta")

                if st.session_state.get("roberta_pos"):
                    filtered = [r for r in roberta_results if r[1] == "positive"]
                elif st.session_state.get("roberta_neg"):
                    filtered = [r for r in roberta_results if r[1] == "negative"]
                elif st.session_state.get("roberta_neu"):
                    filtered = [r for r in roberta_results if r[1] == "neutral"]
                else:
                    filtered = roberta_results

            display_reviews(filtered)

        with tab_tfidf:
            col1, col2 = st.columns(2)
            with col1:
                plot_donut_chart(tfidf_results, "TF-IDF Sentiment Distribution", key="donut_tfidf")

                pos_checked = st.checkbox(
                    "Only show positive",
                    value=False,
                    key="tfidf_pos",
                    on_change=tfidf_pos_changed
                )
                neg_checked = st.checkbox(
                    "Only show negative",
                    value=False,
                    key="tfidf_neg",
                    on_change=tfidf_neg_changed
                )
                neutral_checked = st.checkbox(
                    "Only show neutral",
                    value=False,
                    key="tfidf_neu",
                    on_change=tfidf_neu_changed
                )

            with col2:
                plot_horizontal_bar(roberta_results, tfidf_results, vader_results, key="bar_tfidf")

                if st.session_state.get("tfidf_pos"):
                    filtered = [r for r in tfidf_results if r[1] == "positive"]
                elif st.session_state.get("tfidf_neg"):
                    filtered = [r for r in tfidf_results if r[1] == "negative"]
                elif st.session_state.get("tfidf_neu"):
                    filtered = [r for r in tfidf_results if r[1] == "neutral"]
                else:
                    filtered = tfidf_results

            display_reviews(filtered)

        with tab_vader:
            col1, col2 = st.columns(2)
            with col1:
                plot_donut_chart(vader_results, "VADER Sentiment Distribution", key="donut_vader")

                pos_checked = st.checkbox(
                    "Only show positive",
                    value=False,
                    key="vader_pos",
                    on_change=vader_pos_changed
                )
                neg_checked = st.checkbox(
                    "Only show negative",
                    value=False,
                    key="vader_neg",
                    on_change=vader_neg_changed
                )
                neutral_checked = st.checkbox(
                    "Only show neutral",
                    value=False,
                    key="vader_neu",
                    on_change=vader_neu_changed
                )

            with col2:
                plot_horizontal_bar(roberta_results, tfidf_results, vader_results, key="bar_vader")

                if st.session_state.get("vader_pos"):
                    filtered = [r for r in vader_results if r[1] == "positive"]
                elif st.session_state.get("vader_neg"):
                    filtered = [r for r in vader_results if r[1] == "negative"]
                elif st.session_state.get("vader_neu"):
                    filtered = [r for r in vader_results if r[1] == "neutral"]
                else:
                    filtered = vader_results

            display_reviews(filtered)
    else:
        st.info("Please fetch and analyze sentiment to see results.")
else:
    st.info("Please select a movie first to fetch and analyze reviews.")




