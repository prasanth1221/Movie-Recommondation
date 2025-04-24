import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up page
st.set_page_config(
    page_title="🎬 CineMatcher", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load data
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    movies = pd.read_csv(os.path.join(base_path, 'archive (5)', 'movies.csv'))
    ratings = pd.read_csv(os.path.join(base_path, 'archive (5)', 'ratings.csv'))
    return movies, ratings

movies, ratings = load_data()

# Prepare movie stats
movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
movies = movies.merge(movie_stats, on='movieId', how='left')
movies['avg_rating'] = movies['avg_rating'].round(2)
movies['genres'] = movies['genres'].str.replace('|', ', ', regex=False)

# TF-IDF Matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Hybrid scoring
def hybrid_score(sim_scores, movies_df):
    result = []
    for idx, score in sim_scores:
        rating_count = movies_df.iloc[idx]['rating_count'] if not pd.isna(movies_df.iloc[idx]['rating_count']) else 0
        avg_rating = movies_df.iloc[idx]['avg_rating'] if not pd.isna(movies_df.iloc[idx]['avg_rating']) else 0
        popularity_boost = (rating_count / 1000) + avg_rating
        result.append((idx, score * popularity_boost))
    return sorted(result, key=lambda x: x[1], reverse=True)

# Recommendation function
def recommend_movie(title):
    if title not in movies['title'].values:
        return pd.DataFrame()
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = hybrid_score(sim_scores, movies)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres', 'avg_rating', 'rating_count']].iloc[movie_indices]

# Apply custom CSS
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e6e6e6;
    }
    
    /* Header styles */
    .title-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    .title {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        color: #ff6b6b;
    }
    
    .subtitle {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
        color: #e6e6e6;
        font-size: 1.2rem;
    }
    
    /* Card styles */
    .card {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 1rem;
        border-left: 4px solid #ff6b6b;
    }
    
    /* Button styles */
    .stButton>button {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff8e8e 100%);
        color: #1a1a2e;
        font-weight: 600;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 20px rgba(255, 107, 107, 0.6);
    }
    
    /* Select box styles */
    .stSelectbox>div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .stSelectbox>div>div>div {
        color: #e6e6e6;
    }
    
    /* Results table */
    .results-container {
        background: rgba(26, 26, 46, 0.7);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    /* Star rating */
    .star-rating {
        color: #ffcb6b;
        font-size: 1.2rem;
    }
    
    /* Movie card */
    .movie-card {
        display: flex;
        background: rgba(22, 33, 62, 0.7);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border-left: 4px solid #ff6b6b;
    }
    
    .movie-info {
        padding: 1rem;
    }
    
    .movie-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #ff6b6b;
        margin-bottom: 0.5rem;
    }
    
    .movie-genres {
        font-size: 0.9rem;
        color: #a0a0a0;
        margin-bottom: 0.5rem;
    }
    
    .movie-rating {
        display: flex;
        align-items: center;
        font-size: 0.9rem;
    }
    
    .rating-value {
        font-weight: 600;
        color: #ffcb6b;
        margin-right: 0.5rem;
    }
    
    .rating-count {
        color: #a0a0a0;
        font-size: 0.8rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        color: #a0a0a0;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .title {
            font-size: 2.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.markdown('<h1 class="title">🎬 CineMatcher</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover your next favorite movie with our smart recommendation engine</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Find Your Next Movie</h3>', unsafe_allow_html=True)
    
    # Movie selection
    movie_list = sorted(movies['title'].unique())
    selected_movie = st.selectbox("Select a movie you enjoyed", movie_list, index=100)
    
    # Get movie details
    selected_movie_info = movies[movies['title'] == selected_movie].iloc[0]
    
    # Display selected movie details
    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-info">
            <div class="movie-title">{selected_movie}
