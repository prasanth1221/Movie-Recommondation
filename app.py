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
    
    /* Results container */
    .recommendations-container {
        background: rgba(22, 33, 62, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-top: 1.5rem;
        border-left: 4px solid #4CAF50;
    }
    
    .recommendations-title {
        font-weight: 600;
        font-size: 1.2rem;
        color: #4CAF50;
        margin-bottom: 1rem;
        text-align: center;
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
        width: 100%;
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
    
    /* Recommended movie cards - full width with better styling */
    .recommendation-card {
        background: rgba(22, 33, 62, 0.9);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        display: flex;
        align-items: center;
    }
    
    .recommendation-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4CAF50;
        margin-right: 1rem;
        min-width: 40px;
        text-align: center;
    }
    
    .recommendation-info {
        flex-grow: 1;
    }
    
    .recommendation-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #ff6b6b;
        margin-bottom: 0.3rem;
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

# Main content - First layout for selection
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
            <div class="movie-title">{selected_movie}</div>
            <div class="movie-genres">{selected_movie_info['genres']}</div>
            <div class="movie-rating">
                <span class="rating-value">⭐ {selected_movie_info['avg_rating'] if not pd.isna(selected_movie_info['avg_rating']) else 'No rating'}</span>
                <span class="rating-count">({int(selected_movie_info['rating_count']) if not pd.isna(selected_movie_info['rating_count']) else 'No'} ratings)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation button
    recommend_button = st.button("Get Recommendations", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>How It Works</h3>', unsafe_allow_html=True)
    
    # Create tabs for the explanation
    tab1, tab2, tab3 = st.tabs(["About", "Features", "Technology"])
    
    with tab1:
        st.markdown("""
        **CineMatcher** is an intelligent movie recommendation system that helps you discover new films 
        based on your preferences. Simply select a movie you enjoyed, and our algorithm will find similar 
        films that match your taste.
        
        Our recommendations are based on a combination of:
        - Genre similarity
        - User ratings
        - Movie popularity
        """)
    
    with tab2:
        st.markdown("""
        ✨ **Smart Recommendations** - Get personalized movie suggestions based on what you like
        
        🔍 **Genre Matching** - Find films with similar themes and styles
        
        ⭐ **Rating-Based** - Recommendations include highly-rated films that other users have enjoyed
        
        🎯 **Popularity Boost** - Discover both hidden gems and popular hits
        """)
    
    with tab3:
        st.markdown("""
        CineMatcher uses advanced machine learning techniques to provide the best recommendations:
        
        - **TF-IDF Vectorization** - Converts movie genres into numerical vectors
        - **Cosine Similarity** - Measures how similar movies are to each other
        - **Hybrid Scoring** - Combines content-based similarity with popularity metrics
        - **Optimized Algorithms** - Fast processing for quick recommendations
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Full-width recommendations section - only appears when recommendations are made
if recommend_button:
    with st.spinner("Finding perfect matches for you..."):
        recommendations = recommend_movie(selected_movie)
    
    if recommendations.empty:
        st.error("No recommendations found. Please try another movie.", icon="🚫")
    else:
        # Full-width container for recommendations
        st.markdown("""
        <div class="recommendations-container">
            <div class="recommendations-title">✨ Top Recommendations for You</div>
        """, unsafe_allow_html=True)
        
        # Display recommendations with ranking numbers
        for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-number">{i}</div>
                <div class="recommendation-info">
                    <div class="recommendation-title">{movie['title']}</div>
                    <div class="movie-genres">{movie['genres']}</div>
                    <div class="movie-rating">
                        <span class="rating-value">⭐ {movie['avg_rating'] if not pd.isna(movie['avg_rating']) else 'No rating'}</span>
                        <span class="rating-count">({int(movie['rating_count']) if not pd.isna(movie['rating_count']) else 'No'} ratings)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Top movies stats (optional section)
with st.expander("Explore Movie Statistics"):
    top_movies = movies.sort_values('rating_count', ascending=False).head(10)
    
    st.markdown("<h4>Most Popular Movies</h4>", unsafe_allow_html=True)
    for _, movie in top_movies.iterrows():
        st.markdown(f"""
        <div class="movie-card" style="margin-bottom: 0.5rem">
            <div class="movie-info">
                <div class="movie-title">{movie['title']}</div>
                <div class="movie-rating">
                    <span class="rating-value">⭐ {movie['avg_rating'] if not pd.isna(movie['avg_rating']) else 'No rating'}</span>
                    <span class="rating-count">({int(movie['rating_count']) if not pd.isna(movie['rating_count']) else 'No'} ratings)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('Created by Prasanthkumar | Powered by Streamlit & Scikit-Learn', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
