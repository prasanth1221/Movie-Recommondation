import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up page
st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="wide")

# Load data
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    movies = pd.read_csv('/acrhive (5)/movies.csv')
    ratings = pd.read_csv('/acrhive (5)/ratings.csv')
    return movies, ratings

movies, ratings = load_data()

# Prepare movie stats
movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
movies = movies.merge(movie_stats, on='movieId', how='left')
movies['avg_rating'] = movies['avg_rating'].round(2)
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# TF-IDF Matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Hybrid scoring
def hybrid_score(sim_scores, movies_df):
    result = []
    for idx, score in sim_scores:
        rating_count = movies_df.iloc[idx]['rating_count']
        avg_rating = movies_df.iloc[idx]['avg_rating']
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

# UI
with st.container():
    st.markdown("<h1 style='text-align: center;'>🎥 Smart Movie Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Select a movie and get top 5 similar recommendations based on genres, ratings, and popularity!</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])

    with col1:
        movie_list = sorted(movies['title'].unique())
        selected_movie = st.selectbox("Choose a movie you like:", movie_list)

        if st.button("🔍 Recommend Movies"):
            recommendations = recommend_movie(selected_movie)
            if recommendations.empty:
                st.error("No recommendations found. Please try another movie.")
            else:
                st.success(f"🎯 Top 5 movies similar to **{selected_movie}**")
                st.dataframe(recommendations.reset_index(drop=True), use_container_width=True)

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/744/744922.png", width=300)
