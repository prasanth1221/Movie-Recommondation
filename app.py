import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('archive (5)/movies.csv')
ratings = pd.read_csv('archive (5)/ratings.csv')

movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']

movies = movies.merge(movie_stats, on='movieId', how='left')
movies['avg_rating'] = movies['avg_rating'].round(2)

movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])


cosine_sim = cosine_similarity(tfidf_matrix)


def recommend_movie(title):
    if title not in movies['title'].values:
        return pd.DataFrame()
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres', 'avg_rating', 'rating_count']].iloc[movie_indices]


st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

st.markdown("#### Select a movie to get similar recommendations:")

movie_list = movies['title'].sort_values().unique()
selected_movie = st.selectbox("Choose a movie you like:", movie_list)

if st.button("Recommend"):
    recommendations = recommend_movie(selected_movie)
    if recommendations.empty:
        st.error("No recommendations found!")
    else:
        st.success(f"Top 5 similar movies to **{selected_movie}**:")
        st.dataframe(recommendations.reset_index(drop=True), use_container_width=True)
