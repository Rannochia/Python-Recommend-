import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies_columns = ["MovieID", "Title", "Genres", "Year"]
ratings_columns = ["UserID", "MovieID", "Rating", "Timestamp"]

try:
    movies = pd.read_csv("movies.dat", sep=r"::", engine="python", names=movies_columns, encoding="ISO-8859-1")
    ratings = pd.read_csv("ratings.dat", sep=r"::", engine="python", names=ratings_columns, encoding="ISO-8859-1")
except FileNotFoundError:
    st.error("❌ Could not find 'movies.dat' or 'ratings.dat'. Make sure the files are in the correct location.")
    st.stop()

# Ensure MovieID is an integer
movies['MovieID'] = movies['MovieID'].astype(int)
ratings['MovieID'] = ratings['MovieID'].astype(int)

# Remove Timestamp
ratings = ratings.drop(columns=['Timestamp'])

# One-hot encode genres
genres = movies['Genres'].str.get_dummies('|')
movies = pd.concat([movies, genres], axis=1).drop(columns=['Genres'])

# Extract year from title
movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)').astype('Int64')

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)

# Cosine similarity: ratings
movie_similarity = cosine_similarity(user_item_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Cosine similarity: genres
genre_similarity = cosine_similarity(movies[genres.columns])
genre_similarity_df = pd.DataFrame(genre_similarity, index=movies['MovieID'], columns=movies['MovieID'])

# Normalize similarities with safety check
movie_max = movie_similarity_df.to_numpy().max()
genre_max = genre_similarity_df.to_numpy().max()

if movie_max == 0 or genre_max == 0:
    st.error("❌ Similarity matrices have zero max values. Normalization failed.")
    st.stop()

movie_similarity_df_normalized = movie_similarity_df / movie_max
genre_similarity_df_normalized = genre_similarity_df / genre_max
combined_similarity = 0.6 * movie_similarity_df_normalized + 0.4 * genre_similarity_df_normalized

# Recommendation logic
def recommend_movies_by_title(movie_title, top_n=10):
    movie_entry = movies[movies['Title'] == movie_title]

    if movie_entry.empty:
        return "Movie not found! Please try another title."

    movie_id = movie_entry.iloc[0]['MovieID']

    if movie_id not in combined_similarity.index:
        return "Movie not found in similarity matrix. Try another movie."

    similar_movies = combined_similarity[movie_id].sort_values(ascending=False).iloc[1:top_n+1]
    recommended_movies = movies[movies['MovieID'].isin(similar_movies.index)]['Title'].tolist()

    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_titles = sorted(movies['Title'].unique())

col1, col2 = st.columns([4, 2])
with col1:
    user_input = st.selectbox("Select a Movie Title:", movie_titles)
with col2:
    # Create a form to capture the button click
    with st.form("recommendation_form", clear_on_submit=False):
        trigger = st.form_submit_button(
            label="🎯 Get Recommendations"
        )

        # Inject CSS to style the button
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #4CAF50;
                color: white;
                height: 40px;
                width: 100%;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                transition: 0.3s;
            }
            div.stButton > button:first-child:hover {
                background-color: #45a049;
                transform: scale(1.02);
            }
            </style>
        """, unsafe_allow_html=True)


num_recommendations = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

# Show recommendations
if trigger and user_input:
    recommendations = recommend_movies_by_title(user_input, top_n=num_recommendations)

    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.subheader(f"Top {num_recommendations} Recommended Movies:")
        for movie in recommendations:
            st.write(f"🎥 {movie}")
