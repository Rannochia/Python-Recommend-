import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies_columns = ["MovieID", "Title", "Genres", "Year"]
ratings_columns = ["UserID", "MovieID", "Rating", "Timestamp"]

movies = pd.read_csv("movies.dat", sep=r"::", engine="python", names=movies_columns, encoding="ISO-8859-1")
ratings = pd.read_csv("ratings.dat", sep=r"::", engine="python", names=ratings_columns, encoding="ISO-8859-1")

# Ensure MovieID is an integer
movies['MovieID'] = movies['MovieID'].astype(int)
ratings['MovieID'] = ratings['MovieID'].astype(int)

# Remove the Timestamp column from ratings
ratings = ratings.drop(columns=['Timestamp'])

# One-hot encode the genres for content-based similarity
genres = movies['Genres'].str.get_dummies('|')
movies = pd.concat([movies, genres], axis=1)
movies = movies.drop(columns=['Genres'])  # Drop original column

# Extract the year from the title (for completeness, though not used here)
movies['Year'] = movies['Title'].str.extract(r'(\(\d{4}\))').replace({r'(\(|\))': ''}, regex=True)

# Create a user-item matrix
user_item_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating')
user_item_matrix = user_item_matrix.fillna(0)

# Calculate cosine similarity based on user ratings
movie_similarity = cosine_similarity(user_item_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Calculate cosine similarity based on genres
genre_similarity = cosine_similarity(movies[genres.columns])
genre_similarity_df = pd.DataFrame(genre_similarity, index=movies['MovieID'], columns=movies['MovieID'])

# Normalize and combine similarities
movie_similarity_df_normalized = movie_similarity_df / movie_similarity_df.max().max()
genre_similarity_df_normalized = genre_similarity_df / genre_similarity_df.max().max()
combined_similarity = 0.6 * movie_similarity_df_normalized + 0.4 * genre_similarity_df_normalized

# Recommend movies based on title only
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

# Autocomplete-style dropdown in a cleaner layout
movie_titles = sorted(movies['Title'].unique())

col1, col2 = st.columns([4, 3])
with col1:
    user_input = st.selectbox("Select a Movie Title:", movie_titles)
with col2:
    if st.markdown(
        "<div style='margin-top: 32px;'><button style='width:100%; height:40px;'>🎯 Get Recommendations</button></div>",
        unsafe_allow_html=True
        
    ):
        trigger = True

num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

# Show recommendations
if trigger and user_input:
    recommendations = recommend_movies_by_title(user_input, top_n=num_recommendations)

    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.subheader("Top 10 Recommended Movies:")
        for movie in recommendations:
            st.write(f"🎥 {movie}")

