import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies_columns = ["MovieID", "Title", "Genres"]
ratings_columns = ["UserID", "MovieID", "Rating", "Timestamp"]

movies = pd.read_csv("movies.dat", sep=r"::", engine="python", names=movies_columns, encoding="ISO-8859-1")
ratings = pd.read_csv("ratings.dat", sep=r"::", engine="python", names=ratings_columns, encoding="ISO-8859-1")

# Ensure MovieID is an integer
movies['MovieID'] = movies['MovieID'].astype(int)
ratings['MovieID'] = ratings['MovieID'].astype(int)

# Remove the Timestamp column from ratings (not needed for collaborative filtering)
ratings = ratings.drop(columns=['Timestamp'])

# Create a user-item matrix with user as rows and movie as columns
user_item_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating')

# Fill NaN values with 0 (assuming unrated movies as 0)
user_item_matrix = user_item_matrix.fillna(0)

# Calculate the cosine similarity between movies
movie_similarity = cosine_similarity(user_item_matrix.T)

# Convert the similarity matrix to a DataFrame for easier interpretation
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Function to recommend movies based on title
def recommend_movies_by_title(movie_title, top_n=10):
    # Find the movie by title
    movie_entry = movies[movies['Title'].str.contains(movie_title, case=False, na=False)]

    if movie_entry.empty:
        return "Movie not found! Please try another title."

    movie_id = movie_entry.iloc[0]['MovieID']

    if movie_id not in movie_similarity_df.index:
        return "Movie not found in similarity matrix. Try another movie."

    # Get top N similar movies
    similar_movies = movie_similarity_df[movie_id].sort_values(ascending=False).iloc[1:top_n+1]

    recommended_movies = movies[movies['MovieID'].isin(similar_movies.index)]['Title'].tolist()

    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")

# Movie title input
user_input = st.text_input("Enter a Movie Title:")

# Recommendations button
if st.button("Get Recommendations") and user_input:
    recommendations = recommend_movies_by_title(user_input, top_n=10)

    if isinstance(recommendations, str):  # Error message
        st.error(recommendations)
    else:
        st.subheader("Top 10 Recommended Movies:")
        for movie in recommendations:
            st.write(f"🎥 {movie}")


