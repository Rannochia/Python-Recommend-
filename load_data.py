import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Define column names based on MovieLens documentation
ratings_columns = ["UserID", "MovieID", "Rating", "Timestamp"]
movies_columns = ["MovieID", "Title", "Genres"]

# Load datasets
ratings = pd.read_csv("ratings.dat", sep="::", engine="python", names=ratings_columns, encoding="ISO-8859-1")
movies = pd.read_csv("movies.dat", sep="::", engine="python", names=movies_columns, encoding="ISO-8859-1")

# Convert MovieID to integer (ensures compatibility)
movies["MovieID"] = movies["MovieID"].astype(int)
ratings["MovieID"] = ratings["MovieID"].astype(int)

# Remove the Timestamp column from ratings (not needed)
ratings = ratings.drop(columns=['Timestamp'])

# One-hot encode the genres
genres = movies['Genres'].str.get_dummies('|')
movies = pd.concat([movies, genres], axis=1)
movies = movies.drop(columns=['Genres'])  # Drop original column

# Create a user-item matrix
user_item_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)

# Calculate cosine similarity between movies based on user ratings
movie_similarity_df = pd.DataFrame(
    cosine_similarity(user_item_matrix.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# Calculate genre similarity
genre_similarity_df = pd.DataFrame(
    cosine_similarity(movies[genres.columns]),
    index=movies['MovieID'],
    columns=movies['MovieID']
)

# Normalize and combine similarities (weighting: 40% ratings, 60% genres)
combined_similarity = 0.2 * (movie_similarity_df / movie_similarity_df.max().max()) + \
                      0.8 * (genre_similarity_df / genre_similarity_df.max().max())

# Ensure index and columns are integers
combined_similarity.index = combined_similarity.index.astype(int)
combined_similarity.columns = combined_similarity.columns.astype(int)


# Recommendation function
def recommend_movies_by_title(movie_title, top_n=10):
    # Find Movie ID
    movie_entry = movies[movies['Title'].str.contains(movie_title, case=False, na=False)]

    if movie_entry.empty:
        return "Movie not found! Please try another title."

    movie_id = movie_entry.iloc[0]['MovieID']

    # Check if movie exists in similarity matrix
    if movie_id not in combined_similarity.index:
        print(f"DEBUG: Movie ID {movie_id} not in similarity matrix. Available IDs: {list(combined_similarity.index)[:10]}")
        return "Movie not found in similarity matrix. Try another movie."

    # Get top N similar movies
    similar_movies = combined_similarity[movie_id].sort_values(ascending=False).iloc[1:top_n+1]

    # Get movie titles
    recommended_movies = movies[movies['MovieID'].isin(similar_movies.index)][['Title']]

    return recommended_movies['Title'].tolist()


# Tkinter Popup Function
def show_popup():
    root = tk.Tk()
    root.withdraw()  # Hide main window

    movie_title = simpledialog.askstring("Movie Input", "Enter a Movie Title:")

    if movie_title:
        recommendations = recommend_movies_by_title(movie_title)

        if isinstance(recommendations, str):  # If error message
            messagebox.showerror("Error", recommendations)
        else:
            messagebox.showinfo("Top 10 Recommendations", "\n".join(recommendations))


# Run popup only when setup is complete
show_popup()
