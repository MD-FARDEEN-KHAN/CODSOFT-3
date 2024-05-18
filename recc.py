# Import necessary libraries
import pandas as pd
import numpy as np
data = {
    'movie_id': [101,102,103],
    'title': ['jumanji', 'spider_man', 'super_man'],
    'genres': ['Action|Adventure', 'Comedy|drama', 'Action|Sci-Fi']
}
df_movies = pd.DataFrame(data)

# Create a feature vector for each movie (using genres as features)
genres_matrix = df_movies['genres'].str.get_dummies(sep='|')

# Calculate cosine similarity between movies
movie_similarity = np.dot(genres_matrix, genres_matrix.T)
norms = np.linalg.norm(genres_matrix, axis=1)
cosine_similarity = movie_similarity / np.outer(norms, norms)

# Get recommendations for a specific movie (e.g., Movie A)
target_movie_index = 0
similar_movies_indices = np.argsort(cosine_similarity[target_movie_index])[::-1]
top_n_recommendations = similar_movies_indices[1:6]  # Exclude the target movie

# Print recommended movie titles
print("Top recommendations for jumanji:")
for idx in top_n_recommendations:
    print(df_movies.loc[idx, 'title'])

# Output:
# Top recommendations for Movie A:
# Movie C
# Movie B
