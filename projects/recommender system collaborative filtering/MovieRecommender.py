import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


class MovieRecommender:
    def __init__(
        self,
        df: pd.DataFrame,
        user_id: str = "user_id",
        movie_id: str = "movie_id",
        rating: str = "rating",
        timestamp: str = "timestamp",
    ):
        """
        Initialize MovieRecommender with:

        Args:
        - df: input dataset
        - user_id: column name for user_id (default "user_id")
        - movie_id: column name for movie_id (default "movie_id")
        - rating: column name for rating (default "rating")
        - timestamp: column name for timestamp (default "timestamp")
        """

        self.df = df
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating = rating
        self.timestamp = timestamp

    def data_prep(self) -> pd.DataFrame:
        """
        This function takes the original dataset, preserves only necessary columns
        and renames them to the standard names used further on:
        'user_id', 'movie_id', 'rating' and 'timestamp'.
        """

        # ratings_df = self.df.rename(
        #     columns={
        #         self.user_id: "user_id",
        #         self.movie_id: "movie_id",
        #         self.rating: "rating",
        #         self.timestamp: "timestamp",
        #     }
        # )

        self.df = self.df[[self.user_id, self.movie_id, self.rating, self.timestamp]]

        return self.df

    def remove_duplicates(self) -> pd.DataFrame:
        """
        This function removes duplicated entries wrt to columns,
        row with the highest timestamp is preserved as the most current.

        Parameters:
        - df (pd.DataFrame): DataFrame with columns 'user_id', 'movie_id', 'timestamp'.

        Returns:
        - pd.DataFrame without duplicates
        """

        self.df = self.df.loc[
            self.df.groupby(
                [
                    self.user_id,
                    self.movie_id,
                ]
            )[self.timestamp].idxmax()
        ].reset_index(drop=True)

        return self.df

    def recommend_movies_knn(
        self, user_id: int, k: int = 5, num_recommendations: int = 5
    ) -> list:
        """
        Recommends movies to a user user_id using weighted k-NN (User-Based Collaborative Filtering).

        Parameters:
        - user_id (int): Target user ID.
        - self.df (pd.DataFrame): DataFrame with [self.user_id, self.movie_id, self.rating].
        - k (int): Number of nearest neighbors.
        - num_recommendations (int): Number of movies to recommend.

        Returns:
        - List of recommended movie_ids.
        """
        # Create User-Movie Ratings Matrix
        user_movie_matrix = self.df.pivot(
            index=self.user_id, columns=self.movie_id, values=self.rating
        ).fillna(0)

        if user_id not in user_movie_matrix.index:
            # Cold Start: Recommend movies with highest average rating
            avg_ratings = self.df.groupby(self.movie_id)[self.rating].mean()
            return (
                avg_ratings.sort_values(ascending=False)
                .head(num_recommendations)
                .index.tolist()
            )

        # Compute User Similarity Matrix (Cosine Similarity)
        user_similarity = cosine_similarity(user_movie_matrix)

        # Get the row index of the target user
        user_index = user_movie_matrix.index.get_loc(user_id)

        # Get the k most similar users (excluding self)
        similar_users_indices = np.argsort(user_similarity[user_index])[::-1][1 : k + 1]
        similar_users = user_movie_matrix.iloc[similar_users_indices]

        # Get similarity scores of the k nearest neighbors
        similarity_scores = user_similarity[user_index][similar_users_indices]

        # Compute Weighted Average of Ratings (Weighted by Similarity)
        weighted_ratings = (
            similar_users.T * similarity_scores
        ).T  # Multiply each row by its similarity score
        movie_scores = weighted_ratings.sum(axis=0) / (
            np.abs(similarity_scores).sum() + 1e-9
        )  # Avoid division by zero

        # Get movies the user has already rated
        rated_movies = user_movie_matrix.loc[user_id][
            user_movie_matrix.loc[user_id] > 0
        ].index

        # Recommend top N movies that the user has NOT rated yet
        recommendations = (
            movie_scores.drop(index=rated_movies, errors="ignore")
            .sort_values(ascending=False)
            .head(num_recommendations)
        )

        return recommendations.index.tolist()  # List of recommended movie_ids

    def svd_model_fit(
        self,
        *args,
        **kwargs,
    ) -> list:
        """
        Fits the SVD model to provided dataset (Collaborative Filtering).

        Parameters:
        - self.df (pd.DataFrame): DataFrame with [self.user_id, self.movie_id, self.rating].
        - **kwargs: arguments passed to SVD model creator
            - n_factors (int): The number of factors, default is 100.
            - reg_all (float): The regularization term for all parameters, default is 0.2.
            - etc.
        Returns:
        - SVD model object of type surprise.prediction_algorithms.matrix_factorization.SVD.
        """

        # Define a Reader format for Surprise
        reader = Reader(
            rating_scale=(self.df[self.rating].min(), self.df[self.rating].max())
        )

        # Load data into Surprise dataset
        data = Dataset.load_from_df(
            self.df[[self.user_id, self.movie_id, self.rating]], reader
        )
        full_trainset = data.build_full_trainset()

        # Train SVD model
        model = SVD(*args, **kwargs)
        model.fit(full_trainset)

        return model

    def recommend_movies_svd(
        self, user_id, model, num_recommendations: int = 5
    ) -> list:
        """
        Recommends movies to a user using SVD (Collaborative Filtering) based on
        a fitted model.

        Parameters:
        - self.df (pd.DataFrame): DataFrame with [self.user_id, self.movie_id, self.rating].
        - self.user_id (int): Target user ID.
        - model (surprise.prediction_algorithms.matrix_factorization.SVD): fitted SVD model
        - num_recommendations (int): Number of movies to recommend.

        Returns:
        - List of recommended movie_ids.
        """

        # Get all unique movie IDs
        all_movies = self.df[self.movie_id].unique()

        # Check if the user exists in the dataset
        if user_id not in self.df[self.user_id].values:
            # Cold Start: Recommend movies with the highest average rating
            top_movies = (
                self.df.groupby(self.movie_id)[self.rating]
                .mean()
                .sort_values(ascending=False)
                .head(num_recommendations)
                .index.tolist()
            )
            return top_movies

        # Get movies the user has already rated
        rated_movies = self.df[self.df[self.user_id] == user_id][self.movie_id].values

        # Get movies the user has NOT rated
        unrated_movies = [m for m in all_movies if m not in rated_movies]

        # Predict ratings for unrated movies
        predicted_ratings = [
            (movie, model.predict(user_id, movie).est) for movie in unrated_movies
        ]

        # Sort by highest predicted rating
        top_recommendations = sorted(
            predicted_ratings, key=lambda x: x[1], reverse=True
        )[:num_recommendations]

        return [movie for movie, rating in top_recommendations]
