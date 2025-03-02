# %% imports
import pandas as pd

# import numpy as np
import os
import time
from sklearn.metrics.pairwise import cosine_similarity

# internal functions
import functions_recommend_system as recfun

# %% current directory
# ### Get the current working directory (adjust)
cwd = "/home/michal/silos/Dropbox/programowanie/python_general/projects/recommender system collaborative filtering"

# cwd = "D:/Dropbox/programowanie/python_general"

# TODO: for some reason does not work in Linux
# cwd = os.path.dirname(os.path.abspath("__file__"))

os.chdir(cwd)
print("Current working directory: {0}".format(os.getcwd()))


# %% load dataset
df = pd.read_csv("data 1.csv").drop(columns="Unnamed: 0")
df.head()

# %% data preparation
df = recfun.data_prep(df=df)

# %% check for NAs
df.isna().sum()

# %% check duplicates
boolean = df.duplicated(subset=["user_id", "movie_id"])
dups = df[boolean]

print(f"There are {dups.shape[0]}/{df.shape[0]} duplicated entries")

# dups.to_csv("dups.csv", index=False)

# %% checks

# example there are duplicates
df.loc[(df["user_id"] == 614) & (df["movie_id"] == 100)]

# %% check for duplicates again

df_nodup = recfun.remove_duplicates(ratings_df=df)
df_nodup.shape
df_nodup.loc[(df_nodup["user_id"] == 614) & (df_nodup["movie_id"] == 100)]

# %% KNN example

recommended_movies_knn = recfun.recommend_movies_knn(
    user_id=1, ratings_df=df_nodup, k=6, num_recommendations=5
)
print(f"--- KNN recommended movies: {recommended_movies_knn}")

# %% KNN cold start case
recommended_movies_knn = recfun.recommend_movies_knn(
    user_id=2000, ratings_df=df_nodup, k=6, num_recommendations=5
)
print(f"--- KNN recommended movies (cold start): {recommended_movies_knn}")
# %% SVD example

model_svd = recfun.svd_model_fit(ratings_df=df_nodup, n_factors=50)
recommended_movies_svd = recfun.recommend_movies_svd(
    user_id=1, ratings_df=df_nodup, model=model_svd, num_recommendations=5
)
print(f"--- SVD recommended movies: {recommended_movies_svd}")

# %% SVD cold start case

recommended_movies_svd = recfun.recommend_movies_svd(
    user_id=2000, ratings_df=df_nodup, model=model_svd, num_recommendations=5
)
print(f"--- SVD recommended movies (cold start): {recommended_movies_svd}")

# %% some tests

x = pd.DataFrame(
    {
        "user_id": [1, 1, 2, 2, 3, 3, 3],
        "movie_id": [10, 20, 10, 30, 10, 20, 50],
        "rating": [4, 5, 3, 2, 4, 5, 5],
        "timestamp": [1000, 2000, 1500, 2500, 3000, 4000, 5000],
    }
)
x

# %%
recommended_movies_knn = recfun.recommend_movies_knn(
    user_id=1, ratings_df=x, k=2, num_recommendations=2
)
print(f"--- KNN recommended movies: {recommended_movies_knn}")

# %%
model_svd = recfun.svd_model_fit(ratings_df=x, n_factors=2)
recommended_movies_svd = recfun.recommend_movies_svd(
    user_id=1, ratings_df=x, model=model_svd, num_recommendations=2
)
print(f"--- SVD recommended movies: {recommended_movies_svd}")
# %%
