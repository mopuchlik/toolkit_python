import pandas as pd
import numpy as np
import os
import time

# import class
from MovieRecommender import MovieRecommender

cwd = "/home/michal/silos/Dropbox/programowanie/python_general/projects/recommender system collaborative filtering"

# cwd = "D:/Dropbox/programowanie/python_general"

# TODO: for some reason does not work in Linux
# cwd = os.path.dirname(os.path.abspath("__file__"))

os.chdir(cwd)
print("Current working directory: {0}".format(os.getcwd()))


# load dataset
df = pd.read_csv("data 1.csv").drop(columns="Unnamed: 0")
df.shape

# initialize
recommender = MovieRecommender(df)

# data prep
recommender.data_prep()

# remove duplicates
recommender.remove_duplicates()
recommender.df.shape

n_user = 50
start_time = time.time()
for i in range(1, n_user):
    recommended_movies = recommender.recommend_movies_knn(
        user_id=i, k=6, num_recommendations=5
    )
print(
    f"--- KNN method --- Check for {n_user} users: {round(time.time() - start_time, 2)} seconds"
)

start_time = time.time()
model_svd = recommender.svd_model_fit()
for i in range(1, n_user):
    recommended_movies_svd = recommender.recommend_movies_svd(
        user_id=i, model=model_svd, num_recommendations=5
    )
print(
    f"--- SVD method --- Check for {n_user} users: {round(time.time() - start_time, 2)} seconds"
)
