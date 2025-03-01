# imports
import pandas as pd
import numpy as np
import os
import time

# internal functions
import functions_recommend_system as recfun

cwd = "/home/michal/silos/Dropbox/programowanie/python_general/projects/recommender system collaborative filtering"

# cwd = "D:/Dropbox/programowanie/python_general"

# TODO: for some reason does not work in Linux
# cwd = os.path.dirname(os.path.abspath("__file__"))

os.chdir(cwd)
print("Current working directory: {0}".format(os.getcwd()))


# load dataset
df = pd.read_csv("data 1.csv").drop(columns="Unnamed: 0")

# data prep
df = recfun.data_prep(df=df)

# remove duplicates
df_nodup = recfun.remove_duplicates(ratings_df=df)

n_user = 50
start_time = time.time()
for i in range(1, n_user):
    recommended_movies = recfun.recommend_movies_knn(
        user_id=i, ratings_df=df_nodup, k=6, num_recommendations=5
    )
print(
    f"--- KNN method --- Check for {n_user} users: {round(time.time() - start_time, 2)} seconds"
)

start_time = time.time()
model_svd = recfun.svd_model_fit(ratings_df=df_nodup)
for i in range(1, n_user):
    recommended_movies_svd = recfun.recommend_movies_svd(
        user_id=1, ratings_df=df, model=model_svd, num_recommendations=5
    )
print(
    f"--- SVD method --- Check for {n_user} users: {round(time.time() - start_time, 2)} seconds"
)
