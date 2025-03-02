import pytest
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from functions_recommend_system import (
    data_prep,
    remove_duplicates,
    recommend_movies_knn,
    svd_model_fit,
    recommend_movies_svd,
)


# Sample Data
@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 3],
            "movie_id": [10, 20, 10, 30, 10, 20, 50],
            "rating": [4, 5, 3, 2, 4, 5, 5],
            "timestamp": [1000, 2000, 1500, 2500, 3000, 4000, 5000],
        }
    )


# Test data_prep
def test_data_prep(sample_df):
    # change columns name
    sample_df = sample_df.rename(columns={"user_id2": "user_id"})
    # add a column
    sample_df["test"] = 1
    # change column name to non-standard
    result = data_prep(sample_df, user_id="user_id2")
    # check whether columns were changed
    assert list(result.columns) == ["user_id", "movie_id", "rating", "timestamp"]
    assert len(result.columns) == 4


# Test remove_duplicates
def test_remove_duplicates(sample_df):
    # create a duplicate
    sample_df.loc[1] = [1, 10, 5, 3000]
    result = remove_duplicates(sample_df)
    assert len(result) == 6
    assert result.loc[result["user_id"] == 1, "rating"].tolist() == [5]


# Test recommend_movies_knn
def test_recommend_movies_knn_coldstart(sample_df):
    result = recommend_movies_knn(4, sample_df, k=2, num_recommendations=2)
    assert result == [20, 50]


# Test recommend_movies_knn, user_id not in a set (cold start)
def test_recommend_movies_knn(sample_df):
    result = recommend_movies_knn(1, sample_df, k=2, num_recommendations=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(x, (int, np.int64)) for x in result)
    assert result == [50, 30]


# Test svd_model_fit
def test_svd_model_fit(sample_df):
    model = svd_model_fit(sample_df, n_factors=10, reg_all=0.1)
    assert isinstance(model, SVD)


# Test recommend_movies_svd
def test_recommend_movies_svd(sample_df):
    model = svd_model_fit(sample_df)
    result = recommend_movies_svd(
        user_id=1, ratings_df=sample_df, model=model, num_recommendations=2
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(x, (int, np.int64)) for x in result)
    assert result == [50, 30]


# Test recommend_movies_svd, user_id not in a set (cold start)
def test_recommend_movies_svd_coldstart(sample_df):
    model = svd_model_fit(sample_df)
    result = recommend_movies_svd(
        sample_df, user_id=4, model=model, num_recommendations=3
    )
    assert result == [20, 50, 10]
