# %% imports
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# ### Get the current working directory (adjust)
cwd = "/home/michal/silos/Dropbox/programowanie/python_general/projects/time_series_with_groups//"
os.chdir(cwd)
# cwd = os.getcwd()
print("Current working directory: {0}".format(os.getcwd()))

# %%
# loading a csv file
path_yX_train = "yX_train_clean.csv"
yX_train = pd.read_csv(
    path_yX_train,
)
print(f"number of NAs of original set after cleanup: {yX_train.isna().sum().sum()}")

# %%
df = yX_train.copy()
df["date"] = pd.to_datetime(df["date"])
df.set_index(["date"])


# %% functions
def decompose_group_plot(group, col_name):
    group = group.set_index("date")  # Set date as index
    seasonal_decompose(group[col_name], model="additive", period=4).plot()
    return 0


def decompose_group(group, col_name):
    group = group.set_index("date")  # Set date as index
    result = seasonal_decompose(
        group[col_name], model="additive", period=4
    )  # Adjust period based on seasonality
    return {
        "observed": result.observed,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": result.resid,
    }


## print only one key for simplicity
# df["key"].unique()
# df = df.loc[df["key"] == 10873]

# %%
for key, group in df.groupby("key"):
    # print(f"Key={key}")
    decompose_group_plot(group, "y")
    decompose_group_plot(group, "chann_sum")

# %%
