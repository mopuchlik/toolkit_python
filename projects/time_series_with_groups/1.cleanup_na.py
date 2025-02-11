# %% imports
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# ### Get the current working directory (adjust)
cwd = "/home/michal/Dropbox/programowanie/python_general/projects/time_series_with_groups/"
os.chdir(cwd)
# cwd = os.getcwd()
print("Current working directory: {0}".format(os.getcwd()))

# %%
# loading a csv file
# loading a csv file
path_X_train = "X_train_T2.csv"
path_y_train = "y_train_T2.csv"

X_train = pd.read_csv(path_X_train, sep=";", decimal=",")
y_train = pd.read_csv(path_y_train, sep=";", decimal=",")

# cast date to datetime
X_train["date"] = pd.to_datetime(X_train["date"])
y_train["date"] = pd.to_datetime(y_train["date"])

# merged y and X
yX_train = pd.merge(y_train, X_train, on=["key", "date"])

# key as int -- it is used by a model used further on
yX_train["key"] = yX_train["key"].astype("int")

# list of original features
reg_list = [col for col in X_train.columns if "channel_" in col]

# row sums for all channels
yX_train["channel_sum"] = yX_train[reg_list].sum(axis=1)

# %% check NAs date vs channel for stacked keys
msno.matrix(X_train)

# %% check NAs date vs key for each channel

for i, _ in enumerate(reg_list):

    fig = plt.figure()
    axis = fig.add_axes([0, 0, 2, 1])
    axis.set_title(f"NAs for {reg_list[i]}")

    X_train_p = pd.pivot_table(
        data=X_train, index="date", columns="key", values=reg_list[i]
    )
    # matrix is better but does not work due to some version conflicts
    msno.matrix(X_train_p, ax=axis)


# %% check NAs date vs key for each channel
key_list = X_train["key"].unique()

X_train_c = X_train.copy()
for i, _ in enumerate(key_list):

    fig = plt.figure()
    axis = fig.add_axes([0, 0, 2, 1])
    axis.set_title(f"NAs for {key_list[i]}")

    X_train_key = X_train_c.loc[X_train_c["key"] == key_list[i]].drop(columns="key")
    # matrix is better but does not work due to some version conflicts
    msno.matrix(X_train_key, ax=axis)

# %% remove NaN rows in y and X
# in X replace NaN with zeroes

# original shape of y and X
print(f"original shape of y and X: {yX_train.shape}")

# Number of NAs in y
print(f"number of NAs in y: {yX_train["y"].isna().sum()}")
# interpolate y
yX_train["y"] = yX_train["y"].interpolate(method="linear")
print(f"number of NAs in y after: {yX_train["y"].isna().sum()}")

print(f"number of NAs of original set: {yX_train.isna().sum().sum()}")
# in X replace NA with 0s as there is no pattern in X's NAs
yX_train = yX_train.fillna(0)
print(f"number of NAs of original set after cleanup: {yX_train.isna().sum().sum()}")

# %%
yX_train.to_csv("yX_train_clean.csv", index=False)

# %%
