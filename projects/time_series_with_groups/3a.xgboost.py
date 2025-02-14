# %% imports
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    GridSearchCV,
    train_test_split,
)

import shap


# ### Get the current working directory (adjust)
cwd = "/home/michal/Dropbox/programowanie/python_general/projects/time_series_with_groups/"
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

yX_train["date"] = pd.to_datetime(yX_train["date"])
yX_train.set_index(["date"])

# %% feature preparation

# list of original features
reg_list = [col for col in yX_train.columns if "channel_" in col]

# dummies
yX_train["month"] = yX_train["date"].dt.month
yX_train["quarter"] = yX_train["date"].dt.quarter

month_dummies = pd.get_dummies(yX_train["month"], prefix="month")
# month_dummies.drop(columns=["month_12"], inplace=True)
yX_train = pd.concat([yX_train, month_dummies], axis=1)

quarter_dummies = pd.get_dummies(yX_train["quarter"], prefix="quarter")
# quarter_dummies.drop(columns=["quarter_4"], inplace=True)
yX_train = pd.concat([yX_train, quarter_dummies], axis=1)

# # monthly quarter and monthly dummies using cyclical function
# yX_train["month_sin"] = np.sin(2 * np.pi * yX_train["month"] / 12)
# yX_train["quarter_sin"] = np.sin(2 * np.pi * yX_train["month"] / 4)

yX_train.drop(columns=["month"], inplace=True)
yX_train.drop(columns=["quarter"], inplace=True)

# %% calculate rolling means
window_size = 2
for channel in reg_list:
    yX_train[channel + "_ma"] = yX_train[channel].rolling(window=window_size).mean()


# %% logdiff


def calc_logdiff(
    df,
    col_name,
):

    # logs
    df[f"{col_name}_log"] = np.log(df[f"{col_name}"])
    df[f"{col_name}_loglag"] = np.log(
        df[["key", f"{col_name}"]].groupby("key").shift(1)
    )
    # logdiff
    df[f"{col_name}_logdiff"] = df[f"{col_name}_log"] - df[f"{col_name}_loglag"]

    df.drop(
        columns=[
            f"{col_name}_log",
            f"{col_name}_loglag",
        ],
        inplace=True,
    )

    return df


for i in reg_list:
    yX_train = calc_logdiff(yX_train, f"{i}")

# cleanup after lags
yX_train.replace([np.inf, -np.inf], 0, inplace=True)
yX_train.fillna(0, inplace=True)
print(f"number of NAs of original set after cleanup: {yX_train.isna().sum().sum()}")


# %% plotter
def plot_linear(
    df, series1, series2, grid_columns=3, grid_rows=5
):  ## add series3 to parameters for real serie
    fig, axes = plt.subplots(grid_rows, grid_columns, figsize=(25, 15))
    axes = axes.flatten()

    for ax, (key, group) in zip(axes, df.groupby("key")):
        ax.plot(group["date"], group[series1], label=f"{series1}", c="r")
        ax.plot(group["date"], group[series2], label=f"{series2}", c="b")
        # ax.plot(group["date"], group[series3], label=f"{series3}", c= "b")
        ax.set_title(f"Key: {key}")
        ax.set_xlabel("Date")
        ax.set_ylabel("y")
        ax.set_ylim(
            [
                min(df[series1].min(), df[series2].min()),
                max(df[series1].max(), df[series2].max()),
            ]
        )
        ax.set_xlim([df["date"].min(), df["date"].max()])
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    ## delet unused plots (subplots)
    for i in range(len(df.groupby("key")), grid_rows * grid_columns):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


# %%

y_train = yX_train["y"]
# X_train = yX_train.drop(columns=["key", "date", "y"])
X_train = yX_train.drop(columns=["y", "date"])
# X_train[["date"]] = int(X_train[["date"]])

# %%


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


# # XGBoost-Compatible WMAPE Metric
# def wmape_eval(preds, dtrain):
#     labels = dtrain.get_label()
#     wmape_value = wmape(labels, preds)
#     return "WMAPE", wmape_value


# def wmape_eval(preds, dtrain):
#     labels = dtrain.get_label()  # dtrain is a DMatrix (has get_label)
#     wmape_value = np.sum(np.abs(labels - preds)) / np.sum(np.abs(labels))
#     return "WMAPE", wmape_value


# %%
model = xgb.XGBRegressor(
    objective="reg:squarederror",  # Use standard regression objective
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)


# Grid Search to Optimize for WMAPE
param_grid = {
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.05, 0.075, 0.11],
    "n_estimators": [50, 100, 150, 200],
    "reg_lambda": [0.5, 1, 5],
    "reg_alpha": [0, 0.5, 1],
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",  # Use MAE as a proxy for WMAPE
    cv=3,
    verbose=1,
)

grid_search.fit(X_train, y_train)
print("Best Params:", grid_search.best_params_)

# %%
model = xgb.XGBRegressor(
    objective="reg:squarederror",  # Use standard regression objective
    n_estimators=150,
    learning_rate=0.05,
    max_depth=2,
    reg_lambda=0.5,
    # random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_train)

final_wmape = wmape(y_train, y_pred)
print(f"Final WMAPE: {final_wmape:.4f}")
print(f"MSE: {mean_squared_error(y_train, y_pred):.2f}")
print(f"R² Score: {r2_score(y_train, y_pred):.2f}")

# # ✅ Fit the Model with WMAPE Evaluation
# model.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_train, y_train)],
#     # eval_metric=wmape_eval,  # Custom WMAPE metric
#     verbose=True,
# )

# %% feature importance plot


xgb.plot_importance(model, importance_type="weight")
plt.show()


# %% plot of predict for train

y_pred_name = pd.DataFrame({"y_pred": y_pred})
model_y_fitted = yX_train[["key", "date", "y"]]
model_y_fitted = pd.concat([model_y_fitted, y_pred_name], axis=1)

plot_linear(df=model_y_fitted, series1="y", series2="y_pred")

# %% residuals plt

residuals = y_train - y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color="red", linestyle="--")  # Zero Line
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()


# %% distribution of residuals

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color="blue")
plt.axvline(0, color="red", linestyle="--")  # Zero Line
plt.title("Distribution of Residuals")
plt.show()


# %%
# import xgboost as xgb

# print(xgb.__version__)  # ✅ Should be ≥ 1.3.0

# %%


def plot_learning_curves(model, X, y):
    _xtrain, _xval, _ytrain, _yval = train_test_split(X, y, test_size=0.2)
    train_errors = []
    val_errors = []
    for m in range(1, len(_xtrain)):
        model.fit(_xtrain[:m], _ytrain[:m])
        _ytrain_predict = model.predict(_xtrain[:m])
        _yval_predict = model.predict(_xval)
        train_errors.append(mean_squared_error(_ytrain_predict, _ytrain[:m]))
        val_errors.append(mean_squared_error(_yval_predict, _yval))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.xlabel("Training set size")
    plt.xlabel("RMSE")


# %%

model = xgb.XGBRegressor(
    objective="reg:squarederror",  # Use standard regression objective
    n_estimators=150,
    learning_rate=0.05,
    max_depth=2,
    reg_lambda=0.5,
    # random_state=42,
)
plot_learning_curves(model, X_train, y_train)

# %% ##########################
# prepare X_test
# calculate and plot prediction
# and other plots

# loading a csv file
path_X_test = "X_test_T2.csv"

X_test = pd.read_csv(path_X_test, sep=";", decimal=",")

# cast date to datetime
X_test["date"] = pd.to_datetime(X_test["date"])

# cleanup NAs
X_test = X_test.fillna(0)

# row sums for all channels
reg_list = [col for col in X_test.columns if "channel_" in col]
X_test["channel_sum"] = X_test[reg_list].sum(axis=1)
reg_list = [col for col in X_test.columns if "channel_" in col]

# dummies
X_test["month"] = X_test["date"].dt.month
X_test["quarter"] = X_test["date"].dt.quarter

month_dummies = pd.get_dummies(X_test["month"], prefix="month")
# month_dummies.drop(columns=["month_12"], inplace=True)
X_test = pd.concat([X_test, month_dummies], axis=1)

quarter_dummies = pd.get_dummies(X_test["quarter"], prefix="quarter")
# quarter_dummies.drop(columns=["quarter_4"], inplace=True)
X_test = pd.concat([X_test, quarter_dummies], axis=1)

# # monthly quarter and monthly dummies using cyclical function
# X_test["month_sin"] = np.sin(2 * np.pi * X_test["month"] / 12)
# X_test["quarter_sin"] = np.sin(2 * np.pi * X_test["month"] / 4)

X_test.drop(columns=["month"], inplace=True)
X_test.drop(columns=["quarter"], inplace=True)

# calculate rolling means
window_size = 2
for channel in reg_list:
    X_test[channel + "_ma"] = X_test[channel].rolling(window=window_size).mean()

for i in reg_list:
    X_test = calc_logdiff(X_test, f"{i}")

# cleanup after lags
X_test.replace([np.inf, -np.inf], 0, inplace=True)
X_test.fillna(0, inplace=True)
print(f"number of NAs of original set after cleanup: {X_test.isna().sum().sum()}")
X_test_orig = X_test.copy()
X_test = X_test.drop(columns=["date"])

# %% SHAP plot
# Shapley Additive Explanations
# algorithms to explain ensemble tree models

# Red: High feature values.
# Blue: Low feature values.
# Left/Right: Negative/positive impact on the prediction.
# NOTE: useful mostly for categorical variables, for numerical use bar plot

explainer = shap.Explainer(model)
shap_values = explainer(X_test)
# ### Beeswarm
# mean absolute value of the SHAP values for each feature. (default)
# This order however places more emphasis on broad average impact,
# and less on rare but high magnitude impacts.
shap.plots.beeswarm(shap_values, order=shap_values.abs.mean(0))
# find features with high impacts for individual people we can instead
# sort by the max absolute value
shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0))

# default beeswarm
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.plots.bar(shap_values.abs.mean(0))

shap.dependence_plot("channel_101", shap_values.values, X_test)

# decision plot TBC
base_val = explainer.expected_value
shap_val = explainer.shap_values(X_test)
shap.decision_plot(base_val, shap_val)

# %% partial dependence plot
# Purpose: Understand how a feature affects predictions, holding others constant.
# Flat Line: Feature has little impact.
# Curved Line: Strong relationship with the target.

from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_test, features=["channel_101", "channel_17"]
)
plt.show()


# %% plot of predict
y_test_pred = model.predict(X_test)

y_test_pred_name = pd.DataFrame({"y_pred": y_test_pred})
model_y_fitted = X_test_orig[["key", "date"]]
model_y_fitted = pd.concat([model_y_fitted, y_pred_name], axis=1)

plot_linear(df=model_y_fitted, series1="y_pred", series2="y_pred")

# %% cross validation

# Perform n-Fold Cross-Validation
cv = KFold(n_splits=20, shuffle=True, random_state=42)
# negative because sklearn minimizes scores
scores = cross_val_score(
    model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
)
print(f"Cross-Validation MAE: {-np.mean(scores):.4f}")


# %%
# version with WMAPE
# Define a custom WMAPE scorer
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


wmape_scorer = make_scorer(wmape, greater_is_better=False)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=wmape_scorer)
print(f"Cross-Validation WMAPE: {-np.mean(scores):.4f}")

# %%
