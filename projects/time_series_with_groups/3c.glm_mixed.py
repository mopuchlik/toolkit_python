# %% imports
import pandas as pd
import os
import numpy as np

# import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import (
    cross_val_score,
    KFold,
    GridSearchCV,
    train_test_split,
)

# ### Get the current working directory (adjust)
# cwd = "Q:/Dropbox/programowanie/nestle"
cwd = "/home/michal/silos/Dropbox/programowanie/python_general/projects/time_series_with_groups/"
os.chdir(cwd)
print("Current working directory: {0}".format(os.getcwd()))

# %%
# loading a csv file
path_yX_train = "yX_train_clean.csv"
yX_train = pd.read_csv(
    path_yX_train,
)
print(f"number of NAs of original set after cleanup: {yX_train.isna().sum().sum()}")

# cast date to datetime
yX_train["date"] = pd.to_datetime(yX_train["date"])

# list of original regressors
reg_list = [col for col in yX_train.columns if "channel_" in col]

# key as int -- it is used by a model used further on
yX_train["key"] = yX_train["key"].astype("int")

# %% for y interpolate NAs
# in X replace NaN with zeroes

# original shape of y and X
print(f"original shape of y and X: {yX_train.shape}")

# Number of NAs in y
print(f"number of NAs in y: {yX_train["y"].isna().sum()}")
# interpolate y
yX_train["y"] = yX_train["y"].interpolate(method="linear")
print(f"number of NAs in y after: {yX_train["y"].isna().sum()}")

print(f"number of NAs of original set: {yX_train.isna().sum().sum()}")
# in X replace NA with 0s
yX_train = yX_train.fillna(0)
print(f"number of NAs of original set after cleanup: {yX_train.isna().sum().sum()}")

# %% feature preparation

# NOTE: separately for train and test
yX_train["key_target"] = yX_train.groupby("key")["y"].transform("mean")

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

# %%
# calculate rolling means for regressors/features
window_size = 2
for channel in reg_list:
    yX_train[channel + f"_ma{window_size}"] = (
        yX_train[channel].rolling(window=window_size).mean()
    )

# %% lagged variables

shift_size = 1
for channel in reg_list:
    yX_train[channel + f"_lag{shift_size}"] = (
        yX_train[["key", channel]].groupby("key").shift(shift_size)
    )

# %% prepare monthly and quarter dummies

yX_train["month"] = yX_train["date"].dt.month
yX_train["quarter"] = yX_train["date"].dt.quarter

month_dummies = pd.get_dummies(yX_train["month"], prefix="month")
# month_dummies.drop(columns=["month_12"], inplace=True)
yX_train = pd.concat([yX_train, month_dummies], axis=1)

quarter_dummies = pd.get_dummies(yX_train["quarter"], prefix="quarter")
# quarter_dummies.drop(columns=["quarter_4"], inplace=True)
yX_train = pd.concat([yX_train, quarter_dummies], axis=1)

# # monthly quarter and monthly dummies using cyclical function
yX_train["sin_month"] = np.sin(2 * np.pi * yX_train["month"] / 12)
yX_train["sin_quarter"] = np.sin(2 * np.pi * yX_train["month"] / 4)

yX_train.drop(columns=["month"], inplace=True)
yX_train.drop(columns=["quarter"], inplace=True)

# %% feature lists
# prepare sets of regressors/features

# moving averages (withous channel_sum)
reg_list_ma = [col for col in yX_train.columns if (f"_ma{window_size}" in col)]

# lags
reg_list_lag = [col for col in yX_train.columns if f"_lag{shift_size}" in col]

# logdiff
reg_list_logdiff = [col for col in yX_train.columns if f"_logdiff" in col]

# quarter dummies
reg_list_dummy_q = [col for col in yX_train.columns if f"quarter_" in col]

# month dummies
reg_list_dummy_m = [col for col in yX_train.columns if f"month_" in col]


# %% functions
def create_model_y_fitted(results):
    """this function creates table with realisations y and model fitted values"""

    model = pd.DataFrame(
        {
            "key": yX_train["key"],
            "date": yX_train["date"],
            "y": yX_train["y"],
            "fitted_values": results.fittedvalues,
        }
    )

    return model


def create_model_params(results):
    """this function creates table with params names, param values and respective p-values"""

    model_par = pd.DataFrame(
        {"param_val": results.params, "pvalue": results.pvalues}
    ).reset_index()
    model_par.rename(columns={"index": "param_name"}, inplace=True)

    return model_par


def wmape(y_true, y_pred):
    """
    This function calculates WMAPE (Weighted Mean Absolute Percentage Error)
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


# plotter
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


# %% final dataset

# remove NAs that may have been created by applying lags for regressors
yX_train = yX_train.dropna()
print(f"shape of the final inpt dataset: {yX_train.shape}")

# %% regressor strings to pass to model
reg_list_str = "+".join(reg_list)
reg_list_ma_str = "+".join(reg_list_ma)
reg_list_lag_str = "+".join(reg_list_lag)
reg_list_dummy_q_str = "+".join(reg_list_dummy_q)
reg_list_logdiff_str = "+".join(reg_list_logdiff)
reg_list_dummy_m_str = "+".join(reg_list_dummy_m)

# %%

reg_list_str_current = f"{reg_list_str} + {reg_list_dummy_q_str} + key_target"

# mixedLM runs
# Random intercept + slope for all regressors in your string
model = smf.mixedlm(
    f"y ~ {reg_list_str_current}",
    yX_train,
    groups=yX_train["key"],
    # re_formula="1 + quarter_1 + quarter_2 + quarter_3",
    re_formula=f"1 + {reg_list_str}",
)

mdf = model.fit()
print(mdf.summary())

model_y_fitted = create_model_y_fitted(mdf)
model_params = create_model_params(mdf)

# basic stats

y_train = model_y_fitted["y"]
fitted_train = model_y_fitted["fitted_values"]

final_wmape = wmape(y_train, fitted_train)
final_wmape = wmape(y_train, fitted_train)
print(f"Final WMAPE: {final_wmape:.4f}")
print(f"MSE: {mean_squared_error(y_train, fitted_train):.2f}")
print(f"R² Score: {r2_score(y_train, fitted_train):.2f}")

plot_linear(df=model_y_fitted, series1="y", series2="fitted_values")

# %% validation on train set

lmplt = sns.lmplot(
    x="y",
    y="fitted_values",
    data=model_y_fitted,
    aspect=1,
    line_kws={"color": "black", "linewidth": 2},
)
ax = lmplt.ax
ax.plot(
    [model_y_fitted["y"].min(), model_y_fitted["y"].max()],
    [model_y_fitted["y"].min(), model_y_fitted["y"].max()],
    linestyle="--",
    color="red",
    linewidth=2,
    label="45° Line",
)

# %% out-of-time plot
# NOTE for oot key_target should be calculated for trimmed X_train
# here it is based on full sample, i.e. yX_train
yX_oot = yX_train.copy()

N_obs = 6

# train on observations without last N_obs
# test with last N_obs
yX_oot_trim = (
    yX_oot.groupby("key", group_keys=False)
    .apply(lambda x: x.iloc[:-N_obs])
    .reset_index()
).drop(columns=["index"])

yX_oot_trim_last = (
    yX_oot.groupby("key", group_keys=False)
    .apply(lambda x: x.iloc[-N_obs:])
    .reset_index()
).drop(columns=["index"])

# check
print(f"CHECK: {yX_oot.shape[0] - yX_oot_trim.shape[0] - yX_oot_trim_last.shape[0]}")

y_oot_trim = yX_oot_trim[["y"]]
X_oot_trim = yX_oot_trim.drop(columns=["y", "date"])

y_oot_trim_last = yX_oot_trim_last[["y"]]
X_oot_trim_last = yX_oot_trim_last.drop(columns=["y", "date"])

# fit model
model_oot = smf.mixedlm(
    f"y ~ {reg_list_str_current}",
    yX_oot_trim,
    groups=yX_oot_trim["key"],
    # re_formula="1 + quarter_1 + quarter_2 + quarter_3",
    re_formula=f"1 + {reg_list_str}",
)
mdf_oot = model_oot.fit()
print(mdf_oot.summary())

# TODO problem with predict on new data
# https://chatgpt.com/share/68948b08-1f08-8010-b7fc-5d23039c7387

# # key_last_series = yX_oot_trim.loc[X_oot_trim_last.index, "key"]
# # new = X_oot_trim_last.copy()
# # new["key"] = key_last_series
# # # new = X_oot_trim_last.copy()

# y_pred_last = mdf_oot.predict(exog=X_oot_trim_last, groups=X_oot_trim_last["key"])

# # predict on last N_obs
# y_pred_last = model_oot.predict(exog=X_oot_trim_last, params=mdf_oot.params)
# # model_y_fitted = create_model_y_fitted(mdf)

# y_pred_last_name = pd.DataFrame({"y_pred": y_pred_last})
# yX_oot_trim_last = pd.concat(
#     [yX_oot_trim_last[["key", "date"]], y_pred_last_name], axis=1
# )

# # join with main set
# yX_oot = pd.merge(yX_oot, yX_oot_trim_last, on=["key", "date"], how="left")
# yX_oot.fillna(0)

# # plot
# plot_linear(df=yX_oot, series1="y", series2="y_pred")

# # %% cross validation (must be done by habd as scikit cannot handle it)

# # # Perform n-Fold Cross-Validation
# # cv = KFold(n_splits=20, shuffle=True, random_state=42)
# # # negative because sklearn minimizes scores

# # X_train = yX_train[reg_list + reg_list_dummy_q + ["key_target"]]
# # y_train = yX_train["y"]


# # scores = cross_val_score(
# #     model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
# # )
# # print(f"Cross-Validation MAE: {-np.mean(scores):.4f}")


# # %% make predict

# # loading a csv file
# path_X_test = "X_test_T2.csv"
# path_y_test = "y_test_T2.csv"

# X_test = pd.read_csv(path_X_test, sep=";", decimal=",")
# y_test = pd.read_csv(path_y_test, sep=";", decimal=",")

# # cast date to datetime
# X_test["date"] = pd.to_datetime(X_test["date"])
# # X_train["date"] = X_train["date"].dt.date
# y_test["date"] = pd.to_datetime(y_test["date"])
# # y_train["date"] = y_train["date"].dt.date


# # cleanup NAs
# X_test = X_test.fillna(0)

# # ### create logdiff
# # chann_sum
# reg_list = [col for col in X_test.columns if "channel_" in col]

# for channel in reg_list:
#     X_test[channel + f"_ma{window_size}_"] = (
#         X_test[channel].rolling(window=window_size).mean()
#     )

# X_test["quarter"] = X_test["date"].dt.quarter
# quarter_dummies = pd.get_dummies(X_test["quarter"], prefix="quarter")
# quarter_dummies.drop(columns=["quarter_4"], inplace=True)
# X_test = pd.concat([X_test, quarter_dummies], axis=1)
# X_test.drop(columns=["quarter"], inplace=True)

# predict = pd.DataFrame({"y": mdf.predict(exog=X_test)})
# y_test.drop(columns=["y"], inplace=True)
# y_test = pd.concat([y_test, predict], axis=1)


# # %%
# yX_test = pd.merge(y_test, X_test, on=["key", "date"])
# plot_linear(df=yX_test, series1=f"chann_sum_ma{window_size}_", series2="y")
# plot_linear(df=yX_test, series1=f"y", series2="y")


# MixedLMResults.predict doesn’t accept groups= (that’s why you got the error). It only takes exog (and transform=). To include random effects in predictions you need to add the group-specific BLUPs yourself:

#     get fixed-effect preds from the fitted results;

#     build the random-effects design for the new rows;

#     add ZbgZbg​ for groups that were seen in training (for unseen groups, bg=0bg​=0).


# import numpy as np
# import pandas as pd
# from patsy import dmatrix

# # ---- inputs ----
# # mdf_oot : statsmodels MixedLMResults from .fit()
# # X_oot_trim_last : DataFrame of new rows, MUST include the 'key' column
# # reg_list_str_current : RHS of your fixed-effects formula (e.g., "x1 + x2 + ...")
# # reg_list_str         : RHS used in re_formula (e.g., "x1 + x3 + quarter_1 + ...")
# # -----------------

# new = X_oot_trim_last.copy()

# # 1) fixed-effects prediction (formula is applied automatically if transform=True)
# yhat_fe = mdf_oot.predict(exog=new, transform=True)   # shape (n,)

# # 2) random-effects design for the new data (match your re_formula)
# re_formula = "1 + " + reg_list_str                    # include random intercept if you used "1 + ..."
# Z_new = dmatrix(re_formula, new, return_type="dataframe")  # shape (n, q)

# # name alignment for random-effects vector b_g
# # try to get the random-effect names from the fitted model; else fall back to Z_new columns
# re_names = getattr(getattr(mdf_oot.model, "data", object()), "exog_re_names", Z_new.columns)

# # 3) add Z b_g for groups seen in training
# re_contrib = np.zeros(len(new))
# re_map = mdf_oot.random_effects  # dict: group -> vector/Series of random effects (BLUPs)

# for g, idx in new.groupby("key").groups.items():
#     b = re_map.get(g, None)
#     if b is None:
#         continue  # unseen group -> random effect = 0
#     # make b a Series indexed by random-effect names, then align to Z_new columns
#     b_ser = pd.Series(np.asarray(b).ravel(), index=re_names)
#     b_ser = b_ser.reindex(Z_new.columns)  # align order; missing -> NaN
#     re_contrib[list(idx)] = np.nan_to_num(Z_new.loc[idx].to_numpy() @ b_ser.to_numpy())

# # final prediction including random effects
# y_pred_last = yhat_fe.to_numpy() + re_contrib         # shape (n,)
# # if you need (n,1):
# y_pred_last = y_pred_last[:, None]
