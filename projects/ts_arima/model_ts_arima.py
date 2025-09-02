#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:31:41 2023

example project with implementation of ARIMAX model]

@author: michal
"""


# %% lib imports
import pandas as pd
import os
import matplotlib.pyplot as plt
# from datetime import datetime
import holidays
import numpy as np
import time

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %% load dataset
cwd = "/home/michal/silos/Dropbox/programowanie/python_general/projects/ts_arima/"
# cwd = "D:\Dropbox\programowanie\projekt_ts\project-time-series"

# TODO: for some reason does not work in Linux
# cwd = os.path.dirname(os.path.abspath("__file__"))

os.chdir(cwd)
print("Current working directory: {0}".format(os.getcwd()))


df = pd.read_csv("yahoo_stock.csv", parse_dates=["Date"])

# prints info about datatypes for each columns and how many null values each column contains
print(df.info())
print(df.dtypes)

max(df['Close'])
min(df['Close'])

max(df['Volume'])
min(df['Volume'])

df.isna().sum().sum()


# %% stationarity test (Augmented Dickey-Fuller)

def check_stationarity(ts):
    adfuller_test = adfuller(ts)
    print(f'ADF Statistic:{adfuller_test[0]}')
    print(f'p-value: {adfuller_test[1]}')
    print('Critical Values:')
    for key, value in adfuller_test[4].items():
        print((key, value))
    if adfuller_test[1] <= 0.05:
        print('Series is stationary')
    else:
        print('Series is not stationary')

    return None


print(check_stationarity(df['Close']))


# %% some plots
# plt.plot(df['Close']);
plt.plot(df['Volume'])
plt.show()

# %% create features

# time features

df['year'] = pd.to_datetime(df['Date']).dt.year
df['month'] = pd.to_datetime(df['Date']).dt.month
df['day'] = pd.to_datetime(df['Date']).dt.day
# Monday is 0, Sunday is 6
df['week'] = pd.to_datetime(df['Date']).dt.isocalendar().week
df['weekday'] = pd.to_datetime(df['Date']).dt.weekday


def is_weekend(weekday):
    if weekday == 5 or weekday == 6:
        return 1
    else:
        return 0


df['is_weekend'] = df['weekday'].apply(is_weekend)
# ----------------------------------------------------------------------------
# def is_war(date):
#     if date >= pd.Timestamp('2022-02-24'):
#         return 1
#     else:
#         return 0

# df['is_war'] = df['Date'].apply(is_war)
# ----------------------------------------------------------------------------


def is_holiday(date):
    us_holidays = holidays.US()
    return int(date in us_holidays)

df['is_holiday'] = df['Date'].apply(is_holiday)
# ----------------------------------------------------------------------------


def logarithmise(var):
    return np.log(var)


df['log_price'] = df['Close'].apply(logarithmise)
df['log_vol'] = df['Volume'].apply(logarithmise)

# ----------------------------------------------------------------------------


def is_monday(weekday):
    if weekday == 0:
        return 1
    else:
        return 0

df['is_monday'] = df['weekday'].apply(is_monday)
# ----------------------------------------------------------------------------


def is_tuesday(weekday):
    if weekday == 1:
        return 1
    else:
        return 0

df['is_tuesday'] = df['weekday'].apply(is_tuesday)
# ----------------------------------------------------------------------------


def is_wednesday(weekday):
    if weekday == 2:
        return 1
    else:
        return 0

df['is_wednesday'] = df['weekday'].apply(is_wednesday)
# ----------------------------------------------------------------------------


def is_thursday(weekday):
    if weekday == 3:
        return 1
    else:
        return 0

df['is_thursday'] = df['weekday'].apply(is_thursday)
# ----------------------------------------------------------------------------


def is_friday(weekday):
    if weekday == 4:
        return 1
    else:
        return 0

df['is_friday'] = df['weekday'].apply(is_friday)
# ----------------------------------------------------------------------------


def is_saturday(weekday):
    if weekday == 5:
        return 1
    else:
        return 0

df['is_saturday'] = df['weekday'].apply(is_saturday)
# ----------------------------------------------------------------------------

def is_sunday(weekday):
    if weekday == 6:
        return 1
    else:
        return 0

df['is_sunday'] = df['weekday'].apply(is_sunday)

# %%----------------------------------------------------------------------------
# months

df['is_Jan'] = 0
df.loc[df['month'] == 1, 'is_Jan'] = 1

df['is_Feb'] = 0
df.loc[df['month'] == 2, 'is_Feb']= 1

df['is_Mar'] = 0
df.loc[df['month'] == 3, 'is_Mar'] = 1

df['is_Apr'] = 0
df.loc[df['month'] == 4, 'is_Apr'] = 1

df['is_May'] = 0
df.loc[df['month'] == 5, 'is_May'] = 1

df['is_Jun'] = 0
df.loc[df['month'] == 6, 'is_Jun'] = 1

df['is_Jul'] = 0
df.loc[df['month'] == 7, 'is_Jul'] = 1

df['is_Aug'] = 0
df.loc[df['month'] == 8, 'is_Aug'] = 1

df['is_Sep'] = 0
df.loc[df['month'] == 9, 'is_Sep'] = 1

df['is_Oct'] = 0
df.loc[df['month'] == 10, 'is_Oct'] = 1

df['is_Nov'] = 0
df.loc[df['month'] == 11, 'is_Nov'] = 1

df['is_Dec'] = 0
df.loc[df['month'] == 12, 'is_Dec'] = 1

# %% price and vol features

df['log_price_lag_1'] = df['log_price'].shift(1)
df['log_diff_1'] = df['log_price'] - df['log_price_lag_1']

# clean any NAs
df = df.dropna()
df.reset_index(drop=True, inplace=True)


plt.plot(df['log_diff_1'])
plt.show()

check_stationarity(df['log_diff_1'])

window_size = 14  # in days --> it's one week
roll_vol_lag = 7

# Calculate rolling mean for all rows except last 24
df['roll_mean_log_vol'] = df['log_vol'].rolling(window_size).mean()
df['roll_mean_log_vol_lag'] = df['roll_mean_log_vol'].shift(roll_vol_lag)

# #%% plots 
# df.head(10)

# plt.plot(df['log_vol'])
# plt.show();

# plt.plot(df['roll_mean_log_vol'])
# plt.show();

# # ACFs and PACFs
# plot_pacf(df['log_diff_1'])
# plt.show();

# plot_pacf(df['log_price'])
# plt.show();

# plot_pacf(df['log_vol'])
# plt.show();

# %% diagnostic functions

def mape(actual, predicted):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between two lists of values.

    Args:
        actual (list): A list of actual values.
        predicted (list): A list of predicted values.

    Returns:
        float: The MAPE value.
    """
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

#%% logdiff 7 and 30
df['log_price_lag_7'] = df['log_price'].shift(7)
df['log_diff_7'] = df['log_price'] - df['log_price_lag_7']
df['log_price_lag_30'] = df['log_price'].shift(30)
df['log_diff_30'] = df['log_price'] - df['log_price_lag_30']

# %% clean any NAs
df = df.dropna()
df.reset_index(drop=True, inplace=True)

df_bak = df.copy()

#%% some model tweeking and backtesting (AR-2)
# check against the grid

start_time = time.time()

mapes_vol_iter = []

for win_size_mult in range(7):

    print('-------- win_size_mult = ', win_size_mult, '-----')

    for roll_vol_lag in range(7):
        print('------ roll_vol_lag = ', roll_vol_lag, '-----')

        df = df_bak.copy()
        # Calculate rolling mean for all rows except last 24
        window_size = 7 * (1 + win_size_mult)   # in days --> it's one week
        roll_vol_lag = 7 * (1 + roll_vol_lag) # lag for rolling window
        df['roll_mean_log_vol'] = df['log_vol'].rolling(window_size).mean()
        df['roll_mean_log_vol_lag'] = df['roll_mean_log_vol'].shift(roll_vol_lag)
        df['log_vol_diff'] = df['roll_mean_log_vol'] - df['roll_mean_log_vol_lag']
        # clean any NAs
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)

        mapes_means = []

        for n_forecasted in range(1, 10):

            # print('----- n_forecasted = ', n_forecasted, '-----')

            mapes_for_n_forecasts = []

            for model_iteration in range(5):

                # print('----- model_iteration = ', model_iteration, '-----')

                end = df.shape[0]
                start = end - n_forecasted - model_iteration

                train = df.iloc[:start]
                test = df.iloc[start:start + n_forecasted]

                # fit an ARIMA model with exogenous variables
                order = (1, 0, 0)
                exog_vars = [
                    # 'is_monday',
                    # 'is_tuesday',
                    # 'is_wednesday',
                    # 'is_thursday',
                    # 'is_friday',
                    # 'is_weekend',
                    # 'is_holiday',
                    'log_vol_diff',
                ]
                model = ARIMA(train['log_diff_7'], order=order, exog=train[exog_vars], trend='n' )
                # , trend='n'
                results = model.fit()
                # print(results.summary())

                # predictions on test set
                df['log_preds'] = results.forecast(
                    steps=n_forecasted, exog=test[exog_vars])

                # return from logdiff to original prices
                df['preds'] = np.exp(
                    df['log_preds']) * np.exp(df['log_price_lag_7'])

                # remember mape for this run
                ind_last_pred = start + n_forecasted - 1
                ind_last_pred_next = start + n_forecasted

                # print('-- realization for target: ',
                #       df.Close[ind_last_pred:ind_last_pred_next])
                # print('-- last prediction for target: ',
                #       df[f'preds_{model_iteration}'][ind_last_pred:ind_last_pred_next])

                # if model_iteration % 20 == 0 and n_forecasted >= 5:
                #     plt.plot(df.Close[start:ind_last_pred_next])
                #     plt.plot(df['preds'][start:ind_last_pred_next])
                #     plt.show();

                mape_calc = mape(df.Close[ind_last_pred:ind_last_pred_next],
                                df['preds'][ind_last_pred:ind_last_pred_next])
                mapes_for_n_forecasts.append(mape_calc)
                # print(
                #     f'{model_iteration}: mape = {round(mapes_for_n_forecasts[model_iteration],3)}')

            # mapes_for_n_forecasts = [x for x in mapes_for_n_forecasts if x >= 0.5 ]
            mapes_means.append(np.mean(mapes_for_n_forecasts))
            print('-- mean mape for', n_forecasted, ' ahead forecasts:',
                np.mean(mapes_for_n_forecasts))

        print('mean mapes for respective ahead forecasts :', mapes_means)
        mapes_vol_iter.append(mapes_means)

end_time = time.time()
time_diff = (end_time - start_time) / 60
print('Execution time:', time_diff, 'minutes')



# %% diagnostics

# TODO regress results on realizations
# TODO learning curve
# TODO struc breaks


