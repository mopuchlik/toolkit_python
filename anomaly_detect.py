# %%

# anomaly detection using fit to a predetermined distrobution and then loglikelihood value cut at some threshold

# UNQ_C1
# GRADED FUNCTION: estimate_gaussian

import numpy as np
import pandas as pd

# %%

# x = pd.DataFrame(
#     [
#         [13.04681517, 14.74115241],
#         [13.40852019, 13.7632696],
#         [14.19591481, 15.85318113],
#         [14.91470077, 16.17425987],
#         [13.57669961, 14.04284944],
#     ]
# )

x = np.array(
    [
        [13.04681517, 14.74115241],
        [13.40852019, 13.7632696],
        [14.19591481, 15.85318113],
        [14.91470077, 16.17425987],
        [13.57669961, 14.04284944],
    ]
)


print(x)


# %%


def estimate_gaussian(X):
    """
    Calculates mean and variance of all features
    in the dataset

    Args:
        X (ndarray): (m, n) Data matrix

    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape

    ### START CODE HERE ###

    mu = np.zeros(n, dtype=float)
    var = np.zeros(n, dtype=float)

    for i in range(n):
        x_feat = X[:, i]

        mu_feat = x_feat.sum() / m
        mu[i] = mu_feat

        x_feat_cent = x_feat - mu_feat
        x_feat_cent_sq = (x_feat_cent) ** 2
        var_feat = x_feat_cent_sq.sum() / m
        var[i] = var_feat

    ### END CODE HERE ###

    return mu, var


# %%

a, b = estimate_gaussian(x)

print(a)
print(b)


# %%
# todo at some point

# # Returns the density of the multivariate normal
# # at each data point (row) of X_train
# p = multivariate_gaussian(X_train, mu, var)

# #Plotting code
# visualize_fit(X_train, mu, var)


# %%
def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (p_val)
    and the ground truth (y_val)

    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set

    Returns:
        epsilon (float): Threshold chosen
        F1 (float):      F1 score by choosing epsilon as threshold
    """

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        ### START CODE HERE ###

        tp_flag = (p_val < epsilon) & (y_val == 1)
        tp = tp_flag.sum()

        fp_flag = (p_val < epsilon) & (y_val == 0)
        fp = fp_flag.sum()

        fn_flag = (p_val >= epsilon) & (y_val == 1)
        fn = fn_flag.sum()

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        F1 = 2 * prec * rec / (prec + rec)

        ### END CODE HERE ###

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1
