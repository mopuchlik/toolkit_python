#%%

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# %%

# create dataset
X = np.array([[0, 2.1, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])

# add NaNs
X_with_nan = np.array([[np.nan, 0.87, 1.31],
                       [np.nan, -0.67, -0.22]])

#%% knn

# explanatory var
# X[:, 1:]
# var to be explained
# X[:, 0]

class_knn = KNeighborsClassifier(3, weights='distance')
trained_model = class_knn.fit(X[:, 1:], X[:, 0])

# %% apply model to impute nans

imputed_vals = trained_model.predict(X_with_nan[:, 1:])
# hstack is column-wise concat, takes tuple as input
X_with_imputed = np.hstack((imputed_vals.reshape(-1, 1), X_with_nan[:, 1:]))

# vstack row-wise concat, takes tuple as input
X_final = np.vstack((X, X_with_imputed))
print(X_final)

# %%
