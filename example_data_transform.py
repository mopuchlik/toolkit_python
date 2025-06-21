
#%% libs
import numpy as np
import pandas as pd
from sklearn import preprocessing as prep


#%% funkcja zmienia argument jako default

def dod(list, num):
    list.append(num)

    return list

#%% utworz dataframe

x = {'id': [1, 2, 3], 
    'gugu': ('a', 'b', 'c'), 
    'lala': [3, 4, 8]}

x = pd.DataFrame(x)

y = {'id': [4, 5, 6, 7], 
    'gugu': ('a', 'b', 'c', 'd'), 
    'lala': [3, 4, 8, 9]}

y = pd.DataFrame(y)

# x.append(y)

# alternatywnie
z = pd.concat([x, y], axis=0, ignore_index=True)

# %% scalers %%%
x = np.array([[1, 2, 3, 4, 5, 6]]).T

#%%
scaler = prep.MinMaxScaler()
a = scaler.fit_transform(x)
print(a)

# %%

scaler = prep.StandardScaler()
a = scaler.fit_transform(x)
print(a)

#%% dziwne ale sumuje po wierszach zeby bylo 1 (sa inne opcje)
scaler = prep.Normalizer()
a = scaler.fit_transform(x)
print(a)

# %% transformacje zmiennych %%

x = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])

def add_100(x):
    x += 100
    return x

trans_1 = prep.FunctionTransformer(add_100)
trans_1.transform(x[:, :2])
print(x)

#%%  alternatively for pandas
y = pd.DataFrame(x)
y.iloc[:,:2] = y.iloc[:,:2].apply(add_100)
print(y)

#%% filtrowanie numpy vs pandas

x = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])

y_pd = pd.DataFrame(x)

# dla numpy jest jest pelne info a dla pandas skrotowo w logice kolumnowej 
x[x[:, 1] == 2, :]

y_pd[y_pd.iloc[:, 1] == 2]


# %%
