# easy handout
# https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
# https://www.shanelynn.ie/pandas-iloc-loc-select-rows-and-columns-dataframe/
# https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/


# %% imports
import pandas as pd
import os
import numpy as np

# %% current directory
# ### Get the current working directory (adjust)
# cwd = "/home/trurl/Dropbox/programowanie/python_general/"

cwd = "D:/Dropbox/programowanie/python_general"

# TODO: for some reason does not work in Linux
# cwd = os.path.dirname(os.path.abspath("__file__"))

os.chdir(cwd)
print("Current working directory: {0}".format(os.getcwd()))

# %% loading a csv file
path = 'housing.csv'
df = pd.read_csv(path)

# %% various prints
print(df)

# prints info about datatypes for each columns and how many null values each column contains
print(df.info())
df.info()
df.dtypes

# prints statistical summary of the data
print(df.describe())

# column names
print(df.columns)

# ### number of rows and columns, line dim in R
df.shape
df.shape[0]
df.shape[1]

# print column content
print(df["median_income"])
print(df[['median_income', 'median_house_value', 'households']])

df["median_income"].head()
df["median_income"].tail()

# or
df.median_income.head()
df.median_income.tail()

# %% some viewing options
# pd.options.display.width – the width of the display in characters –
# use this if your display is wrapping rows over more than one line.
#
# pd.options.display.max_rows – maximum number of rows displayed.
#
# pd.options.display.max_columns – maximum number of columns displayed.


# %% some data.frame stats
# number of all obs
len(df)
len(df["median_income"])

### retrieve column by index
# NOTE: indices x:y goes from x to y-1 !!!
df.iloc[:, 2:4]
# but loc is inclusive (!!!)
df.loc[:, "longitude":"housing_median_age"]

## the same for rows 
df.iloc[2:4, :] # --> does not include 4
df.loc[2:4, :] # --> includes 4

# first row
df.iloc[[0]] # outputs dataframe
df.iloc[0] # outputs series

# last row
df.iloc[[-1]]
df.iloc[-1]

# unique values with counts
# WARNING: does not count NAs
df.total_bedrooms.value_counts()
df.total_bedrooms.nunique()

# # number of all NAs
df.isna().sum().sum()

# # number of all nulls
df.isnull().sum().sum()

# number of NAs in a column
sum(df["total_bedrooms"].isna())
sum(df.median_income.isna())
df.median_income.isna().sum()


# %% categorize values

# just give number of bins and function will provide bins with equal number of obs
df["bins"] = pd.cut(df["median_house_value"], 5)
# crosstab --> as table in R
pd.crosstab(df["ocean_proximity"], df["bins"])

# one may also exxplicitly give bins
min_bin = df["median_house_value"].min()
max_bin = df["median_house_value"].max()
bins2 = np.linspace(min_bin, max_bin, 5)
df["bins2"] = pd.cut(df["median_house_value"], bins2)
pd.crosstab(df["ocean_proximity"], df["bins2"])

# %%  filling NA values
# TODO: see .fillna() and .dropna() methods from pandas
#
# ...
#
# %% sorting
df.sort_values(by = "median_house_value", inplace = True)

# %% apply function dataset 
cols = ['median_house_value', 'median_income']

df1 = df[cols].copy()
df2 = df[cols].copy()
df3 = df[cols].copy()

floorer = lambda x: np.floor(x)

# apply works for rows/columns, for dataframe and series
df1 = df1.apply(floorer)

# map works element-wise but only for series
df2 = df2.map(floorer) # <-- will not work because input is a df
df2['median_income'] = df2['median_income'].map(floorer)

# applymap works element-wise for dataframe only (map is for series)
df3 = df3.applymap(floorer)

# %% droping columns by name
cols_1 = ['latitude', 'longitude']

# real copy
df_drop = df.copy()

df_drop.drop(columns=cols_1, inplace=True)
# # + some alternatives
# df.drop(cols_1, axis = 1, inplace = True) # inplace = True means that there is no need for assigning the result to a variable
# df.drop(cols_1, axis = 'columns', inplace = True)

df_drop.columns
df.columns

# serialization - saving an object as a byte stream to
# https://data-flair.training/blogs/python-pickle/

# ### subsetting a dataframe

# example 1 --> list of some columns
cols_2 = ['median_income', 'median_house_value', 'households']
sub_df = df[cols_2]
print(sub_df)

# example 2 --> subsetting with a condition
df.ocean_proximity.value_counts()

df1 = df[df['ocean_proximity'] == 'NEAR BAY']
df2 = df.loc[df["ocean_proximity"] == "NEAR BAY"]

# example 2 with subsetting
cols = ["longitude", "latitude", "population"]
df_ocean_prox_near_bay = df.loc[df["ocean_proximity"] == "NEAR BAY", cols]
print(df_ocean_prox_near_bay)

#%% replace values
df5 = df.copy()

# elegant way
df5['ocean_proximity'].value_counts()
df5['ocean_proximity'].replace(('INLAND', 'NEAR BAY'), ('aa', 'bb'), inplace=True)
df5['ocean_proximity'].value_counts()

# like in R
df6 = df.copy()
df6.loc[df6['ocean_proximity'] == 'INLAND', 'ocean_proximity'] = 'aa'
df6['ocean_proximity'].value_counts()


# %% compares

# and nice compare (pandas)
df_diff = pd.concat([df1, df2]).drop_duplicates(keep=False)
# or
# TODO what are differences
df1.compare(df2)


# compare for lists
def compare_list(l1, l2):
    l1.sort()
    l2.sort()
    if (l1 == l2):
        return "Equal"
    else:
        return "Not equal"


# %% renames
# rename columns
df_rename = df.copy()

df_rename.rename(
    columns={
        "housing_median_age": "juss1",
        "households": "juss2"
    },
    inplace=True
)
df_rename.columns

# ### group by

# series
df_group1 = df.groupby("ocean_proximity")["housing_median_age"].mean()
# data.frame
df_group2 = df.groupby("ocean_proximity")[["housing_median_age"]].mean()

# with renames
df_group_many = df.groupby("ocean_proximity").agg(
    mean_housing_median_age=("housing_median_age", "mean"),
    sum_population=("population", sum),
    sum_isna_total_bedrooms=("total_bedrooms", lambda x: (x.isna().sum()))
)
# by dict
# silly but just for an example 
# merge columns longitude and latitude and calculate sum
df.columns
group_dict = {'ocean_proximity': 'prox', 
              'longitude': 'tude', 
              'latitude': 'tude'}

df_grouped = df.groupby(group_dict, axis=1)
df_grouped.sum()

### apply function for each group
# top gives n top obs in a column 
def top(df, n=5, column='median_income'):
    return df.sort_values(by=column)[-n:]

top(df)
# creates hierarchical row index
df.groupby('ocean_proximity').apply(top, n=3)
# index in a column group_keys=False
df.groupby('ocean_proximity', group_keys=False).apply(top, n=3)

# exhaustive description of merges for pandas
# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
df_group3 = df.groupby("ocean_proximity").agg(
    mean_housing_median_age=("housing_median_age", "mean"))
# add to have products of groupby in rows in original table; see also transform
df_with_agg = pd.merge(df, df_group3,
                       on="ocean_proximity",
                       how="left")
df_with_agg[["ocean_proximity",
             "housing_median_age",
             "mean_housing_median_age"]]

# transform: calculate a function on grouped table but do not change input
# like mutate/transform in R
# this is done instead of agg
df.columns
df_group5 = df.groupby('ocean_proximity').housing_median_age
df['mean_housing_median_age'] = df_group5.transform('mean')
df.columns

# pivot table (mean is a default)

# index is rows, columns is columns
df.pivot_table(index=['ocean_proximity', 'housing_median_age'], aggfunc="mean")
df.pivot_table(index=['ocean_proximity'],
               columns='housing_median_age',
               aggfunc="mean")

# reshape
# gdy w reshapie jest -1, to python traktuje to jak nieznany wymiar i
# wnioskuje (zgaduje) jaki to musi byc wymiar sam w locie
# często spotykane reshapy:
# (-1, 1) liczba kolumn = 1, liczba wierszy nieznana, do wywnioskowania przez interpreter/kompilator
# (1, -1) liczba wierszy = 1, liczba kolumn nieznana, do wywnioskowania przez interpreter/kompilator

# przykład 1
a = np.arange(10)
print(a)
a = a.reshape((5, 2))
print(a)

# przykład 2
a = np.array([[10, 20, 30], [40, 50, 60]])
print(a)
b = np.reshape(a, (6, 1))
print(b)
# implicitly takes first value as x
c = np.reshape(a, 6)
print(c)

# przykład 3 - 2d array początkowo i robimy reshape (1,-1) lub reshape(-1,1)
a = np.array([[-10, -20, -30], [-40, -50, -60]])
print(a)
print(a.reshape(1, -1))
print(a.reshape(-1, 1))


# ### przekształcanie danych nienumerycznch
print(df.head())

# PRZYPADEK SZCZEGOLNY
# kolumna "pseudonienumeryczna" -- czyli string, którego wartością jest liczba w postaci napisu a nie numeru
# jezeli string jest (pseudo)numeryczny, np. '1', to mozna zamienic jego wartosc na typ numeryczny - int float uzywajac metody astype('int') lub 'float' itp.
# zeby to pokazac zmodyfikuje najpierw kolumne latitude na typ string
# TODO a co jak sa NaN?
df_typechange = df.copy()
df_typechange['lat_as_string'] = df_typechange['latitude'].astype('string')
print(df_typechange['lat_as_string'])

# i ponownie z typu "pseudonumerycznego stringa" na typ float64
df_typechange['lat_as_float64'] = \
    df_typechange['lat_as_string'].astype('float64')
print(df_typechange['lat_as_float64'])

# podobna metoda -> pd.to_numeric(kolumna, error='coerce')
# error='coerce' transforms invalid values to NaN

# INNE PRZYPADKI
# TODO to juz bylo
# dla zbioru housing kolumna TYPOWO nienumeryczna, to np. ocean_proximity
# mamy 5 róznych wartości ['NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND']
print(df.ocean_proximity.unique())
df['oc_prox'] = df['ocean_proximity']
print(df['oc_prox'])

# ### create data.frame

x = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "b": ['lala', 'lulu', 'huhu'],
        "c": [True, False, True],
        "d": [1, np.nan, 4]
    }
)

# TODO add numpy ndarray

#%% melt and pivot

# melt --> wide to long
melted = pd.melt(df, id_vars=('longitude', 'latitude'), 
              value_vars=('housing_median_age','ocean_proximity'))

# # pivot --> long to wide
# # does not work due to some duplicate in index (TODO or find another dataset) 

# melted2 = melted.drop_duplicates(melted.loc[:, ['longitude', 'latitude']])
# pivoted = melted2.pivot(index=('longitude', 'latitude'),
#                         columns='variable',
#                         values='value')

