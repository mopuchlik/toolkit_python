# %% imports
import pandas as pd
import os
import numpy as np
import plotly.express as px


# %% current directory
# ### Get the current working directory (adjust)
cwd = "/home/michal/Dropbox/Frontex_prep//"

# cwd = "D:/Dropbox/programowanie/python_general"

# TODO: for some reason does not work in Linux
# cwd = os.path.dirname(os.path.abspath("__file__"))

os.chdir(cwd)
print("Current working directory: {0}".format(os.getcwd()))

# %% loading a csv file
path = "housing.csv"
df = pd.read_csv(path)

# %% histogram

fig = px.histogram(
    df, x="median_income", template="plotly_white", opacity=0.8, color="ocean_proximity"
)
fig.show()
fig.write_image("images/hist_med_income.png")

# %% line


# fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
# fig.show()

# %% bar plot
# NOTE: stupid example, takes sum of mediam_income probably as default

fig = px.bar(df, x="ocean_proximity", y="median_income")
fig.write_image("images/bar_med_income.png")
fig.show()


# %%
