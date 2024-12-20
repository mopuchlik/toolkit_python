# -*- coding: utf-8 -*-
"""charts.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yBjj-fEEdBTe48t0Z86pveXNZXd_lu4Y
"""
#%%
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import plotly.graph_objects as go

# %% current directory
# ### Get the current working directory (adjust)
# cwd = "/home/michal/Dropbox/Frontex_prep//"
cwd = "Q:/Dropbox/Frontex_prep/"
# cwd = "D:/Dropbox/programowanie/python_general"

# TODO: for some reason does not work in Linux
# cwd = os.path.dirname(os.path.abspath("__file__"))

os.chdir(cwd)
print("Current working directory: {0}".format(os.getcwd()))

# %% loading a csv file
path = "housing.csv"
df = pd.read_csv(path)

df["mean_age_in_geo"] = df.groupby(["latitude", "longitude"])["housing_median_age"].transform("mean")
df.head()

#%% geospacial plot California

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Set the coordinate reference system (CRS) to EPSG:4326 (WGS 84 latitude/longitude)
gdf.set_crs(epsg=4326, inplace=True)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Reproject to a meter-based CRS for compatibility with contextily
gdf = gdf.to_crs(epsg=3857)

# Plot the points, with sizes based on mean_age_in_geo
gdf.plot(
    ax=ax,
    markersize=gdf['mean_age_in_geo'] * 2,  # Adjust marker size as needed
    color='blue',
    alpha=0.4,
    edgecolor='black'
)

# Add a basemap of California
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Add labels
ax.set_title('California Map with Circles Sized by Mean Age in Geo')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()
fig.savefig("images/map_scatter_chart.png")

#%% pie plot

# Categorize 'HouseAge' into 3 groups: Young, Middle-aged, Old
bins = [0, 10, 25, 50, float('inf')]
labels = ['Young', "Lower-middle-aged", 'Upper-middle-aged', 'Old']
df['HouseAgeCategory'] = pd.cut(df['housing_median_age'], bins=bins, labels=labels)

# Count the occurrences of each category
category_counts = df['HouseAgeCategory'].value_counts()

# Update the labels with the age ranges
category_labels = [
    'Young (0-10)',
    "Lower-middle-aged (11-25)",
    "Upper-middle-aged (26-50)",
    'Old (51+)'
]
# Custom colors to match the FRONTEX color palette
frontex_colors = ['#02339d', '#f2c52b', '#0394d5', '#7ab722']  # Blue, Yellow, Light Blue, Green
frontex_colors1 = ['#02339d', '#ffffff', '#0394d5', '#7ab722']  # Blue, White, Light Blue, Green

# Plot using Plotly for a fancier, 3D pie chart
fig = go.Figure(data=[go.Pie(
    labels=category_labels,  # Updated labels with age ranges
    values=category_counts.values,
    hole=0.3,  # Donut chart (optional)
    pull=[0.1, 0, 0, 0],  # Explode the first slice
    textinfo='percent+label',  # Display percentages with labels
    marker=dict(colors=frontex_colors1),  # Custom colors based on FRONTEX palette
    opacity=0.9
)])

# Update layout to make the chart more visually appealing
fig.update_layout(
    title='Distribution of House Age Categories',
    title_x=0.5,  # Center the title
    template='plotly_dark',  # plotly_dark --> Dark theme for a fancier look
    showlegend=True,  # Show legend
    width=900,  # Set a narrower width
    height=600  # Adjust the height if needed
)

fig.show()

fig.write_html("images/interactive_pie_chart.html")


# %%
