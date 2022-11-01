from google.colab import drive
drive.mount('/content/drive')

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc , col, max 
import matplotlib.pyplot as plts

# create spar session
spark = SparkSession.builder.appName('spark_app').getOrCreate()

# load csv
listening_csv_path = '/content/drive/MyDrive/dataset/listenings.csv'
listening_df = spark.read.format('csv').option('InferSchema', True).option('header', True).load(listening_csv_path)

# drop column 
listening_df = listening_df.drop('date')

# drop NAs 
listening_df = listening_df.na.drop()

# show table stats
listening_df.show()

# see table schema
listening_df.printSchema()

# get shape of a table 
shape = (listening_df.count(), len(listening_df.columns))
print(shape)

# select separate columns
q0 = listening_df.select('artist', 'track')
q0.show()

# filtering 
q1 = listening_df.select('*').filter(listening_df.artist == 'Rihanna')
q1.show()

### top 10 users who are fan of Rihanna

# get counts
q2 = listening_df.select('user_id').filter(listening_df.artist == 'Rihanna').groupby('user_id').agg(count('user_id').alias('count'))
q2.show()
# top 10
q2 = q2.orderBy(desc('count')).limit(10)
q2.show()

# top 10 famous tracks (artist-track buckets)
q3 = listening_df.select('artist', 'track').groupby('artist', 'track').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
q3.show()


# top 10 famous tracks (artist-track buckets) of Rihanna
q4 = listening_df.select('artist', 'track').filter(listening_df.artist == 'Rihanna').groupby('artist', 'track').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
q4.show()

### join

data = listening_df.join(genre_df, how = 'inner', on = ['artist'])
data.show()

# top 10 genre
q5 = data.select('genre').groupby('genre').agg(count('genre').alias('count')).orderBy(desc('count')).limit(10)
q5.show()

# for each user get the most popular genre 

q6 = data.select('user_id', 'genre').groupby('user_id', 'genre').agg(count('*').alias('count'))
q6.show()

q7 = q6.groupby('user_id').agg(max(struct(col('count'), col('genre'))).alias('max'))
q7.show()

q8 = q7.select('user_id', col(max.genre))
q8.show()

# also works
q8 = q7.select('user_id', 'max.genre')
q8.show()

### matplotlib examples

q9 = genre_df.select('genre').filter((col('genre') == 'pop') | (col('genre') == 'rock') | (col('genre') == 'metal') | (col('genre') == 'hip hop')).groupby('genre').agg(count('genre').alias('count'))
q9.show()

q9_list = q9.collect()

labels = [row['genre'] for row in q9_list]
values = [row['count'] for row in q9_list]

print(labels)
print(values)

# barplot 
plts.bar(labels, values)






