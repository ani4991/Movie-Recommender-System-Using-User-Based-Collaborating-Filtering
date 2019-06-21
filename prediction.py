# LIBRARIES IMPORT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT DATASET
movie_titles_df = pd.read_csv("Movie_Id_Titles")
movie_titles_df.head(20)

movies_rating_df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

print(movies_rating_df.head(10),movies_rating_df.tail())

# dropped the timestamp
movies_rating_df.drop(['timestamp'], axis = 1, inplace = True)
print(movies_rating_df)
print(movies_rating_df.describe())

# merged both dataframes together to have the ID with the movie name
movies_rating_df = pd.merge(movies_rating_df, movie_titles_df, on = 'item_id')
print(movies_rating_df.shape)

# VISUALIZE DATASET

movies_rating_df.groupby('title')['rating'].describe()

ratings_df_mean = movies_rating_df.groupby('title')['rating'].describe()['mean']
ratings_df_count = movies_rating_df.groupby('title')['rating'].describe()['count']

ratings_mean_count_df = pd.concat([ratings_df_count, ratings_df_mean], axis = 1)
print(ratings_mean_count_df.reset_index())

ratings_mean_count_df['mean'].plot(bins=100, kind='hist', color = 'r')
ratings_mean_count_df['count'].plot(bins=100, kind='hist', color = 'r')

# found the highest rated movies!

print(ratings_mean_count_df[ratings_mean_count_df['mean'] == 5])

# Listed all the movies that are most rated
print(ratings_mean_count_df.sort_values('count', ascending = False).head(100))

# Tried ITEM-BASED COLLABORATIVE FILTERING ON ONE MOVIE SAMPLE
userid_movietitle_matrix = movies_rating_df.pivot_table(index = 'user_id', columns = 'title', values = 'rating')
print(userid_movietitle_matrix)
titanic = userid_movietitle_matrix['Titanic (1997)']
print(titanic)

# calculated the correlations
titanic_correlations = pd.DataFrame(userid_movietitle_matrix.corrwith(titanic), columns=['Correlation'])
titanic_correlations = titanic_correlations.join(ratings_mean_count_df['count'])
print(titanic_correlations)

titanic_correlations.dropna(inplace=True)
print(titanic_correlations)

# sort the correlations vector
print(titanic_correlations.sort_values('Correlation', ascending=False))

print(titanic_correlations[titanic_correlations['count']>80].sort_values('Correlation',ascending=False).head())

# CREATED AN ITEM-BASED COLLABORATIVE FILTER ON THE ENTIRE DATASET
print(userid_movietitle_matrix)

movie_correlations = userid_movietitle_matrix.corr(method = 'pearson', min_periods = 80)
# pearson : standard correlation coefficient
# Obtained the correlations between all movies in the dataframe
















