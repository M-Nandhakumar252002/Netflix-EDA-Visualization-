# Netflix-EDA-Visualization-

## Overview
This project explores the Netflix dataset using **Python, Pandas, and Seaborn** to find insights on content distribution, genres, and ratings.

## Dataset
Source: (https://github.com/TheiScale/YouTube-Video-Notes/blob/main/New%20Netflix%20Data%20Analysis%20Project%202025/mymoviedb.csv)

## Objectives
- Data cleaning (handle nulls, duplicates).
- Exploratory analysis on:
  - Genre distribution
  - Release year trends
- Visualizations with Seaborn & Matplotlib.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("mymoviedb.csv" ,engine= 'python')

df.head()

df.info()

#dropping rows which has null values
df.dropna(inplace=True) 

df.shape[0]

# converting vote count and vote average to integer and float respectively
df['Vote_Count']= df['Vote_Count'].astype(int)
df['Vote_Average']= df['Vote_Average'].astype(np.float64)


df.info()

# checking for duplicates
df.duplicated().sum()

df.describe()

DATA PREPROCESSING

# dropping the columns not required for EDA -- overview, language and poster
columns = ['Overview','Original_Language','Poster_Url']
df.drop(columns, axis = 1, inplace = True)

df.head()

# keeping only year and dropping other details in data
df['Release_Date']= pd.to_datetime(df['Release_Date'])
df['Release_Date']= df['Release_Date'].dt.year

df.head()

# creating a function for categorizing the vote_average column into 4 sections
def cat_col(df, col, labels):
    edges = list(df[col].quantile([0, 0.25, 0.5, 0.75, 1]))
    df[col] = pd.cut(df[col], labels=labels, duplicates = 'drop', bins = edges)
    return df

labels = ['worst','bad','average','popular']
df = cat_col(df, 'Vote_Average', labels)

df.head()

df['Vote_Average'].value_counts()

df.dropna(inplace=True)

df.isna().sum()

We split genres 

df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre').reset_index(drop = True)
df['Genre'] = df['Genre'].str.strip()

df.head()

# categorising genre
df['Genre']= df['Genre'].astype('category')

Data visualization

sns.set_style('whitegrid')

# Most frequent genre of movies released in **netflix** 

df['Genre'].describe()

genre distribution

sns.catplot(y = 'Genre', data = df, kind = 'count', order = df['Genre'].value_counts().index) 
plt.title('Genre distribution')
plt.show()

plotting graph for finding highest voted movies

sns.catplot(y = 'Vote_Average', data = df, kind = 'count', order = df['Vote_Average'].value_counts().index)
plt.title('Highest Voted ')
plt.show()

Movie with highest Popularity

df[df['Popularity']== df['Popularity'].max()]

Movie with lowest Popularity

df[df['Popularity']== df['Popularity'].min()]

Year for which more movies released

df['Release_Date'].hist()
plt.title('Year/movie distribution')

