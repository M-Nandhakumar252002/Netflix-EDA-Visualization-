# Netflix-EDA-Visualization-


#  Movie Dataset Analysis (EDA)
# Author: Nandha Kumar M


# === Importing Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Dataset ===
df = pd.read_csv("mymoviedb.csv", engine='python')

# Preview dataset
print(df.head())
print(df.info())


#  Data Preprocessing


# Drop rows with null values
df.dropna(inplace=True)
print(f"Rows after dropping nulls: {df.shape[0]}")

# Convert datatypes
df['Vote_Count'] = df['Vote_Count'].astype(int)
df['Vote_Average'] = df['Vote_Average'].astype(np.float64)
print(df.info())

# Check duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Summary statistics
print(df.describe())

# Drop irrelevant columns
columns = ['Overview', 'Original_Language', 'Poster_Url']
df.drop(columns, axis=1, inplace=True)

# Extract only year from release date
df['Release_Date'] = pd.to_datetime(df['Release_Date'])
df['Release_Date'] = df['Release_Date'].dt.year


#  Categorizing Vote_Average


def cat_col(df, col, labels):
    """
    Function to categorize a numeric column into bins based on quantiles
    """
    edges = list(df[col].quantile([0, 0.25, 0.5, 0.75, 1]))
    df[col] = pd.cut(df[col], labels=labels, duplicates='drop', bins=edges)
    return df

labels = ['worst', 'bad', 'average', 'popular']
df = cat_col(df, 'Vote_Average', labels)
print(df['Vote_Average'].value_counts())


#  Genre Preprocessing


# Split multiple genres into rows
df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre').reset_index(drop=True)
df['Genre'] = df['Genre'].str.strip()
df['Genre'] = df['Genre'].astype('category')


#  Data Visualization


sns.set_style('whitegrid')

# Genre distribution
sns.catplot(y='Genre', data=df, kind='count', order=df['Genre'].value_counts().index)
plt.title('Genre Distribution')
plt.show()

# Distribution of Vote_Average categories
sns.catplot(y='Vote_Average', data=df, kind='count', order=df['Vote_Average'].value_counts().index)
plt.title('Vote_Average Distribution')
plt.show()

# Movie with highest popularity
print(" Movie with Highest Popularity:")
print(df[df['Popularity'] == df['Popularity'].max()])

# Movie with lowest popularity
print("Movie with Lowest Popularity:")
print(df[df['Popularity'] == df['Popularity'].min()])

# Yearly movie distribution
df['Release_Date'].hist()
plt.title('Number of Movies Released per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()


