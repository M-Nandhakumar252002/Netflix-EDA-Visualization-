# Netflix-EDA-Visualization-

## Overview
This project explores the Netflix dataset using **Python, Pandas, and Seaborn** to find insights on content distribution, genres, and ratings.

## Dataset
Source: [Kaggle - Netflix Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)

## Objectives
- Data cleaning (handle nulls, duplicates).
- Exploratory analysis on:
  - Content by country
  - Genre distribution
  - Release year trends
- Visualizations with Seaborn & Matplotlib.

## Example Code
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("netflix_titles.csv")

# Top 10 countries by content
top_countries = df['country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 Countries by Netflix Content")
plt.show()
