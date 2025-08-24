# 🎬 Movie Dataset Exploratory Data Analysis (EDA)

## 📖 Project Overview
This project explores and analyzes a custom movie dataset (`mymoviedb.csv`).  
The focus is on **data preprocessing, cleaning, and visualization** to uncover insights about genres, ratings, popularity, and release trends.  

---

## ⚙️ Steps Performed
1. **Data Cleaning**
   - Removed null values and duplicates  
   - Converted datatypes for consistency  
   - Extracted release year  

2. **Feature Engineering**
   - Categorized movies into **worst / bad / average / popular** based on `Vote_Average`  
   - Split multi-genre entries into separate rows  

3. **EDA & Visualization**
   - Most frequent genres  
   - Rating distribution  
   - Popularity extremes (highest/lowest)  
   - Yearly movie release trends  

---

## 📊 Tools Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---

## 🔮 Possible Extensions
- Predict **box office revenue** using regression  
- Build a **Hit vs Flop classifier**  
- Develop a **movie recommendation system**  

---

## 🚀 How to Run
```bash
pip install pandas numpy matplotlib seaborn
python movie_eda.py


