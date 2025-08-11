# Real-Time Trend Analysis

A machine learning pipeline for analyzing and predicting trending topics in real-time using Reddit post data.

## 📂 Project Structure

```

REAL\_TIME\_TREND\_ANALYSIS/
│
├── datasets/
│   ├── reddit\_posts.csv        # Raw Reddit data (title + text)
│   ├── cleaned\_posts.csv       # Preprocessed and cleaned text data
│   ├── clustered\_posts.csv     # Data with clustering results (KMeans)
│
├── notebook/
│   └── Real-Time-Trend-Analysis.ipynb  # Jupyter notebook with the full pipeline

````

## 📊 Workflow Overview

1. **Data Collection**  
   - The raw data (`reddit_posts.csv`) contains post titles and text scraped from Reddit.

2. **Data Cleaning**  
   - Removes URLs, punctuation, numbers, and stopwords.  
   - Saves cleaned output in `cleaned_posts.csv`.

3. **Clustering (KMeans)**  
   - Groups posts into topic clusters using TF-IDF vectorization.  
   - Results stored in `clustered_posts.csv`.

4. **Trending Prediction**  
   - Marks certain clusters as "trending" (example: cluster 0 & 1).  
   - Uses **Logistic Regression** with **SMOTE** to handle class imbalance.  
   - Evaluates accuracy on train/test splits.

5. **Visualization**  
   - Bar charts comparing train vs test accuracy.  
   - Optionally, confusion matrices for better performance insight.

## 🛠️ Requirements

Install the dependencies using:

```bash
pip install pandas scikit-learn imbalanced-learn nltk matplotlib
````

Also download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

## ▶️ Running the Project

1. Place your `reddit_posts.csv` file in the `datasets/` folder.
2. Open `notebook/Real-Time-Trend-Analysis.ipynb`.
3. Run all cells to:

   * Clean the dataset
   * Cluster posts
   * Train the trending prediction model
   * Visualize results

## 📌 Notes

* You can adjust the number of clusters in the **KMeans** step.
* Change the logic for what counts as "trending" based on your dataset.
* The model currently uses a maximum of 1000 TF-IDF features for efficiency.
