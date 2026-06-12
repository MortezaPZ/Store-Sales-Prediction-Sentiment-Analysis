# 📊 Machine Learning Projects: Sentiment Analysis & Store Sales Prediction

This repository contains two machine learning notebooks: a **sentiment classification** project comparing multiple text vectorization techniques, and a **retail sales forecasting** project using regression models. 🤖

## 📂 Project Structure

```
├── Sentiment_Analysis.ipynb       # Review sentiment classification (positive/negative)
└── Store_Sales_Prediction.ipynb   # Daily sales prediction for retail stores
```

---

## 💬 Sentiment Analysis

Classifies product/store reviews as **positive** or **negative** based on their text and rating, comparing three different text representation methods.

### 🧠 Pipeline

1. **Preprocessing** — lowercasing, removing URLs, numbers, and punctuation, tokenization, and stopword removal with `nltk`
2. **Labeling** — reviews are labeled as positive (`1`) if the rating is ≥ 4, otherwise negative (`0`)
3. **Vectorization** — each review is represented using three different methods:
   - 🔤 **TF-IDF**
   - 🧩 **Word2Vec** (averaged word embeddings)
   - 🤖 **BERT** sentence embeddings (`all-MiniLM-L6-v2`)
4. **Modeling** — each vector representation is evaluated with:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Random Forest

   All models are tuned with `GridSearchCV` (5-fold cross-validation, optimizing F1 score).

### 📈 Results (F1 Score, tuned)

| Vectorization | Logistic Regression | KNN |
|----------------|---------------------|-----|
| TF-IDF | 0.861 | 0.860 |
| Word2Vec | 0.855 | 0.858 |
| BERT | 0.853 | 0.858 |

> ⚠️ The dataset is heavily imbalanced toward positive reviews (rating ≥ 4), so all models perform similarly. BERT embeddings are expected to generalize best overall, followed by Word2Vec and TF-IDF.

### 📦 Requirements

```bash
pip install pandas regex nltk scikit-learn gensim sentence-transformers
```

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 📄 Input Data

- `train_sentiment.csv` — must contain `review` (text) and `rating` (numeric) columns

---

## 🏪 Store Sales Prediction

Predicts **daily sales** for retail stores using historical sales data, store metadata, and engineered time-based features.

### 🧠 Pipeline

1. **Data merging** — combines store metadata (`store.csv`) with daily sales records (`train.csv`)
2. **Feature engineering**:
   - 📅 Extracts `Year`, `Month`, `Day`, and `WeekOfYear` from the date
   - 📊 Adds **lag features**: sales from the previous 1–10 days
   - 📉 Adds a **10-day rolling average** of sales
   - 🏬 Fills missing `DistanceToRivalStore` values with the median
   - 🔢 Encodes categorical features (`RetailType`, `Stock variety`, `ContinuousBogoMonths`) with `LabelEncoder`
3. **Modeling**:
   - 🌲 **Random Forest Regressor**
   - 📈 **Linear Regression** (tuned with `GridSearchCV`)
4. **Evaluation** — models are scored using **RMSPE** (Root Mean Square Percentage Error)
5. **Feature Importance** — visualizes which features matter most for the Random Forest model 📊

### 📦 Requirements

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 📄 Input Data

- `train.csv` — daily sales records per store
- `store.csv` — store metadata (retail type, stock variety, distance to rival store, etc.)

---

## 📝 Notes

- Both notebooks are exploratory/educational and were developed iteratively — some cells include tuning experiments and commented-out alternatives.
- Make sure the required CSV files are placed in the same directory as the notebooks before running.

## 👤 Author

Morteza Pazhoum — @MortezaPZ

K.N. Toosi University of Technology — Computer Science
