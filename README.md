# Machine Learning Notebooks: Sentiment and Retail Sales

Two exploratory notebooks cover review sentiment classification and retail sales forecasting.

## Contents

- [Sentiment Analysis](Sentiment%20Analysis.ipynb): compares TF-IDF, averaged Word2Vec, and sentence-transformer representations.
- [Store Sales Prediction](Store%20Sales%20Prediction.ipynb): explores daily store-sales regression and time-based features.

## Data

Datasets and generated artifacts are intentionally excluded from this repository.

### Review sentiment

The source of the previously committed review sample was not recorded, so it is not redistributed here. To use the notebook, provide a dataset you are allowed to use with `review` (text) and `rating` (numeric) columns at:

```text
data/raw/sentiment/train_sentiment.csv
```

The notebook labels ratings of 4 or 5 as positive and lower ratings as negative.

### Retail sales

The sales and store schemas match Kaggle's [Rossmann Store Sales competition data](https://www.kaggle.com/competitions/rossmann-store-sales/data). Kaggle lists the dataset license as **Subject to Competition Rules**. Review the competition's current terms before using the data; the files are not included here. After obtaining the files through Kaggle, place them at:

```text
data/raw/rossmann/train.csv
data/raw/rossmann/store.csv
```

## Setup

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```

Open either notebook from the repository root after placing its input data in the paths above.

## Sentiment results

The notebook records F1 values from its exploratory comparisons. TF-IDF scores highest in the reported logistic-regression and KNN results; these runs do not establish which representation generalizes best.

| Representation | Logistic regression | KNN |
| --- | ---: | ---: |
| TF-IDF | 0.861 | 0.860 |
| Averaged Word2Vec | 0.855 | 0.858 |
| Sentence embeddings | 0.853 | 0.858 |

## Retail-sales workflow

The notebook merges daily sales with store metadata, builds date and lag features, fits a random forest and a tuned linear-regression baseline, and evaluates with RMSPE. Results depend on the competition data and the split used in the notebook.

## Author

Morteza Pazhoum — [@MortezaPZ](https://github.com/MortezaPZ)

Computer Science, K. N. Toosi University of Technology
