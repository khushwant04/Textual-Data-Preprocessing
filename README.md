# Textual Data Preprocessing

A notebook that walks through basic NLP preprocessing of news articles: filling missing values,
building a text field, stemming, stopword removal and TF-IDF vectorization. It stops at the feature
matrix; the follow-up classifier lives in
[fake-news-prediction](https://github.com/khushwant04/fake-news-prediction).

## Dataset

`train.csv` (~98 MB, 20,800 rows) with columns `id, title, author, text, label`, where
`label = 1` is fake and `0` is real. This matches the Kaggle "Fake News" competition training set.

## Steps (`DataPreprocessing.ipynb`)

1. Load `train.csv`, replace missing values with empty strings.
2. Build `content = author + " " + title`.
3. Clean `content`: keep letters only, lowercase, remove NLTK English stopwords, apply the Porter
   stemmer.
4. Fit a `TfidfVectorizer` on the cleaned text to produce a sparse feature matrix `X` and labels `Y`.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook DataPreprocessing.ipynb
```

The notebook calls `nltk.download('stopwords')` on first run. Stemming 20,800 rows is slow because
the stopword list is rebuilt for every word; caching `set(stopwords.words('english'))` once speeds it
up considerably.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
