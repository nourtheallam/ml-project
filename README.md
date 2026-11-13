# Comprehensive Box Office Revenue Prediction & Ablation Study Pipeline

This document serves as the complete technical reference and operational guide for the automated end-to-end machine learning pipeline built to predict domestic and foreign box office revenue using structural metadata, sentiment statistics, emotion scores, and textual TF-IDF features.

---

## 1. System Overview & Methodology

* **Target Variable Transformation:** Box office revenues exhibit a massive right-skew (ranging from indie films earning $50,000 to blockbusters earning $2,000,000,000). The pipeline applies a `log1p` transformation to stabilize variance and prevent outlier dominance during training.
* **Core Regressor:** Utilizes **LightGBM** (`LGBMRegressor`), configured with robust hyperparameter tuning via `RandomizedSearchCV` across a 3-fold cross-validation split, coupled with early stopping to prevent overfitting.
* **Ablation Study Structure:** Systematically evaluates six distinct feature configurations to quantify the performance lift provided by NLP text semantics over baseline structural metadata.

---

## 2. Complete File & Script Directory

Here is the operational breakdown of every script in the pipeline:

* **`run_all_experiments.py` (The Orchestrator)**
  * Manages the end-to-end execution flow.
  * Sequentially invokes preprocessing, sentiment extraction, emotion extraction, and TF-IDF vectorization.
  * Iterates through all 6 ablation configurations for both `Domestic` and `Foreign` revenue targets, passing appropriate feature flags to `predictor.py`.
  * Automatically triggers `plot_results.py` upon completion.

* **`data-processor.py` (Metadata Feature Extractor)**
  * Cleans the raw dataset and drops unneeded fields.
  * Encodes categorical metadata: maps single-value categorical columns (e.g., original language) and generates multi-hot encoded vectors for multi-value fields (e.g., genres, production countries).
  * Outputs `processed_features_x.pkl` and `processed_targets_y.pkl`.

* **`sentiment-analyzer.py` (Sentiment Feature Extractor)**
  * Parses review text using a transformer model (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
  * Computes aggregated numerical statistics per movie: sentiment standard deviation, positive sentiment ratio, mean word count, and exclamation mark density.
  * Outputs `sentiment_features.pkl`.

* **`emotion-analyzer.py` (Emotion Feature Extractor)**
  * Extracts specialized emotional indicators from review corpora (specifically focusing on mean joy and mean anger scores).
  * Captures polarity nuances that standard positive/negative sentiment metrics miss.
  * Outputs `emotion_features.pkl`.

* **`tfidf-processor.py` (Text Vectorizer)**
  * Tokenizes review text, strips English stop words, and fits a `TfidfVectorizer` (unigrams and bigrams, limited by max features and minimum document frequency thresholds).
  * Generates a sparse matrix bundle stored as `tfidf_bundle.pkl`.

* **`predictor.py` (Model Trainer & Tuner)**
  * Aligns indices across metadata, sentiment, and sparse TF-IDF feature arrays.
  * Stacks tabular arrays and sparse matrices using `scipy.sparse.hstack`.
  * Tunes hyperparameters (`n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, regularization terms) via `RandomizedSearchCV`.
  * Trains the final LightGBM model with early stopping and saves model artifacts inside `models/`.

* **`plot_results.py` (Diagnostic Visualization Suite)**
  * Scans saved model pickles in the `models/` directory (robustly parsing configuration names and targets from metadata dictionaries or filenames).
  * Automatically generates configuration-specific parity plots, residual plots, and cross-ablation comparative bar charts.
  * Saves all figures to the designated plots directory.

---

## 3. The 6-Tier Ablation Study Matrix

The orchestrator runs experiments across these exact feature configurations:

1. **`metadata_only`**: Structural metadata features only (baseline).
2. **`sentiment_only`**: Metadata + transformer-based sentiment statistics.
3. **`emotions_only`**: Metadata + emotion scores (joy and anger).
4. **`tfidf_only`**: Metadata + sparse TF-IDF review n-grams.
5. **`sentiment_plus_emotions`**: Metadata + sentiment statistics + emotion scores.
6. **`all_combined`**: Full model combining metadata, sentiment, emotions, and TF-IDF features.

---

## 4. Environment Setup & Execution Instructions

### Prerequisites
Ensure Python 3.x is installed along with the required machine learning and data science libraries:
```bash
pip install lightgbm transformers scikit-learn tqdm matplotlib pandas scipy