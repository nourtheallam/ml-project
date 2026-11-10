# Movie Box Office & Review Analysis Pipeline

A modular pipeline to scrape movie reviews from Letterbox and train a LightGBM classification model to predict box office revenue using the scarped reviews and other features.

---

## Prerequisites & Installation

This folder is packaged with a custom virtual python environment. Make sure to run the code within it. 

---

## Pipeline Usage Instructions

Run the scripts in the following order to execute the full data processing and training workflow.

### 1. Scrape Letterboxd Reviews (letterboxd.py)
Downloads the box office dataset from Kaggle and scrapes popular reviews from Letterboxd for each movie.

* **Usage:**
  python letterboxd.py --output movies_dataset_updated.csv [--num_reviews 3]
* **Arguments:**
  * `--output` *(Required)*: Path to save the output CSV file containing the scraped reviews.
  * `--num_reviews` *(Optional)*: Number of popular reviews to scrape per movie (defaults to `3`).

---

### 2. Numerical Encoding (numerical_encoding.py)
Processes the updated dataset, handles categorical features, multi-hot encodes lists (Genres, Production Countries), bins continuous revenue targets into classes (`0` to `3`), and exports local pickle files (`.pkl`).

* **Usage:**
  python numerical_encoding.py --input movies_dataset_updated.csv
* **Arguments:**
  * `--input` *(Required)*: Path to the input CSV dataset generated from the scraping step.

---

### 3. LightGBM Classification Predictor (predictor.py)
Ingests the preprocessed pickle files, flattens multi-hot lists into a numeric matrix, and trains/evaluates a LightGBM multi-class classifier.

* **Usage:**
  python3 predictor.py --features processed-datasets/x-basic.pkl --targets processed-datasets/y.pkl --target Domestic
* **Arguments:**
  * `--features` *(Required)*: Path to the features pickle file (e.g., `movies_dataset_x.pkl`).
  * `--targets` *(Required)*: Path to the targets pickle file (e.g., `movies_dataset_y.pkl`).
  * `--target` *(Required)*: Name of the target column to predict (e.g., `Domestic` or `Foreign`).