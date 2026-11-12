# Movie Box Office & Review Analysis Pipeline

A modular pipeline to scrape movie reviews from Letterbox and train a LightGBM classification model to predict box office revenue using the scarped reviews and other features.


## Prerequisites & Installation

Make sure to pip install the requirements.txt


## Pipeline Usage Instructions

Run the scripts in the following order to execute the full data processing and training workflow.


### 1. Scrape Letterboxd Reviews (letterboxd-scraper.py)
Downloads the box office dataset from Kaggle and scrapes popular reviews from Letterboxd for each movie.

**Usage:**
  python3 letterboxd-scraper.py --output <OUTPUT_FILE_PATH> [--num_reviews 15] [--max_pages 3] [--delay 1.0] [--checkpoint 50] [--workers 6]

**Arguments:**
  * `--output` *(Required)*: Path to save the output CSV (and `.pkl` checkpoint) file containing the scraped reviews.
  * `--num_reviews` *(Optional)*: Maximum number of reviews to collect per movie (defaults to `15`).
  * `--max_pages` *(Optional)*: Maximum number of review pagination pages to traverse per movie (defaults to `3`).
  * `--delay` *(Optional)*: Sleep delay in seconds between page requests to avoid rate-limiting (defaults to `1.0`).
  * `--checkpoint` *(Optional)*: Frequency of completed movies between periodic backup saves (defaults to `50`).
  * `--workers` *(Optional)*: Number of concurrent worker threads running in parallel (defaults to `6`).


### 2. Numerical Encoding (data-processor.py)
Processes the updated dataset, handles categorical features, multi-hot encodes lists (Genres, Production Countries), bins continuous revenue targets into classes (`0` to `3`), and exports local pickle files (`.pkl`).

* **Usage:**
  python3 data-processor.py --input <RAW CSV FILE PATH>
* **Arguments:**
  * `--input` *(Required)*: Path to the input CSV dataset generated from the scraping step.


### 3. LightGBM Classification Predictor (predictor.py)
Ingests the preprocessed pickle files, flattens multi-hot lists into a numeric matrix, and trains/evaluates a LightGBM multi-class classifier.

* **Usage:**
  python3 predictor.py --features <X .pkl filename> --targets <y .pkl filename> --target <either "Domestic" or "Foreign">
* **Arguments:**
  * `--features` *(Required)*: Path to the features pickle file (e.g., `movies_dataset_x.pkl`).
  * `--targets` *(Required)*: Path to the targets pickle file (e.g., `movies_dataset_y.pkl`).
  * `--target` *(Required)*: Name of the target column to predict (e.g., `Domestic` or `Foreign`).