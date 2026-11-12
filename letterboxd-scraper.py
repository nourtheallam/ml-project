import argparse
import kagglehub
import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests
import re
import unicodedata
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def remove_accents(text):
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join(
        c for c in nfkd_form
        if not unicodedata.combining(c)
    )


def make_URL(title):
    title = remove_accents(str(title))
    title = re.sub(r"[^\w\s\-]", "", title.lower())
    return "-".join(title.split())


def extract_reviews(soup):
    """Extract review text strings from a Letterboxd review page."""
    reviews = []
    articles = soup.select("div.viewing-list div.listitem article.production-viewing")
    for article in articles:
        body = article.select_one(".js-review-body")
        if body:
            text = body.get_text(separator=" ", strip=True)
            if text:
                reviews.append(text)
    return reviews


def scrape_movie_reviews(
    session,
    title,
    year,
    num_reviews=15,
    max_pages=3,
    delay=2
):
    """Collect reviews directly from a movie's Letterboxd review pages."""
    reviews = []

    # Try the movie page with and without year
    year_and_title = f"{make_URL(title)}-{year}"
    title_only = make_URL(title)
    
    candidate_urls = [
        f"https://letterboxd.com/film/{year_and_title}/reviews/",
        f"https://letterboxd.com/film/{title_only}/reviews/"
    ]
    
    base_review_url = None

    for url in candidate_urls:
        try:
            response = session.get(url, timeout=20)
            if response.status_code == 200:
                base_review_url = url
                soup = BeautifulSoup(response.text, "html.parser")
                page_reviews = extract_reviews(soup)
                if page_reviews:
                    reviews.extend(page_reviews)
                break
            elif response.status_code not in (404,):
                return []
        except requests.exceptions.RequestException:
            return []

    if not base_review_url:
        return []

    # Collect reviews directly from review pages
    for page in range(2, max_pages + 1):
        if len(reviews) >= num_reviews:
            break
            
        url = f"{base_review_url}page/{page}/"
        try:
            # Try to reach the review page
            response = session.get(url, timeout=20)
            if response.status_code != 200:
                break

            # Try to scrape reviews from the page
            soup = BeautifulSoup(response.text, "html.parser")
            page_reviews = extract_reviews(soup)
            if not page_reviews:
                break            

            reviews.extend(page_reviews)
            time.sleep(delay)
            
        except requests.exceptions.RequestException:
            break

    return reviews[:num_reviews]


def process_single_movie(index, title, year, num_reviews, max_pages, delay):
    """Worker function to scrape a single movie using its own thread-safe session."""
    session = requests.Session(impersonate="chrome")
    try:
        reviews = scrape_movie_reviews(
            session=session,
            title=title,
            year=year,
            num_reviews=num_reviews,
            max_pages=max_pages,
            delay=delay
        )
    finally:
        session.close()
    return index, title, reviews


def main(output, num_reviews=15, max_pages=3, delay=2, checkpoint=25, workers=5):

    # Load the movies dataset
    print("Loading movie dataset...")
    path = kagglehub.dataset_download("aditya126/movies-box-office-dataset-2000-2024")
    movies_dataset = pd.read_csv(path + "/enhanced_box_office_data(2000-2024)u.csv")

    # Initialize reviews array for every movie
    results = [[] for _ in range(len(movies_dataset))]
    # And track movies whose reviews we couldn't find
    missed_movies = []
    # And track total number of reviews scraped
    total_reviews = 0
    completed_count = 0

    print(f"Starting concurrent scrape with {workers} workers...")
    
    lock = threading.Lock()

    def task_wrapper(i, row):
        nonlocal total_reviews, completed_count
        title = row["Release Group"]
        year = row["Year"]

        # User worker to scrape reviews
        idx, title_res, reviews = process_single_movie(
            i, title, year, num_reviews, max_pages, delay
        )
        
        with lock:
            completed_count += 1
            results[idx] = reviews
            total_reviews += len(reviews)
            if not reviews:
                missed_movies.append(str(title_res))

            # And print progress
            sys.stdout.write(
                f"\rMovies: {completed_count}/{len(movies_dataset)} "
                f"| Reviews: {total_reviews}"
            )
            sys.stdout.flush()

            # Save reviews dataset periodically
            if completed_count % checkpoint == 0:
                movies_dataset["Reviews"] = results
                movies_dataset.to_pickle(
                    output.replace(".csv", ".pkl")
                )
                print("\nCheckpoint saved.")

        return reviews

    # Execute scraping in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(task_wrapper, i, row)
            for i, row in movies_dataset.iterrows()
        ]
        for future in as_completed(futures):
            future.result()

    # Final save
    movies_dataset["Reviews"] = results
    movies_dataset.to_csv(output, index=False)
    movies_dataset.to_pickle(output.replace(".csv", ".pkl"))
    print(f"\nSaved to {output}")
    print(f"Movies without reviews: {len(missed_movies)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--num_reviews",
        type=int,
        default=15
    )

    parser.add_argument(
        "--max_pages",
        type=int,
        default=3
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=50
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of concurrent worker threads"
    )

    args = parser.parse_args()

    main(
        output=args.output,
        num_reviews=args.num_reviews,
        max_pages=args.max_pages,
        delay=args.delay,
        checkpoint=args.checkpoint,
        workers=args.workers
    )