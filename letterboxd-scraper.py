import argparse
import kagglehub
import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests
import re
import unicodedata
import sys

def remove_accents(text: str) -> str:
    """Decomposes unicode characters and strips accent marks (diacritics)."""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def main(output: str, num_reviews: int = 3):
    """Scrapes Letterboxd for movie reviews of movies in chosen dataset."""

    # Grab basic movies dataset from Kaggle
    path = kagglehub.dataset_download("aditya126/movies-box-office-dataset-2000-2024")
    movies_dataset = pd.read_csv(path + "/enhanced_box_office_data(2000-2024)u.csv")

    # Loop to find reviews for each movie, and keep track of missing movie pages/reviews
    reviews_dataset = []
    missed_movies = []
    counter = 0
    for title, year in zip(movies_dataset["Release Group"], movies_dataset["Year"]):
        counter += 1
        # Remove any weird characters and turn title into URL-friendly forms
        title = remove_accents(str(title))
        query_with_year = "-".join((re.sub(r'[^\w\s\-]', '', f"{title}-{year}".lower())).split())
        query = "-".join((re.sub(r'[^\w\s\-]', '', f"{title}".lower())).split())

        # Look for the movie's Letterboxd page
        search_url = f"https://letterboxd.com/film/{query_with_year}/"
        response = requests.get(search_url)
        if response.status_code == 404:
            # Try URL with and without year of release
            search_url = f"https://letterboxd.com/film/{query}/"
            response = requests.get(search_url)
            if response.status_code == 404 or response == None:
                missed_movies.append(search_url)
                continue
        elif response == None:
            missed_movies.append(search_url)
            continue

        # If page found, parse the "Popular Reviews" section for top number of reviews
        soup = BeautifulSoup(response.text, "html.parser")
        popular_section = soup.select_one('section.js-popular-reviews')
        # If no popular reviews exist, move on
        if popular_section == None:
            missed_movies.append(search_url)
            continue
        reviews = []
        articles = popular_section.select('article.js-production-viewing')[:num_reviews]
        for article in articles:
            body_elem = article.select_one('.js-review-body')
            review_text = body_elem.get_text(separator=' ', strip=True) if body_elem else ''
            reviews.append(review_text)
        reviews_dataset.append(reviews)

        # Print progress bar 
        sys.stdout.write(f"\rParsing movies: {counter}/{len(movies_dataset)}")
        sys.stdout.flush()

    # Add a new reviews column to given dataset, and save as a new CSV file
    movies_dataset["Reviews"] = reviews_dataset
    movies_dataset.to_csv(output, index=False)
    print(f"\nSaved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Letterboxd reviews for movies dataset.")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--num_reviews", type=int, default=3, help="Number of popular reviews to scrape per movie (default: 3)")
    args = parser.parse_args()
    main(output=args.output, num_reviews=args.num_reviews)