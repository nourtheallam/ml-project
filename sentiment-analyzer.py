import argparse
import ast
import gc
import sys
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from tqdm import tqdm
from joblib import Parallel, delayed

tqdm.pandas(desc="")

def classify_row(classifier, reviews_row):
    """Translates each row of reviews into a list and classifies each review in it."""
    sentiments = []
    reviews_row_parsed = ast.literal_eval(reviews_row)
    reviews_list = [
        review for review in reviews_row_parsed
        if isinstance(review, str) and review.strip()
    ]
    sentiments = classifier(
        reviews_list,
        truncation=True,
        max_length=512,
        batch_size=16
    )
    return sentiments

def main(input_path, output_path):
    # Accept both .csv and .pkl input
    print(f"Loading dataset from {input_path}...")
    df = pd.read_pickle(input_path) if input_path.endswith(".pkl") else pd.read_csv(input_path)

    # Make sure input file has "Reviews" column
    if "Reviews" not in df.columns:
        print("Error: Input dataset must contain a 'Reviews' column.")
        sys.exit(1)

    # Load a pre-trained RoBERTa model
    print(f"Loading model...")
    classifier = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest", dtype=torch.float16, 
                          device="mps" if torch.backends.mps.is_available() else "cpu")

    # Label each review with a positive/negative sentiment label with some confidence, apply to one row at a time
    df["review_sentiments"] = df["Reviews"].progress_apply(lambda reviews: classify_row(classifier, reviews))

    # Save output to either .pkl or .csv
    df.to_pickle(output_path) if output_path.endswith(".pkl") else df.to_csv(output_path, index=False)
    print(f"Successfully saved sentiment features to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifies text into positive, neutral, or negative sentiment labels using CardiffNLP RoBERTa")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)