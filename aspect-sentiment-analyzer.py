import argparse
import ast
import sys

import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

tqdm.pandas(desc="")

ASPECTS = (
    "acting",
    "plot",
    "characters",
    "direction",
    "cinematography",
    "music",
    "visual effects",
    "pacing",
    "dialogue",
    "ending",
)


def classify_row(classifier, reviews_row):
    # Parse list of reviews
    reviews_row_parsed = ast.literal_eval(reviews_row)
    reviews_list = [
        review for review in reviews_row_parsed
        if isinstance(review, str) and review.strip()
    ]

    # Return empty list if no reviews exist
    annotations = []
    if not reviews_list:
        return annotations

    # Apply sentiment analysis to every pair of reviews and aspects
    predictions = classifier(
        [{"text": review, "text_pair": aspect}
            for review in reviews_list
            for aspect in ASPECTS
        ],
        truncation=True,
        max_length=512,
        batch_size=15 #= number of reviews, anyway
    )

    # Organize the output of the above as
    # [[{aspect name, label, score for aspect in aspects} for review] for review in reviews_list]
    annotations = [
        [
            {
                "aspect": aspect,
                "label": predictions[
                    review_index * len(ASPECTS) + aspect_index
                ]["label"].lower(),
                "score": predictions[
                    review_index * len(ASPECTS) + aspect_index
                ]["score"],
            }
            for aspect_index, aspect in enumerate(ASPECTS)
        ]
        for review_index, review in enumerate(reviews_list)
    ]

    return annotations



def main(input_path, output_path):
    print(f"Loading dataset from {input_path}...")
    df = pd.read_pickle(input_path) if input_path.endswith(".pkl") else pd.read_csv(input_path)

    # Check that input file has a "Reviews" column
    if "Reviews" not in df.columns:
        print("Error: Input dataset must contain a 'Reviews' column.")
        sys.exit(1)

    # Load DeBERTa model
    print("Loading aspect-based sentiment model...")
    classifier = pipeline(
        task="text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        dtype=torch.float16,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Calculate aspect-separated sentiments for the reviews of one row at a time (helps see progress)
    df["review_aspect_sentiments"] = None
    try:
        for index in tqdm(df.index):
            df.at[index, "review_aspect_sentiments"] = classify_row(
                classifier,
                df.at[index, "Reviews"]
            )
    except KeyboardInterrupt:
        # Save whatever's been collected so far if interrupted
        print("\nProcess interrupted. Saving completed annotations...")
        df[["review_aspect_sentiments"]].to_csv(
            "interrupted_results.csv",
            index=False
        )
        print("Saved to interrupted_results.csv")
        sys.exit(0)

    # Save final file as either .csv or .pkl
    df.to_pickle(output_path) if output_path.endswith(".pkl") else df.to_csv(
        output_path,
        index=False
    )

    print(f"Successfully saved aspect sentiment features to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classifies review aspects as positive, neutral, or negative."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)