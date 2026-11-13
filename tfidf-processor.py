import argparse
import ast
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def parse_reviews_column(val):
    if isinstance(val, list):
        return " ".join([str(r) for r in val if r and str(r).strip()])
    if pd.isna(val) or not isinstance(val, str):
        return ""
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return " ".join([str(r) for r in parsed if r and str(r).strip()])
    except (ValueError, SyntaxError):
        pass
    return str(val)


def main(input_path, output_path, max_features, min_df):
    print(f"Loading dataset from {input_path}...")
    df = pd.read_pickle(input_path) if input_path.endswith(".pkl") else pd.read_csv(input_path)

    if "Reviews" not in df.columns:
        raise ValueError("Input dataset must contain a 'Reviews' column.")

    print("Parsing review corpora...")
    corpus = [parse_reviews_column(val) for val in tqdm(df["Reviews"], desc="Parsing Reviews")]

    print(f"Fitting TF-IDF vectorizer (max_features={max_features}, min_df={min_df})...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english",
        min_df=min_df
    )
    X_tfidf = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out().tolist()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "tfidf_matrix": X_tfidf,
            "feature_names": feature_names
        }, f)

    print(f"Successfully saved TF-IDF bundle to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract TF-IDF sparse matrix with progress bar.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_features", type=int, default=100)
    parser.add_argument("--min_df", type=int, default=10)
    args = parser.parse_args()
    main(args.input, args.output, args.max_features, args.min_df)