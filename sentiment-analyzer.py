import argparse
import ast
import gc
import sys
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm


def parse_reviews_column(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def get_optimal_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return 0
    return -1


def main(input_path, output_path, model_name, batch_size):
    print(f"Loading dataset from {input_path}...")
    df = pd.read_pickle(input_path) if input_path.endswith(".pkl") else pd.read_csv(input_path)

    if "Reviews" not in df.columns:
        print("Error: Input dataset must contain a 'Reviews' column.")
        sys.exit(1)

    device = get_optimal_device()
    print(f"Loading sentiment model ({model_name}) on device: {device}...")
    dtype = torch.float16 if device in ("mps", 0) else torch.float32
    
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model=model_name, device=device, torch_dtype=dtype
    )

    flat_items = []
    movie_word_counts = {idx: [] for idx in range(len(df))}
    movie_exclamations = {idx: [] for idx in range(len(df))}

    for movie_idx, row in df.iterrows():
        reviews = parse_reviews_column(row["Reviews"])
        for rev in reviews:
            if rev and str(rev).strip():
                text = str(rev)
                flat_items.append((len(flat_items), movie_idx, text[:512]))
                words = text.split()
                movie_word_counts[movie_idx].append(len(words))
                movie_exclamations[movie_idx].append(text.count("!"))

    total_reviews = len(flat_items)
    if total_reviews == 0:
        print("No valid reviews found.")
        df.to_pickle(output_path)
        return

    flat_items.sort(key=lambda x: len(x[2]))
    sorted_texts = [item[2] for item in flat_items]
    label_mapping = {"negative": 0.0, "neutral": 0.5, "positive": 1.0}
    sorted_scores = []

    print("Running batched sentiment inference...")
    with torch.inference_mode(), tqdm(total=total_reviews, desc="Sentiment Inference", unit="rev") as pbar:
        for i in range(0, total_reviews, batch_size):
            chunk = sorted_texts[i : i + batch_size]
            try:
                results = sentiment_analyzer(chunk, batch_size=batch_size, truncation=True, max_length=256)
                scores = [label_mapping.get(res["label"].lower(), 0.5) * res["score"] for res in results]
                sorted_scores.extend(scores)
            except Exception:
                sorted_scores.extend([0.5] * len(chunk))

            if device == "mps" and (i // batch_size) % 20 == 0:
                torch.mps.empty_cache()

            pbar.update(len(chunk))

    movie_scores = {idx: [] for idx in range(len(df))}
    for (item_idx, movie_idx, _), score in zip(flat_items, sorted_scores):
        movie_scores[movie_idx].append(score)

    sentiment_std = []
    positive_ratio = []
    mean_word_count = []
    exclamation_density = []
    reviews_processed = []

    for idx in range(len(df)):
        scores = movie_scores[idx]
        w_counts = movie_word_counts[idx]
        excl = movie_exclamations[idx]
        
        n = len(scores)
        reviews_processed.append(n)
        if n > 0:
            s_series = pd.Series(scores)
            sentiment_std.append(s_series.std() if n > 1 else 0.0)
            positive_ratio.append((s_series > 0.5).mean())
            mean_word_count.append(pd.Series(w_counts).mean())
            exclamation_density.append(pd.Series(excl).mean())
        else:
            sentiment_std.append(0.0)
            positive_ratio.append(0.0)
            mean_word_count.append(0.0)
            exclamation_density.append(0.0)

    df["sentiment_std"] = sentiment_std
    df["positive_ratio"] = positive_ratio
    df["mean_word_count"] = mean_word_count
    df["exclamation_density"] = exclamation_density
    df["reviews_processed"] = reviews_processed

    del sentiment_analyzer
    gc.collect()

    df.to_pickle(output_path)
    print(f"Successfully saved sentiment features to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract lowercase sentiment features with progress bar.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--batch_size", type=int, default=24)
    args = parser.parse_args()
    main(args.input, args.output, args.model, args.batch_size)