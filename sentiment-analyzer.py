import argparse
import ast
import gc
import sys
import pandas as pd
import torch
from transformers import pipeline


def parse_reviews_column(val):
    """Safely evaluate stringified lists of reviews loaded from CSV."""
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
    """Select Apple Silicon MPS GPU if available, otherwise CUDA or CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return 0
    return -1


def main(input_path, output_path, model_name, batch_size):
    print(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)

    if "Reviews" not in df.columns:
        print("Error: Input CSV must contain a 'Reviews' column.")
        sys.exit(1)

    device = get_optimal_device()
    print(f"Loading sentiment model ({model_name}) on device: {device} (FP16)...")
    
    # Use float16 on MPS/CUDA to cut unified memory usage in half
    dtype = torch.float16 if device in ("mps", 0) else torch.float32
    
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        torch_dtype=dtype
    )

    # 1. Flatten reviews while tracking (original_order_idx, movie_idx, text)
    print("Parsing and flattening reviews...")
    flat_items = []
    for movie_idx, row in df.iterrows():
        reviews = parse_reviews_column(row["Reviews"])
        for rev in reviews:
            if rev and str(rev).strip():
                flat_items.append((len(flat_items), movie_idx, str(rev)[:512]))

    total_reviews = len(flat_items)
    print(f"Total individual reviews to analyze: {total_reviews}")

    if total_reviews == 0:
        print("No valid reviews found to process.")
        df["sentiment_std"] = 0.0
        df["positive_ratio"] = 0.0
        df["reviews_processed"] = 0
        df.to_csv(output_path, index=False)
        return

    # 2. Sort by string length so batches have minimal padding overhead
    flat_items.sort(key=lambda x: len(x[2]))
    sorted_texts = [item[2] for item in flat_items]

    label_mapping = {"negative": 0.0, "neutral": 0.5, "positive": 1.0}
    sorted_scores = []

    print(f"Running MPS batched inference (batch_size={batch_size})...")
    
    with torch.inference_mode():
        for i in range(0, total_reviews, batch_size):
            chunk = sorted_texts[i : i + batch_size]
            try:
                results = sentiment_analyzer(
                    chunk,
                    batch_size=batch_size,
                    truncation=True,
                    max_length=256  # 256 tokens covers almost all Letterboxd reviews and cuts attention RAM by 4x
                )
                scores = [
                    label_mapping.get(res["label"].lower(), 0.5) * res["score"]
                    for res in results
                ]
                sorted_scores.extend(scores)
            except Exception:
                sorted_scores.extend([0.5] * len(chunk))

            # Periodically clear MPS cache to prevent 8GB unified memory swap
            if device == "mps" and (i // batch_size) % 20 == 0:
                torch.mps.empty_cache()

            progress = min(i + batch_size, total_reviews)
            sys.stdout.write(
                f"\rProcessed Reviews: {progress}/{total_reviews} "
                f"({(progress / total_reviews) * 100:.1f}%)"
            )
            sys.stdout.flush()

    print("\nAggregating results back to movies...")

    # 3. Map scores back to their respective movies
    movie_scores = {idx: [] for idx in range(len(df))}
    for (_, movie_idx, _), score in zip(flat_items, sorted_scores):
        movie_scores[movie_idx].append(score)

    sentiment_stds = []
    positive_ratios = []
    processed_counts = []

    for idx in range(len(df)):
        scores = movie_scores[idx]
        if not scores:
            sentiment_stds.append(0.0)
            positive_ratios.append(0.0)
            processed_counts.append(0)
        else:
            series = pd.Series(scores)
            sentiment_stds.append(series.std() if len(series) > 1 else 0.0)
            positive_ratios.append((series > 0.7).mean())
            processed_counts.append(len(scores))

    df["sentiment_std"] = sentiment_stds
    df["positive_ratio"] = positive_ratios
    df["reviews_processed"] = processed_counts

    # Clean up GPU memory before saving
    del sentiment_analyzer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    df.to_csv(output_path, index=False)
    print(f"Successfully saved enhanced dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Apple Silicon MPS-optimized sentiment features from a Letterboxd reviews CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file containing a 'Reviews' column."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the resulting enhanced CSV file."
    )
    parser.add_argument(
        "--model",
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        help="Hugging Face model ID for sentiment analysis."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for M2 8GB inference (16-32 recommended to avoid swap)."
    )
    args = parser.parse_args()
    main(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size
    )