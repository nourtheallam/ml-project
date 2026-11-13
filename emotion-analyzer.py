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
    print(f"Loading emotion/sentiment model ({model_name}) on device: {device}...")
    dtype = torch.float16 if device in ("mps", 0) else torch.float32
    
    analyzer = pipeline("sentiment-analysis", model=model_name, device=device, torch_dtype=dtype)

    flat_items = []
    for movie_idx, row in df.iterrows():
        reviews = parse_reviews_column(row["Reviews"])
        for rev in reviews:
            if rev and str(rev).strip():
                flat_items.append((len(flat_items), movie_idx, str(rev)[:512]))

    total_reviews = len(flat_items)
    if total_reviews == 0:
        df.to_pickle(output_path)
        return

    flat_items.sort(key=lambda x: len(x[2]))
    sorted_texts = [item[2] for item in flat_items]
    sorted_joy = []
    sorted_anger = []

    print("Running batched emotion inference...")
    with torch.inference_mode(), tqdm(total=total_reviews, desc="Emotion Inference", unit="rev") as pbar:
        for i in range(0, total_reviews, batch_size):
            chunk = sorted_texts[i : i + batch_size]
            try:
                results = analyzer(chunk, batch_size=batch_size, truncation=True, max_length=256)
                sorted_joy.extend([res.get("score", 0.0) if res["label"].lower() == "positive" else 0.0 for res in results])
                sorted_anger.extend([res.get("score", 0.0) if res["label"].lower() == "negative" else 0.0 for res in results])
            except Exception:
                sorted_joy.extend([0.0] * len(chunk))
                sorted_anger.extend([0.0] * len(chunk))

            if device == "mps" and (i // batch_size) % 20 == 0:
                torch.mps.empty_cache()

            pbar.update(len(chunk))

    movie_joy = {idx: [] for idx in range(len(df))}
    movie_anger = {idx: [] for idx in range(len(df))}

    for (item_idx, movie_idx, _), j_score, a_score in zip(flat_items, sorted_joy, sorted_anger):
        movie_joy[movie_idx].append(j_score)
        movie_anger[movie_idx].append(a_score)

    mean_joy = []
    mean_anger = []

    for idx in range(len(df)):
        j_list = movie_joy[idx]
        a_list = movie_anger[idx]
        mean_joy.append(pd.Series(j_list).mean() if j_list else 0.0)
        mean_anger.append(pd.Series(a_list).mean() if a_list else 0.0)

    df["mean_joy"] = mean_joy
    df["mean_anger"] = mean_anger

    del analyzer
    gc.collect()

    df.to_pickle(output_path)
    print(f"Successfully saved emotion features to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract lowercase emotion (joy/anger) features with progress bar.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--batch_size", type=int, default=24)
    args = parser.parse_args()
    main(args.input, args.output, args.model, args.batch_size)