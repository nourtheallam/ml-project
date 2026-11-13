import argparse
import ast
import numpy as np
import pandas as pd


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


def main(input_path: str, output_x: str, output_y: str):
    """Cleans tabular dataset, encodes categorical metadata, and prepares feature/target pickles for LightGBM."""

    # Load from CSV or PKL depending on input format
    if input_path.endswith(".pkl"):
        movies_dataset = pd.read_pickle(input_path)
    else:
        movies_dataset = pd.read_csv(input_path)
    
    # Drop revenue & other review/rating indicators from the feature dataset, 
    # but KEEP 'Reviews' so downstream analyzers can access it if needed.
    movies_dataset_x = movies_dataset.drop(
        columns=["Rank", "Release Group", "$Worldwide", "$Domestic", "$Foreign", "Foreign %", "Domestic %", "Vote_Count", "Rating", "Year"], 
        errors="ignore"
    )

    # We want to predict domestic and foreign income
    movies_dataset_y = movies_dataset[["$Domestic", "$Foreign"]].copy()

    # Rename columns to remove the '$' sign
    movies_dataset_y = movies_dataset_y.rename(columns={r"$Domestic": "Domestic", r"$Foreign": "Foreign"})

    # Turn original language feature values into numerical categories
    single_val_cols = ['Original_Language']
    for col in single_val_cols:
        if col in movies_dataset_x.columns:
            types = movies_dataset_x[col].unique()
            types_enum = {val: i for i, val in enumerate(types)}
            for t in types:
                movies_dataset_x[col] = movies_dataset_x[col].replace(t, types_enum[t])
        
    # Turn genres and production countries entries into multi-hot encoded entries
    multi_val_cols = ['Genres', 'Production_Countries']
    for col in multi_val_cols:
        if col in movies_dataset_x.columns:
            types = []
            for entry in movies_dataset_x[col].unique(): 
                if isinstance(entry, str):
                    entry_types = entry.split(", ")
                    for t in entry_types:
                        if t not in types:
                            types.append(t)
            types_enum = {val: i for i, val in enumerate(types)} 
            new_col = []
            for entry in movies_dataset_x[col]:
                multihot_vector = [0] * len(types)
                if isinstance(entry, str):
                    entry_types = entry.split(", ")
                    for t in entry_types:
                        if t in types_enum:
                            multihot_vector[types_enum[t]] = 1
                new_col.append(multihot_vector)
            movies_dataset_x[col] = new_col    

    # Save the final feature and target datasets as pickle files using command-line arguments
    movies_dataset_x.to_pickle(output_x)
    movies_dataset_y.to_pickle(output_y)
    print("\nProcessed Features X columns:", movies_dataset_x.columns.tolist())
    print("Processed Targets Y shape:", movies_dataset_y.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process structural metadata into numeric and multi-hot features.")
    parser.add_argument("--input", required=True, help="Path to input movies dataset CSV or PKL")
    parser.add_argument("--output_x", required=True, help="Path to save the processed features X pickle file")
    parser.add_argument("--output_y", required=True, help="Path to save the target labels Y pickle file")
    
    args = parser.parse_args()
    main(input_path=args.input, output_x=args.output_x, output_y=args.output_y)