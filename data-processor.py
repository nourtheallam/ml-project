import argparse
import pandas as pd

def main(input_path: str, output_x: str, output_y: str):
    """Cleans the given dataset so that it's ready to be used as a model's input."""

    movies_dataset = pd.read_csv(input_path)
    # Drop revenue indicators from the feature dataset
    movies_dataset_x = (movies_dataset.drop(columns=["Rank", "Release Group", "$Worldwide", "$Domestic", "$Foreign", "Foreign %", "Domestic %", "Vote_Count", "Rating"]))
    if("Reviews" in movies_dataset_x.columns): movies_dataset_x = (movies_dataset_x.drop(columns=["Reviews"])) 
    # We want to predict domestic and foreign income ranges
    movies_dataset_y = (movies_dataset[["$Domestic", "$Foreign"]])
    # Drop the scraped reviews column (save it for sentiment analysis)
    movies_dataset_x = movies_dataset_x.iloc[:, :-1]

    # Rename columns to remove the '$' sign
    movies_dataset_y = movies_dataset_y.rename(columns={r"$Domestic": "Domestic", r"$Foreign": "Foreign"})
    # Turn domestic and foreign profit column values into range categories 
    num_bins = 4
    bin_labels = [0, 1, 2, 3]  
    for col in ["Domestic", "Foreign"]:
        movies_dataset_y[col], bin_edges = pd.qcut(
            movies_dataset_y[col], q=num_bins, labels=bin_labels, retbins=True
        )
        print(f"Non-trivial bin edges for {col}: {bin_edges}")

    # Turn original language feature values into numerical categories
    single_val_cols = ['Original_Language']
    for col in single_val_cols:
        types = movies_dataset_x[col].unique()
        types_enum = {val: i for i, val in enumerate(types)}
        for t in types:
            movies_dataset_x[col] = movies_dataset_x[col].replace(t, types_enum[t])
        
    # Turn genres and production countries entries into multi-hot encoded entries
    multi_val_cols = ['Genres', 'Production_Countries']
    for col in multi_val_cols:
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

    # # Just remove the "/10" term for the rating entries
    # movies_dataset_x["Rating"] = movies_dataset_x["Rating"].str[:-3]

    # Save the final feature and target datasets as pickle files using command-line arguments
    movies_dataset_x.to_pickle(output_x)
    movies_dataset_y.to_pickle(output_y)
    print(movies_dataset_x)
    print(movies_dataset_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset to make input vectors numerical and of fixed length.")
    parser.add_argument("--input", required=True, help="Path to input movies dataset CSV")
    parser.add_argument("--output_x", required=True, help="Path to save the processed features X pickle file")
    parser.add_argument("--output_y", required=True, help="Path to save the target labels Y pickle file")
    
    args = parser.parse_args()
    main(input_path=args.input, output_x=args.output_x, output_y=args.output_y)