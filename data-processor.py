
import argparse
import ast
import re
import numpy as np
import pandas as pd


def clean_category(value):
    """Clean and properly capitalize category names."""

    if pd.isna(value):
        return None

    value = re.sub(r'\s+', ' ', str(value).strip())

    if not value:
        return None

    # Capitalize category names
    return value.title()


def make_feature_name(prefix, value):
    """Create readable feature names from category values."""

    value = clean_category(value)

    if value is None:
        return None

    value = re.sub(r'[^A-Za-z0-9]+', '_', value).strip('_')

    return f"{prefix}_{value}"


def transform_emotions(emotions):
    if isinstance(emotions, str):
        try:
            emotions = ast.literal_eval(emotions)
        except (ValueError, SyntaxError):
            return {}

    if not isinstance(emotions, list):
        return {}

    scores = {}

    for emotion in emotions:
        if not isinstance(emotion, dict):
            continue

        label = emotion.get('label')
        score = emotion.get('score')

        if label is None or score is None:
            continue

        try:
            score = float(score)
        except (ValueError, TypeError):
            continue

        label = clean_category(label)

        if label is not None:
            scores[label] = max(scores.get(label, float('-inf')), score)

    return scores


def remove_duplicate_input_columns(dataset):
    """Remove duplicate columns only when their contents are identical."""

    duplicate_names = dataset.columns[
        dataset.columns.duplicated(keep=False)
    ].unique()

    for name in duplicate_names:
        positions = [
            i for i, col in enumerate(dataset.columns)
            if col == name
        ]

        first_column = dataset.iloc[:, positions[0]]

        for position in positions[1:]:
            other_column = dataset.iloc[:, position]

            if not first_column.equals(other_column):
                raise ValueError(
                    f"Duplicate input column {name!r} contains conflicting data. "
                    f"Occurrences at positions {positions}. "
                    "Inspect the input CSV before processing."
                )

        print(
            f"Duplicate input column '{name}' found "
            f"{len(positions)} times. Identical copies detected; keeping one."
        )

    return dataset.loc[:, ~dataset.columns.duplicated()].copy()


def main(input_path: str, output_x: str, output_y: str):
    """Cleans tabular dataset, encodes categorical metadata, and prepares feature/target pickles for LightGBM."""

    # Accept both .csv and .pkl as input
    if input_path.endswith('.pkl'):
        movies_dataset = pd.read_pickle(input_path)
    else:
        movies_dataset = pd.read_csv(input_path)


    # Remove duplicate input columns when their contents are identical
    movies_dataset = remove_duplicate_input_columns(movies_dataset)

    # Remove all zero revenue entries
    movies_dataset = movies_dataset[
        (movies_dataset["$Domestic"] != 0) &
        (movies_dataset["$Foreign"] != 0)
    ]

    # Drop the "Reviews" column, if it exists
    movies_dataset = movies_dataset.drop(columns=['Reviews'], errors='ignore')

    # Drop revenue & other review/rating indicators from the feature dataset
    movies_dataset_x = movies_dataset.drop(
        columns=["Rank", "Release Group", "$Worldwide", "$Domestic", "$Foreign", "Foreign %", "Domestic %", "Rating", "Vote_Count"],
        errors="ignore"
    )

    # We want to predict domestic and foreign income
    movies_dataset_y = movies_dataset[["$Domestic", "$Foreign", "$Worldwide"]].copy()

    # Rename columns to remove the '$' sign
    movies_dataset_y = movies_dataset_y.rename(
        columns={r"$Domestic": "Domestic", r"$Foreign": "Foreign", r"$Worldwide": "Worldwide"}
    )

    # Ensure revenue columns are numeric
    for col in movies_dataset_y.columns:
        movies_dataset_y[col] = pd.to_numeric(movies_dataset_y[col], errors='coerce')

    # Turn production budget into a numerical feature
    if 'Budget' in movies_dataset_x.columns:
        movies_dataset_x['Budget'] = pd.to_numeric(
            movies_dataset_x['Budget'].astype(str).str.replace(r'[$,]', '', regex=True),
            errors='coerce'
        )

    # Turn review emotions into numerical features
    if 'review_emotions' in movies_dataset_x.columns:
        emotion_features = (
            movies_dataset_x['review_emotions']
            .apply(transform_emotions)
            .apply(pd.Series)
            .fillna(0)
        )

        emotion_features.columns = [
            make_feature_name('Review_Emotion', col)
            for col in emotion_features.columns
        ]

        movies_dataset_x = pd.concat([
            movies_dataset_x.drop(columns=['review_emotions']),
            emotion_features
        ], axis=1)

    # Turn review sentiments into numerical features
    if 'review_sentiments' in movies_dataset_x.columns:
        sentiment_features = (
            movies_dataset_x['review_sentiments']
            .apply(transform_emotions)
            .apply(pd.Series)
            .fillna(0)
        )

        sentiment_features.columns = [
            make_feature_name('Review_Sentiment', col)
            for col in sentiment_features.columns
        ]

        movies_dataset_x = pd.concat([
            movies_dataset_x.drop(columns=['review_sentiments']),
            sentiment_features
        ], axis=1)

    # Turn original language feature values into named one-hot encoded columns
    if 'Original_Language' in movies_dataset_x.columns:
        language_features = pd.get_dummies(
            movies_dataset_x['Original_Language'].apply(clean_category),
            prefix='Original_Language',
            dtype=int
        )

        movies_dataset_x = pd.concat([
            movies_dataset_x.drop(columns=['Original_Language']),
            language_features
        ], axis=1)

    # Turn genres, production countries, and production studios into named one-hot encoded columns
    multi_val_cols = {
        'Genres': ('Genre', ', '),
        'Production_Countries': ('Production_Country', ', '),
        'Production_Studios': ('Production_Studio', '; ')
    }

    for col, (prefix, separator) in multi_val_cols.items():
        if col in movies_dataset_x.columns:

            types = []

            for entry in movies_dataset_x[col].dropna().unique():
                if isinstance(entry, str):
                    entry_types = entry.split(separator)

                    for t in entry_types:
                        t = clean_category(t)

                        if t and t not in types:
                            types.append(t)

            new_col = {}

            for t in types:
                feature_name = make_feature_name(prefix, t)

                new_col[feature_name] = movies_dataset_x[col].apply(
                    lambda entry: int(
                        t in [clean_category(x) for x in entry.split(separator)]
                    ) if isinstance(entry, str) else 0
                )

            encoded_features = pd.DataFrame(
                new_col,
                index=movies_dataset_x.index
            )

            movies_dataset_x = pd.concat([
                movies_dataset_x.drop(columns=[col]),
                encoded_features
            ], axis=1)

    # Turn any remaining categorical features into named one-hot encoded columns
    categorical_cols = movies_dataset_x.select_dtypes(
        include=['object', 'category', 'string']
    ).columns.tolist()

    if categorical_cols:
        movies_dataset_x = pd.get_dummies(
            movies_dataset_x,
            columns=categorical_cols,
            dtype=int
        )

    # Standardize remaining feature names
    movies_dataset_x.columns = [
        re.sub(
            r'_+',
            '_',
            re.sub(r'[^A-Za-z0-9]+', '_', str(col).strip())
        ).strip('_')
        for col in movies_dataset_x.columns
    ]

    # Convert boolean columns to integers
    bool_cols = movies_dataset_x.select_dtypes(include=['bool']).columns
    movies_dataset_x[bool_cols] = movies_dataset_x[bool_cols].astype(int)

    # Ensure all features are numerical
    movies_dataset_x = movies_dataset_x.apply(pd.to_numeric, errors='coerce')

    # Replace missing and infinite feature values
    movies_dataset_x = movies_dataset_x.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    # Check that all feature names are unique
    duplicate_columns = movies_dataset_x.columns[
        movies_dataset_x.columns.duplicated(keep=False)
    ]

    if len(duplicate_columns) > 0:
        print("\nDuplicate feature names detected:")

        for name in sorted(duplicate_columns.unique()):
            print(
                f"  {name} "
                f"(appears {sum(movies_dataset_x.columns == name)} times)"
            )

        raise ValueError("Duplicate feature names detected after encoding.")

    # Check that all revenue targets are valid
    if movies_dataset_y.isna().any().any():
        raise ValueError("Revenue targets contain missing or invalid values.")

    # Save the final feature and target datasets as .pkl files using command-line arguments
    movies_dataset_x.to_pickle(output_x)
    movies_dataset_y.to_pickle(output_y)

    print("\nProcessed Features X columns:", movies_dataset_x.columns.tolist())
    print("\nNumber of features:", len(movies_dataset_x.columns))
    print("\nProcessed Targets:", movies_dataset_y.columns.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process structural metadata into numeric and multi-hot features.")
    parser.add_argument("--input", required=True, help="Path to input movies dataset CSV or PKL")
    parser.add_argument("--output_x", required=True, help="Path to save the processed feature X pickle file")
    parser.add_argument("--output_y", required=True, help="Path to save the target labels Y pickle file")

    args = parser.parse_args()
    main(input_path=args.input, output_x=args.output_x, output_y=args.output_y)
