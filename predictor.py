import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

def main(features_path: str, targets_path: str, target: str):
    """Uses PKL files of feature and target values to train a LightGBM classifier to predict movie revenue categories."""

    print("Loading preprocessed pickle files...")
    X = pd.read_pickle(features_path)
    y_raw = pd.read_pickle(targets_path)
    y = y_raw[target].astype(int)

    print("Flattening multi-hot lists and categorical features into a numeric matrix...")
    processed_dfs = []
    for col in X.columns:
        if X[col].apply(lambda x: isinstance(x, list)).any():
            expanded_df = pd.DataFrame(X[col].tolist(), index=X.index)
            expanded_df.columns = [f"{col}_idx_{i}" for i in range(expanded_df.shape[1])]
            processed_dfs.append(expanded_df)
        else:
            processed_dfs.append(X[[col]])

    X_flat = pd.concat(processed_dfs, axis=1)
    X_flat = X_flat.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_np = X_flat.values.astype(np.float32)
    y_np = y.values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )

    print("Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y_np)),
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    print("\nEvaluating Model...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n===============================")
    print(f"     MODEL EVALUATION")
    print(f"===============================")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    model_filename = "movie_revenue_lgb_classifier.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained LightGBM classifier saved successfully as '{model_filename}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM classification model using preprocessed movie data.")
    parser.add_argument("--features", required=True, help="Path to features pickle file (e.g., movies_dataset_x.pkl)")
    parser.add_argument("--targets", required=True, help="Path to targets pickle file (e.g., movies_dataset_y.pkl)")
    parser.add_argument("--target", required=True, help="Target column name from y pickle (e.g., Domestic)")
    args = parser.parse_args()

    main(features_path=args.features, targets_path=args.targets, target=args.target)