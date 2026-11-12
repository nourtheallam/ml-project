
import argparse
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


def main(features_path: str, targets_path: str, target: str):
    """Tune and train a LightGBM multiclass movie revenue classifier."""

    # Load datasets
    print("Loading datasets...")
    X = pd.read_pickle(features_path)
    y_raw = pd.read_pickle(targets_path)

    # Align target and feature indices
    y = y_raw[target].reindex(X.index)

    if y.isna().any():
        raise ValueError("Missing or unmatched target values detected.")

    # Encode class labels as consecutive integers
    encoder = LabelEncoder()
    y_np = encoder.fit_transform(y.astype(int))

    print("\nRevenue categories:")
    print(pd.Series(y_np).value_counts().sort_index())

    # Flatten multi-hot encoded features
    print("\nFlattening features...")
    processed_dfs = []

    for col in X.columns:
        if X[col].apply(lambda x: isinstance(x, list)).any():
            expanded_df = pd.DataFrame(
                X[col].tolist(),
                index=X.index
            )

            expanded_df.columns = [
                f"{col}_idx_{i}"
                for i in range(expanded_df.shape[1])
            ]

            processed_dfs.append(expanded_df)
        else:
            processed_dfs.append(X[[col]])

    X_flat = pd.concat(processed_dfs, axis=1)

    # Convert all features to numeric
    X_flat = (
        X_flat
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )

    # Remove constant features
    X_flat = X_flat.loc[:, X_flat.nunique() > 1]

    # Check class distribution
    class_counts = np.bincount(y_np)

    if class_counts.min() < 4:
        raise ValueError(
            "At least 4 examples per class are recommended "
            "for this train/validation/test and CV setup."
        )

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y_np,
        test_size=0.20,
        random_state=42,
        stratify=y_np
    )

    # Split training data into training and validation sets
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    # Initial model
    base_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(encoder.classes_),
        random_state=42,
        n_jobs=1,
        verbosity=-1
    )

    # Hyperparameter search space
    param_distributions = {
        "n_estimators": randint(500, 2001),
        "learning_rate": loguniform(0.01, 0.10),
        "num_leaves": randint(15, 64),
        "max_depth": [5, 6, 7, 8, 10, -1],
        "min_child_samples": randint(15, 101),
        "subsample": uniform(0.65, 0.35),
        "subsample_freq": [1],
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": loguniform(1e-3, 10),
        "reg_lambda": loguniform(1e-3, 10),
        "min_split_gain": uniform(0, 0.2)
    }

    # Cross-validation
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    print("\nStarting hyperparameter tuning...")

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=False
    )

    search.fit(X_fit, y_fit)

    print("\nBest hyperparameters:")
    print(search.best_params_)

    print(f"\nBest CV Macro F1: {search.best_score_:.4f}")

    # Train final model using best parameters
    print("\nTraining final model...")

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(encoder.classes_),
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **search.best_params_
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=True
            ),
            lgb.log_evaluation(period=100)
        ]
    )

    print(f"\nBest iteration: {model.best_iteration_}")

    # Evaluate on untouched test set
    print("\nEvaluating final model...")

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        preds,
        labels=np.arange(len(encoder.classes_)),
        target_names=[str(c) for c in encoder.classes_],
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        preds,
        labels=np.arange(len(encoder.classes_))
    ))

    # Feature importance
    importance = pd.DataFrame({
        "feature": X_flat.columns,
        "importance": model.feature_importances_
    }).sort_values(
        by="importance",
        ascending=False
    )

    print("\nTop 20 Features:")
    print(importance.head(20).to_string(index=False))

    # Save model and label encoder together
    os.makedirs("models", exist_ok=True)

    model_filename = (
        f"models/movie_revenue_lgb_classifier_{target}.pkl"
    )

    with open(model_filename, "wb") as f:
        pickle.dump({
            "model": model,
            "label_encoder": encoder,
            "feature_columns": X_flat.columns.tolist(),
            "best_params": search.best_params_
        }, f)

    print(f"\nModel saved to: {model_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tuned LightGBM movie revenue classifier."
    )

    parser.add_argument(
        "--features",
        required=True,
        help="Path to feature pickle file"
    )

    parser.add_argument(
        "--targets",
        required=True,
        help="Path to target pickle file"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name"
    )

    args = parser.parse_args()

    main(
        features_path=args.features,
        targets_path=args.targets,
        target=args.target
    )
