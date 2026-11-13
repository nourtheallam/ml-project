import argparse
import ast
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm


def flatten_and_clean_dataframe(df, prefix=""):
    processed_dfs = []
    for col in df.columns:
        col_name = f"{prefix}_{col}" if prefix and not col.startswith(prefix) else col
        if df[col].apply(lambda x: isinstance(x, list)).any():
            expanded_df = pd.DataFrame(df[col].tolist(), index=df.index)
            expanded_df.columns = [f"{col_name}_idx_{i}" for i in range(expanded_df.shape[1])]
            processed_dfs.append(expanded_df)
        else:
            temp_df = df[[col]].copy()
            if prefix and col != "Reviews":
                temp_df.columns = [col_name]
            processed_dfs.append(temp_df)

    flat_df = pd.concat(processed_dfs, axis=1)
    if "Reviews" in flat_df.columns:
        flat_df = flat_df.drop(columns=["Reviews"])

    flat_df = (
        flat_df
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    flat_df = flat_df.loc[:, flat_df.nunique() > 1]
    return flat_df


def main(features_path: str, targets_path: str, target: str, config: str, sentiments_path: str, tfidf_path: str):
    print(f"\n--- Training Model for Target: {target} | Config: {config} ---")
    print("Loading datasets...")
    X = pd.read_pickle(features_path)
    y_raw = pd.read_pickle(targets_path)

    y = y_raw[target].reindex(X.index)

    if y.isna().any():
        raise ValueError("Missing or unmatched target values detected.")

    y_np = np.log1p(y.astype(float)).values

    X_flat_df = flatten_and_clean_dataframe(X)
    tab_feature_names = X_flat_df.columns.tolist()
    X_tab_array = X_flat_df.values

    feature_arrays = [X_tab_array]
    all_feature_names = tab_feature_names

    if sentiments_path and os.path.exists(sentiments_path):
        print(f"Loading sentiment features from {sentiments_path}...")
        X_sent_raw = pd.read_pickle(sentiments_path)
        lowercase_cols = [c for c in X_sent_raw.columns if str(c)[0].islower()]
        X_sent_filtered = X_sent_raw[lowercase_cols].reindex(X.index)
        X_sent_flat = flatten_and_clean_dataframe(X_sent_filtered, prefix="sent")
        
        feature_arrays.append(X_sent_flat.values)
        all_feature_names.extend(X_sent_flat.columns.tolist())

    if tfidf_path and os.path.exists(tfidf_path):
        print(f"Loading TF-IDF bundle from {tfidf_path}...")
        with open(tfidf_path, "rb") as f:
            tfidf_bundle = pickle.load(f)
        feature_arrays.append(tfidf_bundle["tfidf_matrix"])
        all_feature_names.extend(tfidf_bundle["feature_names"])

    X_final = hstack([csr_matrix(arr) if not isinstance(arr, csr_matrix) else arr for arr in feature_arrays]).tocsr()

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_np, test_size=0.20, random_state=42)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    base_model = lgb.LGBMRegressor(objective="regression", random_state=42, n_jobs=1, verbosity=-1)
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

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    print("Starting hyperparameter tuning (RandomizedSearchCV)...")
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="r2",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=False
    )

    search.fit(X_fit, y_fit)

    print("Training final regressor model with early stopping...")
    model = lgb.LGBMRegressor(objective="regression", random_state=42, n_jobs=-1, verbosity=1, **search.best_params_)
    
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"Test R² Score: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    os.makedirs("models", exist_ok=True)
    # Save model uniquely tagged by target and configuration name
    model_filename = f"models/movie_revenue_lgb_regressor_{target}_{config}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump({
            "model": model,
            "target": target,
            "config": config,
            "feature_columns": all_feature_names,
            "preds": preds,
            "y_test": y_test
        }, f)

    print(f"Model saved to: {model_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuned LightGBM Regressor.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--config", default="default", help="Ablation configuration name.")
    parser.add_argument("--sentiments", required=False, default=None)
    parser.add_argument("--tfidf", required=False, default=None)
    args = parser.parse_args()
    main(args.features, args.targets, args.target, args.config, args.sentiments, args.tfidf)