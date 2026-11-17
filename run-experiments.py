
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = [
    "reviews-expanded.csv",
    "review-sentiments-expanded.csv",
    "review-emotions-expanded.csv",
    "reviews-sentiments-emotions-expanded.csv"
]

TARGETS = ["Domestic", "Foreign", "Worldwide"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

N_ESTIMATORS = 100
EARLY_STOPPING_ROUNDS = 70

MODEL_PARAMS = {
    "objective": "regression",
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories(plots_dir):

    directories = {
        "actual_vs_predicted": plots_dir / "actual_vs_predicted",
        "performance": plots_dir / "performance",
        "feature_importance": plots_dir / "feature_importance",
        "learning_curves": plots_dir / "learning_curves"
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


# ============================================================
# DATA PREPROCESSING
# ============================================================

def process_dataset(dataset_path, processed_dir):

    dataset_name = dataset_path.stem

    features_path = processed_dir / f"{dataset_name}_X.pkl"
    targets_path = processed_dir / f"{dataset_name}_Y.pkl"

    print(f"\nProcessing dataset: {dataset_path.name}")

    subprocess.run(
        [
            sys.executable,
            "data-processor.py",
            "--input", str(dataset_path),
            "--output_x", str(features_path),
            "--output_y", str(targets_path)
        ],
        check=True
    )

    X = pd.read_pickle(features_path)
    Y = pd.read_pickle(targets_path)

    return X, Y


# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate(X, Y, target):

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Split training data into fitting and validation sets
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    # Transform revenue targets using log1p
    y_fit_log = np.log1p(y_fit[target])
    y_val_log = np.log1p(y_val[target])

    model = lgb.LGBMRegressor(**MODEL_PARAMS)

    print(f"Training model for {target}...")

    model.fit(
        X_fit,
        y_fit_log,

        eval_set=[
            (X_fit, y_fit_log),
            (X_val, y_val_log)
        ],

        eval_names=["training", "validation"],
        eval_metric="rmse",

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False
            )
        ]
    )

    # Predict test set
    predictions = model.predict(X_test)

    actual = np.log1p(y_test[target].to_numpy())

    # Calculate metrics
    r2 = r2_score(actual, predictions)

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    mae = mean_absolute_error(actual, predictions)

    print(
        f"{target}: "
        f"R2={r2:.4f} | "
        f"Log RMSE={rmse:.4f} | "
        f"Log MAE={mae:.4f} | "
        f"Best iteration={model.best_iteration_}"
    )

    results = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "actual": actual,
        "predictions": predictions,
        "model": model,
        "evals_result": model.evals_result_,
        "feature_names": model.booster_.feature_name()
    }

    return results


# ============================================================
# ACTUAL VS PREDICTED GRAPHS
# ============================================================

def plot_actual_vs_predicted(dataset_name, results, output_dir):

    fig, axes = plt.subplots(
        1, 3,
        figsize=(19, 6)
    )

    fig.suptitle(
        f"Actual vs Predicted Revenue - {dataset_name}",
        fontsize=16
    )

    for ax, target in zip(axes, TARGETS):

        result = results[target]

        actual = result["actual"]
        predictions = result["predictions"]

        # Logarithmic transformation makes highly skewed
        # movie revenue distributions easier to visualize.

        ax.scatter(
            actual,
            predictions,
            alpha=0.5,
            s=20
        )

        lower = min(
            actual.min(),
            predictions.min()
        )

        upper = max(
            actual.max(),
            predictions.max()
        )

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=2
        )

        ax.set_title(
            f"{target}\n"
            f"R² = {result['r2']:.3f}"
        )

        ax.set_xlabel("Actual Log Revenue")
        ax.set_ylabel("Predicted Log Revenue")

        ax.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{dataset_name}_actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FEATURE IMPORTANCE GRAPHS
# ============================================================

def plot_feature_importance(dataset_name, results, output_dir):

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))

    fig.suptitle(
        f"Feature Importance - {dataset_name}",
        fontsize=16,
    )

    for ax, target in zip(axes, TARGETS):

        model = results[target]["model"]

        importance = model.booster_.feature_importance(
            importance_type="gain"
        )

        feature_names = results[target]["feature_names"]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        })

        # Exclude features with zero gain
        importance_df = importance_df[importance_df["Importance"] > 0]

        importance_df = (
            importance_df
            .sort_values("Importance", ascending=False)
            .head(20)
            .sort_values("Importance", ascending=True)
        )

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"],
        )

        ax.set_title(f"{target}")
        ax.set_xlabel("Total Gain")
        ax.set_ylabel("Feature")

        ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{dataset_name}_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# LEARNING CURVES
# ============================================================

def plot_learning_curves(dataset_name, results, output_dir):

    fig, axes = plt.subplots(
        1, 3,
        figsize=(19, 6)
    )

    fig.suptitle(
        f"Learning Curves - {dataset_name}",
        fontsize=16
    )

    for ax, target in zip(axes, TARGETS):

        eval_results = results[target]["evals_result"]

        training_rmse = eval_results["training"]["rmse"]
        validation_rmse = eval_results["validation"]["rmse"]

        iterations = range(1, len(training_rmse) + 1)

        ax.plot(
            iterations,
            training_rmse,
            label="Training RMSE"
        )

        ax.plot(
            iterations,
            validation_rmse,
            label="Validation RMSE"
        )

        best_iteration = results[target]["model"].best_iteration_

        if best_iteration:
            ax.axvline(
                best_iteration,
                linestyle="--",
                label=f"Best iteration: {best_iteration}"
            )

        ax.set_title(target)
        ax.set_xlabel("Boosting Iteration")
        ax.set_ylabel("RMSE")

        ax.legend()
        ax.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{dataset_name}_learning_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

def plot_performance_comparison(metrics_df, output_dir):

    # Average each metric across the three revenue targets
    averages = (
        metrics_df
        .groupby("Dataset")[["R2", "RMSE", "MAE"]]
        .mean()
        .reset_index()
    )

    averages.to_csv(
        output_dir / "average_performance.csv",
        index=False
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(19, 6)
    )

    fig.suptitle(
        "Average Model Performance Across Datasets",
        fontsize=16
    )

    metric_info = [
        ("R2", "Average R²", "R²"),
        ("RMSE", "Average RMSE", "RMSE"),
        ("MAE", "Average MAE", "MAE")
    ]

    for ax, (column, title, ylabel) in zip(axes, metric_info):

        bars = ax.bar(
            averages["Dataset"],
            averages[column]
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Dataset")

        ax.tick_params(
            axis="x",
            rotation=25
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3
        )

        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        output_dir / "dataset_performance_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return averages


# ============================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================

def main(raw_dir, plots_dir):

    raw_dir = Path(raw_dir)
    plots_dir = Path(plots_dir)
    processed_dir = Path("processed_data")

    # Clear previous experiment outputs
    for directory in [processed_dir, plots_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    directories = create_directories(plots_dir)

    all_metrics = []

    for filename in DATASETS:

        dataset_path = raw_dir / filename

        if not dataset_path.exists():
            print(f"WARNING: Missing dataset {dataset_path}")
            continue

        dataset_name = dataset_path.stem

        print("\n" + "=" * 65)
        print(f"RUNNING EXPERIMENTS: {filename}")
        print("=" * 65)

        # Preprocess dataset
        X, Y = process_dataset(
            dataset_path,
            processed_dir
        )

        dataset_results = {}

        # Train one model per target
        for target in TARGETS:

            print(f"\n--- Target: {target} ---")

            results = train_and_evaluate(
                X,
                Y,
                target
            )

            dataset_results[target] = results

            all_metrics.append({
                "Dataset": dataset_name,
                "Target": target,
                "R2": results["r2"],
                "RMSE": results["rmse"],
                "MAE": results["mae"],
                "Best Iteration": results["model"].best_iteration_
            })

        # Generate grouped graphs for this dataset
        print(f"\nGenerating graphs for {dataset_name}...")

        plot_actual_vs_predicted(
            dataset_name,
            dataset_results,
            directories["actual_vs_predicted"]
        )

        plot_feature_importance(
            dataset_name,
            dataset_results,
            directories["feature_importance"]
        )

        plot_learning_curves(
            dataset_name,
            dataset_results,
            directories["learning_curves"]
        )

    # Save individual target metrics
    metrics_df = pd.DataFrame(all_metrics)

    metrics_df.to_csv(
        plots_dir / "all_experiment_metrics.csv",
        index=False
    )

    # Generate overall performance comparison
    if not metrics_df.empty:

        averages = plot_performance_comparison(
            metrics_df,
            directories["performance"]
        )

        print("\n" + "=" * 65)
        print("AVERAGE PERFORMANCE ACROSS REVENUE TARGETS")
        print("=" * 65)

        print(averages.to_string(index=False))

    print("\nAll experiments completed!")
    print(f"Plots saved to: {plots_dir.resolve()}")


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run movie revenue prediction experiments."
    )

    parser.add_argument(
        "--raw_dir",
        default="raw-data",
        help="Directory containing raw CSV datasets."
    )

    parser.add_argument(
        "--plots_dir",
        default="plots",
        help="Directory for saving all plots and results."
    )

    args = parser.parse_args()

    main(
        raw_dir=args.raw_dir,
        plots_dir=args.plots_dir
    )
