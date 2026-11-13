import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def main(models_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' does not exist.")
        return

    for filename in os.listdir(models_dir):
        if filename.endswith(".pkl") and "regressor" in filename:
            filepath = os.path.join(models_dir, filename)
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"Skipping corrupted file {filename}: {e}")
                continue
            
            # Robust extraction: try dictionary keys first, then parse filename
            # Expected filename pattern: movie_revenue_lgb_regressor_[Target]_[Config].pkl
            parts = filename.replace(".pkl", "").split("_")
            
            # Extract target and config from filename parts as fallback
            # parts roughly: ['movie', 'revenue', 'lgb', 'regressor', Target, Config...]
            file_target = "Domestic" if "Domestic" in filename else ("Foreign" if "Foreign" in filename else "Unknown")
            
            # Find config by stripping prefix parts
            # e.g., metadata_only, sentiment_only, all_combined, etc.
            known_configs = [
                "metadata_only", "sentiment_only", "emotions_only", 
                "tfidf_only", "sentiment_plus_emotions", "all_combined"
            ]
            file_config = "default"
            for cfg in known_configs:
                if cfg in filename:
                    file_config = cfg
                    break

            target = data.get("target", file_target)
            config = data.get("config", file_config)

            # Skip if it somehow resolved to default or unknown
            if config == "default" or target == "Unknown":
                print(f"Skipping unparseable or legacy model file: {filename}")
                continue

            results.append({
                "Filename": filename,
                "Target": target,
                "Config": config,
                "preds": data["preds"],
                "y_test": data["y_test"]
            })

    if not results:
        print("No valid ablation models found in directory.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    summary_metrics = []

    for res in results:
        target = res["Target"]
        config = res["Config"]
        preds = res["preds"]
        y_test = res["y_test"]

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        summary_metrics.append({"Target": target, "Config": config, "R2": r2, "RMSE": rmse})

        # --- 1. PARITY PLOT ---
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, preds, alpha=0.5, color="#1f77b4", edgecolors="k", s=30)
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", lw=2, label="Ideal Fit")
        
        ax.set_xlabel("Actual Revenue (Log Scale)", fontsize=11)
        ax.set_ylabel("Predicted Revenue (Log Scale)", fontsize=11)
        ax.set_title(f"Parity: {config.upper()} ({target})\n$R^2$ = {r2:.3f}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left")
        plt.tight_layout()
        
        parity_path = os.path.join(output_dir, f"parity_{target}_{config}.png")
        plt.savefig(parity_path, dpi=300)
        plt.close()

        # --- 2. RESIDUAL PLOT ---
        residuals = preds - y_test
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(preds, residuals, alpha=0.5, color="#2ca02c", edgecolors="k", s=30)
        ax.axhline(0, color="red", linestyle="--", lw=2, label="Zero Error")
        
        ax.set_xlabel("Predicted Revenue (Log Scale)", fontsize=11)
        ax.set_ylabel("Residuals (Predicted - Actual)", fontsize=11)
        ax.set_title(f"Residuals: {config.upper()} ({target})", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right")
        plt.tight_layout()
        
        res_path = os.path.join(output_dir, f"residuals_{target}_{config}.png")
        plt.savefig(res_path, dpi=300)
        plt.close()
        print(f"Generated clean plots for target '{target}', config '{config}'.")

    # --- 3. COMPARATIVE ABLATION BAR CHART ---
    if summary_metrics:
        df_metrics = pd.DataFrame(summary_metrics)
        for target in df_metrics["Target"].unique():
            subset = df_metrics[df_metrics["Target"] == target]
            if len(subset) > 1:
                fig, ax = plt.subplots(figsize=(8, 5))
                subset.plot(x="Config", y="R2", kind="bar", ax=ax, color="#ff7f0e", legend=False, edgecolor="k")
                ax.set_ylabel("Test R² Score", fontsize=11)
                ax.set_title(f"Ablation Study Comparison ({target})", fontsize=13, fontweight="bold")
                ax.set_xticklabels(subset["Config"], rotation=30, ha="right")
                ax.set_ylim(0, 1.0)
                plt.tight_layout()
                
                bar_path = os.path.join(output_dir, f"ablation_comparison_{target}.png")
                plt.savefig(bar_path, dpi=300)
                plt.close()
                print(f"Generated comparative ablation bar chart for {target}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate unique configuration-specific diagnostic and comparative plots.")
    parser.add_argument("--models_dir", default="models", help="Directory containing saved model pickles.")
    parser.add_argument("--output", default="paper_plots", help="Directory to save generated figures.")
    args = parser.parse_args()
    main(args.models_dir, args.output)