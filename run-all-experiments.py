import argparse
import os
import subprocess
import pandas as pd


def run_command(cmd):
    print(f"\nExecuting: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        exit(result.returncode)


def main(input_csv: str, plots_dir: str):
    print(f"==================================================")
    print(f" PIPELINE INITIALIZATION: Processing Raw CSV")
    print(f"==================================================")

    features_x_pkl = "processed_features_x.pkl"
    targets_y_pkl = "processed_targets_y.pkl"
    sentiment_pkl = "sentiment_features.pkl"
    emotion_pkl = "emotion_features.pkl"
    tfidf_pkl = "tfidf_bundle.pkl"

    # Step 1: Run data processor[cite: 1]
    run_command([
        "python3", "data-processor.py",
        "--input", input_csv,
        "--output_x", features_x_pkl,
        "--output_y", targets_y_pkl
    ])

    # Step 2: Run sentiment analyzer
    run_command([
        "python3", "sentiment-analyzer.py",
        "--input", input_csv,
        "--output", sentiment_pkl
    ])

    # Step 3: Run emotion analyzer
    run_command([
        "python3", "emotion-analyzer.py",
        "--input", input_csv,
        "--output", emotion_pkl
    ])

    # Step 4: Run TF-IDF processor
    run_command([
        "python3", "tfidf-processor.py",
        "--input", input_csv,
        "--output", tfidf_pkl,
        "--max_features", "100",
        "--min_df", "10"
    ])

    targets_list = ["Domestic", "Foreign"]
    
    # 6-tier ablation study configurations: (config_name, use_sentiment, use_emotion, use_tfidf)
    configs = [
        ("metadata_only", False, False, False),
        ("sentiment_only", True, False, False),
        ("emotions_only", False, True, False),
        ("tfidf_only", False, False, True),
        ("sentiment_plus_emotions", True, True, False),
        ("all_combined", True, True, True)
    ]

    for target in targets_list:
        print(f"\n==================================================")
        print(f" RUNNING ABLATION EXPERIMENTS FOR TARGET: {target}")
        print(f"==================================================")

        for config_name, use_sent, use_emot, use_tfidf in configs:
            print(f"\n--- Configuration: {config_name.upper()} ({target}) ---")
            
            cmd = [
                "python3", "predictor.py",
                "--features", features_x_pkl,
                "--targets", targets_y_pkl,
                "--target", target,
                "--config", config_name
            ]

            if use_sent and os.path.exists(sentiment_pkl):
                cmd.extend(["--sentiments", sentiment_pkl])
            elif use_emot and not use_sent and os.path.exists(emotion_pkl):
                cmd.extend(["--sentiments", emotion_pkl])

            if use_sent and use_emot:
                combined_path = "temp_sentiment_emotion_combined.pkl"
                df_s = pd.read_pickle(sentiment_pkl)
                df_e = pd.read_pickle(emotion_pkl)
                for col in ["mean_joy", "mean_anger"]:
                    if col in df_e.columns:
                        df_s[col] = df_e[col]
                df_s.to_pickle(combined_path)
                cmd[cmd.index("--sentiments") + 1] = combined_path

            if use_tfidf and os.path.exists(tfidf_pkl):
                cmd.extend(["--tfidf", tfidf_pkl])

            run_command(cmd)

    print(f"\n==================================================")
    print(f" GENERATING CONFIG-SPECIFIC EVALUATION PLOTS")
    print(f"==================================================")
    
    plot_cmd = [
        "python3", "plot-results.py",
        "--models_dir", "models",
        "--output", plots_dir
    ]
    run_command(plot_cmd)
    
    print(f"\nAll experiments complete! Unique plots saved to '{plots_dir}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate full end-to-end box office experiment pipeline.")
    parser.add_argument("--input", required=True, help="Path to input raw reviews dataset CSV.")
    parser.add_argument("--plots_dir", default="paper_plots", help="Folder to store generated figures.")

    args = parser.parse_args()
    main(input_csv=args.input, plots_dir=args.plots_dir)