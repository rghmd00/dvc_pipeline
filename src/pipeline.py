import json
import joblib
import dvc.api

from data_loader import load_data
from preprocessing import wrangle
from train_model import setup_tracking, train_models


def main():
    cfg = dvc.api.params_show("params.yaml")

    # Step 1: Load data
    train_df = load_data(cfg["data"]["train_csv"])

    # Step 2: Preprocess data
    processed_df = wrangle(train_df)

    # Step 3: Train models
    setup_tracking()
    results = train_models(processed_df, cfg)

    # Step 4: Save each model under its own name
    for name, r in results.items():
        path = f"models/{name}_model.pkl"
        joblib.dump(r["model"], path)
        print(f"Saved {name} -> {path}")

    # Step 5: Persist metrics for reference
    metrics_only = {name: r["metrics"] for name, r in results.items()}
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_only, f, indent=2)

    print("Pipeline complete. Both models saved.")


if __name__ == "__main__":
    main()