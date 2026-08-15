

import json
import joblib

MODEL_NAMES = ["logistic_regression", "random_forest"]


def load_trained_model(name: str):
    return joblib.load(f"models/{name}_model.pkl")


def select_best_model(metrics_by_model: dict, metric: str = "accuracy") -> str:
    return max(metrics_by_model, key=lambda name: metrics_by_model[name][metric])


if __name__ == "__main__":

    with open("models/metrics.json") as f:
        metrics_by_model = json.load(f)

    best_name = select_best_model(metrics_by_model, metric="accuracy")
    best_model = load_trained_model(best_name)

    joblib.dump(best_model, "models/best_model.pkl")
    print(f"Best model: {best_name} (accuracy={metrics_by_model[best_name]['accuracy']:.4f})")
    print("Saved to models/best_model.pkl")