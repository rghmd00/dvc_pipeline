

import os
import joblib
import mlflow
import mlflow.sklearn
import dagshub
import dvc.api
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def setup_tracking():
    load_dotenv()
    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        raise ValueError("DAGSHUB_TOKEN environment variable is not set.")

    mlflow.set_tracking_uri("https://dagshub.com/rghmd00/test.mlflow")
    dagshub.init(repo_owner="rghmd00", repo_name="test", mlflow=True)


def train_models(train_df: pd.DataFrame, cfg: dict) -> dict:
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=cfg["model"]["logistic_regression"]["max_iter"]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg["model"]["random_forest"]["n_estimators"],
            random_state=cfg["data"]["random_state"],
        ),
    }

    trained_models = {}
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }
            print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(model, f"model/{name}")  # type: ignore

            joblib.dump(model, f"models/{name}_model.pkl")
            trained_models[name] = {"model": model, "metrics": metrics}

    return trained_models


if __name__ == "__main__":
    setup_tracking()

    cfg = dvc.api.params_show("params.yaml")
    file_path = cfg["processed_data"]["train"]
    train_df = pd.read_csv(file_path)
    print("Data loaded successfully")

    results = train_models(train_df, cfg)


