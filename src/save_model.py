import dvc.api
import pandas as pd
import pickle
from train_model import train_models  # Adjust path if needed


def save_model(predictor, path):
    with open(path, "wb") as f:
        pickle.dump(predictor, f)


if __name__ == "__main__":
    # Load configuration and data
    cfg = dvc.api.params_show("d:/ITI/10-MLOPS/dvc_test/ITI-MLOps/params.yaml")
    file_path = cfg["processed_data"]["train"]
    train_df = pd.read_csv(file_path)

    # Train models and retrieve the Random Forest model
    models = train_models(train_df, cfg)
    rf_model = models["random_forest"]

    logistic_model = models["logistic_regression"]
    # Save the trained Logistic Regression model        
    save_model(logistic_model, "models/logistic_regression_model.pkl")
    print("Logistic Regression model saved successfully.")

    # Save the trained Random Forest model
    save_model(rf_model, "models/random_forest_model.pkl")
    print("Random Forest model saved successfully.")
