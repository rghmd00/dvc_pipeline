import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import dvc.api
import pandas as pd
import joblib 


dagshub.auth.add_app_token(token="64dba0bfb5a93060a3285d63f7a58f6527ecbe61")

# $env:DAGSHUB_TOKEN="64dba0bfb5a93060a3285d63f7a58f6527ecbe61"
# $env:MLFLOW_TRACKING_URI="https://dagshub.com/rghmd00/test.mlflow"



dagshub.init(repo_owner='rghmd00', repo_name='test', mlflow=True)
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

mlflow.set_tracking_uri('https://dagshub.com/rghmd00/test.mlflow')  # Or your MLflow server URI




def train_models(train_df, cfg):
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg['data']['test_size'], random_state=cfg['data']['random_state']
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=cfg['model']['logistic_regression']['max_iter']),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg['model']['random_forest']['n_estimators'],
            random_state=cfg['data']['random_state']
        )
    }

    trained_models = {}

    # Start MLflow tracking
    with mlflow.start_run():

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {acc:.4f}")

            # Log model to MLflow
            mlflow.sklearn.log_model(model, f"model/{name}")
            mlflow.log_metric(f"{name}_accuracy", acc)

            # Save model locally (optional)
            joblib.dump(model, f"models/{name}_model.pkl")
            trained_models[name] = model



            if name == "random_forest":  # For example, register only the random forest model
                mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model/{name}", f"{name}_model")


    return trained_models

# Main execution
if __name__ == "__main__":
    cfg = dvc.api.params_show("d:/ITI/10-MLOPS/dvc_test/ITI-MLOps/params.yaml")
    file_path = cfg["processed_data"]["train"]
    train_df = pd.read_csv(file_path)
    print("Data loaded successfully")
    models = train_models(train_df, cfg)

