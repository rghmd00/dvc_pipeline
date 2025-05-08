import dvc.api
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle


def save_model(predictor, path):
    with open(path, "wb") as f:
        pickle.dump(predictor, f)



if __name__ == "__main__":

    cfg = dvc.api.params_show("d:/ITI/10-MLOPS/dvc_test/ITI-MLOps/params.yaml")
    file_path = cfg["processed_data"]["train"]
    train_df = pd.read_csv(file_path)
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    model = RandomForestClassifier(n_estimators=cfg['model']['random_forest']['n_estimators'], random_state=cfg['data']['random_state'])
    model.fit(X, y)

    save_model(model, "models/random_forest_model.pkl") 