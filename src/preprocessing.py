from sklearn.preprocessing import LabelEncoder
import pandas as pd
import dvc.api

def wrangle(df):

    df = df.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"])

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

    return df


if __name__ == "__main__":
    cfg = dvc.api.params_show("d:/ITI/10-MLOPS/dvc_test/ITI-MLOps/params.yaml")
    file_path = cfg["data"]["train_csv"]
    data = pd.read_csv(file_path)
    processed_data = wrangle(data)
    print("Data wrangling completed successfully")

    processed_data.to_csv("data/processed/processed_train.csv", index=False)
    print("Processed data saved to data/processed/processed_train.csv")