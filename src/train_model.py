from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_model(train_df,cfg):
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= cfg.data.test_size, random_state= cfg.data.random_state
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=cfg.model.logistic_regression.max_iter),
        "random_forest": RandomForestClassifier(n_estimators=cfg.model.random_forest.n_estimators, random_state=cfg.data.random_state)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
