from sklearn.preprocessing import LabelEncoder


def wrangle(df):

    df = df.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"])

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

    return df
