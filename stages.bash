dvc stage add -n preprocess_data `
  -d data/raw/train.csv `
  -d src/preprocessing.py `
  -d params.yaml `
  -o data/processed/processed_train.csv `
  python src/preprocessing.py





dvc stage add -n train_model `
  -d data/processed/processed_train.csv `
  -d src/train.py `
  -d params.yaml `
  python src/train.py



dvc stage add -n save_model `
  -d data/processed/processed_train.csv `
  -d src/save_model.py `
  -d params.yaml `
  -o model/logistic_regression_model.pkl `
  -o model/random_forest_model.pkl `
  python src/save_model.py