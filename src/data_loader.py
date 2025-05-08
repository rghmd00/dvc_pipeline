import pandas as pd
import dvc.api


def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    cfg = dvc.api.params_show("d:/ITI/10-MLOPS/dvc_test/ITI-MLOps/params.yaml") 
    file_path = cfg["data"]["train_csv"]
    data = load_data(file_path)
    print("Data loaded successfully")
