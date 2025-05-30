# src/pipeline.py
import dvc.api


import hydra
from omegaconf import DictConfig

from data_loader import load_data
from preprocessing import wrangle
from save_model import save_model
from train_model import train_models

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):


    # Step 1: Load data
    train_df = load_data(cfg.data.train_csv)

    # Step 2: Preprocess data
    train_df_Preprocessed = wrangle(train_df)

    # Step 3: Train model
    predictor = train_models(train_df_Preprocessed,cfg)

    # Step 4: Save model using pickle
    save_model(predictor, cfg.model_output)


if __name__ == "__main__":
    main()
