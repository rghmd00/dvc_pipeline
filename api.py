import os
import logging

import joblib
import pandas as pd
import litserve as ls
from pydantic import BaseModel

from src.preprocessing import wrangle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_api")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
ENCODER_DIR = os.path.join(MODEL_DIR, "encoders")


class PredictRequest(BaseModel):
    columns: list[str]
    rows: list[list]


class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        model_name = os.getenv("MODEL_NAME", "random_forest_model.pkl")
        model_path = os.path.join(MODEL_DIR, model_name)

        self._model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Expected features: {list(self._model.feature_names_in_)}")
        self._encoder_dir = ENCODER_DIR

    def decode_request(self, request: PredictRequest):
        df = pd.DataFrame(request.rows, columns=request.columns)
        df = wrangle(df)

        expected = list(self._model.feature_names_in_)
        missing = set(expected) - set(df.columns)
        if missing:
            raise ls.HTTPException(
                status_code=400,
                detail=f"Missing required columns after preprocessing: {sorted(missing)}",
            )

        return df[expected]

    def predict(self, x):
        return self._model.predict(x)

    def encode_response(self, output):
        return {
            "message": "Response Produced Successfully",
            "data": output.tolist(),
        }