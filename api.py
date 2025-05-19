import os
import pickle
import joblib
import litserve as ls
import pandas as pd

class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        # with open(
        #     os.path.join(
        #         os.path.dirname(__file__),
        #         "models",
        #         "logistic_regression_model.pkl",),
        #         "rb") as pkl:
        #     self._model = pickle.load(pkl)
        self._model = joblib.load(
            os.path.join(
                os.path.dirname(__file__),
                "models",
                "random_forest_model.pkl"))
        # print feature names that the model was trained on
        print("Model loaded successfully")
        print("Feature names:", self._model.feature_names_in_)

    def decode_request(self, request):
        try:
            print(request)
            columns = request["columns"]
            rows = request["rows"]
            df = pd.DataFrame(rows, columns=columns)
            print(df)
            return df
        except Exception:
            return None

    def predict(self, x):
        print(x)
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        if output is None:
            message = "Error Occurred"
            response = {
                "message": message,
                "data": None,
            }
        else:
            message = "Response Produced Successfully"
            response = {
                "message": message,
                "data": output.tolist(),
            }
        return response