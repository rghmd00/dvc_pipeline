# http://127.0.0.1:8000/docs


import os
import logging
import litserve as ls
from api import InferenceAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    api = InferenceAPI()
    server = ls.LitServer(api, accelerator="cpu")

    logger.info(f"Starting server on {host}:{port}")
    server.run(host=host, port=port, generate_client_file=False)