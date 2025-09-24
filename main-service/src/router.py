from fastapi import FastAPI, File
import logging
import sys
from datetime import datetime
import requests
from models import ResponsePathsModel
from qdrant_client import QdrantClient
from redis import Redis

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            filename=f'logs/main_service_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.log'
        ),
        logging.StreamHandler(stream=sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
app = FastAPI(title="Main Microservise")
redis = Redis(host="redis", port=6379, decode_responses=True)
client = QdrantClient(host="qdrant", port=6333)


@app.get("/api/v1/files", response_model=ResponsePathsModel)
def get_files(query: str, top_n: int = 5):
    logger.info(f"Got /api/v1/files request")
    # test
    status = requests.post(
        f"http://embedding-service:8000/api/v1/text_embedding", json={"text": query}
    ).status_code
    response = ResponsePathsModel(files=[f"{status}"] * top_n)
    logger.info(f"Finished /api/v1/files")
    return response
