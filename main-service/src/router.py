import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from hashlib import sha256
from threading import Thread

import requests
from fastapi import FastAPI, File, HTTPException, status
from models import IndexRequestModel, ResponsePathsModel, ScoredFileModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from redis import Redis
from utils import consumer, index_processor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(filename=f'logs/main_service_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.log'),
        logging.StreamHandler(stream=sys.stdout),
    ],
)
EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_ADDRESS", "")
EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", "512"))

logger = logging.getLogger(__name__)

qdrant = QdrantClient(host="qdrant", port=6333)
redis = Redis(host="redis", port=6379, decode_responses=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Thread(target=consumer, args=(redis, qdrant), daemon=True).start()

    yield


app = FastAPI(title="Main Microservise", lifespan=lifespan)

if not qdrant.collection_exists("files"):
    logger.info(f"Creating db 'files' with embedding size {EMBEDDING_SIZE}")
    qdrant.create_collection(
        collection_name="files",
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )


@app.get("/api/v1/text_search", response_model=ResponsePathsModel)
def get_text_search(query: str, top_n: int = 5):
    logger.info(f"Got /get_text_search request")
    resp = requests.post(
        f"http://{EMBEDDING_SERVICE_URL}/api/v1/text_embedding",
        json={"text": query},
    )
    if resp.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=500,
            detail=f"http://{EMBEDDING_SERVICE_URL}/api/v1/text_embedding returned status {resp.status_code}",
        )
    embedding = resp.json()["embedding"]
    result = qdrant.query_points(
        collection_name="files",
        query=embedding,
        limit=top_n,
    ).points
    response = ResponsePathsModel(
        files=[ScoredFileModel(file=item.payload["path"], score=item.score) for item in result]
    )
    logger.info(f"Finished get_text_search")
    return response


@app.post("/api/v1/image_search", response_model=ResponsePathsModel)
def post_image_search(image: bytes = File(), top_n: int = 5):
    logger.info(f"Got /get_image_search request")
    resp = requests.post(
        f"http://{EMBEDDING_SERVICE_URL}/api/v1/image_embedding",
        files=[("image", image)],
    )
    if resp.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=500,
            detail=f"http://{EMBEDDING_SERVICE_URL}/api/v1/image_embedding returned status {resp.status_code}",
        )
    embedding = resp.json()["embedding"]
    result = qdrant.query_points(
        collection_name="files",
        query=embedding,
        limit=top_n,
    ).points
    response = ResponsePathsModel(
        files=[ScoredFileModel(file=item.payload["path"], score=item.score) for item in result]
    )
    logger.info(f"Finished get_image_search")
    return response


@app.post("/api/v1/index")
def post_index(request: IndexRequestModel):
    logger.info(f"Got /api/v1/index request")
    file_paths = request.files

    try:
        index_processor(file_paths, qdrant)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    logger.info(f"Finished /api/v1/index request")
