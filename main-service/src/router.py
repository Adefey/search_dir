import ctypes
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from hashlib import sha256
from threading import Thread

from fastapi import FastAPI, File, Form, HTTPException, status
from models import IndexRequestModel, ResponsePathsModel, ScoredFileModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from redis import Redis
from utils import consumer, index_processor, search

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


@app.post("/api/v1/search", response_model=ResponsePathsModel)
def post_search(
    text_query: str | None = Form(default=None), image_query: bytes | None = File(default=None), top_n: int = 5
):
    logger.info(f"Got /search request")

    # Protection against ''
    if not text_query:
        text_query = None

    # Protection against b''
    if not image_query:
        image_query = None

    try:
        result = search(qdrant, top_n, text_query, image_query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    response = ResponsePathsModel(files=[ScoredFileModel(file=file, score=score) for file, score in result])

    logger.info(f"Finished /search ")
    return response


@app.post("/api/v1/index")
def post_index(request: IndexRequestModel):
    logger.info(f"Got /api/v1/index request")
    file_paths = request.files

    try:
        index_processor(file_paths, qdrant)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    logger.info(f"Finished /api/v1/index request")
