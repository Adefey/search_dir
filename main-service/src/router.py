import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from itertools import batched
from threading import Thread

from fastapi import FastAPI, File, Form, HTTPException, status
from models import FilePathModel, IndexRequestModel, ResponsePathsModel, ScoredFileModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from redis import Redis
from utils import consumer, index_processor, remove_file, search

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(filename=f'logs/main_service_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.log'),
        logging.StreamHandler(stream=sys.stdout),
    ],
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "30"))
EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", "512"))
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "files")

logger = logging.getLogger(__name__)
qdrant = QdrantClient(host="qdrant", port=6333)
redis = Redis(host="redis", port=6379, decode_responses=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run consumer in background
    """
    Thread(target=consumer, args=(redis, qdrant), daemon=True).start()

    yield


app = FastAPI(
    title="Main Microservise",
    description="Microservice used to manage search and indexation, stores indexed giles in vector DB",
    version=os.environ.get("APP_VERSION", "0.1"),
    lifespan=lifespan,
)

if not qdrant.collection_exists(QDRANT_COLLECTION_NAME):
    logger.info(f"Creating db 'files' with embedding size {EMBEDDING_SIZE}")
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )


@app.post("/api/v1/search", response_model=ResponsePathsModel)
def post_search(
    text_query: str | None = Form(default=None),
    image_query: bytes | None = File(default=None),
    top_n: int = 5,
):
    """
    Search files on text or/and image query
    """
    logger.info(f"Got /api/v1/search request")

    # Protection against ''
    if not text_query:
        text_query = None

    # Protection against b''
    if not image_query:
        image_query = None

    try:
        result = search(qdrant, top_n, text_query, image_query)
    except ValueError as exc:
        logger.error(f"Incorrect request, got exception: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Incorrect request, got exception: {str(exc)}",
        )
    except RuntimeError as exc:
        logger.error(f"Error while processing request, got exception: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while processing request, got exception: {str(exc)}",
        )
    response = ResponsePathsModel(files=[ScoredFileModel(file=file, score=score) for file, score in result])

    logger.info(f"Finished /api/v1/search request")
    return response


@app.post("/api/v1/index")
def post_index(request: IndexRequestModel):
    """
    Index text and image files
    """
    logger.info(f"Got /api/v1/index request")
    file_paths = request.files

    if len(file_paths) > BATCH_SIZE:
        logger.info(
            f"Got {len(file_paths)} files, while batch size is {BATCH_SIZE}, will be processed in multiple batches"
        )

    batches = batched(file_paths, BATCH_SIZE)

    try:
        for i, batch in enumerate(batches):
            logger.info(f"Indexing batch {i+1}")
            index_processor(batch, qdrant)
    except RuntimeError as exc:
        logger.error(f"Error while processing request, got exception: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while processing request, got exception: {str(exc)}",
        )

    logger.info(f"Finished /api/v1/index request")


@app.delete("/api/v1/index")
def delete_index(request: FilePathModel):
    """
    Deletes file (by filename) from index
    """
    logger.info(f"Got delete /api/v1/index request")
    filename = request.file
    try:
        remove_file(filename, qdrant)
    except RuntimeError as exc:
        logger.error(f"Error while removing file, got exception: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while removing file, got exception: {str(exc)}",
        )

    logger.info(f"Finished delete /api/v1/index request")
