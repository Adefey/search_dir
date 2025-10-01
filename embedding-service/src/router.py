import ctypes
import logging
import os
import sys
from datetime import datetime

from fastapi import BackgroundTasks, Depends, FastAPI, File
from model import Model
from models import (RequestTextModel, RequestTextsModel,
                    ResponseEmbeddingModel, ResponseEmbeddingsModel)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(filename=f'logs/embedding_service_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.log'),
        logging.StreamHandler(stream=sys.stdout),
    ],
)

TRIM_EVERY_N_MODEL_REQUESTS = int(os.environ.get("TRIM_EVERY_N_MODEL_REQUESTS", "10"))
MODEL_REQUEST_COUNTER = 0

logger = logging.getLogger(__name__)
model = Model()
app = FastAPI(title="Embedding Microservise")

libc = ctypes.CDLL("libc.so.6")


def trim_memory():
    logger.info("Performing background memory trim...")
    libc.malloc_trim(0)
    logger.info("Memory trim finished.")


def manage_memory_trim(background_tasks: BackgroundTasks):
    global BATCH_COUNTER

    BATCH_COUNTER += 1
    if BATCH_COUNTER >= TRIM_EVERY_N_MODEL_REQUESTS:
        logger.info(f"Reached {BATCH_COUNTER} batches, scheduling a memory trim.")
        background_tasks.add_task(trim_memory)
        BATCH_COUNTER = 0


@app.post(
    "/api/v1/text_embedding",
    response_model=ResponseEmbeddingModel,
)
def post_text_embedding(request: RequestTextModel, _trim_task: None = Depends(manage_memory_trim)):
    logger.info(f"Got /api/v1/text_embedding request")
    text = request.text
    emb = model.encode_text(text)
    response = ResponseEmbeddingModel(embedding=emb)
    logger.info(f"Finished /api/v1/text_embedding")
    return response


@app.post(
    "/api/v1/texts_embeddings",
    response_model=ResponseEmbeddingsModel,
)
def post_texts_embeddings(request: RequestTextsModel, _trim_task: None = Depends(manage_memory_trim)):
    logger.info(f"Got /api/v1/texts_embeddings request")
    texts = request.texts
    emb = model.encode_texts(texts)
    response = ResponseEmbeddingsModel(embeddings=emb)
    logger.info(f"Finished /api/v1/texts_embeddings")
    return response


@app.post(
    "/api/v1/image_embedding",
    response_model=ResponseEmbeddingModel,
)
def post_image_embedding(image: bytes = File(), _trim_task: None = Depends(manage_memory_trim)):
    logger.info(f"Got /api/v1/image_embedding request")
    emb = model.encode_image(image)
    response = ResponseEmbeddingModel(embedding=emb)
    logger.info(f"Finished /api/v1/image_embedding")
    return response


@app.post(
    "/api/v1/images_embeddings",
    response_model=ResponseEmbeddingsModel,
)
def post_images_embeddings(images: list[bytes] = File(), _trim_task: None = Depends(manage_memory_trim)):
    logger.info(f"Got /api/v1/images_embeddings request")
    emb = model.encode_images(images)
    response = ResponseEmbeddingsModel(embeddings=emb)
    logger.info(f"Finished /api/v1/images_embeddings")
    return response
