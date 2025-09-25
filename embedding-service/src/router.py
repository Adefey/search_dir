import logging
import sys
from datetime import datetime

from fastapi import FastAPI, File
from model import Model
from models import (RequestTextModel, RequestTextsModel,
                    ResponseEmbeddingModel, ResponseEmbeddingsModel)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            filename=f'logs/embedding_service_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.log'
        ),
        logging.StreamHandler(stream=sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
model = Model()
app = FastAPI(title="Embedding Microservise")


@app.post(
    "/api/v1/text_embedding",
    response_model=ResponseEmbeddingModel,
)
def post_text_embedding(request: RequestTextModel):
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
def post_texts_embeddings(request: RequestTextsModel):
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
def post_image_embedding(image: bytes = File()):
    logger.info(f"Got /api/v1/image_embedding request")
    emb = model.encode_image(image)
    response = ResponseEmbeddingModel(embedding=emb)
    logger.info(f"Finished /api/v1/image_embedding")
    return response


@app.post(
    "/api/v1/images_embeddings",
    response_model=ResponseEmbeddingsModel,
)
def post_images_embeddings(images: list[bytes] = File()):
    logger.info(f"Got /api/v1/images_embeddings request")
    emb = model.encode_images(images)
    response = ResponseEmbeddingsModel(embeddings=emb)
    logger.info(f"Finished /api/v1/images_embeddings")
    return response
