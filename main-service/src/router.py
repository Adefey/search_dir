import logging
import mimetypes
import os
import sys
import uuid
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, status
from models import IndexRequestModel, ResponsePathsModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

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
qdrant = QdrantClient(host="qdrant", port=6333)
if not qdrant.collection_exists("files"):
    qdrant.create_collection(
        collection_name="files",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_ADDRESS", "")


@app.get("/api/v1/files", response_model=ResponsePathsModel)
def get_files(query: str, top_n: int = 5):
    logger.info(f"Got /api/v1/files request")
    resp = requests.post(
        f"http://{EMBEDDING_SERVICE_URL}/api/v1/text_embedding",
        json={"text": query},
    )
    if resp.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=500,
            detail=(
                f"http://{EMBEDDING_SERVICE_URL}/api/v1/text_embedding"
                f" returned status {resp.status_code}"
            ),
        )
    embedding = resp.json()["embedding"]
    result = qdrant.query_points(
        collection_name="files",
        query=embedding,
        limit=top_n,
    ).points
    files = [item.payload["path"] for item in result]
    response = ResponsePathsModel(files=files)
    logger.info(f"Finished /api/v1/files")
    return response


@app.post("/api/v1/index")
def post_index(request: IndexRequestModel):
    logger.info(f"Got /api/v1/index request")
    file_paths = request.files
    # Separate files into texts and images
    images = []
    texts = []
    for file_path in file_paths:
        file_type, _ = mimetypes.guess_file_type(file_path)
        if "image" in file_type:
            images.append(file_path)
            logger.debug(f"File {file_path} is image")
        elif "text" in file_type:
            texts.append(file_path)
            logger.debug(f"File {file_path} is text")
        else:
            logger.warning(f"File {file_path} is not supported")

    logger.debug(f"{images=} {texts=}")

    # Get embeddings for images
    images_payload = []
    for image in images:
        with open(image, "rb") as file:
            images_payload.append(("images", file.read()))
    if images_payload:
        resp = requests.post(
            f"http://{EMBEDDING_SERVICE_URL}/api/v1/images_embeddings",
            files=images_payload,
        )
        if resp.status_code != status.HTTP_200_OK:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"http://{EMBEDDING_SERVICE_URL}/api/v1/images_embeddings"
                    f" returned status {resp.status_code}"
                ),
            )
        embeddings = resp.json()["embeddings"]
        processed_images = list(zip(images, embeddings))
    else:
        processed_images = []

    # Get embeddings for texts
    texts_payload = []
    for text in texts:
        with open(text) as file:
            texts_payload.append(file.read())
    if texts_payload:
        resp = requests.post(
            f"http://{EMBEDDING_SERVICE_URL}/api/v1/texts_embeddings",
            json={"texts": texts_payload},
        )
        if resp.status_code != status.HTTP_200_OK:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"http://{EMBEDDING_SERVICE_URL}/api/v1/texts_embeddings"
                    f" returned status {resp.status_code}"
                ),
            )
        embeddings = resp.json()["embeddings"]
        processed_texts = list(zip(texts, embeddings))
    else:
        processed_texts = []

    logger.debug(f"{processed_images=} {processed_texts=}")

    qdrant.upload_points(
        "files",
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"path": path},
            )
            for path, emb in processed_images + processed_texts
        ],
    )

    logger.info(f"Finished /api/v1/index request")
