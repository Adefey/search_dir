import logging
import mimetypes
import os
from hashlib import sha256
from time import sleep
from uuid import UUID

import requests
from fastapi import status
from models import IndexRequestModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from redis import Redis

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 30))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "queue")
EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_ADDRESS", "")
logger = logging.getLogger(__name__)


def consumer(redis: Redis, qdrant: QdrantClient):
    files = []

    server_io_success = True

    while True:
        filename = None
        if server_io_success:
            queue_filename = redis.brpop(QUEUE_NAME, timeout=30)
            if queue_filename is not None:
                _, filename = queue_filename

        logger.info(f"[CONSUMER] retrieved file {filename}")

        if filename is not None:
            files.append(filename)

        if len(files) >= BATCH_SIZE or filename is None:
            logger.info(f"[CONSUMER] Found {len(files)} files")
            if files:
                logger.info(f"[CONSUMER] Sending to index")

                try:
                    index_processor(files, qdrant)
                    logger.info("[CONSUMER] Index is processed")
                    files = []
                    logger.info("[CONSUMER] Buffer is cleared")
                    server_io_success = True
                except RuntimeError as exc:
                    logger.error(f"[CONSUMER] Drop buffer with potential problematic data and retry ({exc})")
                    files = []
                    logger.info("[CONSUMER] Buffer is cleared")
                    server_io_success = True
                except requests.exceptions.ConnectionError as exc:
                    logger.error(f"[CONSUMER] service call failed ({exc}), will retry in 5 sec")
                    server_io_success = False
                    sleep(5)

            if filename is None:
                logger.info("[CONSUMER] No files were retrieved from queue")


def index_processor(file_paths: list[str], qdrant: QdrantClient):
    logger.info(f"Got index request")
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
            raise RuntimeError(
                f"http://{EMBEDDING_SERVICE_URL}/api/v1/images_embeddings returned status {resp.status_code}",
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
            raise RuntimeError(
                f"http://{EMBEDDING_SERVICE_URL}/api/v1/texts_embeddings returned status {resp.status_code}",
            )
        embeddings = resp.json()["embeddings"]
        processed_texts = list(zip(texts, embeddings))
    else:
        processed_texts = []

    logger.debug(f"{processed_images=} {processed_texts=}")
    logger.info(f"Processed {len(processed_images)} images and {len(processed_texts)}")

    qdrant.upload_points(
        "files",
        points=[
            PointStruct(
                id=str(UUID(hex=sha256(path.encode()).hexdigest()[:32])),
                vector=emb,
                payload={"path": path},
            )
            for path, emb in processed_images + processed_texts
        ],
    )

    logger.info(f"Finished processing index")
