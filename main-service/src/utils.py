import logging
import mimetypes
import os
from hashlib import sha256
from pathlib import PosixPath
from time import sleep
from uuid import UUID

import gradio as gr
import requests
from fastapi import status
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList, PointStruct
from redis import Redis

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 30))
CONSUMER_EMPTY_RESCAN_DELAY = int(os.environ.get("CONSUMER_EMPTY_RESCAN_DELAY", 30))
CONNECTION_ERROR_CONSUMER_RETRY_DELAY = int(os.environ.get("CONNECTION_ERROR_CONSUMER_RETRY_DELAY", 5))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "queue")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "files")
EMBEDDING_SERVICE_URL = os.environ.get(
    "EMBEDDING_SERVICE_ADDRESS",
    "EMBEDDING_SERVICE_ADDRESS",
)
ACTION_CREATE_ID = int(os.environ.get("ACTION_CREATE_ID", "1"))
ACTION_UPDATE_ID = int(os.environ.get("ACTION_UPDATE_ID", "2"))
ACTION_DELETE_ID = int(os.environ.get("ACTION_DELETE_ID", "3"))
logger = logging.getLogger(__name__)

redis = Redis(host="redis", port=6379, decode_responses=True)
qdrant = QdrantClient(host="qdrant", port=6333)


def consumer():
    """
    Endless function that runs in background to get jobs from queue
    """
    index_files = []

    server_io_success = True

    while True:
        action = None
        filename = None
        if server_io_success:
            queue_element = redis.brpop(
                QUEUE_NAME,
                timeout=CONSUMER_EMPTY_RESCAN_DELAY,
            )
            if queue_element is not None:
                _, queue_element = queue_element
                queue_element = queue_element.split(",")
                filename, action = queue_element[0], int(queue_element[1])

        logger.info(f"[CONSUMER] retrieved file {filename}, action: {action}")

        if filename is not None and action in (
            ACTION_CREATE_ID,
            ACTION_UPDATE_ID,
        ):
            index_files.append(filename)
        elif action == ACTION_DELETE_ID:
            # Fast delete, no waiting
            logger.info(f"Removing file {filename} from index")
            remove_file(filename)

        if len(index_files) >= BATCH_SIZE or filename is None:
            logger.info(f"[CONSUMER] Found {len(index_files)} files")
            if index_files:
                logger.info(f"[CONSUMER] Sending to index")

                try:
                    index_processor(index_files)
                    logger.info("[CONSUMER] Index is processed")
                    index_files = []
                    logger.info("[CONSUMER] Buffer is cleared")
                    server_io_success = True
                except RuntimeError as exc:
                    logger.error(f"[CONSUMER] Drop buffer with potential problematic data: ({exc})")
                    index_files = []
                    logger.info("[CONSUMER] Buffer is cleared")
                    server_io_success = True
                except requests.exceptions.ConnectionError as exc:
                    logger.error(
                        "[CONSUMER] service call"
                        f" failed ({exc}), will"
                        " retry in"
                        f" {CONNECTION_ERROR_CONSUMER_RETRY_DELAY} sec"
                    )
                    server_io_success = False
                    sleep(CONNECTION_ERROR_CONSUMER_RETRY_DELAY)

            if filename is None:
                logger.info("[CONSUMER] No files were retrieved from queue")


def index_processor(file_paths: list[str]):
    """
    Add files to index
    """
    logger.info(f"Got index request")

    files = []
    for file_path in file_paths:
        if isinstance(file_path, PosixPath):
            file_path = file_path.as_posix()
        with open(file_path, "rb") as file:
            files.append((
                "files",
                (file_path, file.read()),
            ))

    resp = requests.post(
        f"http://{EMBEDDING_SERVICE_URL}/api/v1/file_embeddings",
        files=files,
    )

    if resp.status_code != 200:
        logger.error(f"http://{EMBEDDING_SERVICE_URL}/api/v1/file_embeddings returned status {resp.status_code}")
        raise RuntimeError(f"http://{EMBEDDING_SERVICE_URL}/api/v1/file_embeddings returned status {resp.status_code}")

    json_resp = resp.json()

    processed_files = []
    for processed_file in json_resp["file_records"]:
        processed_files.append((
            processed_file["file_path"],
            processed_file["embedding"],
        ))

    unprocessed_files = json_resp["unprocessed_files"]
    if unprocessed_files:
        logger.error(f"Unprocessed files: {unprocessed_files}")

    qdrant.upload_points(
        QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(UUID(hex=sha256(path.encode()).hexdigest()[:32])),
                vector=emb,
                payload={"path": path},
            )
            for path, emb in processed_files
        ],
    )

    logger.info(f"Finished index request")


def remove_file(file_path: str):
    """
    Remove file from index
    """
    logger.info(f"Got remove request")
    if isinstance(file_path, PosixPath):
        file_path = file_path.as_posix()
    file_id = str(UUID(hex=sha256(file_path.encode()).hexdigest()[:32]))
    qdrant.delete(
        QDRANT_COLLECTION_NAME,
        points_selector=PointIdsList(points=[file_id]),
    )
    logger.info(f"Finished remove request")


def search(
    text_query: str | None = None,
    image_query: bytes | None = None,
    top_n: int = 5,
):
    """
    Search files by text and/or image query
    """
    if text_query is None and image_query is None:
        logger.error("Text Query and Image query cannot both be None")
        raise ValueError("Text Query and Image query cannot both be None")

    logger.info(f"Got search request")

    if text_query:
        resp = requests.post(
            f"http://{EMBEDDING_SERVICE_URL}/api/v1/text_embedding",
            data={"text": text_query},
        )
        if resp.status_code != status.HTTP_200_OK:
            logger.error(f"http://{EMBEDDING_SERVICE_URL}/api/v1/text_embedding returned status {resp.status_code}")
            raise RuntimeError(
                f"http://{EMBEDDING_SERVICE_URL}/api/v1/text_embedding returned status {resp.status_code}",
            )
        text_embedding = resp.json()["embedding"]

    if image_query:
        resp = requests.post(
            f"http://{EMBEDDING_SERVICE_URL}/api/v1/image_embedding",
            files=[("image", image_query)],
        )
        if resp.status_code != status.HTTP_200_OK:
            logger.error(f"http://{EMBEDDING_SERVICE_URL}/api/v1/image_embedding returned status {resp.status_code}")
            raise RuntimeError(
                f"http://{EMBEDDING_SERVICE_URL}/api/v1/image_embedding returned status {resp.status_code}",
            )
        image_embedding = resp.json()["embedding"]

    if text_query is None:
        logger.info("Using only image embedding")
        search_embedding = image_embedding
    elif image_query is None:
        logger.info("Using only text embedding")
        search_embedding = text_embedding
    else:
        logger.info("Using average between text and image embeddings")
        search_embedding = [(e1 + e2) / 2 for e1, e2 in zip(text_embedding, image_embedding)]

    result = qdrant.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=search_embedding,
        limit=top_n,
    ).points
    files_and_scores = [(item.payload["path"], item.score) for item in result]

    logger.info(f"Finished search")
    return files_and_scores


def search_ui(
    text_query: str = None,
    image_query: bytes = None,
    top_n: int = 5,
):
    """
    Search function for Gradio
    """
    # Protection against ''
    if not text_query:
        text_query = None

    # Protection against b''
    if not image_query:
        image_query = None

    result = []
    try:
        result = search(text_query, image_query, top_n)
    except ValueError as exc:
        logger.error(f"[UI] Incorrect request, got exception: {str(exc)}")
    except RuntimeError as exc:
        logger.error(f"[UI] Error while processing request, got exception: {str(exc)}")

    result_paths_with_score = []
    result_paths = []

    for filename, score in result:
        result_paths.append(filename)

        mime_type, _ = mimetypes.guess_file_type(filename)

        if mime_type is None:
            logger.warning(f"File {filename} misses MIME type suffix")
            continue

        if "image" in mime_type:
            result_paths_with_score.append((filename, f"Score: {score}"))
        else:
            logger.warning(f"File {filename} of type {mime_type} is not an image")

    return result_paths_with_score, result_paths


def upload_files(
    filenames: list[str],
    file_contents: list[bytes],
):
    """
    Check and upload files
    """
    if len(filenames) != len(file_contents):
        raise RuntimeError(f"Filenames should match file contents. {len(filenames)=}!={len(file_contents)}")

    for filename, content in zip(filenames, file_contents):
        mime_type, _ = mimetypes.guess_file_type(filename)

        if mime_type is None:
            logger.warning(f"File {filename} misses MIME type suffix")
            continue

        open_file_mode = "wb"

        if mime_type is None:
            logger.warning(f"File {filename} misses MIME type suffix")

        if "text" in mime_type:
            logger.info(f"Received text: {filename}")
            content = content.decode("utf-8", errors="ignore")
            open_file_mode = "w"
        elif "image" in mime_type:
            logger.info(f"Received image: {filename}")
        else:
            logger.warning(f"File {filename} of type {mime_type} is not supported")
            raise RuntimeError(
                f"File {filename} of type {mime_type} is not supported",
            )

        target_filename = os.path.join("/data/upload", filename)
        with open(target_filename, open_file_mode) as file:
            logger.info(f"Writing to new file {target_filename}")
            file.write(content)


def upload_files_ui(files: list[gr.File]):
    """
    Upload function for Gradio
    """
    if not files:
        return "No files received"

    filenames = []
    contents = []

    for file in files:
        filename = os.path.basename(file.name)
        filenames.append(filename)
        with open(file.name, "rb") as file:
            content = file.read()
        contents.append(content)

    upload_files(filenames, contents)

    return f"{len(files)} file have been uploaded" if len(files) == 1 else (f"{len(files)} files have been uploaded")
