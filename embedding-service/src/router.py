import ctypes
import logging
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, UploadFile, status
from model import Model
from models import ResponseEmbeddingOnlyModel, ResponseFileEmbeddingModel, ResponseFileEmbeddingsModel

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
MODEL_CALL_COUNTER = 0

logger = logging.getLogger(__name__)
model = Model()
app = FastAPI(
    title="Embedding Microservise",
    description="Microservice used to get text and image embeddings from model",
    version=os.environ.get("APP_VERSION", "0.1"),
)

libc = ctypes.CDLL("libc.so.6")


def trim_memory():
    """
    Perform malloc_trim from libc
    """
    logger.info("Performing background memory trim...")
    libc.malloc_trim(0)
    logger.info("Memory trim finished.")


def manage_memory_trim(background_tasks: BackgroundTasks):
    """
    Background task that performs malloc_trim every TRIM_EVERY_N_MODEL_REQUESTS reuests
    """
    global MODEL_CALL_COUNTER

    MODEL_CALL_COUNTER += 1
    if MODEL_CALL_COUNTER >= TRIM_EVERY_N_MODEL_REQUESTS:
        logger.info(f"Reached {MODEL_CALL_COUNTER} batches, scheduling a memory trim.")
        background_tasks.add_task(trim_memory)
        MODEL_CALL_COUNTER = 0


@app.post(
    "/api/v1/text_embedding",
    response_model=ResponseEmbeddingOnlyModel,
)
def post_text_embedding(text: str = Form(), _trim_task: None = Depends(manage_memory_trim)):
    """
    Get embedding from text
    """
    logger.info(f"Got /api/v1/text_embedding request")
    logger.info(f"{text=} {type(text)=}")
    try:
        embedding = model.encode_text(text)
    except Exception as exc:
        logger.error(f"Processing failed with exception {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed with exception {str(exc)}",
        )

    logger.info(f"Finished/api/v1/text_embedding request")

    return ResponseEmbeddingOnlyModel(embedding=embedding)


@app.post(
    "/api/v1/image_embedding",
    response_model=ResponseEmbeddingOnlyModel,
)
def post_image_embedding(image: UploadFile, _trim_task: None = Depends(manage_memory_trim)):
    """
    Get embedding from image
    """
    logger.info(f"Got /api/v1/image_embedding request")

    try:
        image = image.file.read()
    except Exception as exc:
        logger.error(f"Cannot read image: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot read image: {str(exc)}",
        )

    try:
        embedding = model.encode_image(image)
    except Exception as exc:
        logger.error(f"Processing failed with exception: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed with exception: {str(exc)}",
        )

    logger.info(f"Finished/api/v1/image_embedding request")

    return ResponseEmbeddingOnlyModel(embedding=embedding)


@app.post(
    "/api/v1/file_embeddings",
    response_model=ResponseFileEmbeddingsModel,
)
def post_file_embeddings(files: list[UploadFile], _trim_task: None = Depends(manage_memory_trim)):
    """
    Get embeddings from files. Returns filenames with embeddings and unprocessed files
    """
    logger.info(f"Got /api/v1/file_embeddings request")

    try:
        filenames = [file.filename for file in files]
        contents = [file.file.read() for file in files]
    except Exception as exc:
        logger.error(f"Unprocessable files. Exception while reading data: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unprocessable files. Exception while reading data: {str(exc)}",
        )

    logger.debug(f"Filenames: {filenames}")

    images = []
    texts = []
    unprocessed = []

    for filename, content in zip(filenames, contents):
        mime_type, _ = mimetypes.guess_file_type(filename)
        if mime_type is None:
            logger.warning(f"File {filename} misses MIME type suffix")

        if "text" in mime_type:
            texts.append((filename, content.decode("utf-8", errors="ignore")))
        elif "image" in mime_type:
            images.append((filename, content))
        else:
            logger.warning(f"File {filename} of type {mime_type} is not supported")
            unprocessed.append(filename)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_texts = executor.submit(model.encode_text_files, texts)
            future_images = executor.submit(model.encode_image_files, images)

            text_filename_embeddings = future_texts.result()
            image_filename_embeddings = future_images.result()
    except Exception as exc:
        logger.error(f"Processing failed with exception: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed with exception: {str(exc)}",
        )

    logger.info(f"Processed {len(text_filename_embeddings)} texts and {len(image_filename_embeddings)} images")

    response = ResponseFileEmbeddingsModel(
        file_records=[
            ResponseFileEmbeddingModel(file_path=filename, embedding=embedding)
            for filename, embedding in text_filename_embeddings + image_filename_embeddings
        ],
        unprocessed_files=unprocessed,
    )
    logger.info(f"Finished /api/v1/file_embeddings")
    return response
