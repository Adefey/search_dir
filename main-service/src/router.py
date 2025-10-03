import logging
import mimetypes
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from itertools import batched
from threading import Thread

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.staticfiles import StaticFiles
from models import FilePathsModel, IndexRequestModel, ResponsePathsModel, ScoredFileModel
from qdrant_client.models import Distance, VectorParams
from utils import consumer, gradio_search_ui, index_processor, qdrant, redis, remove_file, search

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run consumer in background
    """
    Thread(target=consumer, daemon=True).start()

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

gradio_app = gr.Interface(
    fn=gradio_search_ui,
    inputs=[
        gr.Textbox(label="Text query"),
        gr.File(label="Image query", type="binary", file_types=["image"]),
        gr.Slider(minimum=1, maximum=500, value=5, step=1, label="Result count"),
    ],
    outputs=[
        gr.Gallery(label="Search result - image render", height="auto"),
        gr.Files(label="Search result - files", type="filepath"),
    ],
    title="File search",
    description="File search based on file content. Supports images and texts",
)

app = gr.mount_gradio_app(app, gradio_app, path="/ui")
app.mount("/data", StaticFiles(directory="/data"), name="data")


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
        result = search(text_query, image_query, top_n)
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
            index_processor(batch)
    except RuntimeError as exc:
        logger.error(f"Error while processing request, got exception: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while processing request, got exception: {str(exc)}",
        )

    logger.info(f"Finished /api/v1/index request")


@app.delete("/api/v1/index")
def delete_index(request: FilePathsModel):
    """
    Deletes file (by filename) from index
    """
    logger.info(f"Got delete /api/v1/index request")
    filenames = request.files
    try:
        for filename in filenames:
            remove_file(filename)
    except RuntimeError as exc:
        logger.error(f"Error while removing file, got exception: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while removing file, got exception: {str(exc)}",
        )

    logger.info(f"Finished delete /api/v1/index request")


@app.post("/api/v1/files")
def post_files(files: list[UploadFile]):
    logger.info(f"Got post /api/v1/files request")

    try:
        filenames = [file.filename for file in files]
        contents = [file.file.read() for file in files]
    except Exception as exc:
        logger.error(f"Unprocessable files. Exception while reading data: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unprocessable files. Exception while reading data: {str(exc)}",
        )

    for filename, content in zip(filenames, contents):

        mime_type, _ = mimetypes.guess_file_type(filename)
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
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File {filename} of type {mime_type} is not supported",
            )

        target_filename = os.path.join("/data", filename)
        with open(target_filename, open_file_mode) as file:
            file.write(content)

    logger.info(f"Finished post /api/v1/files request")
