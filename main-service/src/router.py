import logging
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
from utils import consumer, index_processor, qdrant, remove_file, search, search_ui, upload_files, upload_files_ui

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
GRADIO_USER = os.environ.get("GRADIO_USER", None)
GRADIO_PASSWORD = os.environ.get("GRADIO_PASSWORD", None)
APP_VERSION = os.environ.get("APP_VERSION", "0.1")


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
    version=APP_VERSION,
    lifespan=lifespan,
)

if not qdrant.collection_exists(QDRANT_COLLECTION_NAME):
    logger.info(f"Creating db 'files' with embedding size {EMBEDDING_SIZE}")
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE,
        ),
    )

with gr.Blocks(title="Search") as gradio_app:
    with gr.Tab("Search"):
        gr.Interface(
            fn=search_ui,
            inputs=[
                gr.Textbox(label="Text query"),
                gr.File(
                    label="Image query",
                    type="binary",
                    file_types=["image"],
                ),
                gr.Slider(
                    minimum=1,
                    maximum=500,
                    value=5,
                    step=1,
                    label="Result count",
                ),
            ],
            outputs=[
                gr.Gallery(
                    label="Search result - image render",
                    height="auto",
                ),
                gr.Files(
                    label="Search result - files",
                    type="filepath",
                ),
            ],
            title="File search",
            description="File search based on file content. Supports images and texts. Currently works only in English",
        )
    with gr.Tab("Upload"):
        gr.Interface(
            fn=upload_files_ui,
            inputs=[
                gr.File(
                    file_count="multiple",
                    label="Load new files to be indexed",
                    file_types=["image", "text"],
                )
            ],
            outputs=[gr.Label(label="Upload status")],
            title="File upload",
            description="Upload file that will be indexed and available for search",
            submit_btn="Upload",
        )

    gr.Markdown(f"""
    ---
    <p align="center">Source code: <a href="https://github.com/Adefey/search_dir" target="_blank">Adefey/search_dir</a> on GitHub</p>
    <p align="center">App version: {APP_VERSION}</p>
    """)

    app = gr.mount_gradio_app(
        app,
        gradio_app,
        path="/ui",
        pwa=True,
        auth=([(GRADIO_USER, GRADIO_PASSWORD)] if GRADIO_USER is not None and GRADIO_PASSWORD is not None else None),
    )

app.mount(
    "/data",
    StaticFiles(directory="/data", follow_symlink=True),
    name="data",
)


@app.post(
    "/api/v1/search",
    response_model=ResponsePathsModel,
)
def post_search(
    text_query: str | None = Form(default=None),
    image_query: bytes | None = File(default=None),
    top_n: int = 5,
):
    """
    Search files with text or/and image query
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
    Index provided files (filenames). Supports JPG, PNG, TXT
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
    Deletes file (by filename) from index and from storage. Actual file will not be deleter, it will be reindexed on app relaunch
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
    """
    Add new files that will be indexed up to 50 MB per one upload
    """
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

    try:
        upload_files(filenames, contents)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error happened while processing uploaded files: {str(exc)}",
        )

    logger.info(f"Finished post /api/v1/files request")
