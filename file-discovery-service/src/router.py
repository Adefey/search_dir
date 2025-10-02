import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from threading import Thread

from fastapi import FastAPI
from redis import Redis
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(filename=f'logs/file_discovery_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.log'),
        logging.StreamHandler(stream=sys.stdout),
    ],
)

QUEUE_NAME = os.environ.get("QUEUE_NAME", "queue")
METADATA_HASH_KEY = os.environ.get("METADATA_HASH_KEY", "file_metadata")
MONITOR_PATH = os.environ.get("MONITOR_PATH", "/data")
ACTION_CREATE_ID = int(os.environ.get("ACTION_CREATE_ID", "1"))
ACTION_UPDATE_ID = int(os.environ.get("ACTION_UPDATE_ID", "2"))
ACTION_DELETE_ID = int(os.environ.get("ACTION_DELETE_ID", "3"))

logger = logging.getLogger(__name__)
redis = Redis(host="redis", port=6379, decode_responses=True)


def queue_add_wrapper(file_path: str, action: int):
    """
    Check file and add to queue
    """
    try:
        file_mtime = int(os.path.getmtime(file_path))
    except FileNotFoundError:
        logger.warning(f"[PRODUCER] File {file_path} not found during processing, skipping")
        return

    last_known_mtime_str = redis.hget(METADATA_HASH_KEY, file_path)

    if last_known_mtime_str is not None and file_mtime == int(last_known_mtime_str):
        logger.info(f"[PRODUCER] File {file_path} is unchanged. Skipping")
        return

    redis.hset(METADATA_HASH_KEY, file_path, file_mtime)
    redis.lpush(QUEUE_NAME, f"{file_path},{action}")
    logger.info(f"[PRODUCER] File {file_path} was added, action: {action}")


def queue_deletion_wrapper(file_path: str):
    """
    Send delete task and reset mtime
    """
    logger.info(f"[PRODUCER] Deleting file {file_path}")
    redis.hdel(METADATA_HASH_KEY, file_path)
    redis.lpush(QUEUE_NAME, f"{file_path},{ACTION_DELETE_ID}")
    logger.info(f"[PRODUCER] Delete task added {file_path}")


class FileChangeHandler(FileSystemEventHandler):
    """
    Observer to check file updates in real time
    """

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"[WATCHER] Found file {event.src_path}")
            queue_add_wrapper(event.src_path, ACTION_CREATE_ID)

    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"[WATCHER] Found file change {event.src_path}")
            queue_add_wrapper(event.src_path, ACTION_UPDATE_ID)

    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"[WATCHER] Found file deleted {event.src_path}")
            queue_deletion_wrapper(event.src_path)


def producer():
    """
    Endless function that runs in background, exists if observer got error
    """
    logger.info(f"[PRODUCER] sends all files to indexation")
    for root, _, fnames in os.walk(MONITOR_PATH):
        for fname in fnames:
            full_filename = os.path.join(root, fname)
            logger.info(f"[PRODUCER] Found file {full_filename}")
            queue_add_wrapper(full_filename, ACTION_CREATE_ID)
    logger.info(f"[PRODUCER] Starting watchdog")
    observer = Observer()
    observer.schedule(FileChangeHandler(), MONITOR_PATH, recursive=True)
    observer.start()
    logger.info(f"[PRODUCER] Started watchdog")
    observer.join()
    logger.info(f"[PRODUCER] Exited (Probably watchdog got error)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run producer in background
    """
    Thread(target=producer, daemon=True).start()

    yield


app = FastAPI(
    title="File Discovery Microservise",
    description="Microservice used to scan directory and send file updates to main service",
    version=os.environ.get("APP_VERSION", "0.1"),
    lifespan=lifespan,
)
