import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from threading import Thread
from time import sleep, time

from fastapi import BackgroundTasks, FastAPI
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

logger = logging.getLogger(__name__)
redis = Redis(host="redis", port=6379, decode_responses=True)

QUEUE_NAME = os.environ.get("QUEUE_NAME", "queue")
FILE_RELEVANT_SECONDS = int(os.environ.get("FILE_RELEVANT_SECONDS", 21600))
METADATA_HASH_KEY = "file_metadata"
MONITOR_PATH = "/data"


def lpush_wrapper(file_path: str):
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
    redis.lpush(QUEUE_NAME, file_path)


class FileChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"[WATCHER] New file detected: {event.src_path}")
            lpush_wrapper(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"[WATCHER] File modified: {event.src_path}")
            lpush_wrapper(event.src_path)


def producer():
    logger.info(f"[PRODUCER] sends all files to indexation")
    for root, _, fnames in os.walk(MONITOR_PATH):
        for fname in fnames:
            full_filename = os.path.join(root, fname)
            logger.info(f"[PRODUCER] found file {full_filename}")
            lpush_wrapper(full_filename)
    logger.info(f"[PRODUCER] starting watchdog")
    observer = Observer()
    observer.schedule(FileChangeHandler(), MONITOR_PATH, recursive=True)
    observer.start()
    logger.info(f"[PRODUCER] started watchdog")
    observer.join()
    logger.info(f"[PRODUCER] exited")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Thread(target=producer, daemon=True).start()

    yield


app = FastAPI(title="File Discovery Microservise", lifespan=lifespan)
