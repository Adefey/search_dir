from fastapi import FastAPI, BackgroundTasks
import logging
import sys
from datetime import datetime
import requests
import os
from redis import Redis
from time import sleep

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            filename=f'logs/file_discovery_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.log'
        ),
        logging.StreamHandler(stream=sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
app = FastAPI(title="File Discovery Microservise")
redis = Redis(host="redis", port=6379, decode_responses=True)

MAIN_SERVICE_URL = os.environ.get("MAIN_SERVICE_ADDRESS", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 30))


def producer():
    for root, _, fnames in os.walk("/data"):
        for fname in fnames:
            full_filename = os.path.join(root, fname)
            logger.info(f"[PRODUCER] found file {full_filename}")
            redis.rpush("queue", full_filename)


def consumer():
    files = []

    while True:
        filename = redis.rpop("queue")

        logger.info(f"[CONSUMER] retrieved file {filename}")

        if filename is not None:
            files.append(filename)

        if len(files) == BATCH_SIZE or filename is None:
            logger.info(
                f"Flound {len(files)} files, sending http://{MAIN_SERVICE_URL}/api/v1/index"
            )
            resp = requests.post(
                f"http://{MAIN_SERVICE_URL}/api/v1/index",
                json={"files": files},
            )

            if resp.status_code != 200:
                logger.error(
                    f"http://{MAIN_SERVICE_URL}/api/v1/index returned status {resp.status_code}"
                )
            else:
                files = []

            if filename is None:
                logger.info("No files were retrieved from queue, sleeping 30 seconds")
                sleep(30)


def walk_filesystem_and_send_to_index():
    producer()
    consumer()


@app.post("/api/v1/start_discovery")
def post_start_discovery(background_tasks: BackgroundTasks):
    background_tasks.add_task(walk_filesystem_and_send_to_index)
