import ctypes
import io
import logging
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor


def trim_memory():
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


logger = logging.getLogger(__name__)


class Model:

    def __init__(self):
        self.model_checkpoint = os.environ.get("EMBEDDING_MODEL_CHECKPOINT", "openai/clip-vit-base-patch32")
        self.image_resolution = os.environ.get("TARGET_IMAGE_SIZE", "224,224").split(",")
        self.image_resolution = (int(self.image_resolution[0]), int(self.image_resolution[1]))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Start setting up model {self.model_checkpoint} on {self.device}")
        self.model = AutoModel.from_pretrained(self.model_checkpoint, trust_remote_code=True, cache_dir="/model").to(
            self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint, trust_remote_code=True, cache_dir="/model", use_fast=False
        )
        logger.info(f"Finished setting up model {self.model_checkpoint} on {self.device}")

    def _preprocess_image(self, image: bytes):
        image = Image.open(io.BytesIO(image))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        width, height = image.size

        if width == height:
            return image.resize(self.image_resolution)

        if width > height:
            new_width = self.image_resolution[0]
            new_height = int(height * (new_width / width))
        else:
            new_height = self.image_resolution[1]
            new_width = int(width * (new_height / height))

        resized_image = image.resize((new_width, new_height))

        new_image = Image.new("RGB", self.image_resolution, (0, 0, 0))
        paste_x = (self.image_resolution[0] - new_width) // 2
        paste_y = (self.image_resolution[1] - new_height) // 2
        new_image.paste(resized_image, (paste_x, paste_y))
        return new_image

    def _encode(self, inputs: dict) -> list[float]:
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "pixel_values" in inputs:
            features = self.model.get_image_features(**inputs)
        else:
            features = self.model.get_text_features(**inputs)

        # Normalization
        features = F.normalize(features, p=2, dim=1)

        result = features.cpu().detach().numpy().tolist()

        del inputs
        del features
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # ??????????
        trim_memory()

        return result

    def encode_text(self, text: str) -> list[float]:
        """
        Process text into embedding
        """
        logger.info(f"Start encoding text")
        with torch.inference_mode():
            inputs = self.processor(text=[text], padding=True, truncation=True, return_tensors="pt")
            result = self._encode(inputs)[0]
        logger.info(f"Finished encoding text")
        return result

    def encode_image(self, image: bytes) -> list[float]:
        """
        Process image into embedding
        """
        logger.info(f"Start encoding image")
        image = self._preprocess_image(image)
        with torch.inference_mode():
            inputs = self.processor(images=[image], return_tensors="pt")
            result = self._encode(inputs)[0]
        image.close()
        logger.info(f"Finished encoding image")
        return result

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Process texts into embeddings
        """
        logger.info(f"Start encoding texts")
        with torch.inference_mode():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            result = self._encode(inputs)
        logger.info(f"Finished encoding texts")
        return result

    def encode_images(self, images: list[bytes]) -> list[list[float]]:
        """
        Process images into embeddings
        """
        logger.info(f"Start encoding images")
        image_list = [self._preprocess_image(image) for image in images]
        with torch.inference_mode():
            inputs = self.processor(
                images=image_list,
                return_tensors="pt",
            )
            result = self._encode(inputs)
        for image in image_list:
            image.close()
        logger.info(f"Finished encoding images")
        return result
