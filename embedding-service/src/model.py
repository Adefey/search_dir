import io
import logging
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class Model:

    def __init__(self):
        """
        Prepare the AutoModel and AutoProcessor
        """
        self.model_checkpoint = os.environ.get("EMBEDDING_MODEL_CHECKPOINT", "openai/clip-vit-base-patch32")
        self.image_resolution = os.environ.get("TARGET_IMAGE_SIZE", "224,224").split(",")
        self.image_resolution = (
            int(self.image_resolution[0]),
            int(self.image_resolution[1]),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Start setting up model {self.model_checkpoint} on {self.device}")
        self.model = AutoModel.from_pretrained(self.model_checkpoint, trust_remote_code=True, cache_dir="/model").to(
            self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint,
            trust_remote_code=True,
            cache_dir="/model",
            use_fast=False,
        )
        logger.info(f"Finished setting up model {self.model_checkpoint} on {self.device}")

    def _preprocess_image(self, image: bytes):
        """
        Ensure that image is RGB and correctly transposed, resize to target model input size
        """
        image = Image.open(io.BytesIO(image))
        ImageOps.exif_transpose(image, in_place=True)
        image = image.convert("RGB")

        new_image = Image.new("RGB", self.image_resolution, (0, 0, 0))

        image.thumbnail(self.image_resolution, Image.Resampling.LANCZOS)
        paste_x = (self.image_resolution[0] - image.width) // 2
        paste_y = (self.image_resolution[1] - image.height) // 2

        new_image.paste(image, (paste_x, paste_y))

        image.close()

        return new_image

    def _encode(self, inputs: dict) -> list[float]:
        """
        Encode text or image based on inputs dict, normalize result
        """
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

        return result

    def encode_text(self, text: str) -> list[float]:
        """
        Process text into embedding
        """
        logger.info(f"Start encoding text")
        with torch.inference_mode():
            inputs = self.processor(
                text=[text],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
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

    def encode_text_files(self, text_files: list[tuple[str, str]]) -> list[list[float]]:
        """
        Process list of tuples (filename, text) into tuples (filename, embedding)
        """
        logger.info(f"Start encoding texts")
        if not text_files:
            logger.info(f"No texts, return empty list")
            return []

        filenames, texts = list(zip(*text_files))

        with torch.inference_mode():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            result = self._encode(inputs)

        logger.info(f"Finished encoding texts")

        return zip(filenames, result)

    def encode_image_files(self, image_files: list[tuple[str, bytes]]) -> list[list[float]]:
        """
        Process list of tuples (filename, raw_image_bytes) into tuples (filename, embedding)
        """
        logger.info(f"Start encoding images")

        if not image_files:
            logger.info(f"No images, return empty list")
            return []

        filenames, images = list(zip(*image_files))

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

        return zip(filenames, result)
