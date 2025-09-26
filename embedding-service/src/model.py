import ctypes
import io
import logging

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def trim_memory():
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


logger = logging.getLogger(__name__)


class Model:

    def __init__(self):
        self.model_checkpoint = "openai/clip-vit-base-patch32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Start setting up model {self.model_checkpoint} on {self.device}")
        self.model = CLIPModel.from_pretrained(self.model_checkpoint).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_checkpoint, use_fast=False)
        logger.info(f"Finished setting up model {self.model_checkpoint} on {self.device}")

    def _encode(self, inputs: dict) -> list[float]:
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "pixel_values" in inputs:
            features = self.model.get_image_features(**inputs)
        else:
            features = self.model.get_text_features(**inputs)

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
            inputs = self.processor(text=text, return_tensors="pt")
            result = self._encode(inputs)[0]
        logger.info(f"Finished encoding text")
        return result

    def encode_image(self, image: bytes) -> list[float]:
        """
        Process image into embedding
        """
        logger.info(f"Start encoding image")
        image = Image.open(io.BytesIO(image))
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt")
            result = self._encode(inputs)[0]
        image.close()
        del image
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
        image_list = [Image.open(io.BytesIO(image)) for image in images]
        with torch.inference_mode():
            inputs = self.processor(
                images=image_list,
                return_tensors="pt",
                padding=True,
            )
            result = self._encode(inputs)
        for image in image_list:
            image.close()
        logger.info(f"Finished encoding images")
        return result
