from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import logging
import io

logger = logging.getLogger(__name__)


class Model:

    def __init__(self):
        self.model_checkpoint = "openai/clip-vit-base-patch32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Start setting up model {self.model_checkpoint} on {self.device}")
        self.model = CLIPModel.from_pretrained(self.model_checkpoint).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_checkpoint, use_fast=True)
        logger.info(f"Finished setting up model {self.model_checkpoint} on {self.device}")

    def encode_text(self, text: str) -> list[float]:
        """
        Process text into embedding
        """
        logger.info(f"Start encoding text")
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs).to(self.device)
        result = text_features.numpy().tolist()[0]
        logger.info(f"Finished encoding text")
        return result

    def encode_image(self, image: bytes) -> list[float]:
        """
        Process image into embedding
        """
        logger.info(f"Start encoding image")
        image = Image.open(io.BytesIO(image))
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs).to(self.device)
        result = image_features.numpy().tolist()[0]
        logger.info(f"Finished encoding image")
        return result
