import os

from pydantic import BaseModel, Field

EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", "512"))


class RequestTextModel(BaseModel):
    text: str = Field(max_length=2048)


class RequestTextsModel(BaseModel):
    # Estimate 200MB per text = 10GB max
    texts: list[str] = Field(min_length=1, max_length=50)


class ResponseEmbeddingModel(BaseModel):
    embedding: list[float] = Field(min_length=EMBEDDING_SIZE, max_length=EMBEDDING_SIZE)


class ResponseEmbeddingsModel(BaseModel):
    # Estimate 200MB per text = 10GB max
    embeddings: list[list[float]] = Field(max_length=50)
