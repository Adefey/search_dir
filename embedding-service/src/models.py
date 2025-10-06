import os

from pydantic import BaseModel, Field

EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", "512"))


class ResponseEmbeddingOnlyModel(BaseModel):
    embedding: list[float] = Field(
        min_length=EMBEDDING_SIZE,
        max_length=EMBEDDING_SIZE,
    )


class ResponseFileEmbeddingModel(BaseModel):
    file_path: str
    embedding: list[float] = Field(
        min_length=EMBEDDING_SIZE,
        max_length=EMBEDDING_SIZE,
    )


class ResponseFileEmbeddingsModel(BaseModel):
    file_records: list[ResponseFileEmbeddingModel]
    unprocessed_files: list[str] = Field(default=[])
