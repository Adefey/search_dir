from pydantic import BaseModel, Field


class RequestTextModel(BaseModel):
    text: str = Field(max_length=2048)


class RequestTextsModel(BaseModel):
    texts: list[str] = Field(max_length=256)


class ResponseEmbeddingModel(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)


class ResponseEmbeddingsModel(BaseModel):
    embeddings: list[list[float]] = Field(max_length=256)
