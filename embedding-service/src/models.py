from pydantic import BaseModel, Field


class RequestTextModel(BaseModel):
    # 77 tokens
    text: str = Field(max_length=500)


class RequestTextsModel(BaseModel):
    # Estimate 200MB per text = 10GB max
    texts: list[str] = Field(max_length=50)


class ResponseEmbeddingModel(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)


class ResponseEmbeddingsModel(BaseModel):
    # Estimate 200MB per text = 10GB max
    embeddings: list[list[float]] = Field(max_length=50)
