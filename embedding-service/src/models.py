from pydantic import BaseModel, Field


class RequestTextModel(BaseModel):
    text: str = Field(max_length=2048)


class ResponseEmbeddingModel(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)
