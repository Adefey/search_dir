from pydantic import BaseModel, FilePath, Field


class ResponsePathsModel(BaseModel):
    files: list[str] = Field(max_length=512)
