from pydantic import BaseModel, Field, FilePath


class IndexRequestModel(BaseModel):
    files: list[FilePath] = Field(max_length=50)


class ScoredFileModel(BaseModel):
    file: FilePath
    score: float


class ResponsePathsModel(BaseModel):
    files: list[ScoredFileModel] = Field(max_length=2048)
