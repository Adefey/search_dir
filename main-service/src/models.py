from pydantic import BaseModel, Field, FilePath


class IndexRequestModel(BaseModel):
    files: list[FilePath] = Field(max_length=50)


class ScoredFileModel(BaseModel):
    file: FilePath
    score: float = Field(ge=-1, le=1)


class ResponsePathsModel(BaseModel):
    files: list[ScoredFileModel] = Field(max_length=2048)
