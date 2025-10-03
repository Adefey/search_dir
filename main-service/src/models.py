from pydantic import BaseModel, FilePath


class FilePathsModel(BaseModel):
    files: list[FilePath]


class IndexRequestModel(BaseModel):
    files: list[FilePath]


class ScoredFileModel(BaseModel):
    file: FilePath
    score: float


class ResponsePathsModel(BaseModel):
    files: list[ScoredFileModel]
