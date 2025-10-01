from pydantic import BaseModel, FilePath


class FilePathModel(BaseModel):
    file: FilePath


class IndexRequestModel(BaseModel):
    files: list[FilePath]


class ScoredFileModel(BaseModel):
    file: FilePath
    score: float


class ResponsePathsModel(BaseModel):
    files: list[ScoredFileModel]
