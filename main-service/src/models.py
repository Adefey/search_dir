from pydantic import BaseModel, Field, FilePath


class IndexRequestModel(BaseModel):
    files: list[FilePath] = Field(max_length=50)


class ResponsePathsModel(BaseModel):
    files: list[FilePath] = Field(max_length=512)
