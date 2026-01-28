from pydantic import BaseModel
from typing import List

class QueryModel(BaseModel):
    query: str

class ResponseModel(BaseModel):
    content: str
    questions: List[str]