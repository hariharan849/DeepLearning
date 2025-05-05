from pydantic import BaseModel
from typing import List

class ArxivPaper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    published_date: str
    url: List[str]
    chunks: List[str]