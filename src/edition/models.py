import datetime
from pydantic import BaseModel, ConfigDict
from typing import List, Optional


class Metadata(BaseModel):
    model_config = ConfigDict(json_encoders={datetime.date: lambda v: v.isoformat()})
    title_main: str
    title_alt: str
    volume: Optional[str] = None
    numero: Optional[str] = None
    publisher: Optional[str] = None
    publication_place: Optional[str] = None
    publication_date: datetime.date
    source_description: Optional[str] = None
    authors: List[str] = []
    genres: List[str] = []
    issn: Optional[str] = None
    identifiers: dict = {}


class Document(BaseModel):
    metadata: Metadata
    text_body: List[str]
