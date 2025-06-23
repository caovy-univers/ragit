from langchain.schema import Document
from pydantic import BaseModel
from typing import List, Optional

class Metadata(BaseModel):
    title_main: str
    title_alt: str = ""
    publisher: Optional[str] = None
    publication_place: Optional[str] = None
    publication_date: Optional[str] = None
    source_description: Optional[str] = None
    authors: list = []
    genres: list = []
    issn: Optional[str] = None
    identifiers: dict = {}

def create_documents(chunks: List[str], metadata: dict) -> List[Document]:
    """
    Convert raw text chunks and metadata into LangChain Document objects.

    Args:
        chunks (List[str]): List of text segments.
        metadata (dict): Metadata to attach to each document.

    Returns:
        List[Document]: List of LangChain-compatible document objects.
    """
    parsed_metadata = Metadata(**metadata).dict()
    return [Document(page_content=chunk, metadata=parsed_metadata) for chunk in chunks]
