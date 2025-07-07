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
    Convert raw text chunks and metadata into Document objects,
    embedding metadata into the content header so it's both searchable
    and available to the LLM during generation.

    Args:
        chunks (List[str]): List of text segments to be indexed.
        metadata (dict): Metadata dictionary containing fields like
            title_main, authors, publication_date, publisher, genres, etc.

    Returns:
        List[Document]: A list of LangChain Document objects with metadata
            prefixed in the page_content and metadata attached.
    """
    parsed = Metadata(**metadata).dict()
    docs: List[Document] = []

    # Build metadata header
    header_lines = [f"Title: {parsed['title_main']}"]
    if parsed.get('authors'):
        header_lines.append(f"Authors: {', '.join(parsed['authors'])}")
    if parsed.get('publication_date'):
        header_lines.append(f"Publication Date: {parsed['publication_date']}")
    if parsed.get('publisher'):
        header_lines.append(f"Publisher: {parsed['publisher']}")
    if parsed.get('genres'):
        header_lines.append(f"Genres: {', '.join(parsed['genres'])}")
    header = "\n".join(header_lines) + "\n\n"

    # Prefix each chunk with header
    for chunk in chunks:
        docs.append(
            Document(
                page_content=header + chunk,
                metadata=parsed
            )
        )
    return docs
