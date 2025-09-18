from typing import Any, Dict, List, Optional

from langchain.schema import Document
from pydantic import BaseModel, Field, ConfigDict


class Metadata(BaseModel):
    """
    Normalized metadata schema for JSON inputs.

    Notes:
    - Many fields may be missing or set to null in source JSON. We therefore
      declare them Optional and provide safe defaults where appropriate.
    - We also allow extra fields to pass through (extra="allow") so we keep
      useful keys like authors_id/publisher_id when present.
    - Use Field(default_factory=...) for list/dict to avoid mutable defaults.
    """
    model_config = ConfigDict(extra="allow")

    title_main: str = "Unknown title"
    title_alt: Optional[str] = None
    publisher: Optional[str] = None
    publication_place: Optional[str] = None
    publication_date: Optional[str] = None
    source_description: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    genres: List[str] = Field(default_factory=list)
    issn: Optional[str] = None
    identifiers: Dict[str, Any] = Field(default_factory=dict)


def _safe_join(values: List[Any]) -> str:
    """
    Join list items into a readable string:
    - Coerces non-string values to strings.
    - Skips falsy values.
    """
    return ", ".join(str(v).strip() for v in values if v)


def create_documents(chunks: List[str], metadata: Dict[str, Any]) -> List[Document]:
    """
    Convert raw text chunks and metadata into LangChain Document objects.

    Strategy:
    - Validate metadata via Pydantic (tolerant of null/missing fields).
    - Build a short header (inline provenance) prepended to each chunk so that
      the LLM can cite and the content remains searchable.
    - Return a list of Document with the validated metadata preserved.

    Args:
        chunks (List[str]): Flat list of text lines or paragraphs.
        metadata (dict): Raw metadata dictionary from JSON.

    Returns:
        List[Document]: One Document per chunk.
    """
    parsed = Metadata(**(metadata or {})).model_dump()

    header_lines: List[str] = []
    # Title(s)
    header_lines.append(f"Title: {parsed.get('title_main') or 'Unknown title'}")
    if parsed.get("title_alt"):
        header_lines.append(f"Alt title: {parsed['title_alt']}")

    # Authors
    if parsed.get("authors"):
        header_lines.append(f"Authors: {_safe_join(parsed['authors'])}")

    # Publication info
    pub_date = parsed.get("publication_date")
    pub_place = parsed.get("publication_place")
    publisher = parsed.get("publisher")
    if pub_date or pub_place or publisher:
        parts = []
        if pub_date:
            parts.append(pub_date)
        if pub_place:
            parts.append(pub_place)
        if publisher:
            parts.append(publisher)
        header_lines.append("Publication: " + " — ".join(parts))

    # Genres (optional)
    if parsed.get("genres"):
        header_lines.append(f"Genres: {_safe_join(parsed['genres'])}")

    header = "\n".join(header_lines) + "\n\n"

    docs: List[Document] = []
    for chunk in chunks:
        if not chunk:
            continue
        docs.append(
            Document(
                page_content=header + chunk,
                metadata=parsed,
            )
        )
    return docs
