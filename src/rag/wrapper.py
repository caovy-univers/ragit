from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS  # or your actual type
from langchain_community.llms import HuggingFacePipeline
from langchain_ollama import OllamaLLM


DEFAULT_PROMPT = """You are a helpful RAG assistant.
Use the provided context to answer the question concisely and faithfully.
ALWAYS answer in the target language: {TARGET_LANG}.

# Rules
- If the context is insufficient, say so briefly.
- Cite short source info the system will append; do not fabricate references.

# Question
{QUESTION}

# Context
{CONTEXT}
"""


def _build_context(docs: Sequence[Document], max_chars: int = 8000) -> str:
    """
    Build a concatenated context from retrieved documents, capped by length.

    Args:
        docs (Sequence[Document]): Retrieved documents.
        max_chars (int): Safeguard for oversized contexts.

    Returns:
        str: Concatenated context payload.
    """
    parts: List[str] = []
    size = 0
    for d in docs:
        text = d.page_content.strip()
        if not text:
            continue
        if size + len(text) > max_chars:
            remaining = max_chars - size
            if remaining <= 0:
                break
            text = text[:remaining]
        parts.append(text)
        size += len(text)
        if size >= max_chars:
            break
    return "\n\n---\n\n".join(parts)


def _unique_sources(docs: Sequence[Document], limit: int = 5) -> List[Tuple[str, str, str]]:
    """
    Deduplicate and compress document metadata into (title, authors, date).

    Dedup key:
        (title_main, publication_date)

    Args:
        docs (Sequence[Document]): Retrieved documents.
        limit (int): Max number of unique entries to keep.

    Returns:
        List[Tuple[str, str, str]]: List of (title, authors, date).
    """
    seen = set()
    uniq: List[Tuple[str, str, str]] = []
    for d in docs:
        m = d.metadata or {}
        title = m.get("title_main") or "Unknown title"
        date = m.get("publication_date") or "Unknown date"
        key = (title, date)
        if key in seen:
            continue
        seen.add(key)
        authors_list = m.get("authors") or []
        if isinstance(authors_list, list):
            authors = ", ".join(a for a in authors_list if a) or "Unknown authors"
        else:
            authors = str(authors_list) if authors_list else "Unknown authors"
        uniq.append((title, authors, date))
        if len(uniq) >= limit:
            break
    return uniq


def _format_sources(docs: Sequence[Document]) -> str:
    """
    Produce a clean, non-duplicated "Sources:" section without markdown bold.

    Example:
        Sources:
        - Nam Phong tạp chí — Phạm Quỳnh, Nguyễn Bá Trác (1917-07-01)
        - Những sự gặp gỡ... — Vũ Bội Liêu (1944)
    """
    items = _unique_sources(docs)
    if not items:
        return ""
    lines = ["\nSources:"]
    for title, authors, date in items:
        lines.append(f"- {title} — {authors} ({date})")
    return "\n".join(lines)


class SimpleRAG:
    """
    Minimal RAG wrapper:
      - Builds a retriever from a vector store
      - Assembles a prompt with retrieved context
      - Calls the provided LLM backend (Ollama or HF pipeline)
      - Appends a clean, deduplicated sources section
    """

    def __init__(
        self,
        vectorstore: FAISS,
        *,
        llm: OllamaLLM | HuggingFacePipeline,
        model_name: str,
        target_lang: str = "fr",
        search_k: int = 5,
    ) -> None:
        """
        Args:
            vectorstore (FAISS): Vector store instance.
            llm (OllamaLLM | HuggingFacePipeline): Language model wrapper.
            model_name (str): Kept for logging/debug.
            target_lang (str): Target answer language (e.g., 'fr', 'vi', 'en').
            search_k (int): Number of documents to retrieve.
        """
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
        self.llm = llm
        self.model_name = model_name
        self.target_lang = target_lang
        logger.info(f"Initializing SimpleRAG (target_lang='{self.target_lang}')")

        self.prompt = PromptTemplate(
            input_variables=["TARGET_LANG", "QUESTION", "CONTEXT"],
            template=DEFAULT_PROMPT,
        )

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Retrieve, generate, and format a response with sources.

        Args:
            query (str): User question.

        Returns:
            Dict[str, Any]: {"result": str, "source_documents": List[Document]}
        """
        logger.info(f"Processing query: '{query}'")
        # Retrieve candidates
        sources: List[Document] = self.retriever.invoke(query)

        # Build context
        context = _build_context(sources)

        # Build prompt
        final_prompt = self.prompt.format(
            TARGET_LANG=self.target_lang,
            QUESTION=query,
            CONTEXT=context,
        )

        # Generate
        raw = self.llm.invoke(final_prompt)
        answer = raw if isinstance(raw, str) else str(raw)

        # Append clean sources (no bold, no duplicates)
        src_text = _format_sources(sources)
        if src_text:
            answer += "\n" + src_text

        logger.debug(f"Retrieved {len(sources)} source documents for the query")
        return {"result": answer, "source_documents": sources}
