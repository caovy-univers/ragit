import time
from typing import Any, Dict, List, Sequence, Tuple
from loguru import logger
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
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


def _build_context(docs: Sequence[Document], max_chars: int = 4000) -> str:
    """
    Build a concatenated context from retrieved documents, capped by length.

    Args:
        docs (Sequence[Document]): Retrieved documents.
        max_chars (int): Safeguard for oversized contexts (smaller = faster).

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

    def __init__(self, vectorstore: FAISS, *, llm, model_name: str, target_lang: str = "fr", search_k: int = 3) -> None:
        """
        Args:
            vectorstore (FAISS): Vector store instance.
            llm: Language model wrapper (Ollama or HF pipeline).
            model_name (str): Kept for logging/debug.
            target_lang (str): Target answer language (e.g., 'fr', 'vi', 'en').
            search_k (int): Number of docs to retrieve (smaller = faster).
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
        t0 = time.time()
        logger.info(f"Processing query: '{query}'")

        # Retrieve
        t_retr_0 = time.time()
        sources: List[Document] = self.retriever.invoke(query)
        t_retr_1 = time.time()
        logger.debug(f"[Timing] retrieval={t_retr_1 - t_retr_0:.2f}s; docs={len(sources)}")

        # Build context
        t_ctx_0 = time.time()
        context = _build_context(sources)
        t_ctx_1 = time.time()
        logger.debug(f"[Timing] build_context={t_ctx_1 - t_ctx_0:.2f}s; context_len={len(context)} chars")

        # Prompt
        final_prompt = self.prompt.format(
            TARGET_LANG=self.target_lang,
            QUESTION=query,
            CONTEXT=context,
        )

        # Generate
        t_gen_0 = time.time()
        logger.info(f"[Generation] calling model='{self.model_name}' (greedy, max_new_tokens=256)")
        raw = self.llm.invoke(final_prompt)
        t_gen_1 = time.time()
        logger.debug(f"[Timing] generation={t_gen_1 - t_gen_0:.2f}s (model={self.model_name})")

        answer = raw if isinstance(raw, str) else str(raw)

        # Sources
        src_text = _format_sources(sources)
        if src_text:
            answer += "\n" + src_text

        logger.debug(f"[Timing] total={time.time() - t0:.2f}s (end-to-end)")
        logger.debug(f"Retrieved {len(sources)} source documents for the query")
        return {"result": answer, "source_documents": sources}
