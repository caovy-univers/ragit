from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_ollama import OllamaLLM
from loguru import logger
from typing import Any, Dict

class SimpleRAG:
    """
    A flexible Retrieval-Augmented Generation (RAG) wrapper using OllamaLLM
    and a vectorstore retriever. Includes metadata integration in responses.

    Attributes:
        retriever (BaseRetriever): Configured retriever with specified search type and k-neighbors.
        llm (OllamaLLM): Ollama language model instance.
        qa_chain (RetrievalQA): RAG chain combining the retriever and the LLM.
    """
    def __init__(self,
        vectorstore: Any,
        model_name: str = "mistral",
        k: int = 5,
        chain_type: str = "stuff",
        search_type: str = "similarity"):
        """
        Initialize the SimpleRAG pipeline.

        Args:
            vectorstore (Any): A vectorstore instance (e.g., FAISS, Chroma) supporting `as_retriever()`.
            model_name (str): Name of the Ollama model to use.
            k (int): Number of top documents to retrieve.
            chain_type (str): Method for combining retrieved documents.
            search_type (str): Vector search algorithm.

        Returns:
            None
        """
        logger.info("Initializing SimpleRAG with Ollama model '{}'", model_name)

        # Configure the retriever
        self.retriever: BaseRetriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        # Initialize the Ollama language model
        self.llm = OllamaLLM(model=model_name)

        # Build the RetrievalQA chain with explicit chain_type
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever
        )

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Execute a query through the RAG pipeline, then append source metadata.

        Args:
            query (str): The user's question.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "result": Generated answer with embedded references.
                - "source_documents": List of retrieved document objects.
        """
        logger.info("Processing query: '{}'", query)

        # Invoke the QA chain
        result = self.qa_chain.invoke({"query": query})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        # Build a reference section with metadata
        if sources:
            # Use literal '\n' for newlines
            refs = ["\n**Sources:**"]
            for doc in sources:
                m = doc.metadata
                title = m.get('title_main', 'Unknown title')
                authors = ", ".join(m.get('authors', [])) or 'Unknown authors'
                date = m.get('publication_date', 'Unknown date')
                refs.append(f"- **{title}** by {authors} ({date})")
            answer += "\n" + "\n".join(refs)

        logger.debug(
            "Retrieved %d source documents for the query",
            len(sources)
        )
        return {"result": answer, "source_documents": sources}
