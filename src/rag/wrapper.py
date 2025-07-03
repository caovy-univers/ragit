from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_ollama import OllamaLLM
from loguru import logger

class SimpleRAG:
    """
    Wrapper class for Retrieval-Augmented Generation (RAG) using Ollama and a retriever.
    """

    def __init__(self, retriever: BaseRetriever, model_name: str = "mistral"):
        """
        Initialize the RAG wrapper with a retriever and Ollama model.

        Args:
            retriever (BaseRetriever): Vector store retriever (e.g., FAISS retriever).
            model_name (str): Name of the Ollama model (default: "mistral").
        """
        logger.info(f"Initializing SimpleRAG with Ollama model: {model_name}")
        self.llm = OllamaLLM(model=model_name)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever.as_retriever()
        )

    def ask(self, query: str) -> dict:
        """
        Run a query through the RAG pipeline and return the result.

        Args:
            query (str): User query string.

        Returns:
            dict: The result dictionary containing the generated answer and source docs.
        """
        return self.qa_chain.invoke({"query": query})
