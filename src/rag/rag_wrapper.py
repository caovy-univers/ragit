from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from loguru import logger

class SimpleRAG:
    """
    Wrapper class for a simple Retrieval-Augmented Generation pipeline using Ollama.
    """
    def __init__(self, vectorstore: FAISS, model_name: str = "mistral"):
        logger.info("Initializing SimpleRAG with Ollama model: {}", model_name)
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.llm = Ollama(model=model_name)
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)

    def ask(self, query: str) -> str:
        """
        Run a question through the RAG pipeline.

        Args:
            query (str): The user's question.

        Returns:
            str: LLM-generated answer.
        """
        logger.info(f"Processing query: {query}")
        return self.qa_chain.run(query)
