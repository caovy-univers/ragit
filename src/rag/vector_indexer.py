from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
from loguru import logger

def build_vectorstore(documents: List[Document]) -> FAISS:
    """
    Generate multilingual embeddings and build a FAISS vector store.

    Args:
        documents (List[Document]): Input documents.

    Returns:
        FAISS: LangChain-compatible FAISS index.
    """
    logger.info("Building vector store with multilingual sentence-transformers embeddings")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.from_documents(documents, embedding_model)
