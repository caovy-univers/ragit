import argparse
from pathlib import Path
from rag.document_builder import create_documents
from rag.loader import load_all_json_files
from rag.vector_indexer import build_vectorstore
from rag.wrapper import SimpleRAG


def run_test_queries(rag: SimpleRAG):
    """
    Run predefined test queries against the RAG system.

    Args:
        rag (SimpleRAG): The RAG system instance.
    """
    queries = [
        "Văn minh học thuật của nước Pháp được miêu tả như thế nào trong Nam Phong tạp chí?",
        "Quel est le rôle de l'Académie française selon les articles de la revue Nam Phong ?",
        "How does Nam Phong magazine discuss the conflict between material and spiritual progress in modern civilization?"
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        print("Answer:", rag.ask(q))


def interactive_loop(rag: SimpleRAG):
    """
    Interactive command-line interface for querying the RAG system.

    Args:
        rag (SimpleRAG): The RAG system instance.
    """
    print("Enter your query (type 'q' or 'exit' to leave):")
    while True:
        query = input(">> ").strip()
        if query.lower() in {"q", "exit"}:
            break
        if not query:
            continue
        print("Answer:")
        # print(f"{rag.ask(query)}\n")
        print(f"{rag.ask(query)["result"]}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the RAG pipeline over a JSON corpus."
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data/Nam-Phong/",
        help="Path to the root folder containing JSON files (default: %(default)s)"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run predefined test queries instead of interactive mode."
    )
    args = parser.parse_args()

    # JSON root path, config via --data-dir
    folder_path = Path(args.data_dir)
    if not folder_path.exists():
        raise FileNotFoundError(f"Data directory not found: {folder_path}")

    # Load and prepare documents
    all_entries = load_all_json_files(folder_path)
    all_documents = []
    for chunks, metadata in all_entries:
        all_documents.extend(create_documents(chunks, metadata))

    # Build vector index
    vectorstore = build_vectorstore(all_documents)

    # Initialize RAG system
    rag = SimpleRAG(vectorstore, model_name="mistral")

    # Choose mode
    if args.test:
        run_test_queries(rag)
    else:
        interactive_loop(rag)
