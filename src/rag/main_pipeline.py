from pathlib import Path
from loader import load_all_json_files
from document_builder import create_documents
from vector_indexer import build_vectorstore
from rag_wrapper import SimpleRAG

if __name__ == "__main__":
    folder_path = Path("data/Nam-Phong/Quyen-1/So-1/output_json")

    # Load multiple files
    all_entries = load_all_json_files(folder_path)
    all_documents = []
    for chunks, metadata in all_entries:
        all_documents.extend(create_documents(chunks, metadata))

    # Build vector index
    vectorstore = build_vectorstore(all_documents)

    # Initialize RAG system with Ollama Mistral
    rag = SimpleRAG(vectorstore, model_name="mistral")

    # Test multilingual queries
    queries = [
        "Văn minh học thuật của nước Pháp được miêu tả như thế nào trong Nam Phong tạp chí?",
        # == "Nam Phong describes French scholarly civilization as an expression of deep humanism, centered on the human being, clearly reflected in literature, philosophy, religion, and history. French literature is praised for being simple, clear, and highly educational."
        "Quel est le rôle de l'Académie française selon les articles de la revue Nam Phong ?",
        "How does Nam Phong magazine discuss the conflict between material and spiritual progress in modern civilization?"
    ]

    for q in queries:
        print(f"\n🔎 Query: {q}")
        print("Answer:", rag.ask(q))
