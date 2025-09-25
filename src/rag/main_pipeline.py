import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from rag.document_builder import create_documents
from rag.loader import load_all_json_files, load_all_json_files_multi
from rag.vector_indexer import build_vectorstore
from rag.wrapper import SimpleRAG
from rag.llm_factory import build_llm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def load_test_queries_from_dir(tests_dir: Path, scope: str) -> Optional[List[str]]:
    """
    Load test queries from a YAML/JSON file located in `tests_dir`.

    Expected filenames (by scope):
      - all.yaml / all.json
      - nam.yaml / nam.json
      - tan.yaml / tan.json

    YAML format:
      queries:
        - "question 1"
        - "question 2"
        - "question 3"

    JSON format:
      { "queries": ["q1", "q2", "q3"] }

    Returns:
        Optional[List[str]]: The list of queries, or None if file not found.
    """
    import yaml  # lightweight; required by Poetry anyway

    candidates = [tests_dir / f"{scope}.yaml", tests_dir / f"{scope}.json"]
    for file in candidates:
        if not file.exists():
            continue
        if file.suffix == ".yaml":
            with file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
        qs = data.get("queries") or []
        # Normalize to non-empty strings
        queries = [str(q).strip() for q in qs if str(q).strip()]
        return queries if queries else None
    return None


def default_test_queries(scope: str) -> List[str]:
    """
    Built-in fallback test queries in case tests files are missing.
    """
    if scope == "nam":
        return [
            "Nam Phong tạp chí bàn về 'triết-học là gì' như thế nào? Tóm tắt các luận điểm chính.",
            "Dans Nam Phong, quel rôle attribue-t-on à la langue nationale pour le progrès de l'instruction ?",
            "Summarize how Nam Phong discusses the balance between tradition and modern scientific thought.",
        ]
    if scope == "tan":
        return [
            "Trong các văn bản Tân Việt, những đặc điểm nổi bật của phê bình văn học là gì?",
            "Quels auteurs ou œuvres étrangers sont présentés dans les extraits Tân Việt et dans quel contexte ?",
            "From Tân Việt excerpts, what are the main themes about theatre and long-form fiction?",
        ]
    # 'all'
    return [
        "So sánh cách bàn luận về văn học giữa Nam Phong và Tân Việt (giống và khác).",
        "Quelles références européennes apparaissent dans ces corpus et comment sont-elles réinterprétées localement ?",
        "Across both subcorpora, identify recurring topics about education, language, and modernization.",
    ]


def run_test_queries(rag: SimpleRAG, scope: str, tests_dir: Optional[Path]) -> None:
    """
    Run three predefined test queries depending on the selected scope.

    Args:
        rag (SimpleRAG): The RAG system instance.
        scope (str): 'all' | 'nam' | 'tan'
        tests_dir (Optional[Path]): Directory containing YAML/JSON test files.
    """
    queries = None
    if tests_dir:
        queries = load_test_queries_from_dir(tests_dir, scope)
    if not queries:
        queries = default_test_queries(scope)

    for q in queries:
        print(f"\nQuery: {q}")
        out = rag.ask(q)
        print("Answer:", out["result"])


def interactive_loop(rag: SimpleRAG) -> None:
    """
    Simple interactive loop to query the RAG system from the console.
    """
    print("Enter your query (type 'q' or 'exit' to leave):")
    while True:
        query = input(">> ").strip()
        if query.lower() in {"q", "exit"}:
            break
        if not query:
            continue
        out = rag.ask(query)
        print("Answer:\n", out["result"], "\n")


def resolve_corpus_dirs(corpus: str) -> List[Path]:
    """
    Map a logical corpus choice to concrete directories on disk.

    Args:
        corpus (str): One of {'all', 'nam', 'tan'}.

    Returns:
        List[Path]: Directories to scan for JSON files.
    """
    base = {
        "all": [Path("data/Tan-Viet/output"), Path("data/Nam-Phong/output")],
        "nam": [Path("data/Nam-Phong/output")],
        "tan": [Path("data/Tan-Viet/output")],
    }
    return base[corpus]


def make_rag(
    data_dirs: List[Path],
    backend: str,
    model_name: str,
    target_lang: str,
    use_4bit: bool,
) -> SimpleRAG:
    """
    Build the vector index and initialize a SimpleRAG instance.

    Args:
        data_dirs (List[Path]): One or more directories with JSON files.
        backend (str): 'ollama' or 'phogpt'.
        model_name (str): Model name or HF repo id.
        target_lang (str): Output language (e.g., 'fr', 'vi', 'en').
        use_4bit (bool): Whether to load PhoGPT in 4-bit quantization.

    Returns:
        SimpleRAG: A fully configured RAG wrapper ready to answer.
    """
    # Load and convert to LangChain Documents
    if len(data_dirs) == 1:
        entries = load_all_json_files(data_dirs[0])
    else:
        entries = load_all_json_files_multi(data_dirs)

    all_documents = []
    for chunks, metadata in entries:
        all_documents.extend(create_documents(chunks, metadata))

    # Build vector store (multilingual embeddings)
    vectorstore = build_vectorstore(all_documents)

    # Select LLM backend
    if backend == "ollama":
        llm = build_llm("ollama", model_name=model_name)
    elif backend == "phogpt":
        llm = build_llm(
            "phogpt",
            model_name=model_name,
            use_4bit=use_4bit,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Assemble the RAG pipeline
    return SimpleRAG(
        vectorstore,
        llm=llm,
        model_name=model_name,
        target_lang=target_lang,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline over the JSON corpus.")
    parser.add_argument(
        "--corpus",
        "-c",
        choices=["all", "nam", "tan", "custom"],
        default="all",
        help="Select subcorpus to load: 'all' (default), 'nam', 'tan', or 'custom' (use one or more -d).",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        action="append",
        help="Custom data directory (repeatable). When provided, --corpus must be 'custom'.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=["ollama", "phogpt"],
        default="ollama",
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        default="mistral",
        help="Model name (e.g. 'mistral' for Ollama, or 'vinai/PhoGPT-7B5' for PhoGPT).",
    )
    parser.add_argument(
        "--answer-lang",
        "-L",
        default="fr",
        help="Target answer language (e.g. 'fr', 'vi', 'en').",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit loading for PhoGPT (requires bitsandbytes + supported GPU).",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run three predefined test queries instead of interactive mode.",
    )
    parser.add_argument(
        "--test-scope",
        choices=["all", "nam", "tan"],
        help="Override which set of 3 test queries to run. Defaults to the chosen --corpus (maps 'custom' -> 'all').",
    )
    parser.add_argument(
        "--tests-dir",
        type=str,
        default="src/rag/tests",
        help="Directory where test queries are stored (YAML/JSON). Default: src/rag/tests",
    )
    args = parser.parse_args()

    # Resolve directories to scan
    if args.corpus == "custom":
        if not args.data_dir:
            raise ValueError("When --corpus=custom, at least one --data-dir/-d must be provided.")
        data_dirs = [Path(p) for p in args.data_dir]
    else:
        data_dirs = resolve_corpus_dirs(args.corpus)

    # Sanity checks
    for p in data_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Data directory not found: {p}")

    # Build the RAG pipeline
    rag = make_rag(
        data_dirs=data_dirs,
        backend=args.backend,
        model_name=args.model_name,
        target_lang=args.answer_lang,
        use_4bit=args.use_4bit,
    )

    # Run tests or interactive loop
    if args.test:
        scope = args.test_scope or (args.corpus if args.corpus in {"all", "nam", "tan"} else "all")
        run_test_queries(rag, scope=scope, tests_dir=Path(args.tests_dir))
    else:
        interactive_loop(rag)
