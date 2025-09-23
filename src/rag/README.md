# RAG Pipeline for Nam Phong & Tân-Việt

This package contains a Retrieval-Augmented Generation (RAG) pipeline built with
LangChain and FAISS. It supports Vietnamese, French, and English queries on a
mixed-language JSON corpus. Two LLM backends are supported:

* **Ollama** (e.g., `mistral`)
* **PhoGPT** from Hugging Face (e.g., `vinai/PhoGPT-4B`)

## Project Structure

```
src/
└── rag/
    ├── __init__.py
    ├── main_pipeline.py        ← CLI entry point
    ├── loader.py               ← JSON loader (robust text_body normalization)
    ├── document_builder.py     ← Converts JSON records to LangChain Documents
    ├── vector_indexer.py       ← FAISS vector store builder
    ├── wrapper.py              ← Minimal RAG wrapper (clean sources formatting)
    ├── tests/                  ← Externalized test queries (YAML/JSON)
    │   ├── all.yaml
    │   ├── nam.yaml
    │   └── tan.yaml
    └── README.md               ← This README
```

### JSON expectations

Each JSON file should contain:

* `metadata`: object with fields like `title_main`, `authors`, `publication_date`, etc.
* `text_body`: can be `list[str]`, `list[list[str]]`, `dict[str, list[str]|str]` (paged), or `str`.

`loader.py` flattens the `text_body` to a list of non-empty chunks.

## Usage

### 0) Prepare data

By default, the pipeline looks in:

* `data/Tan-Viet/output`
* `data/Nam-Phong/output`

Use `--corpus all|nam|tan`, or `--corpus custom -d path1 -d path2 ...`.

### 1) Run with **Ollama** (Mistral)

Make sure Ollama is running and `mistral` is available:

```bash
ollama pull mistral
```

### 2) Run with **PhoGPT**

You can use: `-b phogpt -m vinai/PhoGPT-4B` or point to a local folder.
This repo loads PhoGPT via Transformers **without remote code**, forcing a safe
Torch attention implementation (no FlashAttention/Triton required).

### 3) Launch the pipeline

Interactive mode:

```bash
poetry run python -m rag.main_pipeline
```

Test mode with external queries (reads `src/rag/tests/*.yaml`):

```bash
poetry run python -m rag.main_pipeline --test
```

Select backend, corpus, and language:

```bash
poetry run python -m rag.main_pipeline --corpus all -b ollama -m mistral -L fr --test
poetry run python -m rag.main_pipeline --corpus all -b phogpt -m vinai/PhoGPT-4B -L vi --test
```

Custom tests directory:

```bash
poetry run python -m rag.main_pipeline --tests-dir path/to/tests --test
```

### 4) Makefile shortcuts

From the project root:

```bash
make                 # 1.a (All+Ollama) then 1.b (All+PhoGPT)
make full            # 2.a, 2.b, 3.a, 3.b
make help
```

Override defaults:

```bash
make OLLAMA_MODEL=llama3 LANG_OLLAMA=en
make PHOGPT_MODEL=vinai/PhoGPT-4B LANG_PHOGPT=vi
make TESTS_DIR=my_tests
```

## Output format

Answers include a clean, deduplicated **Sources** section (no bold; no duplicates), e.g.:

```
Sources:
- Nam Phong tạp chí — Phạm Quỳnh, Nguyễn Bá Trác (1917-07-01)
- Những sự gặp gỡ của Đông phương và Tây phương trong ngôn ngữ và văn chương — Vũ Bội Liêu (1944)
```

## Notes

* The loader no longer logs each file; it only logs directory scans and warns on failed files.
* The wrapper deduplicates sources by `(title_main, publication_date)` and limits to 5 entries.
