# RAG Pipeline for Nam Phong

This repository contains a Retrieval-Augmented Generation (RAG) pipeline built with LangChain, FAISS, and the Ollama `mistral` model. It supports Vietnamese, French, and English queries on a JSON corpus of mixed-language content.

## Project Structure

```
src/
└── rag/
    ├── __init__.py             ← Module initializer
    ├── main_pipeline.py        ← Main script with CLI options
    ├── loader.py               ← Recursive JSON loader
    ├── document_builder.py     ← LangChain Document builder
    ├── vector_indexer.py       ← FAISS vector store builder
    ├── wrapper.py              ← SimpleRAG wrapper class
    └── README.md               ← This README file
```

## Usage

### 1. **Run the Mistral model with Ollama**

   Make sure Ollama is installed and running. Then:

   ```bash
   ollama pull mistral
   ollama run mistral
   ```

   ⚠️ Keep the Ollama server running in a separate terminal.


### 2. **Launch the RAG Pipeline (from the project root)**

⚠️ First, make sure you are in the project root directory then you can run the RAG pipeline with different modes:

- **Interactive Mode**: This will start an interactive prompt for querying the RAG system.
- **Test Mode**: Run predefined queries to test the system.

#### Interactive Mode (default)

```bash
poetry run python -m rag.main_pipeline
```
This will start an interactive prompt where you can enter your queries in Vietnamese, French, or English. The system will return answers based on the indexed data.
If you want to specify a different data directory, you can use the `--data-dir` option:

```bash
poetry run python -m rag.main_pipeline --data-dir /path/to/your/data
```

#### Test Mode

```bash
poetry run python -m rag.main_pipeline --test
```

Executes predefined queries in Vietnamese, French, and English to show how the system responds to different languages and topics.

These are the predefined queries:
* **Vietnamese**: "Văn minh học thuật của nước Pháp được miêu tả như thế nào trong Nam Phong tạp chí?" (*How is the academic civilization of France described in Nam Phong magazine?*)
* **French**: "Quel est le rôle de l'Académie française selon les articles de la revue Nam Phong ?" (*What is the role of the French Academy according to the articles in Nam Phong magazine?*)
* **English**: "How does Nam Phong magazine discuss the conflict between material and spiritual progress in modern civilization?"
