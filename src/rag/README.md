# RAG Pipeline for Nam Phong

This folder contains a Retrieval-Augmented Generation (RAG) pipeline using LangChain, FAISS, and Ollama with the `mistral` model. It supports queries in Vietnamese, French, or English against a JSON corpus containing mixed-language content.

## Structure

* `src/`
  * `rag/`
    * `__init__.py` <- init to declare rag module
    * `main_pipeline.py` <- pipeline script to run full rag process with sample queries
    * `loader.py` <- loader to read and parse json files from directory
    * `document_builder.py` <- builder of langchain documents from chunks and metadata
    * `vector_indexer.py` <- vector indexer using multilingual sentence embeddings and faiss
    * `rag_wrapper.py` <- rag wrapper for ollama mistral with retrieval pipeline
    * `README.md`

## Setup

###  1. Run Mistral model with Ollama

Make sure Ollama is installed and running. Then pull the model:

```bash
ollama pull mistral
ollama run mistral  # this keeps the model server active
```

**/!\ Warning:** Keep the Ollama server running in a separate terminal while executing the pipeline. The model will be queried by the RAG pipeline.

### 2. Run the pipeline (in a separate terminal)

From the project root:

```bash
python src/rag/main_pipeline.py
```

This will:
1. Load all `.json` files in the `data/Nam-Phong/Quyen-1/So-1/output_json` folder
2. Build the FAISS vector index with multilingual sentence embeddings
3. Query the Ollama `mistral` model with example questions in Vietnamese, French, and English

## Example queries used in pipeline

>* "Văn minh học thuật của nước Pháp được miêu tả như thế nào trong Nam Phong tạp chí?"
>   * *translation:* "How is the academic civilization of France described in Nam Phong magazine?"
>* "Quel est le rôle de l'Académie française selon les articles de la revue Nam Phong ?"
>   * *translation:* "What is the role of the French Academy according to articles in Nam Phong magazine?"
>* "How does Nam Phong magazine discuss the conflict between material and spiritual progress in modern civilization?"
