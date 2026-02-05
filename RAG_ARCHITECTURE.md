# RAG Architecture Documentation

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system designed for multilingual digital humanities research, specifically targeting Vietnamese and French colonial literature from the Indochinese legal deposit. The system enables semantic search and question-answering across two major historical corpora: **Nam Phong** (南風) and **Tân Việt** (新越).

The RAG pipeline combines modern natural language processing techniques with historical text analysis, supporting queries in Vietnamese, French, and English while maintaining historical context and provenance through rich metadata.

## Architecture Components

### 1. Data Loading Layer (`loader.py`)

The data loading layer handles the ingestion and normalization of JSON-formatted documents from the corpus.

#### Core Functions

**`load_json_file(filepath: Path) -> Tuple[List[str], Dict]`**

Loads and parses a single JSON document containing:
- `metadata`: Bibliographic information (title, authors, publication date, publisher, genres, etc.)
- `text_body`: Raw OCR text in multiple possible formats

**`_flatten_text_body(text_body: Any) -> List[str]`**

This function normalizes the heterogeneous text structures found in the OCR data. It handles:
- **List of strings**: Direct flattening
- **Nested lists**: Recursive flattening (e.g., `[["line1"], ["line2"]]`)
- **Dictionary with page keys**: Sorted page extraction (e.g., `{"page_1": [...], "page_2": [...]}`)
- **Raw strings**: Line-by-line splitting

The function ensures consistent output regardless of the original format, extracting only non-empty lines and preserving reading order through numeric page sorting.

**`load_all_json_files(directory: Path) -> List[Tuple[List[str], Dict]]`**

Recursively scans directories for JSON files, loading all documents while logging errors for corrupted files. Returns a list of (chunks, metadata) tuples ready for document construction.

**`load_all_json_files_multi(directories: List[Path]) -> List[Tuple[List[str], Dict]]`**

Merges data from multiple corpus directories (e.g., both Nam Phong and Tân Việt simultaneously), enabling cross-corpus queries.

### 2. Document Builder Layer (`document_builder.py`)

This layer transforms raw text and metadata into structured LangChain `Document` objects with rich contextual headers.

#### The Metadata Schema

**`Metadata` (Pydantic Model)**

A normalized schema that validates and standardizes bibliographic information:

```python
title_main: str              # Primary title (required)
title_alt: Optional[str]     # Alternative title
publisher: Optional[str]     # Publishing house
publication_place: Optional[str]  # City of publication
publication_date: Optional[str]   # Date (typically YYYY-MM-DD)
source_description: Optional[str]
authors: List[str]           # List of author names
genres: List[str]            # Literary genres (journal, essay, etc.)
issn: Optional[str]          # Serial identifier
identifiers: Dict[str, Any]  # Additional IDs (Gallica, BnF, etc.)
```

The model uses `extra="allow"` to preserve additional metadata fields while providing type safety for core fields. It handles missing or null values gracefully with sensible defaults.

#### Document Creation

**`create_documents(chunks: List[str], metadata: Dict[str, Any]) -> List[Document]`**

For each text chunk, this function:

1. **Validates metadata** through the Pydantic schema
2. **Builds a contextual header** containing:
   - Title(s)
   - Author(s)
   - Publication information (date, place, publisher)
   - Genre classification
3. **Prepends the header** to each chunk, creating self-contained segments that:
   - Enable accurate citation
   - Improve retrieval relevance
   - Provide context to the LLM without requiring metadata injection at query time

This design ensures that each chunk carries its provenance inline, making the content more searchable and semantically rich.

### 3. Vector Indexing Layer (`vector_indexer.py`)

The vector store provides semantic search capabilities across the multilingual corpus.

**`build_vectorstore(documents: List[Document]) -> FAISS`**

This function:

1. **Initializes multilingual embeddings** using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - Supports Vietnamese, French, and English
   - Produces 384-dimensional dense vectors
   - Handles code-switching and mixed-language documents

2. **Builds a FAISS index** for efficient nearest-neighbor search
   - CPU-based implementation (portable across systems)
   - Fast approximate search using Hierarchical Navigable Small World (HNSW) graphs
   - Scales to thousands of documents

The multilingual embedding model is crucial for this corpus, as documents often contain mixed French-Vietnamese content reflecting the colonial bilingual context.

### 4. LLM Factory Layer (`llm_factory.py`)

The factory pattern enables flexible LLM backend selection, supporting both local and cloud-based models.

#### Supported Backends

**Ollama Backend**

The default lightweight option:
- Local inference via Ollama server
- Supports models like Mistral, Llama, etc.
- Zero GPU requirements
- Fast startup and inference

**PhoGPT Backend**

Vietnamese-optimized large language model:
- Based on VinAI's `PhoGPT-4B-Chat` or `PhoGPT-7B5`
- Trained specifically on Vietnamese text
- Superior performance for Vietnamese queries
- Supports 4-bit quantization for memory efficiency

#### Implementation Details

**`_PhoGPTChatLLM` (Custom Wrapper)**

A LangChain-compatible wrapper for PhoGPT that:
- Applies chat templates using the tokenizer's `apply_chat_template` method
- Injects a system prompt for instruction-following behavior
- Handles the prompt-to-generation pipeline
- Strips the input prompt from output (via `return_full_text=False`)

**`_build_phogpt_pipeline(...) -> _PhoGPTChatLLM`**

Constructs a Hugging Face pipeline with careful configuration:

1. **Device Selection**: Auto-detects CUDA availability
2. **Dtype Selection**:
   - `bfloat16` on Ampere+ GPUs (compute capability ≥ 8)
   - `float16` on older CUDA devices
   - `float32` on CPU
3. **Quantization** (optional):
   - 4-bit NF4 quantization via BitsAndBytes
   - Double quantization for additional compression
   - Reduces memory footprint from ~16GB to ~4GB
4. **Generation Parameters**:
   - `max_new_tokens`: 256-512 tokens
   - `temperature`: Low (0.0-0.2) for deterministic, fact-based answers
   - `repetition_penalty`: 1.10 to discourage redundancy
   - `do_sample`: Inferred from temperature/top_p/top_k settings

**`build_llm(backend: Backend, model_name: str, ...) -> Union[OllamaLLM, _PhoGPTChatLLM]`**

Factory function that:
- Routes to the appropriate backend based on the `backend` parameter
- Applies consistent generation parameters across backends
- Returns a LangChain-compatible interface regardless of underlying implementation

### 5. RAG Wrapper Layer (`wrapper.py`)

The orchestration layer that ties retrieval and generation together.

#### Core Components

**Prompt Template**

```
You are a helpful RAG assistant.
Use the provided context to answer the question concisely and faithfully.
ALWAYS answer in the target language: {TARGET_LANG}.

# Rules
- If the context is insufficient, say so briefly.
- Cite short source info the system will append; do not fabricate references.

# Question
{QUESTION}

# Context
{CONTEXT}
```

The prompt emphasizes:
- **Faithfulness to sources**: Discourages hallucination
- **Language consistency**: Ensures answers match the requested language
- **Conciseness**: Prevents verbose, meandering responses
- **Citation transparency**: Separates inline citations from appended source lists

#### Helper Functions

**`_build_context(docs: Sequence[Document], max_chars: int = 4000) -> str`**

Concatenates retrieved documents into a single context string:
- Respects a character limit to prevent context overflow
- Separates chunks with `---` delimiters
- Truncates gracefully when hitting the limit

**`_unique_sources(docs: Sequence[Document], limit: int = 5) -> List[Tuple[str, str, str]]`**

Deduplicates sources based on `(title_main, publication_date)` key:
- Prevents redundant citations when multiple chunks come from the same document
- Extracts (title, authors, date) tuples
- Limits to top 5 sources for readability

**`_format_sources(docs: Sequence[Document]) -> str`**

Produces a clean bibliography section:

```
Sources:
- Nam Phong tạp chí — Phạm Quỳnh, Nguyễn Bá Trác (1917-07-01)
- Tân Việt — Vũ Bội Liêu (1944)
```

**`sanitize_text(text: str) -> str`**

Cleans LLM output for terminal display:
- Normalizes to NFC form (crucial for Vietnamese diacritics)
- Removes replacement characters (U+FFFD)
- Strips control characters while preserving whitespace

#### The SimpleRAG Class

**Initialization**

```python
SimpleRAG(
    vectorstore: FAISS,
    llm: Union[OllamaLLM, _PhoGPTChatLLM],
    model_name: str,
    target_lang: str = "fr",
    search_k: int = 3
)
```

Constructor responsibilities:
- Creates a retriever from the vector store with configurable `k` (number of documents)
- Stores the LLM instance for generation
- Sets the target output language
- Initializes the prompt template

**The `ask` Method**

```python
def ask(query: str) -> Dict[str, Any]:
    """
    Retrieve, generate, and format a response with sources.

    Returns:
        {"result": str, "source_documents": List[Document]}
    """
```

Execution flow:

1. **Retrieval Phase**:
   - Embeds the query using the same multilingual model
   - Performs k-NN search in the FAISS index
   - Returns top-k most semantically similar chunks
   - Logs timing: typically 0.05-0.2 seconds

2. **Context Building**:
   - Concatenates retrieved chunks (max 4000 chars)
   - Adds separators for clarity
   - Logs context length for debugging

3. **Prompt Formatting**:
   - Injects target language, question, and context
   - Produces final prompt string

4. **Generation Phase**:
   - Invokes the LLM with the formatted prompt
   - Uses greedy decoding by default (temperature ≈ 0)
   - Logs generation time: 1-10 seconds depending on backend

5. **Post-Processing**:
   - Sanitizes the generated text
   - Appends the formatted sources section
   - Returns combined answer with metadata

### 6. Main Pipeline (`main_pipeline.py`)

The entry point that orchestrates the entire system.

#### Configuration Management

**`resolve_corpus_dirs(corpus: str) -> List[Path]`**

Maps logical corpus names to file paths:
- `"all"`: Both Nam Phong and Tân Việt
- `"nam"`: Only Nam Phong (`data/Nam-Phong/output`)
- `"tan"`: Only Tân Việt (`data/Tan-Viet/output`)
- `"custom"`: User-specified directories

**`make_rag(...) -> SimpleRAG`**

The main factory function that:

1. **Loads documents** from specified directories
2. **Converts** raw (chunks, metadata) pairs to LangChain Documents
3. **Builds the vector store** with multilingual embeddings
4. **Initializes the LLM** backend (Ollama or PhoGPT)
5. **Constructs and returns** a fully configured SimpleRAG instance

#### Usage Modes

**Interactive Mode** (default)

```bash
poetry run python -m rag.main_pipeline --corpus all --backend ollama
```

Launches a REPL where users can enter queries interactively.

**Test Mode**

```bash
poetry run python -m rag.main_pipeline --corpus nam --test
```

Runs three predefined queries from YAML/JSON test files in `src/rag/tests/`:
- `all.yaml`: Cross-corpus comparison queries
- `nam.yaml`: Nam Phong-specific questions
- `tan.yaml`: Tân Việt-specific questions

Each test file contains multilingual queries (Vietnamese, French, English) to validate cross-lingual retrieval and generation.

**Command-Line Arguments**

```bash
--corpus, -c          # Corpus selection: all, nam, tan, custom
--data-dir, -d        # Custom data directory (repeatable)
--backend, -b         # LLM backend: ollama, phogpt
--model-name, -m      # Model identifier
--answer-lang, -L     # Target answer language: fr, vi, en
--use-4bit            # Enable 4-bit quantization (PhoGPT only)
--test, -t            # Run test queries
--test-scope          # Which test set to run
--tests-dir           # Test queries directory
```

### 7. Streamlit Application (`app/main.py`)

A web-based interface for exploring the corpus.

#### Features

**Configuration Sidebar**

- **Corpus Selection**: Dropdown to choose Nam Phong, Tân Việt, both, or custom paths
- **Backend Selection**: Toggle between Ollama and PhoGPT
- **Model Source**: Hub vs. local model path (auto-detects `models/PhoGPT-4B-Chat`)
- **Target Language**: Dynamically adjusts based on backend (Vietnamese for PhoGPT, French for Ollama by default)
- **Quantization Toggle**: 4-bit mode checkbox for PhoGPT

**Query Interface**

- Text input for natural language questions
- Real-time answer generation
- Source document display with metadata
- Session state management for configuration persistence

**Caching Strategy**

```python
@st.cache_resource(show_spinner=False)
def load_pipeline(...) -> SimpleRAG:
    ...
```

The RAG pipeline is cached per configuration hash, avoiding costly reloads when switching between queries. Only configuration changes trigger reinitialization.

## Data Flow

### End-to-End Query Processing

```
User Query: "Nam Phong tạp chí bàn về triết-học như thế nào?"
    ↓
[1. EMBEDDING]
    Query → Multilingual Encoder → 384-dim vector
    ↓
[2. RETRIEVAL]
    FAISS k-NN search → Top 3 document chunks
    ↓
    Retrieved chunks (example):
    - "Nam Phong tạp chí — Phạm Quỳnh (1917-07-01)
       Triết-học là môn học về sự suy ngẫm về con người..."
    - "Nam Phong tạp chí — Nguyễn Bá Trác (1918-03-15)
       Bàn về quan hệ giữa triết học và khoa học..."
    - ...
    ↓
[3. CONTEXT BUILDING]
    Concatenate chunks with headers → Context string (< 4000 chars)
    ↓
[4. PROMPT FORMATTING]
    Inject: TARGET_LANG=vi, QUESTION, CONTEXT
    ↓
[5. GENERATION]
    LLM.invoke(prompt) → Raw answer text
    ↓
[6. POST-PROCESSING]
    - Sanitize text (NFC normalization, control char removal)
    - Append deduplicated sources
    ↓
[7. OUTPUT]
    {
      "result": "Trong Nam Phong tạp chí, triết-học được...\n
                 Sources:\n
                 - Nam Phong tạp chí — Phạm Quỳnh (1917-07-01)\n
                 - ...",
      "source_documents": [Document(...), ...]
    }
```

### Timing Breakdown (Typical)

- **Retrieval**: 0.05-0.2s (FAISS search)
- **Context Building**: <0.01s (string concatenation)
- **Prompt Formatting**: <0.01s (template substitution)
- **Generation**:
  - Ollama (Mistral): 1-3s
  - PhoGPT (4-bit): 3-8s
  - PhoGPT (FP16): 1-4s (GPU-dependent)
- **Post-Processing**: <0.01s
- **Total**: ~1-10s depending on backend and hardware

## Design Decisions

### Why This Architecture?

**Modularity**: Each layer has a single responsibility, enabling independent testing and replacement of components (e.g., swapping FAISS for Pinecone, or adding Anthropic API support).

**Multilingual-First**: The choice of `paraphrase-multilingual-MiniLM-L12-v2` and PhoGPT reflects the reality of the corpus: documents contain mixed French-Vietnamese text, and queries may come in either language.

**Metadata-Enriched Chunks**: By prepending headers to each chunk, we solve two problems:
1. **Relevance**: Headers contain keywords (author names, dates) that improve retrieval
2. **Citation**: The LLM sees source information inline, reducing hallucinated citations

**Low-Temperature Generation**: For a scholarly RAG system, we prioritize factual accuracy over creativity. Temperature is kept near zero to minimize fabrication.

**4-bit Quantization**: Enables deployment on consumer GPUs (RTX 3060, 4060) without sacrificing much quality, democratizing access to large models.

### Trade-offs

**FAISS vs. Vector Databases**: We use FAISS for simplicity and portability. For production at scale, consider Pinecone, Weaviate, or Qdrant for distributed search and filtering.

**Chunk Headers vs. Metadata Injection**: Headers increase index size (~20%) but improve retrieval quality. Alternatively, we could inject metadata at query time, reducing index size but complicating the prompt.

**Fixed k=3 vs. Dynamic Retrieval**: We retrieve a fixed number of chunks. Advanced systems might use threshold-based retrieval or re-ranking (e.g., cross-encoder models).

**Greedy Decoding vs. Sampling**: Greedy decoding is deterministic and suitable for RAG. For more creative tasks (e.g., generating poetry), increase temperature and enable sampling.

## Extending the System

### Adding a New Backend

1. Implement a LangChain-compatible wrapper (similar to `_PhoGPTChatLLM`)
2. Add a branch in `build_llm` factory function
3. Update CLI arguments and Streamlit UI

### Custom Metadata Fields

1. Extend the `Metadata` Pydantic model in `document_builder.py`
2. Update the header generation logic in `create_documents`
3. Modify `_format_sources` if you want new fields in citations

### Improving Retrieval

**Hybrid Search**: Combine dense (vector) and sparse (BM25) retrieval:
```python
from langchain.retrievers import EnsembleRetriever
dense_retriever = vectorstore.as_retriever()
sparse_retriever = BM25Retriever(...)
hybrid = EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever])
```

**Re-ranking**: Add a cross-encoder to reorder retrieved chunks:
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# Reorder chunks based on query-chunk relevance
```

### Multi-Query Generation

Generate multiple query variations to improve recall:
```python
from langchain.retrievers import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
```

## Performance Optimization

### Memory

- Use 4-bit quantization for PhoGPT: `--use-4bit`
- Reduce `search_k` (fewer chunks)
- Limit `max_chars` in `_build_context`

### Speed

- Use Ollama for faster inference (3x speedup over PhoGPT)
- GPU acceleration: CUDA-enabled setup reduces generation time by 5-10x
- Cache the vector store: Save/load FAISS index to avoid recomputing embeddings

### Accuracy

- Increase `search_k` to retrieve more context
- Use cross-encoder re-ranking
- Fine-tune the embedding model on domain-specific data
- Adjust prompt to emphasize citation and factuality

## Troubleshooting

### Common Issues

**"No module named 'rag'"**
- Run `poetry install` to install the package in editable mode

**Ollama connection refused**
- Ensure Ollama is running: `ollama serve`
- Check model availability: `ollama list`

**PhoGPT out of memory**
- Enable 4-bit quantization: `--use-4bit`
- Reduce batch size or context length
- Use CPU inference (slower but works): `--device cpu`

**Poor answer quality**
- Check retrieved chunks: Are they relevant?
- Adjust `search_k` (try 5-7)
- Verify embeddings are multilingual
- Inspect prompt template (may need refinement)

**Mixed-language output**
- Explicitly set `--answer-lang` to desired language
- Check that the LLM supports instruction-following (e.g., use chat-tuned models)

## Conclusion

This RAG system represents a practical, production-ready architecture for multilingual historical text analysis. It balances academic rigor with technical accessibility, enabling researchers to query colonial Vietnamese literature in their preferred language while maintaining scholarly standards of citation and provenance.

The modular design allows for incremental improvements (whether swapping backends, refining retrieval, or scaling to larger corpora) without requiring a ground-up rewrite. By open-sourcing this architecture, we hope to lower the barrier for digital humanities projects working with multilingual, historically complex datasets.
