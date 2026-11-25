# RAG'It!

At the turn of the twentieth century, Vietnamese was being rapidly standardised and enriched. As knowledge circulated transnationally, new vocabulary for modern scientific, political and philosophical ideas emerged, and periodicals such as *Nam Phong* published trilingual lexical lists in *quốc ngữ*, Chinese, and French, to stabilise these neologisms.

RAG'It is an experimental project, carried out during a 2025 digital residency funded by the DISTAM consortium (DIgital STudies Africa, Asia, Middle East). The goal is to trace how new concepts were named, translated and negotiated in colonial print. It unfolds in three stages:

1. **OCR** of ***Nam Phong*** — a trilingual [lexicon](https://doi.org/10.5281/zenodo.19438119) (*quốc ngữ*, Chinese, French) published by the periodical from 1917-1934; and the ***Triết học*** collection — a series of Vietnamese translations of philosophical thinkers published by Tân Việt in Hanoi in the early 1940s;
2. a **Retrieval-Augmented Generation (RAG)** pipeline built with local LLMs (Mistral, PhoGPT) to retrieve and query neologisms across Vietnamese, French and English;
3. **enrichment** of the retrieved segments by linking them to a lexical database and to bibliographic metadata and, in the longer term, extending the corpus by OCR-ing further texts of the period. This stage links RAG'It to [*invisto*](https://doi.org/10.34847/nkl.2fc7j8g7), the open relational database of the colonial Vietnamese book world built during the author's doctoral research — extending that bibliographic work toward the computational study of concepts. 

This repository only contains the code of the RAG system and its architecture.

## Data

The RAG pipeline runs over a corpus of plain-text (`.txt`) documents **that you provide**. Place your documents under `data/` following the structure in [`data_info.md`](data_info.md). The `data/` folder is git-ignored.

## Environment Setup

This guide explains how to set up your Python environment to work with the RAGIT project, which uses **Poetry** for dependency management. Choose one of the three options below depending on your preferred setup tool: `venv`, `pyenv`, or `conda`.

### 1. Prerequisites

* Python **3.12** installed
* Poetry installed
  (see the official [Poetry installation instructions](https://python-poetry.org/docs/#installation))

### 2. Environment Setup

**Option 1: Using `venv`**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
poetry install
```

**Option 2: Using `pyenv`**

```bash
pyenv install -s 3.12.9 && pyenv virtualenv 3.12.9 ragit && pyenv local ragit
poetry install
```

**Option 3: Using `conda`**

```bash
conda create -n ragit python=3.12 -y && conda activate ragit
poetry install
```

### 3. Models

#### Ollama (default)

We use **Ollama** with the `mistral` model by default.

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
# macOS (Homebrew)
brew install ollama

# Pull the base model
ollama pull mistral
```

#### PhoGPT (optional)

Using **PhoGPT** requires:

1. **Hugging Face account** with access to the gated model **vinai/PhoGPT-4B-Chat**
   - open [PhoGPT-4B-Chat on Hugging Face](https://huggingface.co/vinai/PhoGPT-4B-Chat) and **accept the terms**
2. **Authentication**:
   - Either via **token** (`huggingface-cli login`)
   - Or via **SSH** (add your SSH key to HF; we provide a git+ssh fallback in the Makefile)

Then you can **download the model locally** in two ways:

- **Recommended (HF CLI, non-deprecated)**:

  ```bash
  make download-phogpt-hf
  ```

  This places the files under `models/PhoGPT-4B-Chat/`.

- **Fallback (Git+SSH + Git LFS)**:

  ```bash
  make download-phogpt-ssh
  ```

Once the model is available locally, run the tests with:

```bash
make test-all-phogpt PHOGPT_MODEL=models/PhoGPT-4B-Chat
```

If you prefer to stream from the Hub (not recommended on flaky networks), omit `PHOGPT_MODEL=` so it uses `vinai/PhoGPT-4B-Chat`.

> Tip: If your network blocks Hugging Face’s xet backend, the `download-phogpt-hf` target already sets:
> `HF_HUB_ENABLE_HF_TRANSFER=0 HF_ENDPOINT=https://huggingface.co`

### 4. Running the RAG pipeline

> **Note:** no corpus is bundled with this repository (see [Data](#data)). The example targets below (`nam`, `tan`) expect the corresponding documents under `data/`; point them at your own corpus instead.

#### Streamlit explorer

Launch the user interface with Streamlit:

```bash
poetry run ragit-app
```

#### Interactive mode via Makefile (recommended)

- **Default (Ollama, full corpus)**:

  ```bash
  make
  ```

- **PhoGPT, full corpus**:

  ```bash
  make phogpt
  ```

- **Restricted corpus (interactive)**:

  ```bash
  make nam        # Ollama on Nam-Phong
  make tan        # Ollama on Tan-Viet
  make phogpt-nam # PhoGPT on Nam-Phong
  make phogpt-tan # PhoGPT on Tan-Viet
  ```

#### Batch test mode via Makefile

- **Ollama (1.a on full corpus)**

  ```bash
  make test-all
  ```
- **PhoGPT (1.b on full corpus)**

  ```bash
  make test-all-phogpt PHOGPT_MODEL=models/PhoGPT-4B-Chat
  ```
- **Nam-Phong only**

  ```bash
  make test-nam-ollama   # 2.a – Ollama on Nam-Phong
  make test-nam-phogpt   # 2.b – PhoGPT on Nam-Phong
  ```
- **Tân-Việt only**

  ```bash
  make test-tan-ollama   # 3.a – Ollama on Tân-Việt
  make test-tan-phogpt   # 3.b – PhoGPT on Tân-Việt
  ```
- **Full subcorpus matrix (2.a, 2.b, 3.a, 3.b)**

  ```bash
  make test-full
  ```

#### Direct Python entrypoint

```bash
poetry run python -m rag.main_pipeline --corpus all -b ollama -m mistral -L fr --tests-dir src/rag/tests -t --test-scope all
poetry run python -m rag.main_pipeline --corpus all -b phogpt -m models/PhoGPT-4B-Chat -L vi --tests-dir src/rag/tests -t --test-scope all
```

## Roadmap

- Load a **lexicon CSV** directly and align it against an arbitrary, user-provided text corpus to trace concept occurrences and their evolution across languages.
- Generalise corpus handling so any plain-text collection can be indexed without project-specific preprocessing.

## Contributing

Manage dependencies with Poetry:

```bash
poetry add <package-name>
poetry remove <package-name>
```
