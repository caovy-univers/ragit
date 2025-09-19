# Rag'it!

Rag’it is an exploratory project in digital humanities. It aims to build a knowledge graph from a multilingual colonial corpus drawn from the Indochinese legal deposit. It relies on natural language processing techniques applied to OCR-processed texts to identify concepts and semantic networks.

## Environment Setup for the RAGIT Project

This guide explains how to set up your Python environment to work with the RAGIT project, which uses **Poetry** for dependency management. Choose one of the three options below depending on your preferred setup tool: `venv`, `pyenv`, or `conda`.

### 1. Prerequisites

* Python **3.12** installed
* Poetry installed
  (see the official [Poetry installation instructions](https://python-poetry.org/docs/#installation))

### 2. Environment Setup

**Option 1: Using `venv`**

```bash
python3.12 -m venv ragit && source ragit/bin/activate
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

1. **Hugging Face account** with access to the gated model **vinai/PhoGPT-7B5**
   - open [PhoGPT-7B5 on Hugging Face](https://huggingface.co/vinai/PhoGPT-7B5) and **accept the terms**
2. **Authentication**:
   - Either via **token** (`huggingface-cli login`)
   - Or via **SSH** (add your SSH key to HF; we provide a git+ssh fallback in the Makefile)

Then you can **download the model locally** in two ways:

- **Recommended (HF CLI, non-deprecated)**:

  ```bash
  make download-phogpt-hf
  ```

  This places the files under `models/PhoGPT-7B5/`.

- **Fallback (Git+SSH + Git LFS)**:

  ```bash
  make download-phogpt-ssh
  ```

Once the model is available locally, run the tests with:

```bash
make test-all-phogpt PHOGPT_MODEL=models/PhoGPT-7B5
```

If you prefer to stream from the Hub (not recommended on flaky networks), omit `PHOGPT_MODEL=` so it uses `vinai/PhoGPT-7B5`.

> Tip: If your network blocks Hugging Face’s xet backend, the `download-phogpt-hf` target already sets:
> `HF_HUB_ENABLE_HF_TRANSFER=0 HF_ENDPOINT=https://huggingface.co`

### 4. Running the RAG pipeline

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
  make test-all-phogpt PHOGPT_MODEL=models/PhoGPT-7B5
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
poetry run python -m rag.main_pipeline --corpus all -b phogpt -m models/PhoGPT-7B5 -L vi --tests-dir src/rag/tests -t --test-scope all
```

### 5. Contributing

Contributing

Manage dependencies with Poetry:

```bash
poetry add <package-name>
poetry remove <package-name>
```
