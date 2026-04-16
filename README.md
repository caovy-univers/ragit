# Rag'it!

Rag’it is an exploratory project in digital humanities. In this version, the app focuses on querying a published lexicon against a text supplied by the user at runtime.

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

### 3. Running the lexicon explorer

#### Streamlit explorer

Launch the user interface with Streamlit, then upload a `.txt` or `.md` file:

```bash
poetry run ragit-app
```

#### Direct Python entrypoints

#### Exact lexicon lookup in a selected text

Use the dedicated exact-match mode to:
- look up a term in the lexicon by `vi`, `vi-hani`, or `fr`
- select one text by title fragment or file name
- extract the local context where the term appears in that text

```bash
poetry run python -m rag.main_pipeline \
  --lexicon-exact \
  --term "Anh-hùng-ca" \
  --term-column vi \
  --text-file /path/to/text.txt
```

#### Similar-definition lookup in a selected text

Use the `vi-def` of a known lexicon term, then ask the tool to find close terms that really appear in one text:

```bash
poetry run python -m rag.main_pipeline \
  --lexicon-definition-match \
  --definition-term "Anh-hùng-ca" \
  --definition-column vi \
  --candidate-column vi \
  --min-similarity 0.08 \
  --text-file /path/to/text.txt
```

### 4. Contributing

Contributing

Manage dependencies with Poetry:

```bash
poetry add <package-name>
poetry remove <package-name>
```
