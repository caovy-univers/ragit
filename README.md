# Rag'it!

Rag’it is an exploratory project in digital humanities. It aims to build a knowledge graph from a multilingual colonial corpus drawn from the Indochinese legal deposit. It relies on natural language processing techniques applied to OCR-processed texts to identify concepts and semantic networks.

# Environment Setup for the RAGIT Project

This guide explains how to set up your Python environment to work with the RAGIT project, which uses **Poetry** for dependency management. Choose one of the three options below depending on your preferred setup tool: `venv`, `pyenv`, or `conda`.

## Prerequisites

- Python **3.12** installed (Poetry requires this version for this project)
- Poetry installed
  - [see installation instructions on the Poetry site](https://python-poetry.org/docs/#installation)

## Environment Setup
### Option 1: Using `venv` (standard Python virtual environment)

```bash
# Create and activate virtual environment
python3.12 -m venv .venv && source .venv/bin/activate

# Install dependencies
poetry install
```

---

### Option 2: Using `pyenv`

```bash
# Create ragit virtualenv with pyenv-virtualenv and set the local environment
pyenv virtualenv 3.12.9 ragit && pyenv local ragit

# Install dependencies
poetry install
```

---

### Option 3: Using `conda`

```bash
# Create and activate conda environment
conda create -n ragit python=3.12 -y && conda activate ragit

# Install dependencies
poetry install
```

---

## Contributing

To manage project dependencies:

```bash
# Add a dependency
poetry add <package-name>

# Remove a dependency
poetry remove <package-name>
```
