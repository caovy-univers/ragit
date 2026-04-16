# Lexicon Matching Pipeline

This package contains tools for matching the published lexicon against a text supplied by the user. It supports:
- exact lookup by `vi`, `vi-hani`, or `fr`
- similar-definition lookup from `vi-def`

## Project Structure

```
src/
└── rag/
    ├── __init__.py
    ├── lexicon_index.py
    ├── lexicon_exact_match.py
    ├── lexicon_definition_match.py
    ├── main_pipeline.py        ← CLI entry point
    └── README.md               ← This README
```

## Usage

### 1) Prepare the lexicon

Place the published CSV files under `lexicon/` at the workspace root.

Exact lexicon lookup in one text:

```bash
poetry run python -m rag.main_pipeline \
  --lexicon-exact \
  --term "Anh-hùng-ca" \
  --term-column vi \
  --text-file /path/to/text.txt
```

Similar-definition lookup in one text:

```bash
poetry run python -m rag.main_pipeline \
  --lexicon-definition-match \
  --definition-term "Anh-hùng-ca" \
  --definition-column vi \
  --candidate-column vi \
  --min-similarity 0.08 \
  --text-file /path/to/text.txt
```
