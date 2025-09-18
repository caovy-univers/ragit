import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from loguru import logger


def _flatten_text_body(text_body: Any) -> List[str]:
    """
    Normalize a `text_body` field into a flat list of non-empty lines.

    Supported shapes:
    - list[str]                    -> trimmed non-empty lines
    - list[list[str]]              -> flatten and trim
    - dict[str, list[str] | str]   -> pages like 'page_1', 'page_2', sorted numerically
    - str                          -> splitlines + trim
    """
    chunks: List[str] = []

    def add_line(s: str) -> None:
        s = (s or "").strip()
        if s:
            chunks.append(s)

    if isinstance(text_body, list):
        for item in text_body:
            if isinstance(item, str):
                add_line(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        add_line(sub)
            elif isinstance(item, dict):
                for sub in _flatten_text_body(item):
                    add_line(sub)

    elif isinstance(text_body, dict):
        def page_key(k: str) -> int:
            num = "".join(ch for ch in k if ch.isdigit())
            return int(num) if num else 0

        for k in sorted(text_body.keys(), key=page_key):
            v = text_body[k]
            if isinstance(v, list):
                for s in v:
                    if isinstance(s, str):
                        add_line(s)
            elif isinstance(v, str):
                for s in v.splitlines():
                    add_line(s)

    elif isinstance(text_body, str):
        for s in text_body.splitlines():
            add_line(s)

    else:
        logger.warning(f"Unexpected `text_body` type: {type(text_body)}")

    return chunks


def load_json_file(filepath: Path) -> Tuple[List[str], Dict]:
    """
    Load a single JSON file and normalize its text body.

    Input JSON is expected to contain:
      - "metadata": dict (title_main, authors, publication_date, ...)
      - "text_body": list | dict | str (see `_flatten_text_body`)
    """
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {}) or {}
    chunks = _flatten_text_body(data.get("text_body", []))
    return chunks, metadata


def load_all_json_files(directory: Path) -> List[Tuple[List[str], Dict]]:
    """
    Load and parse all JSON files under a given directory (recursive).

    Notes:
        - No per-file info logs; only a summary and error logs.
    """
    logger.info(f"Scanning for JSON files under: {directory}")
    all_data: List[Tuple[List[str], Dict]] = []
    for file in directory.rglob("*.json"):
        try:
            chunks, metadata = load_json_file(file)
            all_data.append((chunks, metadata))
        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")
    logger.info(f"Loaded {len(all_data)} files from {directory}")
    return all_data


def load_all_json_files_multi(directories: List[Path]) -> List[Tuple[List[str], Dict]]:
    """
    Load and merge JSON entries from multiple directories.
    """
    merged: List[Tuple[List[str], Dict]] = []
    for d in directories:
        merged.extend(load_all_json_files(d))
    logger.info(f"Total loaded entries from {len(directories)} dirs: {len(merged)}")
    return merged
