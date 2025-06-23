import json
from pathlib import Path
from typing import List, Tuple
from loguru import logger

def load_json_file(filepath: Path) -> Tuple[List[str], dict]:
    """
    Load a single JSON file containing Vietnamese/French mixed text and metadata.

    Args:
        filepath (Path): Path to the JSON file.

    Returns:
        Tuple[List[str], dict]: List of cleaned text chunks and metadata.
    """
    logger.info(f"Loading JSON file: {filepath}")
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    chunks = [para.strip() for para in data["text_body"] if para.strip()]
    return chunks, metadata


def load_all_json_files(directory: Path) -> List[Tuple[List[str], dict]]:
    """
    Load and parse all JSON files in a given directory.

    Args:
        directory (Path): Path to the folder containing JSON files.

    Returns:
        List[Tuple[List[str], dict]]: List of (chunks, metadata) pairs for each file.
    """
    logger.info(f"Scanning directory for JSON files: {directory}")
    all_data = []
    for file in directory.glob("*.json"):
        try:
            chunks, metadata = load_json_file(file)
            all_data.append((chunks, metadata))
        except Exception as e:
            logger.warning(f"Failed to load {file.name}: {e}")
    return all_data
