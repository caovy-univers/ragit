import argparse
import json
from pathlib import Path
import re

import pytesseract
from loguru import logger
from PIL import Image

from edition.models import Metadata


def load_metadata(metadata_file: Path) -> list[Metadata]:
    """Load metadata list from a JSON file."""
    logger.info(f"Loading metadata from {metadata_file}")
    with metadata_file.open('r', encoding='utf-8') as file:
        data = json.load(file)
    logger.info(f"Loaded {len(data)} metadata entries")
    return [Metadata(**entry["metadata"]) for entry in data]


def ocr_image(image_path: Path) -> list[str]:
    """Perform OCR on an image and split by line."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='vie+fra')
    return [line for line in text.splitlines() if line.strip()]


def find_images_folders(data_folder: Path) -> list[Path]:
    """Find all images folders under data_folder."""
    folders = [p for p in data_folder.rglob('images') if p.is_dir()]
    logger.info(f"Found {len(folders)} image folders under {data_folder}")
    return folders


def get_page_number(name: str) -> int:
    """Extract numeric part from page name for sorting."""
    match = re.search(r'page[_-]?(\d+)', name)
    return int(match.group(1)) if match else 0


def get_metadata_for_folder(metadata_list: list[Metadata], volume: str, issue: str) -> Metadata | None:
    """Get metadata object matching given volume and issue."""
    for meta in metadata_list:
        if str(meta.volume) == volume and str(meta.numero) == issue:
            return meta
    return None


def process_pages(image_files: list[Path], section_name: str) -> dict[str, list[str]]:
    """Process a list of image files and return a dict of page_id to lines."""
    pages_dict: dict[str, list[str]] = {}
    total = len(image_files)

    for i, image_file in enumerate(image_files, start=1):
        page_name = image_file.stem
        page_number = get_page_number(page_name)
        page_id = f"page_{page_number}"

        lines = ocr_image(image_file)
        pages_dict.setdefault(page_id, []).extend(lines)

        print(f"\rProcessing {section_name} ({i}/{total}) : {page_id}", end="", flush=True)

    print()  # Move to new line after loop
    return pages_dict


def build_json_document(images_folder: Path, metadata: Metadata, output_file: Path) -> None:
    """Build JSON document from OCR results and save to file."""
    logger.info(f"Building JSON document for folder: {images_folder}")

    image_files = [f for f in images_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    image_files = sorted(image_files, key=lambda x: get_page_number(x.stem))

    lexicon_files = [f for f in image_files if f.stem.startswith("lexicon")]
    text_body_files = [f for f in image_files if not f.stem.startswith("lexicon") and "page_" in f.stem]

    lexicon = process_pages(lexicon_files, "lexicon") if lexicon_files else {}
    text_body = process_pages(text_body_files, "text_body") if text_body_files else {}

    doc_dict = {
        "metadata": metadata.model_dump(mode="json"),
    }
    if lexicon:
        doc_dict["lexicon"] = lexicon
    doc_dict["text_body"] = text_body

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing JSON output to {output_file}")
    with output_file.open('w', encoding='utf-8') as file:
        json.dump(doc_dict, file, ensure_ascii=False, indent=2)
    logger.success(f"JSON file saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR images to JSON with metadata.")
    parser.add_argument("data_folder", nargs='?', default="data/Nam-Phong/Quyen-1/", help="Path to data folder.")
    parser.add_argument("metadata_file", nargs='?', default="data/Nam-Phong/metadata.json", help="Path to metadata JSON file.")
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    metadata_file = Path(args.metadata_file)

    metadata_list = load_metadata(metadata_file)
    images_folders = find_images_folders(data_folder)

    for folder in images_folders:
        parent = folder.parent
        volume = parent.parent.name.split('-')[-1]  # e.g., "Quyen-1" -> "1"
        issue = parent.name.split('-')[-1]         # e.g., "So-1" -> "1"

        logger.info(f"Processing folder: {folder} (Volume: {volume}, Issue: {issue})")

        metadata = get_metadata_for_folder(metadata_list, volume, issue)
        if not metadata:
            logger.warning(f"No metadata found for folder {folder}. Skipping.")
            continue

        output_folder = parent / "output_json"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_filename = f"Nam-Phong-Q{volume}-S{issue}.json"
        output_file = output_folder / output_filename

        build_json_document(folder, metadata, output_file)
