import json
import re
import unicodedata
from evaluate import load
from loguru import logger
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Normalize a given text string: Unicode normalization, remove extra spaces, lowercase.

    Args:
        text (str): Input text string.

    Returns:
        str: Normalized text.
    """
    # Normalize Unicode to NFC form to ensure composed character sequences
    # (important for diacritics and consistent text comparison)
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def load_gold(gold_path: Path) -> dict:
    """
    Load the gold standard JSON file.

    Args:
        gold_path (Path): Path to the gold JSON file.

    Returns:
        dict: Parsed JSON content.
    """
    logger.info(f"Loading gold standard file: {gold_path}")
    with gold_path.open(encoding='utf-8') as f:
        return json.load(f)


def load_ocr_text(ocr_folder: Path, filename: str) -> str:
    """
    Load OCR result text file corresponding to a given image filename.

    Args:
        ocr_folder (Path): Path to the folder containing OCR text files.
        filename (str): Image filename whose corresponding OCR result to load.

    Returns:
        str: OCR content as string.
    """
    txt_file = ocr_folder / f"{Path(filename).stem}.txt"
    if not txt_file.exists():
        logger.error(f"Missing OCR file: {txt_file}")
        raise FileNotFoundError(f"Missing OCR file: {txt_file}")

    logger.debug(f"Loading OCR text from: {txt_file}")
    return txt_file.read_text(encoding='utf-8')


def compute_scores(gold_text: str, ocr_text: str) -> dict:
    """
    Compute evaluation metrics (WER, CER) between gold and OCR texts.

    Args:
        gold_text (str): Normalized reference text.
        ocr_text (str): Normalized OCR output text.

    Returns:
        dict: Dictionary with WER and CER scores.
    """
    wer_metric = load("wer")
    cer_metric = load("cer")

    if not gold_text.strip() or not ocr_text.strip():
        raise ValueError("Reference and hypothesis texts must be non-empty after normalization.")

    scores = {
        "WER": wer_metric.compute(predictions=[ocr_text], references=[gold_text]),
        "CER": cer_metric.compute(predictions=[ocr_text], references=[gold_text])
    }
    return scores


def evaluate_ocr(gold_path: Path, ocr_folder: Path) -> None:
    """
    Evaluate OCR outputs against gold standard.

    Args:
        gold_path (Path): Path to the gold JSON file.
        ocr_folder (Path): Path to the folder containing OCR .txt files.
    """
    gold_data = load_gold(gold_path)
    filenames = gold_data.get("filenames", [])
    content_by_page = gold_data.get("content", {})

    full_gold_text = ""
    full_ocr_text = ""

    for fname in filenames:
        page_key = Path(fname).stem
        if page_key not in content_by_page:
            raise ValueError(f"Gold content missing for {page_key}")

        gold_page_lines = content_by_page[page_key]
        gold_page_text = ' '.join(gold_page_lines)
        full_gold_text += ' ' + normalize_text(gold_page_text)

        ocr_page_text = load_ocr_text(ocr_folder, fname)
        full_ocr_text += ' ' + normalize_text(ocr_page_text)

    scores = compute_scores(full_gold_text, full_ocr_text)
    logger.success(f"WER: {scores['WER']:.2%}, CER: {scores['CER']:.2%}")


if __name__ == "__main__":
    gold_path = Path("data/Nam-Phong/Quyen-1/So-1/gold/namphong_so1_gold.json")
    ocr_folder = Path("data/Nam-Phong/Quyen-1/So-1/gold")

    evaluate_ocr(gold_path, ocr_folder)
