# Optical Character Recognition (OCR)

## Description

This folder contains a baseline implementation of Optical Character Recognition (OCR) for processing Vietnamese text (with French influence) from scanned documents. The OCR process is designed to handle the specific challenges of Vietnamese script, including diacritics and special characters.

For now we are using a basic OCR approach with Tesseract for initial text extraction (`ocr_extraction_baseline.py`). This script is intended to be a starting point for further development and refinement of OCR capabilities. The results are calculated within the `ocr_quality_report.py` against a gold standard available in the `data/` directory.


## Structure

* `ocr_extraction_baseline.py`: Python script for baseline OCR extraction using Tesseract.
* `ocr_quality_report.py`: Python script to generate a quality report comparing OCR results against a gold standard.
* `extract_trilingual_ocr_training_data.py`: Script to extract training data from wikisource documents to align with the images.

### Baseline Expected Input

* Scanned documents in image format (e.g., PNG, JPEG)

### Quality Report Expected Input

* OCR output text files
* Gold standard JSON file containing expected text with metadata:

Example of the expected JSON structure for the gold standard:

```json
{
  "title": "Document Title",
  "wiki_title": "Wiki Document Title",
  "url": "https://example.com/document-url",
  "filenames": ["image_01.jpg", "image_02.jpg"],
  "content": {
    "image_01": ["Text content of page 1",
                 "Additional text content of page 1 (optional)"],
    "image_02": ["Text content of page 2"]
  }
}
```
