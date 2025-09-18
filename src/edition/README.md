# Edition

## 1. OCR to JSON Edition

- **Script:** `ocr_to_json.py`
- **Goal:** Aggregate OCR pages into normalized JSON files, processing **Nam-Phong** first, then **Tân-Việt**.

### Inputs

- **OCR files**

  - **OCR folder**: `data/OCR/`
  - **Nam-Phong**:
    - format: `Nam-Phong-So<issue>_<page5>.txt`
  - **Tân-Việt**:
    - format: `<H-ID>_<normalized_title>_<page5>.txt`
    - example: `4394_Ba_muoi_nam_van_hoc_00001.txt`.
- **Metadata**
  - **Nam-Phong (JSON):** `data/Nam-Phong/metadata.json`.
  - **Tân-Việt (CSV):** `data/Tan-Viet/Tan_Viet_metadata.csv`.

### Outputs

* **Nam-Phong:** `data/Nam-Phong/output/`
* **Tân-Việt:** `data/Tan-Viet/output/`

### JSON Structure (snippets)

```json
// Nam-Phong
{
  "metadata": { /* fields from metadata.json */ },
  "text_body": {
    "page_1": ["<line>", "<line>", "..."],
    "page_2": ["..."]
  }
}
```

```json
// Tân-Việt
{
  "metadata": {
    "title_main": "...",
    "title_alt": "...",
    "authors": ["..."],
    "authors_id": ["..."],
    "publisher_id": "...",
    "publisher": "...",
    "publication_place_id": "...",
    "publication_place": "...",
    "publication_date": "YYYY",
    "source_description": null,
    "genres": [],
    "issn": null,
    "identifiers": {
      "H-ID": "4394",
      "url1": "...",
      "url2": "...",
      "PDF": true
    }
  },
  "text_body": {
    "page_1": ["<line>", "<line>"]
  }
}
```

### Notes

* **Nam-Phong** pages are grouped by `So<issue>` extracted from the filename; metadata supplies `volume` and `numero`.
* **Tân-Việt** pages are grouped strictly by `H-ID` (first numeric token in the filename), independent of the normalized title.

## 2. HTML to JSON Edition

- **Script:** `html_to_json.py`
- **Goal:** Convert **HTML** files (e.g., from Wikisource) to JSON, injecting metadata from a YAML file.

### Inputs

* HTML files: `data/Nam-Phong/Quyen-1/So-1/*.html`
* Metadata YAML: `data/Nam-Phong/Quyen-1/So-1/metadata.yaml`

### Processing Steps

* Parse YAML to a Pydantic `Metadata` model.
* Parse each HTML with BeautifulSoup and extract paragraphs `<p>`.
* Build a `Document` with `metadata` and a `text_body` list of paragraphs.
* Save JSON with pretty-printing and UTF-8 (diacritics preserved).
* Log each processed file with `loguru`.
