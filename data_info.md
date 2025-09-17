# Data information

## Dataset

We have two datasets:
1. **NAM-PHONG**
    - Lexicon (`data/Nam-Phong/lexicon/csv/`): contains the lexicon files in CSV format from So1 to SO14 (11 and 12 are missing).
      - 1 468 entries in total.
    - Issues
2. **TAN-VIET (Gallica)**

## Metadata

1. **NAM-PHONG**
    - Metadata file (JSON format): `data/Nam-Phong/metadata.json`
    - extract:
      ```json
      [
        {
          "metadata": {
            "title_main": "Nam Phong tạp chí",
            "title_alt": "Nam Phong Magazine",
            "volume": "1",
            "numero": "1",
            "publisher": "Imprimerie Tonkinoise",
            "publication_place": "Hanoi",
            "publication_date": "1917-07-01",
            "source_description": null,
            "authors": [
              "Phạm Quỳnh",
              "Nguyễn Bá Trác",
              "Tuyết Huy"
            ],
            "genres": [
              "journal",
              "essay",
              "literature"
            ],
            "issn": null,
            "identifiers": {}
          }
        },
        ...
      ]
      ```
    - Technical description:
      - The metadata is a list of dictionaries, each containing a "metadata" key with another dictionary as its value. The inner dictionary holds various metadata fields such as title, volume, publisher, publication date, authors, genres, etc.
    - OCR file name format: `Nam-Phong_So<volume>_<page number with 5 digits>.txt`
    - *Expected output (OCR text + metadata)*:
      - name format: `Nam-Phong_Q<volume>_S<numero>.json`
      - extract:
        ```json
        {
          "metadata": {
            <same metadata fields as above>
          },
          "text_body": {
            "page_1": [
              "<text from page 1>",
              "<line by line>",
            ],
            "page_2": [
              "<text from page 2>",
              "<line by line>",
              "<etc.>"
            ],
            ...
          }
        ```

2. **TAN-VIET (Gallica)**
    - Metadata file (CSV format): `data/Tan-Viet/Tan_Viet_metadata.csv`
    - Columns:
      | Column               | Description                                         |
      |----------------------|-----------------------------------------------------|
      | H-ID                 | Unique identifier for each entry                    |
      | title_main           | Main title of the work                              |
      | title_alt            | Alternative title (if any)                          |
      | author_id            | Unique identifier for the author                    |
      | author               | Name of the author                                  |
      | publisher_id         | Unique identifier for the publisher                 |
      | publisher            | Name of the publisher                               |
      | publication_place_id | Unique identifier for the place of publication      |
      | publication_place    | Name of the place of publication                    |
      | publication_date     | Date of publication                                 |
      | url1                 | URL to the BNF catalogue entry                      |
      | url2                 | URL to the Gallica entry                            |
      | genres               | Genres associated with the work                     |
      | PDF                  | Indicates if a PDF version is available (True/False)|
    - OCR file name decomposition:
      - e.g. `4394_Ba_muoi_nam_van_hoc_00001.txt`
      - Components:
        - `4394`: H-ID
        - `Ba_muoi_nam_van_hoc`: title_main with underscores instead of spaces and **diacritics removed**
          - original name: "Ba mươi năm văn học"
        - `00001`: page number with 5 digits
    - OCR file name format: `<H-ID>_<title_main normalized>_<page number>.txt`
    - Expected output (OCR text + metadata):
      - name format: `Tan-Viet_<H-ID>.json`
      - origin line: 4394;Ba mươi năm văn học;Trente années d'histoire littéraire, essai;2460;Kiều Thanh Quế;3303;Éditions Tân Việt;3216;Hà Nội;1942;http://catalogue.bnf.fr/ark:/12148/cb35755394h;https://gallica.bnf.fr/ark:/12148/bpt6k4239095c;;True
      - extract:
        ```json
        {
          "metadata": {
            "title_main": "Ba mươi năm văn học",
            "title_alt": "Trente années d'histoire littéraire, essai",
            "authors": [
              "Kiều Thanh Quế"
            ],
            "authors_id": [
              "2460"
            ],
            "publisher_id": "3303",
            "publisher": "Éditions Tân Việt",
            "publication_place_id": "3216",
            "publication_place": "Hà Nội",
            "publication_date": "1942",
            "source_description": null,
            "genres": [],
            "issn": null,
            "identifiers": {
              "H-ID": "4394",
              "url1": "http://catalogue.bnf.fr/ark:/12148/cb35755394h",
              "url2": "https://gallica.bnf.fr/ark:/12148/bpt6k4239095c"
            }
          },
          "text_body": {
            "page_1": [
              "<text from page 1>",
              "<line by line>",
              "<etc.>"
            ],
            ...
          }
        }
        ```

## OCR specificities

The OCRed text has some specific issues that need to be addressed during preprocessing:

- **Diacritics**: The OCR process may introduce errors in diacritics, which are crucial in Vietnamese.
- **Line breaks**: The OCRed text may contain irregular line breaks that do not correspond to actual paragraph breaks.
- **LaTeX and markdown artifacts**: Texts contain LaTeX or markdown formatting that needs to be cleaned.
- **Image references**: Some texts contain references to images or figures that are not part of the main text.
