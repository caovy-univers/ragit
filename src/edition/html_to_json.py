import json
import yaml
from bs4 import BeautifulSoup
from datetime import date
from loguru import logger
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import List, Optional


class Metadata(BaseModel):
    model_config = ConfigDict(json_encoders={date: lambda v: v.isoformat()})
    title_main: str
    title_alt: str
    publisher: Optional[str] = None
    publication_place: Optional[str] = None
    publication_date: date
    source_description: Optional[str] = None
    authors: List[str] = []
    genres: List[str] = []
    issn: Optional[str] = None
    identifiers: dict = {}


class Document(BaseModel):
    metadata: Metadata
    text_body: List[str]


def parse_yaml_metadata(yaml_file: Path) -> Metadata:
    """Parse YAML file to metadata object.

    Args:
        yaml_file (Path): Path to the YAML file.

    Returns:
        Metadata: Metadata as a Pydantic object.
    """
    with yaml_file.open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return Metadata(**data)


def html_to_json(html_path: Path, metadata: Metadata, output_path: Path) -> None:
    """Convert an HTML file to JSON using provided metadata.

    Args:
        html_path (Path): Path to the input HTML file.
        metadata (Metadata): Metadata object.
        output_path (Path): Path for the output JSON file.
    """
    # Parse HTML content
    with html_path.open('r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Extract paragraphs from HTML
    text_body = [p.get_text(strip=True) for p in soup.find_all('p')]

    # Create document object
    document = Document(metadata=metadata, text_body=text_body)

    # Save JSON
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(document.model_dump(mode="json"), file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    so1_folder = Path("data/Nam-Phong/Quyen-1/So-1")
    metadata_path = Path("data/Nam-Phong/Quyen-1/So-1/metadata.yaml")
    output_folder = Path("data/Nam-Phong/Quyen-1/So-1/output_json")

    metadata = parse_yaml_metadata(metadata_path)

    output_folder.mkdir(parents=True, exist_ok=True)

    for html_file in so1_folder.glob("*.html"):
        output_file = output_folder / f"{html_file.stem}.json"
        html_to_json(html_file, metadata, output_file)
        logger.info(f"Processed {html_file.name} -> {output_file.name}")
