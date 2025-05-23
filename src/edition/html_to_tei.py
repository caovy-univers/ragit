import re
import yaml
from pathlib import Path
from lxml import etree as ET
from typing import Dict, Any
from bs4 import BeautifulSoup
from loguru import logger


def parse_yaml_metadata(yaml_file: Path) -> Dict[str, Any]:
    """Parse YAML file to metadata dictionary.

    Args:
        yaml_file (Path): Path to the YAML file.

    Returns:
        Dict[str, Any]: Metadata as dictionary.
    """
    with yaml_file.open('r', encoding='utf-8') as file:
        metadata = yaml.safe_load(file)
    return metadata


def html_to_tei(html_path: Path, metadata: Dict[str, Any], output_path: Path) -> None:
    """Convert an HTML file to TEI-XML using provided metadata.

    Args:
        html_path (Path): Path to the input HTML file.
        metadata (Dict[str, Any]): Metadata dictionary.
        output_path (Path): Path for the output TEI file.
    """
    # Parse HTML content
    with html_path.open('r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Create TEI root
    tei_root = ET.Element("TEI", xmlns="http://www.tei-c.org/ns/1.0")

    # TEI header
    tei_header = ET.SubElement(tei_root, "teiHeader")
    file_desc = ET.SubElement(tei_header, "fileDesc")
    title_stmt = ET.SubElement(file_desc, "titleStmt")

    ET.SubElement(title_stmt, "title", type="main", xml_lang="vi").text = metadata.get("title_vi", "")
    ET.SubElement(title_stmt, "title", type="alt", xml_lang="en").text = metadata.get("title_en", "")

    publication_stmt = ET.SubElement(file_desc, "publicationStmt")
    ET.SubElement(publication_stmt, "publisher").text = metadata.get("publisher", "")
    ET.SubElement(publication_stmt, "pubPlace").text = metadata.get("pubPlace", "")
    ET.SubElement(publication_stmt, "date").text = metadata.get("date", "")

    source_desc = ET.SubElement(file_desc, "sourceDesc")
    ET.SubElement(source_desc, "p").text = metadata.get("source_desc", "")

    # Text body
    text = ET.SubElement(tei_root, "text")
    body = ET.SubElement(text, "body")

    # Extract paragraphs from HTML
    for paragraph in soup.find_all('p'):
        p = ET.SubElement(body, "p")
        p.text = paragraph.get_text(strip=True)

    # Save TEI XML
    tree = ET.ElementTree(tei_root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True, pretty_print=True)

if __name__ == "__main__":
    so1_folder = Path("data/Nam-Phong/Quyen-1/So-1")
    metadata_path = Path("data/Nam-Phong/Quyen-1/So-1/metadata.yaml")
    output_folder = Path("data/Nam-Phong/Quyen-1/So-1/output")

    metadata = parse_yaml_metadata(metadata_path)

    output_folder.mkdir(parents=True, exist_ok=True)

    for html_file in so1_folder.glob("*.html"):
        output_file = output_folder / f"{html_file.stem}.xml"
        html_to_tei(html_file, metadata, output_file)
        logger.info(f"Processed {html_file.name} -> {output_file.name}")
