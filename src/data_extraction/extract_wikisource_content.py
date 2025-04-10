import requests
from bs4 import BeautifulSoup
from loguru import logger
from pathlib import Path


def download_html(url: str, output_path: Path) -> None:
    """
    Downloads the HTML content of a given URL and saves it to the specified file path.

    Args:
        url (str): URL of the page to download.
        output_path (Path): Path to save the HTML content.
    """
    if output_path.exists():
        logger.info(f"HTML file already exists: {output_path}")
        return

    response = requests.get(url)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding='utf-8')
    logger.success(f"HTML downloaded and saved to: {output_path}")


def extract_text_from_html(html_path: Path, text_output_path: Path) -> None:
    """
    Extracts text from a saved HTML file and saves it to a text file.

    Args:
        html_path (Path): Path to the saved HTML file.
        text_output_path (Path): Path to save the extracted text.
    """
    html_content = html_path.read_text(encoding='utf-8')
    soup = BeautifulSoup(html_content, 'html.parser')

    main_div = soup.find('div', class_='mw-parser-output')
    if not main_div:
        logger.error("Main content not found in HTML.")
        return

    text = main_div.get_text(separator='\n', strip=True)
    text_output_path.write_text(text, encoding='utf-8')
    logger.success(f"Extracted text saved to: {text_output_path}")


if __name__ == "__main__":
    url = "https://vi.m.wikisource.org/wiki/Nam_Phong_t%E1%BA%A1p_ch%C3%AD/Quy%E1%BB%83n_I/S%E1%BB%91_1/T%E1%BB%B1-v%E1%BB%B1ng"
    output_folder = Path("data/Nam-Phong/Quyen-1/So-1/")
    html_file = output_folder / "Tu-Vung.html"
    text_file = output_folder / "Tu-Vung.txt"

    # Download and save HTML only if not already present
    download_html(url, html_file)

    # Extract and save text
    extract_text_from_html(html_file, text_file)
