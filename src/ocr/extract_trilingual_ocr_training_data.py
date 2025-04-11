import pandas as pd
import re
from bs4 import BeautifulSoup
from pathlib import Path

from data_extraction.extract_wikisource_content import download_html

def extract_trilingual_definitions(html_path: Path) -> pd.DataFrame:
    """
    Extracts structured trilingual definitions from HTML tables in a Wikisource HTML file.

    Args:
        html_path (Path): Path to the HTML file to parse.
        output_folder (Path): Base folder where output text files will be stored.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted definitions with columns 'term', 'han', 'vi', and 'fr'.
    """
    html_content = html_path.read_text(encoding='utf-8')
    soup = BeautifulSoup(html_content, 'html.parser')

    # Iterate over all tables with <tr> elements inside
    tables = soup.find_all('table')
    rows = [row.find_all('td') for table in tables for row in table.find_all('tr')]

    all_data = []
    for row in rows:
        if len(row) == 2:
            term = row[0].get_text(strip=True)
            raw_html = row[1].decode_contents()
            clean_text = re.sub(r'<.*?>', '', raw_html)

            match = re.match(r'(—\s*.+?)\s*(＝)\s*(.+?)\s*(—)\s*(.+)', clean_text)
            if match:
                han = match.group(1).strip()
                vi = f"= {match.group(3).strip()}"
                fr = f"— {match.group(5).strip()}"
                all_data.append({"term": term, "han": han, "vi": vi, "fr": fr})

    return pd.DataFrame(all_data)


if __name__ == "__main__":
    url = "https://vi.m.wikisource.org/wiki/Nam_Phong_t%E1%BA%A1p_ch%C3%AD/Quy%E1%BB%83n_I/S%E1%BB%91_1/T%E1%BB%B1-v%E1%BB%B1ng"
    output_folder = Path("data/Nam-Phong/Quyen-1/So-1/")
    html_file = output_folder / "Tu-Vung.html"

    # Download HTML if not already present
    download_html(url, html_file)

    # Extract trilingual definitions
    df = extract_trilingual_definitions(html_file, output_folder)
    # Save the DataFrame to a CSV file
    output_csv = output_folder / "trilingual_definitions.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8')
