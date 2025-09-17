import csv
import json
import re
import sys
from collections import defaultdict
from loguru import logger
from pathlib import Path
from typing import Dict, List, Optional, Tuple


NAM_PHONG_OCR_PATTERN = re.compile(r"Nam-Phong[-_]So(?P<issue>\d+)[-_](?P<page>\d{5})\.txt$")
TAN_VIET_OCR_PATTERN = re.compile(r"(?P<hid>\d+)_.*_(?P<page>\d{5})\.txt$")


def read_text_file_lines(path: Path) -> List[str]:
    """
    Read a UTF-8 text file and return its content split into lines.

    Args:
        path (Path): Path to the text file.

    Returns:
        List[str]: List of lines preserving original line breaks (without trailing newline chars).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def ensure_dir(path: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path (Path): Directory path to create if it does not exist.

    Returns:
        None
    """
    path.mkdir(parents=True, exist_ok=True)


def safe_json_dump(data: Dict, out_path: Path) -> None:
    """
    Write a Python dictionary as pretty-printed JSON (UTF-8, no ASCII escaping).

    Args:
        data (Dict): Data to serialize.
        out_path (Path): Destination JSON file path.

    Returns:
        None
    """
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_nam_phong_metadata(meta_path: Path) -> List[Dict]:
    """
    Parse Nam-Phong metadata JSON file.

    The input is a list of dicts, each containing a "metadata" key.

    Args:
        meta_path (Path): Path to metadata JSON.

    Returns:
        List[Dict]: Flat list of metadata dictionaries, one per issue.
    """
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    records: List[Dict] = []
    for item in raw:
        md = item.get("metadata", {})
        md.setdefault("authors", [])
        md.setdefault("genres", [])
        md.setdefault("identifiers", {})
        records.append(md)
    return records


def collect_nam_phong_pages(ocr_root: Path) -> Dict[int, List[Tuple[int, Path]]]:
    """
    Collect Nam-Phong OCR pages grouped by issue number (So<N>).

    Args:
        ocr_root (Path): Root directory containing OCR files.

    Returns:
        Dict[int, List[Tuple[int, Path]]]: Mapping issue -> list of (page_num, path) sorted by page number.
    """
    groups: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)
    for path in ocr_root.rglob("Nam-Phong*So*.txt"):
        m = NAM_PHONG_OCR_PATTERN.match(path.name)
        if not m:
            continue
        issue = int(m.group("issue"))
        page = int(m.group("page"))
        groups[issue].append((page, path))
    for issue, items in groups.items():
        items.sort(key=lambda x: x[0])
    return groups


def nam_phong_to_json(metadata: Dict, pages: List[Tuple[int, Path]]) -> Dict:
    """
    Build the Nam-Phong JSON structure for a single (volume, numero) using given pages.

    Args:
        metadata (Dict): Metadata dictionary for the issue.
        pages (List[Tuple[int, Path]]): List of (page_num, Path) tuples sorted by page number.

    Returns:
        Dict: Output JSON structure.
    """
    text_body: Dict[str, List[str]] = {}
    for idx, (page_num, path) in enumerate(pages, start=1):  # page_num kept for clarity
        key = f"page_{idx}"
        text_body[key] = read_text_file_lines(path)
    return {"metadata": metadata, "text_body": text_body}


def normalize_nam_phong(
    ocr_root: Path,
    metadata_path: Path,
    output_dir: Path,
) -> Tuple[int, int, int, List[str]]:
    """
    Normalize Nam-Phong corpus and save one JSON per (volume, numero).

    Args:
        ocr_root (Path): OCR root, containing Nam-Phong files.
        metadata_path (Path): Metadata JSON path.
        output_dir (Path): Output directory for normalized JSON.

    Returns:
        Tuple[int, int, int, List[str]]: (n_issues_saved, n_volumes, total_pages, errors)
    """
    logger.info("Normalization for Nam-Phong started.")
    ensure_dir(output_dir)

    md_list = parse_nam_phong_metadata(metadata_path)
    pages_by_issue = collect_nam_phong_pages(ocr_root)

    volumes = {md.get("volume") for md in md_list if md.get("volume") is not None}
    total_pages = sum(len(p) for p in pages_by_issue.values())

    logger.info(
        "Found {files} OCR files for Nam-Phong: {n_volumes} volumes, {n_pages} pages total.",
        files=sum(len(v) for v in pages_by_issue.values()),
        n_volumes=len(volumes),
        n_pages=total_pages,
    )

    errors: List[str] = []
    issues_saved = 0

    md_by_issue: Dict[int, Dict] = {}
    for md in md_list:
        numero_raw = md.get("numero")
        try:
            issue_num = int(str(numero_raw)) if numero_raw is not None else None
        except ValueError:
            issue_num = None
        if issue_num is None:
            errors.append(f"Metadata missing/invalid 'numero': {md!r}")
            continue
        if issue_num in md_by_issue:
            errors.append(f"Duplicate metadata for issue (numero) {issue_num}.")
            continue
        md_by_issue[issue_num] = md

    for issue in sorted(pages_by_issue.keys()):
        if issue not in md_by_issue:
            errors.append(f"OCR pages found for issue So{issue} but no matching metadata (numero={issue}).")

    for issue, md in md_by_issue.items():
        volume = md.get("volume")
        numero = md.get("numero")
        pages = pages_by_issue.get(issue, [])
        if not pages:
            errors.append(f"No OCR pages for Nam-Phong numero={numero} (volume={volume}). Skipped.")
            continue
        out_name = f"Nam-Phong_Q{volume}_S{numero}.json"
        out_path = output_dir / out_name
        output = nam_phong_to_json(md, pages)
        safe_json_dump(output, out_path)
        issues_saved += 1

    if errors:
        logger.warning("Nam-Phong encountered {n} issue(s):", n=len(errors), k=len(errors))
        for e in errors:
            logger.warning(" - {}", e)
    else:
        logger.info("Nam-Phong: no issues detected.")

    logger.success("{n} files for Nam-Phong saved to {dest}", n=issues_saved, dest=str(output_dir))
    return issues_saved, len(volumes), total_pages, errors


def parse_tan_viet_csv(csv_path: Path) -> Dict[str, Dict]:
    """
    Parse Tân-Việt CSV metadata into a dict keyed by H-ID.

    Args:
        csv_path (Path): Path to the CSV file.

    Returns:
        Dict[str, Dict]: Mapping H-ID -> normalized metadata dictionary matching the expected JSON schema.
    """
    records: Dict[str, Dict] = {}

    # Use utf-8-sig to strip a possible BOM from the header.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        # Simple, robust delimiter detection (prefers ';' when present)
        header = f.readline()
        delimiter = ";" if header.count(";") >= header.count(",") and header.count(";") > 0 else ","
        f.seek(0)

        reader = csv.DictReader(f, delimiter=delimiter)

        # Normalize header names (strip spaces); protects against stray spaces after delimiter.
        if reader.fieldnames:
            reader.fieldnames = [fn.strip() if fn else fn for fn in reader.fieldnames]

        for raw_row in reader:
            # Normalize keys and values (strip); handle missing keys safely
            row = { (k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                    for k, v in raw_row.items() }

            hid = (row.get("H-ID") or "").strip()
            if not hid:
                continue

            def split_maybe(s: Optional[str]) -> List[str]:
                # Support separators: ';' or '|' or ','; strip whitespace; drop empties
                if not s:
                    return []
                parts = re.split(r"[;|,]", s)
                return [p.strip() for p in parts if p.strip()]

            authors = split_maybe(row.get("author"))
            authors_id = split_maybe(row.get("author_id"))
            genres = split_maybe(row.get("genres"))

            md = {
                "title_main": (row.get("title_main") or ""),
                "title_alt": (row.get("title_alt") or "") or None,
                "authors": authors,
                "authors_id": authors_id,
                "publisher_id": (row.get("publisher_id") or "") or None,
                "publisher": (row.get("publisher") or ""),
                "publication_place_id": (row.get("publication_place_id") or "") or None,
                "publication_place": (row.get("publication_place") or ""),
                "publication_date": (row.get("publication_date") or ""),
                "source_description": None,
                "genres": genres,
                "issn": None,
                "identifiers": {
                    "H-ID": hid,
                    "url1": (row.get("url1") or "") or None,
                    "url2": (row.get("url2") or "") or None
                },
            }
            records[hid] = md

    return records

def collect_tan_viet_pages(ocr_root: Path) -> Dict[str, List[Tuple[int, Path]]]:
    """
    Collect Tân-Việt OCR pages grouped by H-ID.

    Args:
        ocr_root (Path): Root directory containing OCR files.

    Returns:
        Dict[str, List[Tuple[int, Path]]]: Mapping H-ID -> list of (page_num, path) sorted by page number.
    """
    groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for path in ocr_root.rglob("[0-9]*_*.txt"):
        m = TAN_VIET_OCR_PATTERN.match(path.name)
        if not m:
            continue
        hid = m.group("hid")
        page = int(m.group("page"))
        groups[hid].append((page, path))
    for hid, items in groups.items():
        items.sort(key=lambda x: x[0])
    return groups


def tan_viet_to_json(metadata: Dict, pages: List[Tuple[int, Path]]) -> Dict:
    """
    Build the Tân-Việt JSON structure for a single H-ID.

    Args:
        metadata (Dict): Metadata dictionary for the work.
        pages (List[Tuple[int, Path]]): List of (page_num, Path) tuples sorted by page number.

    Returns:
        Dict: Output JSON structure.
    """
    text_body: Dict[str, List[str]] = {}
    for idx, (page_num, path) in enumerate(pages, start=1):  # page_num kept for clarity
        key = f"page_{idx}"
        text_body[key] = read_text_file_lines(path)
    return {"metadata": metadata, "text_body": text_body}


def normalize_tan_viet(
    ocr_root: Path,
    metadata_csv: Path,
    output_dir: Path,
) -> Tuple[int, int, int, List[str]]:
    """
    Normalize Tân-Việt corpus and save one JSON per H-ID.

    Args:
        ocr_root (Path): OCR root, containing Tân-Việt files.
        metadata_csv (Path): CSV metadata path.
        output_dir (Path): Output directory for normalized JSON.

    Returns:
        Tuple[int, int, int, List[str]]: (n_works_saved, n_works_found, total_pages, errors)
    """
    logger.info("Normalization for Tan-Viet started.")
    ensure_dir(output_dir)

    md_by_hid = parse_tan_viet_csv(metadata_csv)
    pages_by_hid = collect_tan_viet_pages(ocr_root)

    works_found = len(pages_by_hid)
    total_pages = sum(len(p) for p in pages_by_hid.values())

    logger.info(
        "Found {works} works for Tan-Viet: {works} documents, {n_pages} pages total.",
        works=works_found,
        n_pages=total_pages,
    )

    errors: List[str] = []
    works_saved = 0

    for hid in sorted(pages_by_hid.keys()):
        if hid not in md_by_hid:
            errors.append(f"OCR pages found for H-ID={hid} but no metadata row in CSV.")

    for hid, md in md_by_hid.items():
        pages = pages_by_hid.get(hid, [])
        if not pages:
            errors.append(f"No OCR pages for H-ID={hid} ({md.get('title_main')}). Skipped.")
            continue

        out_name = f"Tan-Viet_{hid}.json"
        out_path = output_dir / out_name
        output = tan_viet_to_json(md, pages)
        safe_json_dump(output, out_path)
        works_saved += 1

    if errors:
        logger.warning("Tan-Viet encountered {n} issues:", n=len(errors), k=len(errors))
        for e in errors:
            logger.warning(" - {}", e)
    else:
        logger.info("Tan-Viet: no issues detected.")

    logger.success("{n} files for Tan-Viet saved to {dest}", n=works_saved, dest=str(output_dir))
    return works_saved, works_found, total_pages, errors


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
    )

    base = Path("data")
    ocr_root = base / "OCR"

    nam_phong_meta = base / "Nam-Phong" / "metadata.json"
    nam_phong_out = base / "Nam-Phong" / "output"

    tan_viet_meta = base / "Tan-Viet" / "Tan_Viet_metadata.csv"
    tan_viet_out = base / "Tan-Viet" / "output"

    # 1. Nam-Phong
    normalize_nam_phong(
        ocr_root=ocr_root,
        metadata_path=nam_phong_meta,
        output_dir=nam_phong_out,
    )

    # 2. Tan-Viet
    normalize_tan_viet(
        ocr_root=ocr_root,
        metadata_csv=tan_viet_meta,
        output_dir=tan_viet_out,
    )
