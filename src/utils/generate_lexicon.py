import csv
import re
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Iterable, List, Optional, Tuple

# Regular expressions

# Any CJK (Hán/Nôm)
CJK_RE = re.compile(r"[\u3400-\u9FFF\uF900-\uFAFF]")

# Roman page markers like “– IX –”
ROMAN_HEADER_RE = re.compile(r"^\s*[-–—]*\s*[IVXLCDM]+\s*[-–—]*\s*$", re.MULTILINE)

# Hyphenation across line breaks: "RELA-\n TION" -> "RELATION"
HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-zÀ-Ỵà-ỵ])-\s*\n\s*([A-Za-zÀ-Ỵà-ỵ])")

# First contiguous run of CJK, allowing inner spaces (e.g. "古 典 學 問")
HANI_RUN_RE = re.compile(r"([\u3400-\u9FFF\uF900-\uFAFF][\u3400-\u9FFF\uF900-\uFAFF\s]*)")

# Optional leading bullet (for cleaning headword)
HEAD_BULLET = r"[-–—•*·]"

# Strip code-fence/artifact runs like ``` or `````` anywhere
BACKTICKS_RE = re.compile(r"`{3,}")

# Trailing punctuation to strip from 'vi' (end of the cell only).
# Includes spaces, ASCII/Unicode punctuation, dashes and '='.
TRAIL_PUNCT_RE = re.compile(r"""[\s\.,;:!?\u2026"'“”‘’»«\)\]\}\-–—=。．，、：；！？”’]+$""")

# Collapse spaces around hyphens inside 'vi'
HYPHEN_SPACES_RE = re.compile(r"\s*-\s*")

# Vietnamese-specific letters (to tell VI vs FR in terminal-sentence fallback)
VIET_DIACRITIC_RE = re.compile(
    r"[àáảãạâấầẩẫậăắằẳẵặèéẻẽẹêếềểễệìíỉĩị"
    r"òóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊ"
    r"ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]"
)

# Data model

class LexiconEntry(BaseModel):
    """Structured lexicon record with required columns."""
    vi: str
    vi_hani: str
    vi_def: str
    fr: str


# Core parsing

def normalize_raw_pages(pages: List[str]) -> str:
    """
    Concatenate pages and normalize obvious noise (page markers, hyphenation).

    Important: do not inject artificial separators except blank lines;
    entries may legitimately continue across page boundaries.

    Args:
        pages (List[str]): Plain text pages for one volume.

    Returns:
        str: Normalized multi-line text.
    """
    # Join pages with exactly one blank line so we don't force splits
    text = ("\n\n").join(pages)

    # Remove roman headers like "- IX -"
    text = ROMAN_HEADER_RE.sub("", text)

    # Fix cross-line hyphenations introduced by wrapping.
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # Drop code-fence artifacts like ```
    text = BACKTICKS_RE.sub(" ", text)

    return text


def split_blocks_on_blank_lines(text: str) -> List[str]:
    """
    Split the volume into blocks separated by one or more blank lines.

    Args:
        text (str): Normalized multi-line text.

    Returns:
        List[str]: Blocks (raw multi-line), including continuation lines.
    """
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    normalized = "\n".join(lines).strip()
    raw_blocks = re.split(r"\n\s*\n+", normalized)
    return [b.strip() for b in raw_blocks if b.strip()]


def normalize_chunk(one: str) -> str:
    """
    Normalize a chunk to one line without losing content:
    - Join lines with single spaces (preserve words).
    - Normalize em/en dashes to ' - ' (punctuation separators).
    - Normalize fullwidth equals to ' = '.
    - Normalize double hyphen separators ' -- ' to ' - ' (to catch FR glosses).
    - Do NOT insert spaces around ASCII '-' inside words (keep 'Anh-hùng-ca').
    - Collapse extra spaces.

    Args:
        one (str): Raw multi-line chunk.

    Returns:
        str: Single-line normalized chunk.
    """
    oneline = " ".join(s.strip() for s in one.splitlines() if s.strip())
    # Normalize dash/equals separators
    oneline = re.sub(r"\s*[—–]\s*", " - ", oneline)   # em/en dash -> spaced hyphen
    oneline = re.sub(r"\s*＝\s*", " = ", oneline)     # fullwidth equals
    oneline = re.sub(r"\s*--\s*", " - ", oneline)     # double hyphen -> spaced hyphen
    # Collapse multi-spaces
    oneline = re.sub(r"\s+", " ", oneline).strip()
    return oneline


def extract_headword(left_of_hani: str) -> str:
    """
    Extract the headword (vi) from the segment before the first CJK character.

    Strategy:
    - Remove optional leading bullet ("- ", "– ", "— ", "• ", etc.).
    - Prefer everything up to the first '.' (common in sources).
    - If there is no '.', cut at the earliest of ' = ' or ' - ' if present.
    - Otherwise take the whole left segment.
    - Normalize hyphen spacing to 'term-term' (no spaces around '-').

    Args:
        left_of_hani (str): Text before the first CJK char.

    Returns:
        str: Headword (vi).
    """
    s = re.sub(rf"^\s*(?:{HEAD_BULLET})\s*", "", left_of_hani).strip()

    dot = s.find(".")
    if dot != -1:
        s = s[:dot]
    else:
        cut_positions: List[int] = []
        for sep in (" = ", " - "):
            pos = s.find(sep)
            if pos != -1:
                cut_positions.append(pos)
        if cut_positions:
            s = s[: min(cut_positions)]

    s = s.strip(" .:;!)?—–-")
    s = re.sub(r"\s*-\s*", "-", s)  # "Anh - hùng - ca" -> "Anh-hùng-ca"
    return s


def split_vi_def_fr(after_hani: str) -> Tuple[str, str]:
    """
    Split the tail after Hán-Nôm into Vietnamese definition and French gloss.

    Primary strategy:
    - Trim a leading separator ( '=' or '-' ).
    - Use the last ' - ' as the FR separator (we normalized '—/–' and ' -- ').

    Args:
        after_hani (str): Text after the Hán-Nôm cluster (single-line).

    Returns:
        Tuple[str, str]: (vi_def, fr)
    """
    s = after_hani.strip()
    s = s.lstrip("=- ").strip()
    last_sep = s.rfind(" - ")
    if last_sep == -1:
        return s.strip().strip(" .-—"), ""
    vi_def = s[:last_sep].strip().strip(" .-—")
    fr = s[last_sep + 3 :].strip().strip(" .-—")
    return vi_def, fr


def fallback_fr_by_linebreak(raw_block: str) -> Tuple[str, str]:
    """
    Fallback: infer FR from trailing non-CJK lines when no dash separator is present.

    Heuristic:
    - Find the first Hán-Nôm run in the RAW multi-line block.
    - Consider the text *after* that run, split by lines.
    - From the bottom, gather consecutive lines that:
        * contain NO CJK, and
        * do NOT contain '=' (which usually introduces the vi-def).
      These lines are treated as the French gloss.
    - The remaining upper part is the Vietnamese definition.

    This captures cases like:
        Tiềm-lực.
        潛力 = Sức ngầm, không phát rõ ra ngoài.
        Force latente, cachée; virtualité.

    Args:
        raw_block (str): Original multi-line block.

    Returns:
        Tuple[str, str]: (vi_def, fr) or ("", "") if not applicable.
    """
    m = HANI_RUN_RE.search(raw_block)
    if not m:
        return "", ""

    raw_tail = raw_block[m.end():]
    lines = [ln.strip() for ln in raw_tail.splitlines() if ln.strip()]
    if not lines:
        return "", ""

    french_rev: List[str] = []
    # Accumulate from the bottom while the line is non-CJK and not an equals-bearing line
    for ln in reversed(lines):
        if CJK_RE.search(ln) or "=" in ln:
            break
        french_rev.append(ln)

    if not french_rev:
        return "", ""

    # Remaining lines (upper part) belong to vi-def
    keep_upto = len(lines) - len(french_rev)
    vi_def_lines = lines[:keep_upto]
    fr_lines = list(reversed(french_rev))

    vi_def = " ".join(vi_def_lines).strip().strip(" .-—")
    fr = " ".join(fr_lines).strip().strip(" .-—")
    return vi_def, fr

def fallback_fr_by_terminal_clause(after_hani_oneline: str) -> Tuple[str, str]:
    """
    Fallback: capture FR when it sits at the very end of the line, with or without
    a final period, and without an explicit dash separator.

    Strategy:
    - Work on the 'right' segment (after the Hanzi) as a single line.
    - Prefer splitting at the last '.'; if none, try after the last '='.
    - The trailing chunk is accepted as FR if:
        * it contains no CJK,
        * it contains no Vietnamese diacritics,
        * and it has at least one Latin letter.
    - The leading chunk then becomes vi-def.
    """
    s = after_hani_oneline.strip()
    if not s:
        return "", ""

    # 1) Prefer last '.' as a sentence boundary
    last_dot = s.rfind(".")
    split_pos: Optional[int] = None
    if last_dot != -1 and last_dot < len(s) - 1:
        split_pos = last_dot + 1
    else:
        # 2) Fallback: split right after the last '=' if present
        last_eq = s.rfind("=")
        if last_eq != -1 and last_eq < len(s) - 1:
            split_pos = last_eq + 1

    if split_pos is None:
        return "", ""

    cand_raw = s[split_pos:].strip()
    cand = cand_raw.strip(" .-—;:!?)“”’'")

    # Reject if it looks like Vietnamese or contains CJK
    if not cand or CJK_RE.search(cand) or VIET_DIACRITIC_RE.search(cand):
        return "", ""

    # Require at least one Latin letter (incl. accents in FR)
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", cand):
        return "", ""

    vi_def = s[:split_pos].strip().strip(" .-—;:,")
    fr = cand
    return vi_def, fr

def parse_entry(raw_block: str, oneline: str) -> Optional[LexiconEntry]:
    """
    Parse a block into (vi, vi-hani, vi-def, fr) with fallback for FR-on-newline.

    Args:
        raw_block (str): Original multi-line block (preserves line breaks).
        oneline (str): Normalized single-line version for primary parsing.

    Returns:
        Optional[LexiconEntry]: Structured entry or None if parsing fails.
    """
    m_hani = HANI_RUN_RE.search(oneline)
    if not m_hani:
        return None

    left = oneline[: m_hani.start()]
    right = oneline[m_hani.end() :]

    vi = extract_headword(left)
    vi_hani = re.sub(r"\s+", "", m_hani.group(1))  # remove inner spaces in Hanzi

    # Primary split (dash-based)
    vi_def, fr = split_vi_def_fr(right)

    # Fallback 1: FR on subsequent line(s) without CJK or '='
    if fr == "":
        alt_vi_def, alt_fr = fallback_fr_by_linebreak(raw_block)
        if alt_fr:
            vi_def, fr = alt_vi_def, alt_fr

    # Fallback 2: FR as terminal clause on the same line (with/without final '.')
    if fr == "":
        alt_vi_def2, alt_fr2 = fallback_fr_by_terminal_clause(right)
        if alt_fr2:
            vi_def, fr = alt_vi_def2, alt_fr2

    return LexiconEntry(vi=vi, vi_hani=vi_hani, vi_def=vi_def, fr=fr)


def parse_volume(pages: List[str]) -> List[LexiconEntry]:
    """
    Parse all pages of one volume into a list of entries using blank-line blocks.

    Steps:
    - Normalize pages (remove roman headers, fix hyphenation artifacts).
    - Split into blocks on one or more blank lines.
    - Drop blocks without any CJK (intro/outro prose).
    - For each block, create both oneline and raw forms to enable fallbacks.

    Args:
        pages (List[str]): List of page texts in order.

    Returns:
        List[LexiconEntry]: Parsed entries for the volume.
    """
    text = normalize_raw_pages(pages)
    blocks = split_blocks_on_blank_lines(text)

    entries: List[LexiconEntry] = []
    for raw in blocks:
        if not CJK_RE.search(raw):
            continue
        one = normalize_chunk(raw)
        e = parse_entry(raw, one)
        if e:
            entries.append(e)
    return entries

def load_pages_for_volume(base_dir: Path, volume: int) -> List[str]:
    """
    Load page files for one 'SoN' volume in numeric order.

    Filenames are expected like: Nam-Phong-So{N}-Lexicon_page_{k}.txt

    Args:
        base_dir (Path): Directory containing all TXT files.
        volume (int): Volume number (1..14).

    Returns:
        List[str]: Page texts ordered by page index.
    """
    pattern = f"Nam-Phong-So{volume}-Lexicon_page_*.txt"
    files = sorted(
        base_dir.glob(pattern),
        key=lambda p: int(re.search(r"_page_(\d+)\.txt$", p.name).group(1))
        if re.search(r"_page_(\d+)\.txt$", p.name)
        else 0,
    )
    return [fp.read_text(encoding="utf-8", errors="ignore") for fp in files]


def write_csv(entries: List[LexiconEntry], out_path: Path) -> None:
    """
    Write entries to CSV with headers: vi, vi-hani, vi-def, fr.

    Args:
        entries (List[LexiconEntry]): Parsed entries.
        out_path (Path): Destination CSV path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["vi", "vi-hani", "vi-def", "fr"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for e in entries:
            writer.writerow({"vi": e.vi, "vi-hani": e.vi_hani, "vi-def": e.vi_def, "fr": e.fr})


# Post-processing (clean + QA)

def clean_vi_field(vi: str) -> str:
    """
    Clean the 'vi' (headword) field:
    - Trim whitespace.
    - Collapse spaces around hyphens to "term-term".
    - Remove trailing punctuation characters (spaces + punctuation, dashes, '=').
    """
    s = (vi or "").strip()
    s = HYPHEN_SPACES_RE.sub("-", s)  # "Anh - hùng - ca" -> "Anh-hùng-ca"
    prev = None
    while prev != s:
        prev = s
        s = TRAIL_PUNCT_RE.sub("", s).strip()
    return s


def row_missing_fields(row: Dict[str, str], fields: List[str]) -> List[str]:
    """
    Return the list of column names that are empty (after stripping).
    """
    missing: List[str] = []
    for col in fields:
        val = row.get(col, "")
        if val is None or str(val).strip() == "":
            missing.append(col)
    return missing


def process_csv_file(path: Path) -> Tuple[int, int, List[Tuple[int, List[str]]]]:
    """
    Clean 'vi' field and detect incomplete rows for a single CSV.

    Returns:
        total_rows, fixed_vi, issues (list of (1-based row index, missing_cols[])).
    """
    rows: List[Dict[str, str]] = []
    fixed_vi = 0
    issues: List[Tuple[int, List[str]]] = []
    fields = ["vi", "vi-hani", "vi-def", "fr"]

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            vi_raw = row.get("vi", "")
            vi_new = clean_vi_field(vi_raw)
            if vi_new != vi_raw:
                row["vi"] = vi_new
                fixed_vi += 1

            missing_cols = row_missing_fields(row, fields)
            if missing_cols:
                issues.append((i, missing_cols))

            rows.append({k: row.get(k, "") for k in fields})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return len(rows), fixed_vi, issues


def postprocess_csv_dir(csv_dir: Path) -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[Tuple[int, List[str]]]]]:
    """
    Run post-processing on all So*.csv in a directory:
    - Clean trailing punctuation in 'vi'.
    - Collect QA per file (rows with missing cells).

    Returns:
        (per_file_counts, per_file_issues)
    """
    per_file_issues: Dict[str, List[Tuple[int, List[str]]]] = {}
    per_file_counts: Dict[str, Dict[str, int]] = {}

    for path in sorted(csv_dir.glob("So*.csv")):
        total_rows, fixed_vi, issues = process_csv_file(path)
        per_file_issues[path.name] = issues
        per_file_counts[path.name] = {
            "rows": total_rows,
            "vi_fixed": fixed_vi,
            "incomplete_rows": len(issues),
        }

    # Write QA CSVs next to outputs
    report_path = csv_dir / "qa_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "row_index", "missing_columns"])
        for fname, issues in sorted(per_file_issues.items()):
            for row_idx, cols in issues:
                writer.writerow([fname, row_idx, ";".join(cols)])

    summary_path = csv_dir / "qa_summary.csv"
    total_rows = sum(v["rows"] for v in per_file_counts.values())
    total_fixed = sum(v["vi_fixed"] for v in per_file_counts.values())
    total_incomplete = sum(v["incomplete_rows"] for v in per_file_counts.values())
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "rows", "vi_fixed", "incomplete_rows"])
        for fname, stats in sorted(per_file_counts.items()):
            writer.writerow([fname, stats["rows"], stats["vi_fixed"], stats["incomplete_rows"]])
        writer.writerow([])
        writer.writerow(["TOTAL", total_rows, total_fixed, total_incomplete])

    return per_file_counts, per_file_issues

def build_all_volumes(
    base_dir: Path = Path("data/Nam-Phong/lexicon"),
    out_dir: Path = Path("data/Nam-Phong/lexicon/csv"),
    volumes: Iterable[int] = range(1, 15),
) -> Dict[int, int]:
    """
    Parse all volumes So01..So14 and write SoXX.csv files.

    Args:
        base_dir (Path): Base directory with TXT files.
        out_dir (Path): Output directory for CSVs.
        volumes (Iterable[int]): Volume numbers to process.

    Returns:
        Dict[int, int]: Mapping {volume_number: number_of_entries_written}.
    """
    stats: Dict[int, int] = {}
    for v in volumes:
        pages = load_pages_for_volume(base_dir, v)
        if not pages:
            continue
        entries = parse_volume(pages)
        out_path = out_dir / f"So{v:02d}.csv"
        write_csv(entries, out_path)
        stats[v] = len(entries)
    return stats


def main() -> None:
    """
    Entry point: build So01..So14 into data/Nam-Phong/lexicon/csv/SoXX.csv,
    then run post-processing (clean vi + QA reporting) on generated CSVs.
    """
    base_dir = Path("data/Nam-Phong/lexicon")
    out_dir = Path("data/Nam-Phong/lexicon/csv")

    stats = build_all_volumes(base_dir=base_dir, out_dir=out_dir)
    for v in sorted(stats):
        print(f"So{v:02d}: {stats[v]} entries")

    # Post-correction + QA
    per_file_counts, _ = postprocess_csv_dir(out_dir)

    # Console summary
    print("\nQA summary per file:")
    for fname, s in sorted(per_file_counts.items()):
        print(f"- {fname}: rows={s['rows']}, vi_fixed={s['vi_fixed']}, incomplete_rows={s['incomplete_rows']}")
    total_rows = sum(v["rows"] for v in per_file_counts.values())
    total_fixed = sum(v["vi_fixed"] for v in per_file_counts.values())
    total_incomplete = sum(v["incomplete_rows"] for v in per_file_counts.values())
    print(f"TOTAL: rows={total_rows}, vi_fixed={total_fixed}, incomplete_rows={total_incomplete}")


if __name__ == "__main__":
    main()
