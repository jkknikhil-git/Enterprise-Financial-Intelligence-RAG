"""
parser.py
---------
Table-aware document parser for SEC EDGAR filings.

EDGAR filings are primarily HTML (not PDF). This parser handles:
  - full-submission.txt  (SGML wrapper — what sec-edgar-downloader v5.x downloads)
  - .htm / .html         (standalone HTML filings)
  - .pdf                 (fallback)

For full-submission.txt the primary document HTML is extracted from the
SGML structure before parsing. All paths converge on the same HTML parser.

Output per filing:
    data/processed/{ticker}_{filing_type}_{accession}.json
    A JSON list of elements: {element_id, content, content_type, metadata}

Usage:
    python -m src.ingestion.parser
    python -m src.ingestion.parser --tickers AAPL MSFT
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from loguru import logger
from tqdm import tqdm

# ── Optional: unstructured ────────────────────────────────────────────────────
try:
    from unstructured.partition.html import partition_html
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("unstructured not importable — using BeautifulSoup-only extraction")

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT_DIR      = Path(__file__).resolve().parents[2]
RAW_DIR       = ROOT_DIR / "data" / "raw" / "sec-edgar-filings"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Skip these when looking for standalone .htm files — they are not the primary doc
HTM_SKIP_FILENAMES: set[str] = {
    "filing-details.htm",
    "filing-details.html",
}

MIN_TABLE_CELLS: int = 6
MIN_TEXT_LENGTH: int = 30


# ── Public API ────────────────────────────────────────────────────────────────

def parse_all_filings(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = PROCESSED_DIR,
    tickers: list[str] | None = None,
) -> dict[str, int]:
    """
    Parse all downloaded EDGAR filings and write structured JSON to output_dir.

    Args:
        raw_dir:    Root of sec-edgar-filings/ (created by edgar_fetcher.py).
        output_dir: Destination for processed JSON files.
        tickers:    Limit to these tickers. None = all available.

    Returns:
        Dict mapping ticker -> number of filings successfully parsed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        logger.info("Run: python -m src.ingestion.edgar_fetcher")
        return {}

    ticker_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if tickers:
        ticker_dirs = [d for d in ticker_dirs if d.name in tickers]

    if not ticker_dirs:
        logger.warning("No ticker directories found. Has edgar_fetcher.py been run?")
        return {}

    logger.info(f"Parsing filings for: {[d.name for d in ticker_dirs]}")
    results: dict[str, int] = {}

    for ticker_dir in tqdm(ticker_dirs, desc="Companies", unit="company"):
        ticker       = ticker_dir.name
        parsed_count = 0

        for filing_type_dir in sorted(ticker_dir.iterdir()):
            if not filing_type_dir.is_dir():
                continue

            for accession_dir in sorted(filing_type_dir.iterdir()):
                if not accession_dir.is_dir():
                    continue

                output_path = output_dir / (
                    f"{ticker}_{filing_type_dir.name}_{accession_dir.name}.json"
                )

                if output_path.exists():
                    logger.debug(f"[{ticker}] Already processed: {accession_dir.name}")
                    parsed_count += 1
                    continue

                try:
                    elements = parse_filing(accession_dir)
                    if elements:
                        output_path.write_text(
                            json.dumps(elements, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        logger.success(
                            f"[{ticker}] {filing_type_dir.name} {accession_dir.name}"
                            f" -> {len(elements)} elements"
                        )
                        parsed_count += 1
                    else:
                        logger.warning(
                            f"[{ticker}] {accession_dir.name} — zero elements extracted"
                        )
                except Exception as exc:
                    logger.error(f"[{ticker}] {accession_dir.name} — {exc}")

        results[ticker] = parsed_count

    _log_summary(results)
    return results


def parse_filing(filing_dir: Path) -> list[dict[str, Any]]:
    """
    Parse a single EDGAR accession directory into structured elements.

    Args:
        filing_dir: e.g. data/raw/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/

    Returns:
        List of element dicts with content, content_type, and metadata.
    """
    metadata    = _extract_metadata(filing_dir)
    primary_doc = _find_primary_document(filing_dir)

    if primary_doc is None:
        logger.warning(f"No parseable document in {filing_dir.name}")
        return []

    size_kb = primary_doc.stat().st_size // 1024
    logger.debug(f"Parsing {primary_doc.name} ({size_kb} KB) for {metadata['ticker']}")

    # Route to the appropriate parser
    if primary_doc.name == "full-submission.txt":
        return _parse_full_submission(primary_doc, metadata)

    suffix = primary_doc.suffix.lower()
    if suffix in {".htm", ".html"}:
        return _parse_html_file(primary_doc, metadata)
    elif suffix == ".pdf":
        return _parse_pdf(primary_doc, metadata)
    else:
        logger.warning(f"Unsupported file type: {suffix}")
        return []


# ── File discovery ────────────────────────────────────────────────────────────

def _find_primary_document(filing_dir: Path) -> Path | None:
    """
    Locate the primary readable document in an EDGAR accession directory.

    Priority:
      1. primary-document.htm / .html  (ideal — sec-edgar-downloader naming)
      2. Largest .htm / .html file     (excluding filing-details pages)
      3. Largest .pdf file
      4. full-submission.txt           (SGML wrapper — fallback for v5.x downloads)
    """
    # 1. Explicit primary-document file
    for name in ("primary-document.htm", "primary-document.html"):
        candidate = filing_dir / name
        if candidate.exists():
            return candidate

    # 2. Largest standalone .htm / .html
    htm_candidates = [
        f for f in (list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html")))
        if f.name not in HTM_SKIP_FILENAMES
    ]
    if htm_candidates:
        return max(htm_candidates, key=lambda f: f.stat().st_size)

    # 3. Largest PDF
    pdf_candidates = list(filing_dir.glob("*.pdf"))
    if pdf_candidates:
        return max(pdf_candidates, key=lambda f: f.stat().st_size)

    # 4. full-submission.txt — contains embedded HTML inside SGML wrapper
    full_sub = filing_dir / "full-submission.txt"
    if full_sub.exists():
        return full_sub

    return None


def _extract_metadata(filing_dir: Path) -> dict[str, str]:
    """
    Pull ticker, filing_type, and accession_number from the directory path.
    Expected layout: .../sec-edgar-filings/{ticker}/{filing_type}/{accession}/
    """
    parts = filing_dir.resolve().parts
    return {
        "ticker":           parts[-3],
        "filing_type":      parts[-2],
        "accession_number": parts[-1],
    }


# ── SGML / full-submission.txt parsing ───────────────────────────────────────

def _parse_full_submission(file_path: Path, metadata: dict[str, str]) -> list[dict[str, Any]]:
    """
    Extract the primary document HTML from an SEC full-submission.txt SGML file
    and parse it exactly as a standalone HTML filing.

    full-submission.txt structure:
        <DOCUMENT>
        <TYPE>10-K
        <SEQUENCE>1
        <FILENAME>aapl-20250329.htm
        <TEXT>
        ...HTML or plain text content...
        </TEXT>
        </DOCUMENT>
    """
    logger.debug(f"Extracting HTML from SGML wrapper: {file_path.name}")

    raw = file_path.read_text(encoding="utf-8", errors="replace")
    filing_type = metadata.get("filing_type", "10-K")

    html_content = _extract_html_from_sgml(raw, filing_type)

    if not html_content:
        logger.warning(
            f"[{metadata['ticker']}] Could not extract {filing_type} document "
            f"from {file_path.name}. The filing may use an unsupported format."
        )
        return []

    logger.debug(
        f"Extracted {len(html_content) // 1024} KB of HTML from full-submission.txt"
    )

    # Parse the extracted HTML using the standard pipeline
    soup = BeautifulSoup(html_content, "html.parser")
    return _parse_html_soup(soup, metadata, source_file=file_path.name)


def _extract_html_from_sgml(content: str, filing_type: str = "10-K") -> str | None:
    """
    Line-by-line SGML parser that extracts the primary document from an SEC
    full-submission.txt file.

    Looks for the DOCUMENT block whose TYPE matches the filing type (e.g. 10-K)
    and returns the content between its <TEXT> and </TEXT> tags.
    Falls back to the SEQUENCE=1 block if no TYPE match is found.
    """
    target_types = {filing_type.upper(), f"{filing_type.upper()}/A"}

    in_document   = False
    current_type  = ""
    current_seq   = ""
    in_text       = False
    html_lines: list[str] = []

    # Collected fallback (sequence=1 block) in case type match fails
    fallback_lines: list[str] = []
    in_fallback    = False

    for line in content.splitlines(keepends=True):
        stripped = line.strip().upper()

        if stripped == "<DOCUMENT>":
            in_document  = True
            current_type = ""
            current_seq  = ""
            in_text      = False
            html_lines   = []
            continue

        if stripped == "</DOCUMENT>":
            if in_text and current_type in target_types and html_lines:
                return "".join(html_lines)
            if in_fallback and fallback_lines:
                pass  # keep fallback_lines as-is
            in_document = False
            in_text     = False
            continue

        if not in_document:
            continue

        # Capture document metadata tags
        if stripped.startswith("<TYPE>"):
            current_type = line.strip()[6:].strip().upper()
            if current_type in target_types:
                in_text = False   # reset — wait for <TEXT>
            continue

        if stripped.startswith("<SEQUENCE>"):
            current_seq = line.strip()[10:].strip()
            continue

        if stripped == "<TEXT>":
            in_text = True
            if current_seq == "1":
                in_fallback = True
            continue

        if stripped == "</TEXT>":
            if current_type in target_types and html_lines:
                return "".join(html_lines)
            in_text     = False
            in_fallback = False
            continue

        if in_text:
            html_lines.append(line)
            if in_fallback:
                fallback_lines.append(line)

    # If exact type match never closed cleanly, use sequence=1 fallback
    if fallback_lines:
        logger.debug("Using SEQUENCE=1 block as fallback primary document")
        return "".join(fallback_lines)

    return None


# ── HTML parsing ──────────────────────────────────────────────────────────────

def _parse_html_file(file_path: Path, metadata: dict[str, str]) -> list[dict[str, Any]]:
    """Parse a standalone HTML filing."""
    raw  = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    return _parse_html_soup(soup, metadata, source_file=file_path.name)


def _parse_html_soup(
    soup: BeautifulSoup,
    metadata: dict[str, str],
    source_file: str,
) -> list[dict[str, Any]]:
    """
    Core HTML parser — shared by both standalone HTML and SGML-extracted HTML.
    Tables via BeautifulSoup; narrative text via unstructured (BS4 fallback).
    """
    elements: list[dict[str, Any]] = []
    idx = 0

    # Step 1: extract tables
    for table_tag in soup.find_all("table"):
        md = _table_to_markdown(table_tag)
        if md:
            elements.append(_make_element(md, "table", metadata, source_file, idx))
            idx += 1

    # Step 2: extract narrative text
    if UNSTRUCTURED_AVAILABLE:
        try:
            html_str = str(soup)
            for el in partition_html(text=html_str):
                if type(el).__name__ == "Table":
                    continue
                text = _clean_text(el.text)
                if len(text) >= MIN_TEXT_LENGTH:
                    elements.append(_make_element(text, "text", metadata, source_file, idx))
                    idx += 1
        except Exception as exc:
            logger.warning(f"unstructured failed ({exc}) — falling back to BeautifulSoup")
            for el in _extract_text_bs4(soup, metadata, source_file, idx):
                elements.append(el)
                idx += 1
    else:
        for el in _extract_text_bs4(soup, metadata, source_file, idx):
            elements.append(el)
            idx += 1

    return elements


def _extract_text_bs4(
    soup: BeautifulSoup,
    metadata: dict[str, str],
    source_file: str,
    start_idx: int,
) -> list[dict[str, Any]]:
    """Pure BeautifulSoup text extraction fallback."""
    elements = []
    idx      = start_idx

    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "div"]):
        if tag.find_parent("table"):
            continue
        if tag.name == "div" and tag.find(["p", "div", "h1", "h2", "h3", "h4"]):
            continue
        text = _clean_text(tag.get_text(separator=" ", strip=True))
        if len(text) >= MIN_TEXT_LENGTH:
            elements.append(_make_element(text, "text", metadata, source_file, idx))
            idx += 1

    return elements


# ── PDF parsing ───────────────────────────────────────────────────────────────

def _parse_pdf(file_path: Path, metadata: dict[str, str]) -> list[dict[str, Any]]:
    """Parse a PDF filing using pdfplumber (tables) + unstructured (text)."""
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return []

    elements: list[dict[str, Any]] = []
    idx = 0

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(tqdm(pdf.pages, desc="Pages", leave=False), start=1):
            for raw_table in page.extract_tables():
                md = _raw_table_to_markdown(raw_table)
                if md:
                    el = _make_element(md, "table", metadata, file_path.name, idx)
                    el["metadata"]["page_number"] = page_num
                    elements.append(el)
                    idx += 1

            text = _clean_text(page.extract_text() or "")
            if len(text) >= MIN_TEXT_LENGTH:
                el = _make_element(text, "text", metadata, file_path.name, idx)
                el["metadata"]["page_number"] = page_num
                elements.append(el)
                idx += 1

    return elements


# ── Table converters ──────────────────────────────────────────────────────────

def _table_to_markdown(table_tag: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = [
            _clean_text(cell.get_text(separator=" ", strip=True)).replace("|", "\\|")
            for cell in tr.find_all(["th", "td"])
        ]
        if any(c.strip() for c in cells):
            rows.append(cells)

    if not rows or sum(len(r) for r in rows) < MIN_TABLE_CELLS:
        return ""

    max_cols = max(len(r) for r in rows)
    rows     = [r + [""] * (max_cols - len(r)) for r in rows]

    header    = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body      = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])

    return "\n".join([header, separator, body]) if body else "\n".join([header, separator])


def _raw_table_to_markdown(table: list[list[str | None]]) -> str:
    if not table or len(table) < 2:
        return ""

    cleaned = []
    for row in table:
        cells = [_clean_text(str(c or "")).replace("|", "\\|") for c in row]
        if any(c.strip() for c in cells):
            cleaned.append(cells)

    if not cleaned or sum(len(r) for r in cleaned) < MIN_TABLE_CELLS:
        return ""

    max_cols = max(len(r) for r in cleaned)
    cleaned  = [r + [""] * (max_cols - len(r)) for r in cleaned]

    header    = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body      = "\n".join("| " + " | ".join(r) + " |" for r in cleaned[1:])

    return "\n".join([header, separator, body]) if body else "\n".join([header, separator])


# ── Utilities ─────────────────────────────────────────────────────────────────

def _make_element(
    content: str,
    content_type: str,
    metadata: dict[str, str],
    source_file: str,
    element_index: int,
) -> dict[str, Any]:
    element_id = hashlib.md5(
        f"{metadata['accession_number']}_{element_index}".encode()
    ).hexdigest()[:12]

    return {
        "element_id":   element_id,
        "content":      content,
        "content_type": content_type,
        "metadata": {
            **metadata,
            "source_file":   source_file,
            "page_number":   None,
            "element_index": element_index,
        },
    }


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", "", text)
    return text.strip()


def _log_summary(results: dict[str, int]) -> None:
    divider = "-" * 52
    total   = sum(results.values())
    logger.info(divider)
    logger.info(f"Parsing complete | {total} filing(s) written to data/processed/")
    for ticker, count in results.items():
        icon = "v" if count > 0 else "x"
        logger.info(f"  {icon}  {ticker:<6}  {count} filing(s)")
    logger.info(divider)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse EDGAR filings into structured JSON elements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER", default=None,
        help="Tickers to parse. Omit to parse all downloaded filings.",
    )
    args = parser.parse_args()
    parse_all_filings(tickers=args.tickers)


if __name__ == "__main__":
    main()
