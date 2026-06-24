"""
parser.py
---------
Table-aware document parser for SEC EDGAR filings.

EDGAR filings are primarily HTML (not PDF). This parser handles both formats:

  HTML (primary):
    - BeautifulSoup for reliable financial table extraction → markdown
    - unstructured partition_html for narrative text (with BS4 fallback)

  PDF (fallback):
    - pdfplumber for table extraction → markdown
    - unstructured partition_pdf for narrative text

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
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("unstructured not importable — using BeautifulSoup-only extraction")

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT_DIR / "data" / "raw" / "sec-edgar-filings"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Files to skip — full-submission.txt is a raw XBRL dump, not a readable doc
SKIP_FILENAMES: set[str] = {
    "full-submission.txt",
    "filing-details.htm",
    "filing-details.html",
}

# Ignore tables with fewer total cells than this (layout / navigation tables)
MIN_TABLE_CELLS: int = 6

# Ignore text elements shorter than this many characters
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
        Dict mapping ticker → number of filings successfully parsed.
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
        ticker = ticker_dir.name
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

                # Skip already-processed filings (idempotent)
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
                            f" → {len(elements)} elements"
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
        filing_dir: e.g. data/raw/sec-edgar-filings/AAPL/10-K/0000320193-23-000106/

    Returns:
        List of element dicts with content, content_type, and metadata.
    """
    metadata = _extract_metadata(filing_dir)
    primary_doc = _find_primary_document(filing_dir)

    if primary_doc is None:
        logger.warning(f"No parseable document in {filing_dir.name}")
        return []

    size_kb = primary_doc.stat().st_size // 1024
    logger.debug(f"Parsing {primary_doc.name} ({size_kb} KB) for {metadata['ticker']}")

    suffix = primary_doc.suffix.lower()

    if suffix in {".htm", ".html"}:
        return _parse_html(primary_doc, metadata)
    elif suffix == ".pdf":
        return _parse_pdf(primary_doc, metadata)
    else:
        logger.warning(f"Unsupported file type: {suffix}")
        return []


# ── File discovery ────────────────────────────────────────────────────────────

def _find_primary_document(filing_dir: Path) -> Path | None:
    """
    Locate the primary readable document inside an EDGAR accession directory.
    Priority: primary-document.htm → largest .htm → largest .pdf
    """
    # sec-edgar-downloader names the main file primary-document.htm
    for name in ("primary-document.htm", "primary-document.html"):
        candidate = filing_dir / name
        if candidate.exists():
            return candidate

    # Fall back to the largest .htm that isn't a known skip file
    htm_candidates = [
        f for f in (list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html")))
        if f.name not in SKIP_FILENAMES
    ]
    if htm_candidates:
        return max(htm_candidates, key=lambda f: f.stat().st_size)

    # Last resort: PDF
    pdf_candidates = list(filing_dir.glob("*.pdf"))
    if pdf_candidates:
        return max(pdf_candidates, key=lambda f: f.stat().st_size)

    return None


def _extract_metadata(filing_dir: Path) -> dict[str, str]:
    """
    Extract structured metadata from the directory path.

    Expected layout: .../sec-edgar-filings/{ticker}/{filing_type}/{accession}/
    """
    parts = filing_dir.resolve().parts
    return {
        "ticker":           parts[-3],
        "filing_type":      parts[-2],
        "accession_number": parts[-1],
    }


# ── HTML parsing ──────────────────────────────────────────────────────────────

def _parse_html(file_path: Path, metadata: dict[str, str]) -> list[dict[str, Any]]:
    """
    Parse an HTML EDGAR filing.
    Tables extracted with BeautifulSoup; narrative text via unstructured (with BS4 fallback).
    """
    raw_html = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")

    elements: list[dict[str, Any]] = []
    idx = 0

    # ── Step 1: extract tables ────────────────────────────────────────────────
    for table_tag in soup.find_all("table"):
        md = _table_to_markdown(table_tag)
        if md:
            elements.append(_make_element(md, "table", metadata, file_path.name, idx))
            idx += 1

    # ── Step 2: extract narrative text ────────────────────────────────────────
    if UNSTRUCTURED_AVAILABLE:
        try:
            for el in partition_html(filename=str(file_path)):
                # Skip Table elements — already handled above
                if type(el).__name__ == "Table":
                    continue
                text = _clean_text(el.text)
                if len(text) >= MIN_TEXT_LENGTH:
                    elements.append(_make_element(text, "text", metadata, file_path.name, idx))
                    idx += 1
        except Exception as exc:
            logger.warning(f"unstructured failed ({exc}), falling back to BeautifulSoup text extraction")
            for el in _extract_text_bs4(soup, metadata, file_path.name, idx):
                elements.append(el)
                idx += 1
    else:
        for el in _extract_text_bs4(soup, metadata, file_path.name, idx):
            elements.append(el)
            idx += 1

    return elements


def _extract_text_bs4(
    soup: BeautifulSoup,
    metadata: dict[str, str],
    source_file: str,
    start_idx: int,
) -> list[dict[str, Any]]:
    """
    Pure BeautifulSoup fallback for narrative text extraction.
    Skips tags inside tables and skips divs that wrap other block elements.
    """
    elements = []
    idx = start_idx

    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "div"]):
        # Skip anything nested inside a table
        if tag.find_parent("table"):
            continue
        # Skip container divs (they'll be covered by their children)
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
        logger.error("pdfplumber not installed — cannot parse PDFs. Run: pip install pdfplumber")
        return []

    elements: list[dict[str, Any]] = []
    idx = 0

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(tqdm(pdf.pages, desc="Pages", leave=False), start=1):

            # Tables
            for raw_table in page.extract_tables():
                md = _raw_table_to_markdown(raw_table)
                if md:
                    el = _make_element(md, "table", metadata, file_path.name, idx)
                    el["metadata"]["page_number"] = page_num
                    elements.append(el)
                    idx += 1

            # Text
            text = page.extract_text() or ""
            text = _clean_text(text)
            if len(text) >= MIN_TEXT_LENGTH:
                el = _make_element(text, "text", metadata, file_path.name, idx)
                el["metadata"]["page_number"] = page_num
                elements.append(el)
                idx += 1

    return elements


# ── Table converters ──────────────────────────────────────────────────────────

def _table_to_markdown(table_tag: Tag) -> str:
    """
    Convert a BeautifulSoup <table> element to a markdown table.
    Returns "" for navigation/layout tables with fewer than MIN_TABLE_CELLS cells.
    """
    rows: list[list[str]] = []

    for tr in table_tag.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            text = _clean_text(cell.get_text(separator=" ", strip=True))
            cells.append(text.replace("|", "\\|"))
        if any(c.strip() for c in cells):
            rows.append(cells)

    total_cells = sum(len(r) for r in rows)
    if not rows or total_cells < MIN_TABLE_CELLS:
        return ""

    # Pad all rows to the same width
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    header    = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body_rows = [("| " + " | ".join(r) + " |") for r in rows[1:]]

    return "\n".join([header, separator] + body_rows)


def _raw_table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber raw table (list[list]) to a markdown table."""
    if not table or len(table) < 2:
        return ""

    cleaned: list[list[str]] = []
    for row in table:
        cells = [_clean_text(str(c or "")).replace("|", "\\|") for c in row]
        if any(c.strip() for c in cells):
            cleaned.append(cells)

    total_cells = sum(len(r) for r in cleaned)
    if not cleaned or total_cells < MIN_TABLE_CELLS:
        return ""

    max_cols = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (max_cols - len(r)) for r in cleaned]

    header    = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body_rows = [("| " + " | ".join(r) + " |") for r in cleaned[1:]]

    return "\n".join([header, separator] + body_rows)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _make_element(
    content: str,
    content_type: str,
    metadata: dict[str, str],
    source_file: str,
    element_index: int,
) -> dict[str, Any]:
    """Build a standard element dict."""
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
    """Normalise whitespace; strip non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    # Keep printable ASCII + common financial symbols preserved in ASCII range
    text = re.sub(r"[^\x20-\x7E]", "", text)
    return text.strip()


def _log_summary(results: dict[str, int]) -> None:
    divider = "─" * 52
    total = sum(results.values())
    logger.info(divider)
    logger.info(f"Parsing complete | {total} filing(s) written to data/processed/")
    for ticker, count in results.items():
        icon = "✓" if count > 0 else "✗"
        logger.info(f"  {icon}  {ticker:<6}  {count} filing(s)")
    logger.info(divider)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse EDGAR filings into structured JSON elements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Tickers to parse. Omit to parse all downloaded filings.",
    )
    args = parser.parse_args()
    parse_all_filings(tickers=args.tickers)


if __name__ == "__main__":
    main()
