"""
chunker.py
----------
Token-aware chunker for parsed EDGAR filing elements.

Strategy:
  Text elements  → RecursiveCharacterTextSplitter (700 tok target, 100 tok overlap)
  Table elements → kept intact; split at row boundaries only if > MAX_TABLE_TOKENS
                   (splitting mid-table destroys financial relationships)

Every sub-table chunk repeats the header row so it is self-contained and
interpretable without context from adjacent chunks.

Output:
  data/processed/chunks/{ticker}_{filing_type}_{accession}_chunks.json

Usage:
    python -m src.ingestion.chunker
    python -m src.ingestion.chunker --tickers AAPL MSFT
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT_DIR      = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHUNKS_DIR    = PROCESSED_DIR / "chunks"

CHUNK_SIZE       = 700    # target tokens per text chunk
CHUNK_OVERLAP    = 100    # token overlap between adjacent chunks
MAX_TABLE_TOKENS = 1500   # tables larger than this are row-split

TIKTOKEN_ENCODING = "cl100k_base"   # reasonable proxy for llama-3 token counts


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_all_filings(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = CHUNKS_DIR,
    tickers: list[str] | None = None,
) -> dict[str, int]:
    """
    Chunk all parsed element JSON files and write results to output_dir.

    Args:
        processed_dir: Contains element JSON files produced by parser.py.
        output_dir:    Destination for chunked JSON files.
        tickers:       Limit to these tickers. None = all available.

    Returns:
        Dict mapping ticker → total chunks produced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(processed_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No element JSON files found in {processed_dir}")
        logger.info("Run: python -m src.ingestion.parser")
        return {}

    if tickers:
        json_files = [f for f in json_files if f.stem.split("_")[0] in tickers]

    logger.info(f"Chunking {len(json_files)} filing(s) | chunk_size={CHUNK_SIZE} | overlap={CHUNK_OVERLAP}")

    splitter = _build_splitter()
    encoder  = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    results: dict[str, int] = {}

    for json_file in tqdm(json_files, desc="Chunking", unit="filing"):
        ticker      = json_file.stem.split("_")[0]
        output_path = output_dir / f"{json_file.stem}_chunks.json"

        # Idempotent — skip already-chunked filings
        if output_path.exists():
            logger.debug(f"Already chunked: {json_file.name}")
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            results[ticker] = results.get(ticker, 0) + len(cached)
            continue

        try:
            elements: list[dict[str, Any]] = json.loads(
                json_file.read_text(encoding="utf-8")
            )
            chunks = chunk_elements(elements, splitter, encoder)

            output_path.write_text(
                json.dumps(chunks, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.success(
                f"{json_file.name} → {len(elements)} elements → {len(chunks)} chunks"
            )
            results[ticker] = results.get(ticker, 0) + len(chunks)

        except Exception as exc:
            logger.error(f"{json_file.name} failed: {exc}")

    _log_summary(results)
    return results


def chunk_elements(
    elements: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
    encoder: tiktoken.Encoding,
) -> list[dict[str, Any]]:
    """
    Convert a list of parsed elements into retrieval-ready chunks.

    Args:
        elements: Output from parser.py.
        splitter: Configured LangChain text splitter.
        encoder:  tiktoken encoder for counting tokens.

    Returns:
        Flat list of chunk dicts.
    """
    chunks: list[dict[str, Any]] = []

    for element in elements:
        content      = element.get("content", "").strip()
        content_type = element.get("content_type", "text")
        metadata     = element.get("metadata", {})
        element_id   = element.get("element_id", "")

        if not content:
            continue

        if content_type == "table":
            chunks.extend(_chunk_table(content, metadata, element_id, encoder))
        else:
            chunks.extend(_chunk_text(content, metadata, element_id, splitter, encoder))

    return chunks


# ── Chunking strategies ───────────────────────────────────────────────────────

def _chunk_text(
    content: str,
    metadata: dict[str, Any],
    element_id: str,
    splitter: RecursiveCharacterTextSplitter,
    encoder: tiktoken.Encoding,
) -> list[dict[str, Any]]:
    """Split a narrative text element into overlapping token-bounded chunks."""
    sub_texts = splitter.split_text(content)
    total     = len(sub_texts)
    chunks    = []

    for i, text in enumerate(sub_texts):
        chunks.append(_make_chunk(
            chunk_id     = _make_chunk_id(element_id, i),
            content      = text,
            content_type = "text",
            token_count  = len(encoder.encode(text)),
            metadata     = metadata,
            element_id   = element_id,
            chunk_index  = i,
            total_chunks = total,
        ))

    return chunks


def _chunk_table(
    content: str,
    metadata: dict[str, Any],
    element_id: str,
    encoder: tiktoken.Encoding,
) -> list[dict[str, Any]]:
    """
    Keep a table as a single chunk when possible.
    Row-split only if the table exceeds MAX_TABLE_TOKENS.
    """
    token_count = len(encoder.encode(content))

    if token_count <= MAX_TABLE_TOKENS:
        return [_make_chunk(
            chunk_id     = _make_chunk_id(element_id, 0),
            content      = content,
            content_type = "table",
            token_count  = token_count,
            metadata     = metadata,
            element_id   = element_id,
            chunk_index  = 0,
            total_chunks = 1,
        )]

    logger.debug(f"Oversized table ({token_count} tok) — row-splitting: {element_id}")
    return _split_table_by_rows(content, metadata, element_id, encoder)


def _split_table_by_rows(
    markdown_table: str,
    metadata: dict[str, Any],
    element_id: str,
    encoder: tiktoken.Encoding,
) -> list[dict[str, Any]]:
    """
    Split an oversized markdown table into sub-tables, each including the
    original header row so every chunk is independently interpretable.
    """
    lines = markdown_table.strip().split("\n")

    if len(lines) < 3:
        # Malformed table — return as a single chunk anyway
        token_count = len(encoder.encode(markdown_table))
        return [_make_chunk(
            chunk_id     = _make_chunk_id(element_id, 0),
            content      = markdown_table,
            content_type = "table",
            token_count  = token_count,
            metadata     = metadata,
            element_id   = element_id,
            chunk_index  = 0,
            total_chunks = 1,
        )]

    header    = lines[0]   # | Col A | Col B | ...
    separator = lines[1]   # | ---   | ---   | ...
    data_rows = lines[2:]

    sub_tables: list[str] = []
    current_rows: list[str] = []

    for row in data_rows:
        candidate = "\n".join([header, separator] + current_rows + [row])
        if len(encoder.encode(candidate)) > MAX_TABLE_TOKENS and current_rows:
            # Flush current batch before adding the new row
            sub_tables.append("\n".join([header, separator] + current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        sub_tables.append("\n".join([header, separator] + current_rows))

    total  = len(sub_tables)
    chunks = []

    for i, sub_table in enumerate(sub_tables):
        chunks.append(_make_chunk(
            chunk_id     = _make_chunk_id(element_id, i),
            content      = sub_table,
            content_type = "table",
            token_count  = len(encoder.encode(sub_table)),
            metadata     = metadata,
            element_id   = element_id,
            chunk_index  = i,
            total_chunks = total,
        ))

    return chunks


# ── Utilities ─────────────────────────────────────────────────────────────────

def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Construct a token-aware splitter with financial-document separators."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name = TIKTOKEN_ENCODING,
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )


def _make_chunk(
    chunk_id: str,
    content: str,
    content_type: str,
    token_count: int,
    metadata: dict[str, Any],
    element_id: str,
    chunk_index: int,
    total_chunks: int,
) -> dict[str, Any]:
    return {
        "chunk_id":     chunk_id,
        "content":      content,
        "content_type": content_type,
        "token_count":  token_count,
        "metadata": {
            **metadata,
            "element_id":   element_id,
            "chunk_index":  chunk_index,
            "total_chunks": total_chunks,
        },
    }


def _make_chunk_id(element_id: str, chunk_index: int) -> str:
    return hashlib.md5(f"{element_id}_{chunk_index}".encode()).hexdigest()[:16]


def _log_summary(results: dict[str, int]) -> None:
    divider = "─" * 52
    total   = sum(results.values())
    logger.info(divider)
    logger.info(f"Chunking complete | {total} total chunks across {len(results)} ticker(s)")
    for ticker, count in results.items():
        logger.info(f"  {ticker:<6}  {count:>6} chunks")
    logger.info(divider)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk parsed EDGAR elements into token-bounded pieces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Tickers to chunk. Omit to chunk all parsed filings.",
    )
    args = parser.parse_args()
    chunk_all_filings(tickers=args.tickers)


if __name__ == "__main__":
    main()
