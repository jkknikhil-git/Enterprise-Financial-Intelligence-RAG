"""
bm25_index.py
-------------
Builds and queries a BM25Okapi index from chunked EDGAR filing chunks.

BM25 complements dense vector search by excelling at:
  - Exact number matching  ("$2.3 billion", "15.2%", "Q3 FY2023")
  - Financial code lookup  ("ASC 606", "GAAP", specific line item names)
  - Keyword queries where the term must appear verbatim

The fitted index is pickled to bm25_cache/ alongside a parallel document
store (JSON) that maps corpus positions back to full chunk dicts.

Usage:
    python -m src.indexing.bm25_index           # build index
    python -m src.indexing.bm25_index --reset   # rebuild from scratch
    python -m src.indexing.bm25_index --tickers AAPL NVDA
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from loguru import logger
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parents[2]
CHUNKS_DIR      = ROOT_DIR / "data" / "processed" / "chunks"
BM25_DIR        = ROOT_DIR / "bm25_cache"
BM25_INDEX_PATH = BM25_DIR / "bm25_index.pkl"
BM25_DOCS_PATH  = BM25_DIR / "bm25_docs.json"


# ── Public API ────────────────────────────────────────────────────────────────

def build_bm25_index(
    chunks_dir: Path = CHUNKS_DIR,
    output_dir: Path = BM25_DIR,
    tickers: list[str] | None = None,
    reset: bool = False,
) -> BM25Okapi:
    """
    Build a BM25Okapi index from all chunk files and persist to disk.

    Args:
        chunks_dir: Directory of *_chunks.json files (from chunker.py).
        output_dir: Where to save the pickled index and document store.
        tickers:    Limit to these tickers. None = all.
        reset:      Rebuild even if an index already exists.

    Returns:
        The fitted BM25Okapi index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reset and BM25_INDEX_PATH.exists() and BM25_DOCS_PATH.exists():
        logger.info("BM25 index already exists. Loading from cache.")
        logger.info("Pass --reset to rebuild from scratch.")
        return load_bm25_index(output_dir)

    chunk_files = sorted(chunks_dir.glob("*_chunks.json"))
    if not chunk_files:
        logger.error(f"No chunk files found in {chunks_dir}")
        logger.info("Run: python -m src.ingestion.chunker")
        raise FileNotFoundError(f"No chunk files in {chunks_dir}")

    if tickers:
        chunk_files = [f for f in chunk_files if f.stem.split("_")[0] in tickers]

    logger.info(f"Loading chunks from {len(chunk_files)} file(s)")

    all_chunks: list[dict[str, Any]] = []
    for chunk_file in tqdm(chunk_files, desc="Loading", unit="filing"):
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)

    logger.info(f"Tokenising {len(all_chunks)} chunks...")
    corpus = [
        _tokenise(c["content"])
        for c in tqdm(all_chunks, desc="Tokenising", unit="chunk")
    ]

    logger.info("Fitting BM25Okapi index...")
    index = BM25Okapi(corpus)

    # Persist index
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Persist parallel document store for result lookup
    BM25_DOCS_PATH.write_text(
        json.dumps(all_chunks, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.success(
        f"BM25 index built | {len(all_chunks)} documents | saved → {output_dir}"
    )
    return index


def load_bm25_index(index_dir: Path = BM25_DIR) -> BM25Okapi:
    """Load a persisted BM25 index from disk."""
    path = index_dir / "bm25_index.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {path}. "
            "Run: python -m src.indexing.bm25_index"
        )
    with open(path, "rb") as f:
        index = pickle.load(f)
    logger.debug(f"BM25 index loaded from {path}")
    return index


def load_bm25_docs(index_dir: Path = BM25_DIR) -> list[dict[str, Any]]:
    """Load the parallel document store used to resolve BM25 result indices."""
    path = index_dir / "bm25_docs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 document store not found at {path}. "
            "Run: python -m src.indexing.bm25_index"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def query_bm25(
    query: str,
    n_results: int = 20,
    index_dir: Path = BM25_DIR,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-n BM25 results for a query.

    Args:
        query:      Natural language or keyword query.
        n_results:  Number of results to return.
        index_dir:  Directory containing the pickled index and doc store.

    Returns:
        List of result dicts — chunk_id, content, metadata, score, source.
    """
    index = load_bm25_index(index_dir)
    docs  = load_bm25_docs(index_dir)

    scores  = index.get_scores(_tokenise(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    results = []
    for idx in top_idx:
        if scores[idx] == 0.0:
            break   # Remaining docs have zero overlap with query tokens
        chunk = docs[idx]
        results.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "content":  chunk.get("content", ""),
            "metadata": chunk.get("metadata", {}),
            "score":    round(float(scores[idx]), 4),
            "source":   "bm25",
        })

    return results


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Whitespace + punctuation tokeniser designed for financial text.

    Preserves:
      $2.3B   15.2%   ASC 606   Q3   FY2023   EBITDA
    Strips:
      single characters and pure punctuation tokens
    """
    tokens = re.findall(r"[\w\$\%\.]+", text.lower())
    return [t for t in tokens if len(t) > 1]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 index from chunked EDGAR filings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER", default=None)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Rebuild index even if one already exists.",
    )
    args = parser.parse_args()
    build_bm25_index(tickers=args.tickers, reset=args.reset)


if __name__ == "__main__":
    main()
