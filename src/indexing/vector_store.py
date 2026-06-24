"""
vector_store.py
---------------
Builds and queries a ChromaDB vector store using BAAI/bge-small-en-v1.5 embeddings.

BGE embedding design:
  Documents → embedded as-is
  Queries   → prefixed with BGE_QUERY_PREFIX
  (Asymmetric retrieval — how BGE models are intended to be used)

Usage:
    python -m src.indexing.vector_store           # build index from all chunks
    python -m src.indexing.vector_store --reset   # drop collection and rebuild
    python -m src.indexing.vector_store --tickers AAPL MSFT
"""

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).resolve().parents[2]
CHUNKS_DIR = ROOT_DIR / "data" / "processed" / "chunks"
CHROMA_DIR = ROOT_DIR / "chroma_db"

EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME  = "financial_filings"
EMBED_BATCH_SIZE = 64    # safe for 16 GB RAM + CPU inference
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── Public API ────────────────────────────────────────────────────────────────

def build_vector_store(
    chunks_dir: Path = CHUNKS_DIR,
    chroma_dir: Path = CHROMA_DIR,
    reset: bool = False,
    tickers: list[str] | None = None,
) -> Any:
    """
    Embed all chunks and upsert them into ChromaDB.

    Args:
        chunks_dir: Directory of *_chunks.json files (from chunker.py).
        chroma_dir: Persistent ChromaDB storage directory.
        reset:      Drop and recreate the collection before indexing.
        tickers:    Limit to these tickers. None = all available.

    Returns:
        The ChromaDB Collection object.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client     = chromadb.PersistentClient(path=str(chroma_dir))
    collection = _get_or_create_collection(client, reset)
    model      = _load_model()

    chunk_files = sorted(chunks_dir.glob("*_chunks.json"))
    if not chunk_files:
        logger.error(f"No chunk files found in {chunks_dir}")
        logger.info("Run: python -m src.ingestion.chunker")
        return collection

    if tickers:
        chunk_files = [f for f in chunk_files if f.stem.split("_")[0] in tickers]

    logger.info(f"Indexing {len(chunk_files)} chunk file(s)")
    total_upserted = 0

    for chunk_file in tqdm(chunk_files, desc="Indexing", unit="filing"):
        chunks: list[dict[str, Any]] = json.loads(
            chunk_file.read_text(encoding="utf-8")
        )
        n = _upsert_chunks(collection, model, chunks)
        total_upserted += n
        logger.success(f"{chunk_file.name} → {n} vectors upserted")

    final_count = collection.count()
    logger.info(
        f"Vector store complete | {total_upserted} upserted | "
        f"{final_count} total vectors in '{COLLECTION_NAME}'"
    )
    return collection


def query_vector_store(
    query: str,
    n_results: int = 20,
    chroma_dir: Path = CHROMA_DIR,
    ticker_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-n semantically similar chunks for a query.

    Args:
        query:         Natural language query string.
        n_results:     Number of candidates to retrieve.
        chroma_dir:    Path to ChromaDB directory.
        ticker_filter: Restrict results to a single ticker (optional).

    Returns:
        List of result dicts — chunk_id, content, metadata, score, source.
    """
    client     = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(COLLECTION_NAME)
    model      = _load_model()

    # BGE: queries need the prefix; documents do not
    prefixed  = BGE_QUERY_PREFIX + query
    embedding = model.encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    where = {"ticker": ticker_filter} if ticker_filter else None

    raw = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return _format_results(raw)


def get_collection(chroma_dir: Path = CHROMA_DIR) -> Any:
    """Return the ChromaDB collection (must be built first)."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(COLLECTION_NAME)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_or_create_collection(client: Any, reset: bool) -> Any:
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Dropped collection: {COLLECTION_NAME}")
        except Exception:
            pass  # Didn't exist yet — fine

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    action = "reset + created" if reset else "opened"
    logger.info(f"Collection '{COLLECTION_NAME}' {action} | {collection.count()} existing vectors")
    return collection


def _load_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.max_seq_length = 512    # BGE-small context window
    return model


def _upsert_chunks(
    collection: Any,
    model: SentenceTransformer,
    chunks: list[dict[str, Any]],
) -> int:
    """Embed and upsert chunks in batches. Returns number of vectors upserted."""
    all_ids = [c["chunk_id"] for c in chunks]

    # Check which IDs already exist to avoid redundant embedding work
    try:
        existing = set(collection.get(ids=all_ids)["ids"])
    except Exception:
        existing = set()

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]
    if not new_chunks:
        logger.debug("All chunks already indexed — skipping file")
        return 0

    texts     = [c["content"] for c in new_chunks]
    ids       = [c["chunk_id"] for c in new_chunks]
    metadatas = [_flatten_metadata(c) for c in new_chunks]

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vecs  = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        all_embeddings.extend(vecs)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )
    return len(new_chunks)


def _flatten_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    ChromaDB requires metadata values to be str | int | float | bool.
    Flatten the nested metadata dict; convert None → "".
    """
    flat: dict[str, Any] = {
        "chunk_id":     chunk.get("chunk_id", ""),
        "content_type": chunk.get("content_type", ""),
        "token_count":  chunk.get("token_count", 0),
    }
    for k, v in chunk.get("metadata", {}).items():
        flat[k] = "" if v is None else v
    return flat


def _format_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert raw ChromaDB query output to a flat list of result dicts."""
    ids       = raw.get("ids",       [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results = []
    for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        results.append({
            "chunk_id": chunk_id,
            "content":  doc,
            "metadata": meta,
            "score":    round(1.0 - dist, 4),   # cosine distance → similarity
            "source":   "vector",
        })
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector store from chunked EDGAR filings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER", default=None)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and rebuild the collection from scratch.",
    )
    args = parser.parse_args()
    build_vector_store(tickers=args.tickers, reset=args.reset)


if __name__ == "__main__":
    main()
