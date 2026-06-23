# Enterprise Financial Intelligence RAG

> A production-grade Retrieval-Augmented Generation pipeline for SEC 10-K and 10-Q filings — hybrid BM25 + vector search, cross-encoder reranking, citation enforcement, and RAGAS-gated continuous evaluation.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://langchain.com/)
[![RAGAS](https://img.shields.io/badge/eval-RAGAS-orange.svg)](https://docs.ragas.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/YOUR_USERNAME/financial-rag/actions/workflows/eval.yml/badge.svg)](https://github.com/YOUR_USERNAME/financial-rag/actions)

---

## The Problem

Financial analysts need to compare quarterly metrics, identify risk indicators, and verify compliance across dozens of 10-K and 10-Q filings simultaneously. Standard semantic search fails here: dense embeddings struggle with granular numbers, exact financial codes, and multi-step reasoning across documents.

A query like *"What was Apple's gross margin in fiscal 2023 compared to Microsoft's?"* requires exact keyword matching, semantic understanding, table parsing, and source attribution — all at once.

---

## What Makes This Production-Grade

| Challenge | Solution |
|---|---|
| PDF tables are destroyed by plain text chunking | `unstructured` + `pdfplumber` for structure-aware extraction |
| Dense embeddings miss exact numbers and financial codes | Hybrid BM25 + vector search via Reciprocal Rank Fusion (RRF) |
| Top-k retrieval returns related but irrelevant chunks | Cross-encoder reranker re-scores all candidates |
| LLMs hallucinate unsupported financial claims | Citation enforcer declines answers not grounded in retrieved chunks |
| No visibility into quality degradation across changes | RAGAS evaluation wired into GitHub Actions CI |

---

## Architecture

```
                      EDGAR API
                          │
                          ▼
               Unstructured Parser
               (table-aware PDF extraction)
                          │
                          ▼
              Chunker (500–800 tok, 100 tok overlap)
                    │               │
                    ▼               ▼
               ChromaDB         BM25 Index
             (BGE-small-en)     (rank_bm25)
                    │               │
                    └───────┬───────┘
                            │  Hybrid search
                            ▼
                       RRF Fusion
                            │
                            ▼
              Cross-Encoder Reranker
                (Cohere Rerank API)
                            │
                            ▼
                  Citation Enforcer
              (decline if unsupported)
                            │
                            ▼
             Groq LLM — llama-3.3-70b-versatile
                (versioned YAML prompt config)
                            │
                            ▼
               Answer + inline citations
```

**Evaluation loop — runs on every pull request:**

```
FiQA dataset → RAGAS (Faithfulness · Answer Relevance · Context Precision) → CI gate
```

---

## Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Document ingestion | `sec-edgar-downloader` | One-line EDGAR access by ticker |
| PDF parsing | `unstructured` + `pdfplumber` | Table-aware extraction, no GPU required |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Token-aware, configurable overlap |
| Embeddings | `BAAI/bge-small-en-v1.5` | CPU-efficient, strong retrieval quality |
| Vector store | ChromaDB | Local persistence, no external server |
| Keyword search | `rank-bm25` | Exact number and financial code matching |
| Reranking | Cohere Rerank API | Production-quality cross-encoder, zero local compute |
| LLM | Groq API — `llama-3.3-70b-versatile` | Fast inference, free tier |
| Orchestration | LangChain + LangGraph | Graph-based pipeline with conditional citation gate |
| Evaluation | RAGAS + FiQA dataset | Domain-specific faithfulness measurement |
| CI | GitHub Actions | Automated quality gate on every PR |

---

## Project Phases

- [x] **Phase 0 — Scaffold**: repo structure, dependencies, CI skeleton, versioned prompt config
- [ ] **Phase 1 — Ingestion**: EDGAR fetcher, table-aware parser, chunker, dual index (ChromaDB + BM25)
- [ ] **Phase 2 — Retrieval & Generation**: hybrid search, RRF, cross-encoder reranker, citation enforcer, Groq LLM
- [ ] **Phase 3 — Evaluation & CI**: RAGAS offline evaluation, FiQA test set, GitHub Actions quality gate

---

## Setup

### Prerequisites

- Python 3.11+
- [Groq API key](https://console.groq.com/) — free tier
- [Cohere API key](https://dashboard.cohere.com/) — free tier (1,000 rerank calls/month)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/financial-rag.git
cd financial-rag

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install PyTorch CPU-only first (avoids pulling in CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
cp .env.example .env           # add your API keys
```

### Ingest filings

```bash
python -m src.ingestion.edgar_fetcher     # downloads 10-K filings to data/raw/
python -m src.ingestion.parser            # extracts and chunks into data/processed/
python -m src.indexing.vector_store       # builds ChromaDB index
python -m src.indexing.bm25_index         # builds and pickles BM25 index
```

### Run a query

```bash
python -m src.pipeline \
  --query "What was Apple's operating income in fiscal year 2023?"
```

### Run evaluation

```bash
python src/evaluation/ragas_eval.py
```

---

## Project Structure

```
financial-rag/
├── .github/
│   └── workflows/
│       └── eval.yml                # RAGAS CI pipeline
├── data/
│   ├── raw/                        # Downloaded 10-K PDFs (gitignored)
│   └── processed/                  # Parsed chunks as JSON (gitignored)
├── src/
│   ├── ingestion/
│   │   ├── edgar_fetcher.py        # EDGAR downloader wrapper
│   │   ├── parser.py               # Table-aware PDF parser
│   │   └── chunker.py              # Token-aware chunker
│   ├── indexing/
│   │   ├── vector_store.py         # ChromaDB + BGE embedding builder
│   │   └── bm25_index.py           # BM25 index builder + persistence
│   ├── retrieval/
│   │   ├── hybrid_search.py        # RRF over vector + BM25 results
│   │   └── reranker.py             # Cohere reranker wrapper
│   ├── generation/
│   │   ├── prompts/
│   │   │   └── v1.yaml             # Versioned prompt config
│   │   ├── citation_enforcer.py    # Validates chunk-answer grounding
│   │   └── generator.py            # Groq LLM call
│   ├── evaluation/
│   │   ├── ragas_eval.py           # RAGAS metric runner
│   │   └── fiqa_loader.py          # FiQA dataset loader
│   └── pipeline.py                 # LangGraph orchestration graph
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Target Companies

Apple (AAPL) · Microsoft (MSFT) · NVIDIA (NVDA) · Alphabet (GOOGL) · Meta (META)

Selected for contrasting financial profiles and high public interest — stress-tests retrieval across different revenue structures, margin profiles, and disclosure styles.

---

## License

MIT
