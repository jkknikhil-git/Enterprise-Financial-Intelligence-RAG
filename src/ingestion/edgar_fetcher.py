"""
edgar_fetcher.py
----------------
Downloads SEC 10-K and 10-Q filings from EDGAR for target companies.

SEC fair-use policy requires:
  - A valid User-Agent header (company name + email)
  - Max 10 requests per second

sec-edgar-downloader handles the rate limit automatically.

Usage:
    # Download last 3 10-K filings for all 5 target companies (default)
    python -m src.ingestion.edgar_fetcher

    # Download last 5 10-Q filings for specific tickers
    python -m src.ingestion.edgar_fetcher --tickers AAPL NVDA --filing-type 10-Q --limit 5
"""

import argparse
import time
from pathlib import Path
from typing import Literal

from loguru import logger
from sec_edgar_downloader import Downloader
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"

TARGET_COMPANIES: dict[str, str] = {
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corporation",
    "NVDA":  "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc.",
    "META":  "Meta Platforms Inc.",
}

FilingType = Literal["10-K", "10-Q"]

# Small buffer between companies on top of sec-edgar-downloader's built-in rate limit
INTER_COMPANY_DELAY_S: float = 0.5


# ── Core function ─────────────────────────────────────────────────────────────

def fetch_filings(
    tickers: list[str] | None = None,
    filing_type: FilingType = "10-K",
    limit: int = 3,
    output_dir: Path = DATA_DIR,
) -> dict[str, int]:
    """
    Download SEC filings for the specified tickers.

    Args:
        tickers:      Tickers to download. Defaults to all five TARGET_COMPANIES.
        filing_type:  "10-K" (annual) or "10-Q" (quarterly).
        limit:        Number of most recent filings per ticker.
        output_dir:   Root directory for downloaded filings.

    Returns:
        Dict mapping ticker → number of filings on disk after the run.
    """
    if tickers is None:
        tickers = list(TARGET_COMPANIES.keys())

    unknown = [t for t in tickers if t not in TARGET_COMPANIES]
    if unknown:
        logger.warning(f"Unknown tickers (will still attempt download): {unknown}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # SEC requires a descriptive User-Agent — use your project name + email
    dl = Downloader(
        company_name="EnterpriseFinancialRAG",
        email_address="research@financialrag.dev",
        download_folder=output_dir,
    )

    results: dict[str, int] = {}

    logger.info(
        f"Starting | companies={len(tickers)} | filing_type={filing_type} | limit={limit}"
    )

    for ticker in tqdm(tickers, desc="Downloading filings", unit="company"):
        company_label = TARGET_COMPANIES.get(ticker, ticker)
        filing_dir = output_dir / "sec-edgar-filings" / ticker / filing_type

        try:
            logger.info(f"[{ticker}] {company_label} — requesting {limit}x {filing_type}")
            dl.get(filing_type, ticker, limit=limit)

            # Count subdirectories — each accession number is one filing
            count = len([p for p in filing_dir.iterdir() if p.is_dir()]) if filing_dir.exists() else 0
            results[ticker] = count
            logger.success(f"[{ticker}] {count} filing(s) saved → {filing_dir}")

        except Exception as exc:
            logger.error(f"[{ticker}] Failed: {exc}")
            results[ticker] = 0

        time.sleep(INTER_COMPANY_DELAY_S)

    _log_summary(results, filing_type)
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_summary(results: dict[str, int], filing_type: str) -> None:
    total = sum(results.values())
    failed = [t for t, n in results.items() if n == 0]
    divider = "─" * 52

    logger.info(divider)
    logger.info(f"Done | {total} {filing_type} filing(s) across {len(results)} company/ies")
    for ticker, count in results.items():
        icon = "✓" if count > 0 else "✗"
        logger.info(f"  {icon}  {ticker:<6}  {count} filing(s)")
    if failed:
        logger.warning(f"  Failed: {failed}")
    logger.info(divider)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download SEC EDGAR 10-K / 10-Q filings for target companies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Tickers to download. Omit to use all five target companies.",
    )
    parser.add_argument(
        "--filing-type",
        choices=["10-K", "10-Q"],
        default="10-K",
        dest="filing_type",
        help="SEC filing type.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Most recent N filings per ticker.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    fetch_filings(
        tickers=args.tickers,
        filing_type=args.filing_type,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
