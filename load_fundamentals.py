"""
CLI wrapper for the fundamentals loader.

For scheduled/headless runs outside Streamlit:
    python load_fundamentals.py --tickers tickers.csv --db data/fundamentals.db
    python load_fundamentals.py --tickers AAPL,MSFT   --db data/fundamentals.db --resume
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from core.input_parser import parse_csv_file, parse_text
from core.loader import run


def _setup_logger(log_path: Path | None) -> logging.Logger:
    logger = logging.getLogger("fundamentals_loader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def main() -> int:
    p = argparse.ArgumentParser(
        description="Load quarterly fundamentals snapshot from yfinance into SQLite"
    )
    p.add_argument("--tickers", required=True,
                   help="Path to a CSV file of tickers, or a comma/space-separated string")
    p.add_argument("--db", default="data/fundamentals.db",
                   help="SQLite database path (default: data/fundamentals.db)")
    p.add_argument("--resume", action="store_true",
                   help="Skip tickers already loaded for today's as_of_date")
    p.add_argument("--as-of", default=date.today().isoformat(),
                   help="Override as_of_date (default: today, YYYY-MM-DD)")
    p.add_argument("--log-file", default="logs/fundamentals_loader.log")
    args = p.parse_args()

    logger = _setup_logger(Path(args.log_file) if args.log_file else None)

    src = Path(args.tickers)
    if src.exists() and src.is_file():
        tickers, rejected = parse_csv_file(src)
    else:
        tickers, rejected = parse_text(args.tickers)

    if rejected:
        logger.warning("Rejected %d symbols: %s", len(rejected),
                       ", ".join(rejected[:10]) + ("…" if len(rejected) > 10 else ""))
    if not tickers:
        logger.error("No valid tickers.")
        return 1

    logger.info("Parsed %d unique tickers.", len(tickers))
    result = run(
        tickers=tickers,
        db_path=Path(args.db),
        resume=args.resume,
        as_of=args.as_of,
        logger=logger,
    )
    return 0 if result.success > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
