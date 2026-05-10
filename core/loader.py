"""
Fundamentals loader — core logic extracted from the CLI script.
Called by both the Streamlit refresh page and the standalone load_fundamentals.py CLI.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf

FIELD_MAP: dict[str, str] = {
    "trailingPE":               "trailing_pe",
    "forwardPE":                "forward_pe",
    "marketCap":                "market_cap",
    "sector":                   "sector",
    "industry":                 "industry",
    "beta":                     "beta",
    "dividendYield":            "dividend_yield",
    "priceToBook":              "price_to_book",
    "targetMedianPrice":        "target_median_price",
    "recommendationKey":        "recommendation_key",
    "recommendationMean":       "recommendation_mean",
    "numberOfAnalystOpinions":  "number_of_analyst_opinions",
    "heldPercentInsiders":      "held_percent_insiders",
    "heldPercentInstitutions":  "held_percent_institutions",
    "shortPercentOfFloat":      "short_percent_of_float",
    "pegRatio":                 "peg_ratio",
}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker                      TEXT    NOT NULL,
    as_of_date                  TEXT    NOT NULL,
    trailing_pe                 REAL,
    forward_pe                  REAL,
    market_cap                  REAL,
    sector                      TEXT,
    industry                    TEXT,
    beta                        REAL,
    dividend_yield              REAL,
    price_to_book               REAL,
    target_median_price         REAL,
    recommendation_key          TEXT,
    recommendation_mean         REAL,
    number_of_analyst_opinions  INTEGER,
    held_percent_insiders       REAL,
    held_percent_institutions   REAL,
    short_percent_of_float      REAL,
    peg_ratio                   REAL,
    fetched_at                  TEXT    NOT NULL,
    PRIMARY KEY (ticker, as_of_date)
)
"""

# Columns added after the initial schema — applied as migrations on older DBs.
_NEW_COLUMNS: dict[str, str] = {
    "target_median_price":        "REAL",
    "recommendation_key":         "TEXT",
    "recommendation_mean":        "REAL",
    "number_of_analyst_opinions": "INTEGER",
    "held_percent_insiders":      "REAL",
    "held_percent_institutions":  "REAL",
    "short_percent_of_float":     "REAL",
    "peg_ratio":                  "REAL",
}

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_fund_ticker   ON fundamentals(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_fund_date     ON fundamentals(as_of_date)",
    "CREATE INDEX IF NOT EXISTS idx_fund_sector   ON fundamentals(sector)",
]

RETRY_ATTEMPTS = 3
RETRY_BACKOFF = (2, 5, 15)
INTER_REQUEST_SLEEP = 0.5


@dataclass
class LoadResult:
    success: int = 0
    failure: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _migrate_db(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema without touching existing rows."""
    cur = conn.execute("PRAGMA table_info(fundamentals)")
    existing = {row[1] for row in cur.fetchall()}
    for col, dtype in _NEW_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE fundamentals ADD COLUMN {col} {dtype}")
    conn.commit()


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_CREATE_TABLE)
    for stmt in _CREATE_INDEXES:
        conn.execute(stmt)
    _migrate_db(conn)
    conn.commit()
    return conn


def already_loaded(conn: sqlite3.Connection, as_of: str) -> set[str]:
    cur = conn.execute(
        "SELECT ticker FROM fundamentals WHERE as_of_date = ?", (as_of,)
    )
    return {row[0] for row in cur.fetchall()}


def _insert(conn: sqlite3.Connection, ticker: str, as_of: str, data: dict) -> None:
    cols = ["ticker", "as_of_date"] + list(FIELD_MAP.values()) + ["fetched_at"]
    sql = (
        f"INSERT OR REPLACE INTO fundamentals ({', '.join(cols)}) "
        f"VALUES ({', '.join(['?'] * len(cols))})"
    )
    values = (
        [ticker, as_of]
        + [data.get(col) for col in FIELD_MAP.values()]
        + [pd.Timestamp.now().isoformat(timespec="seconds")]
    )
    conn.execute(sql, values)
    conn.commit()


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_one(ticker: str, logger: logging.Logger) -> tuple[bool, dict | None, str | None]:
    """Returns (success, data, error_msg)."""
    last_err: str | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            info = yf.Ticker(ticker).info
            if not info or (info.get("symbol") is None and info.get("shortName") is None):
                return False, None, "empty info payload (likely delisted or invalid)"
            extracted = {
                sqlite_col: info.get(yf_key)
                for yf_key, sqlite_col in FIELD_MAP.items()
            }
            return True, extracted, None
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < RETRY_ATTEMPTS - 1:
                wait = RETRY_BACKOFF[attempt]
                logger.warning("[%s] attempt %d failed (%s); retrying in %ds",
                               ticker, attempt + 1, last_err, wait)
                time.sleep(wait)
            else:
                logger.error("[%s] all %d attempts failed", ticker, RETRY_ATTEMPTS)
    return False, None, last_err


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, str], None]


def run(
    tickers: list[str],
    db_path: Path,
    resume: bool,
    as_of: str,
    logger: logging.Logger,
    progress_cb: ProgressCallback | None = None,
) -> LoadResult:
    """
    Load fundamentals for *tickers* into SQLite at *db_path*.

    *progress_cb(current, total, message)* is called after each ticker is
    processed — use it to update a Streamlit progress bar or log to console.
    """
    conn = init_db(db_path)
    logger.info("DB initialized at %s  |  as_of_date=%s", db_path, as_of)

    work = tickers
    if resume:
        skip = already_loaded(conn, as_of)
        if skip:
            logger.info("Resume: skipping %d already-loaded tickers", len(skip))
            work = [t for t in tickers if t not in skip]

    total = len(work)
    result = LoadResult()

    if total == 0:
        logger.info("Nothing to do.")
        conn.close()
        return result

    logger.info("Fetching %d tickers from yfinance …", total)
    start = time.time()

    for idx, ticker in enumerate(work, start=1):
        ok, data, err = _fetch_one(ticker, logger)
        if ok and data is not None:
            _insert(conn, ticker, as_of, data)
            result.success += 1
        else:
            result.failure += 1
            result.failed.append((ticker, err or "unknown"))
            logger.warning("[%s] FAILED: %s", ticker, err)

        if progress_cb:
            progress_cb(idx, total, f"{idx}/{total} — {ticker}")

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - start
            rate = idx / elapsed if elapsed else 0
            eta = (total - idx) / rate if rate else 0
            logger.info(
                "Progress %d/%d  %.1f/s  ETA %.1fmin  failures=%d",
                idx, total, rate, eta / 60, result.failure,
            )

        time.sleep(INTER_REQUEST_SLEEP)

    conn.close()
    elapsed_total = time.time() - start
    logger.info("Done in %.1f min — success=%d  failure=%d",
                elapsed_total / 60, result.success, result.failure)
    return result
