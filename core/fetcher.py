from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

CACHE_DIR = Path("cache/ohlcv")
BATCH_SIZE = 50
MAX_RETRIES = 3
BACKOFF = (2, 5, 10)
STALENESS_HOURS = 24

# Use unadjusted Close so indicator values match what a standard price chart
# shows at the raw price level. Adj Close would be more accurate for long
# historical comparisons but introduces inconsistency vs. live quotes.
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

ProgressCallback = Callable[[int, int, str], None]


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=STALENESS_HOURS)


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if path.exists() and not _is_stale(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    path = _cache_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _fetch_single(ticker: str) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.Ticker(ticker).history(
                period="5y", interval="1d", auto_adjust=False
            )
            if df.empty:
                return None
            return df[OHLCV_COLS].dropna(how="all")
        except Exception as exc:
            log.warning("[%s] single-fetch attempt %d failed: %s", ticker, attempt + 1, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF[attempt])
    return None


def _fetch_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download a batch via yf.download(). Returns whatever succeeded."""
    if not tickers:
        return {}
    if len(tickers) == 1:
        # yf.download of a single ticker returns a flat DataFrame, not MultiIndex.
        df = _fetch_single(tickers[0])
        return {tickers[0]: df} if df is not None else {}

    try:
        raw = yf.download(
            tickers=tickers,
            period="5y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception as exc:
        log.warning("Batch download failed: %s", exc)
        return {}

    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = raw[ticker][OHLCV_COLS].dropna(how="all")
            if not df.empty:
                result[ticker] = df
        except (KeyError, Exception):
            pass
    return result


def fetch_ohlcv(
    tickers: list[str],
    force_refresh: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch 5-year daily OHLCV for each ticker.

    Loads from parquet cache when available and fresh (< 24 h old).
    Batch-downloads stale/missing tickers via yf.download(), with per-ticker
    fallback for anything the batch misses.

    Returns dict[ticker -> DataFrame]. Tickers with < 20 rows are skipped
    (insufficient for any indicator computation).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    to_fetch: list[str] = []
    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        if not force_refresh:
            cached = _load_cache(ticker)
            if cached is not None:
                result[ticker] = cached
                continue
        to_fetch.append(ticker)

    if not to_fetch:
        if progress_cb:
            progress_cb(len(result), len(tickers), "All tickers loaded from cache")
        return result

    batches = [to_fetch[i : i + BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE)]
    batch_failed: list[str] = []

    for b_idx, batch in enumerate(batches):
        batch_data = _fetch_batch(batch)
        for ticker in batch:
            if ticker in batch_data:
                df = batch_data[ticker]
                if len(df) >= 20:
                    _save_cache(ticker, df)
                    result[ticker] = df
                else:
                    log.warning("[%s] only %d rows — skipping", ticker, len(df))
            else:
                batch_failed.append(ticker)

        if progress_cb:
            done = len(result)
            progress_cb(done, len(tickers),
                        f"Batch {b_idx + 1}/{len(batches)} downloaded")

        if b_idx < len(batches) - 1:
            time.sleep(0.5)

    # Per-ticker fallback for anything the batch missed.
    for ticker in batch_failed:
        df = _fetch_single(ticker)
        if df is not None and len(df) >= 20:
            _save_cache(ticker, df)
            result[ticker] = df
        else:
            log.warning("[%s] fallback fetch returned no usable data", ticker)

    if progress_cb:
        progress_cb(len(result), len(tickers), "OHLCV fetch complete")

    return result


def clear_cache() -> int:
    """Delete all parquet files in the cache directory. Returns file count deleted."""
    count = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
            count += 1
    return count
