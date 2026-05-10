from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# Explicit column list — never SELECT *; guards against schema drift.
_SELECT_COLS = (
    "f.ticker, f.trailing_pe, f.forward_pe, f.market_cap, "
    "f.sector, f.industry, f.beta, f.dividend_yield, f.price_to_book, "
    "f.target_median_price, f.recommendation_key, f.recommendation_mean, "
    "f.number_of_analyst_opinions, f.held_percent_insiders, "
    "f.held_percent_institutions, f.short_percent_of_float, f.peg_ratio, "
    "f.as_of_date"
)

FUNDAMENTALS_COLUMNS = [
    "ticker", "trailing_pe", "forward_pe", "market_cap",
    "sector", "industry", "beta", "dividend_yield", "price_to_book",
    "target_median_price", "recommendation_key", "recommendation_mean",
    "number_of_analyst_opinions", "held_percent_insiders",
    "held_percent_institutions", "short_percent_of_float", "peg_ratio",
    "as_of_date",
]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def is_db_populated(db_path: Path) -> bool:
    """Return True if the DB exists and has at least one row."""
    if not db_path.exists():
        return False
    try:
        conn = _connect(db_path)
        cur = conn.execute("SELECT 1 FROM fundamentals LIMIT 1")
        result = cur.fetchone() is not None
        conn.close()
        return result
    except Exception:
        return False


def load_for_tickers(tickers: list[str], db_path: Path) -> pd.DataFrame:
    """
    Return the latest-snapshot fundamentals row for each ticker.
    Tickers not found in the DB are simply absent from the result.
    Returns empty DataFrame (with correct columns) when DB is missing or empty.
    """
    if not tickers or not db_path.exists():
        return pd.DataFrame(columns=FUNDAMENTALS_COLUMNS)
    try:
        conn = _connect(db_path)
        placeholders = ",".join("?" * len(tickers))
        sql = f"""
            SELECT {_SELECT_COLS}
            FROM fundamentals f
            INNER JOIN (
                SELECT ticker, MAX(as_of_date) AS max_date
                FROM fundamentals
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
            ) latest
                ON f.ticker = latest.ticker
               AND f.as_of_date = latest.max_date
        """
        df = pd.read_sql_query(sql, conn, params=tickers)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=FUNDAMENTALS_COLUMNS)


def get_distinct_sectors(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    try:
        conn = _connect(db_path)
        cur = conn.execute(
            "SELECT DISTINCT sector FROM fundamentals "
            "WHERE sector IS NOT NULL ORDER BY sector"
        )
        result = [row[0] for row in cur.fetchall()]
        conn.close()
        return result
    except Exception:
        return []


def get_distinct_industries(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    try:
        conn = _connect(db_path)
        cur = conn.execute(
            "SELECT DISTINCT industry FROM fundamentals "
            "WHERE industry IS NOT NULL ORDER BY industry"
        )
        result = [row[0] for row in cur.fetchall()]
        conn.close()
        return result
    except Exception:
        return []


def get_distinct_recommendation_keys(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    try:
        conn = _connect(db_path)
        cur = conn.execute(
            "SELECT DISTINCT recommendation_key FROM fundamentals "
            "WHERE recommendation_key IS NOT NULL ORDER BY recommendation_key"
        )
        result = [row[0] for row in cur.fetchall()]
        conn.close()
        return result
    except Exception:
        return []
