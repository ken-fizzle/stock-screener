from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as pta  # noqa: F401

log = logging.getLogger(__name__)

# Columns present in the per-ticker technicals row.
INDICATOR_COLUMNS = [
    "ticker", "Close", "Volume",
    "RSI_14",
    "BB_upper", "BB_middle", "BB_lower",
    "High_252", "Low_252",
    "SMA_5", "SMA_50", "SMA_100", "SMA_200", "EMA_20",
    "Volume_SMA_20", "Volume_SMA_50",
    "RVOL_50", "Pct_of_252_High", "Pct_of_252_Low",
]


def _compute_for_ticker(df: pd.DataFrame) -> pd.Series | None:
    """
    Compute all indicators on a full OHLCV history and return the most-recent
    row as a Series.  Returns None when history is too short.
    """
    if len(df) < 20:
        return None

    d = df.copy()

    # Use direct pandas-ta functions (not the .ta accessor) to guarantee a
    # Series return value — the accessor can return a DataFrame on modified dfs.
    close = d["Close"]
    d["RSI_14"] = pta.rsi(close, length=14)

    bb = pta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        # pandas-ta names: BBU_20_2.0 / BBM_20_2.0 / BBL_20_2.0
        d["BB_upper"]  = bb.get("BBU_20_2.0")
        d["BB_middle"] = bb.get("BBM_20_2.0")
        d["BB_lower"]  = bb.get("BBL_20_2.0")
    else:
        d["BB_upper"] = d["BB_middle"] = d["BB_lower"] = float("nan")

    d["High_252"] = d["High"].rolling(252).max()
    d["Low_252"]  = d["Low"].rolling(252).min()

    d["SMA_5"]   = pta.sma(close, length=5)
    d["SMA_50"]  = pta.sma(close, length=50)
    d["SMA_100"] = pta.sma(close, length=100)
    d["SMA_200"] = pta.sma(close, length=200)
    d["EMA_20"]  = pta.ema(close, length=20)

    d["Volume_SMA_20"] = d["Volume"].rolling(20).mean()
    d["Volume_SMA_50"] = d["Volume"].rolling(50).mean()

    d["RVOL_50"]        = d["Volume"] / d["Volume_SMA_50"]
    d["Pct_of_252_High"] = d["Close"] / d["High_252"]
    d["Pct_of_252_Low"]  = d["Close"] / d["Low_252"]

    return d.iloc[-1]


def build_technicals_df(ohlcv: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute indicators for every ticker in *ohlcv* and assemble a one-row-per-
    ticker DataFrame.  Tickers with insufficient history are logged and omitted.
    """
    rows: list[pd.Series] = []

    for ticker, df in ohlcv.items():
        row = _compute_for_ticker(df)
        if row is None:
            log.warning("[%s] skipped — fewer than 20 rows", ticker)
            continue
        row = row.rename({"Close": "Close", "Volume": "Volume"})
        rows.append(row.rename(ticker))

    if not rows:
        return pd.DataFrame(columns=INDICATOR_COLUMNS)

    tech = pd.DataFrame(rows)
    tech.index.name = "ticker"
    tech = tech.reset_index()

    # Keep only the columns we care about; extras from pandas-ta are dropped.
    keep = [c for c in INDICATOR_COLUMNS if c in tech.columns]
    return tech[keep]
