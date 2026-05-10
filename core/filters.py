from __future__ import annotations

from typing import Any

import pandas as pd

FilterSpec = dict[str, Any]

# Keys whose filters operate on fundamentals columns (used to decide whether
# we can skip OHLCV fetching for tickers that fail fundamental screens).
FUNDAMENTAL_KEYS = frozenset({
    "sector", "industry", "market_cap",
    "trailing_pe", "forward_pe", "beta", "dividend_yield", "price_to_book",
    "peg_ratio", "recommendation_key", "recommendation_mean",
    "num_analyst_opinions", "held_pct_insiders", "held_pct_institutions",
    "short_pct_float",
})


def is_any_fundamental_filter_active(spec: FilterSpec) -> bool:
    return any(spec.get(k, {}).get("enabled", False) for k in FUNDAMENTAL_KEYS)


# ---------------------------------------------------------------------------
# Fundamental filters  (applied to the fundamentals DataFrame)
# ---------------------------------------------------------------------------

def apply_fundamental_filters(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    """
    Apply all enabled fundamental filters to *df* (which must have the
    fundamentals columns).  Returns a filtered copy.
    """
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    s = spec.get("sector", {})
    if s.get("enabled") and s.get("values"):
        mask &= df["sector"].isin(s["values"])

    s = spec.get("industry", {})
    if s.get("enabled") and s.get("values"):
        mask &= df["industry"].isin(s["values"])

    s = spec.get("market_cap", {})
    if s.get("enabled"):
        lo, hi = s.get("min", 0.0), s.get("max", float("inf"))
        mask &= df["market_cap"].between(lo, hi)

    for key, col in [("trailing_pe", "trailing_pe"), ("forward_pe", "forward_pe")]:
        s = spec.get(key, {})
        if s.get("enabled"):
            lo, hi = s.get("min", -100.0), s.get("max", 200.0)
            col_mask = df[col].between(lo, hi)
            if s.get("exclude_negative"):
                col_mask &= df[col] >= 0
            mask &= col_mask

    s = spec.get("beta", {})
    if s.get("enabled"):
        mask &= df["beta"].between(s.get("min", -2.0), s.get("max", 5.0))

    s = spec.get("dividend_yield", {})
    if s.get("enabled"):
        lo = s.get("min", 0.0) / 100.0
        hi = s.get("max", 20.0) / 100.0
        mask &= df["dividend_yield"].between(lo, hi)

    s = spec.get("price_to_book", {})
    if s.get("enabled"):
        mask &= df["price_to_book"].between(s.get("min", 0.0), s.get("max", 50.0))

    s = spec.get("peg_ratio", {})
    if s.get("enabled") and "peg_ratio" in df.columns:
        mask &= df["peg_ratio"].between(s.get("min", 0.0), s.get("max", 10.0))

    s = spec.get("recommendation_key", {})
    if s.get("enabled") and s.get("values") and "recommendation_key" in df.columns:
        mask &= df["recommendation_key"].isin(s["values"])

    s = spec.get("recommendation_mean", {})
    if s.get("enabled") and "recommendation_mean" in df.columns:
        mask &= df["recommendation_mean"].between(s.get("min", 1.0), s.get("max", 5.0))

    s = spec.get("num_analyst_opinions", {})
    if s.get("enabled") and "number_of_analyst_opinions" in df.columns:
        mask &= df["number_of_analyst_opinions"].between(
            s.get("min", 0), s.get("max", 200)
        )

    s = spec.get("held_pct_insiders", {})
    if s.get("enabled") and "held_percent_insiders" in df.columns:
        lo = s.get("min", 0.0) / 100.0
        hi = s.get("max", 100.0) / 100.0
        mask &= df["held_percent_insiders"].between(lo, hi)

    s = spec.get("held_pct_institutions", {})
    if s.get("enabled") and "held_percent_institutions" in df.columns:
        lo = s.get("min", 0.0) / 100.0
        hi = s.get("max", 100.0) / 100.0
        mask &= df["held_percent_institutions"].between(lo, hi)

    s = spec.get("short_pct_float", {})
    if s.get("enabled") and "short_percent_of_float" in df.columns:
        lo = s.get("min", 0.0) / 100.0
        hi = s.get("max", 100.0) / 100.0
        mask &= df["short_percent_of_float"].between(lo, hi)

    return df[mask].copy()


# ---------------------------------------------------------------------------
# Technical filters  (applied to the combined fundamentals+technicals DataFrame)
# ---------------------------------------------------------------------------

def apply_technical_filters(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    s = spec.get("rsi_14", {})
    if s.get("enabled"):
        mask &= df["RSI_14"].between(s.get("min", 0.0), s.get("max", 100.0))

    _DIRECTION_MAP = {
        "close_vs_sma5":      "SMA_5",
        "close_vs_sma50":     "SMA_50",
        "close_vs_sma100":    "SMA_100",
        "close_vs_sma200":    "SMA_200",
        "close_vs_ema20":     "EMA_20",
        "close_vs_bb_upper":  "BB_upper",
        "close_vs_bb_lower":  "BB_lower",
    }
    for key, col in _DIRECTION_MAP.items():
        s = spec.get(key, {})
        if s.get("enabled") and col in df.columns:
            direction = s.get("direction", "either")
            if direction == "above":
                mask &= df["Close"] > df[col]
            elif direction == "below":
                mask &= df["Close"] < df[col]

    s = spec.get("pct_252_high", {})
    if s.get("enabled"):
        lo = s.get("min", 0.0) / 100.0
        hi = s.get("max", 100.0) / 100.0
        mask &= df["Pct_of_252_High"].between(lo, hi)

    s = spec.get("pct_252_low", {})
    if s.get("enabled"):
        lo = s.get("min", 0.0) / 100.0
        hi = s.get("max", 100.0) / 100.0
        mask &= df["Pct_of_252_Low"].between(lo, hi)

    s = spec.get("rvol_50", {})
    if s.get("enabled"):
        mask &= df["RVOL_50"].between(s.get("min", 0.0), s.get("max", 10.0))

    # upside_to_target is stored as a decimal (0.153 = 15.3%); slider is in %.
    s = spec.get("target_upside", {})
    if s.get("enabled") and "upside_to_target" in df.columns:
        lo = s.get("min", -100.0) / 100.0
        hi = s.get("max",  200.0) / 100.0
        mask &= df["upside_to_target"].between(lo, hi)

    return df[mask].copy()


# ---------------------------------------------------------------------------
# Combined helper
# ---------------------------------------------------------------------------

def apply_all_filters(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    """Convenience: apply fundamental then technical filters in one call."""
    df = apply_fundamental_filters(df, spec)
    df = apply_technical_filters(df, spec)
    return df
