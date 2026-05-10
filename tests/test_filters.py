import numpy as np
import pandas as pd
import pytest
from core.filters import (
    apply_fundamental_filters,
    apply_technical_filters,
    apply_all_filters,
    is_any_fundamental_filter_active,
)


def _fund_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker":                    ["AAPL",  "MSFT",  "GOOG",   "XOM"],
        "sector":                    ["Tech",  "Tech",  "Tech",   "Energy"],
        "industry":                  ["HW",    "SW",    "Search", "Oil"],
        "market_cap":                [3e12,    2e12,    1.5e12,   4e11],
        "trailing_pe":               [28.0,    35.0,    25.0,     12.0],
        "forward_pe":                [25.0,    30.0,    22.0,     11.0],
        "beta":                      [1.2,     0.9,     1.1,      0.7],
        "dividend_yield":            [0.006,   0.009,   0.0,      0.035],
        "price_to_book":             [45.0,    12.0,    6.0,      2.0],
        # New fundamental fields
        "peg_ratio":                 [2.5,     2.8,     1.8,      1.2],
        "recommendation_key":        ["buy",   "buy",   "hold",   "hold"],
        "recommendation_mean":       [2.0,     1.8,     2.5,      2.8],
        "number_of_analyst_opinions":[50,      45,      35,       20],
        "held_percent_insiders":     [0.07,    0.25,    0.06,     0.01],
        "held_percent_institutions": [0.60,    0.70,    0.65,     0.55],
        "short_percent_of_float":    [0.008,   0.007,   0.015,    0.025],
        "target_median_price":       [210.0,   400.0,   120.0,    130.0],
    })


def _tech_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker":          ["AAPL",  "MSFT",  "GOOG",  "XOM"],
        "Close":           [190.0,   380.0,   140.0,   110.0],
        "Volume":          [60e6,    25e6,    20e6,    15e6],
        "RSI_14":          [65.0,    55.0,    40.0,    30.0],
        "SMA_5":           [188.0,   375.0,   142.0,   108.0],
        "SMA_50":          [180.0,   360.0,   150.0,   105.0],
        "SMA_100":         [175.0,   350.0,   155.0,   100.0],
        "SMA_200":         [170.0,   340.0,   160.0,   95.0],
        "EMA_20":          [185.0,   370.0,   145.0,   107.0],
        "BB_upper":        [200.0,   400.0,   155.0,   115.0],
        "BB_lower":        [178.0,   355.0,   130.0,   100.0],
        "BB_middle":       [189.0,   377.0,   142.0,   107.0],
        "RVOL_50":         [1.5,     1.1,     0.8,     2.0],
        "Pct_of_252_High": [0.95,    0.90,    0.80,    0.70],
        "Pct_of_252_Low":  [1.30,    1.20,    1.10,    1.05],
        "High_252":        [200.0,   420.0,   175.0,   157.0],
        "Low_252":         [146.0,   316.0,   127.0,   105.0],
        "Volume_SMA_20":   [55e6,    22e6,    18e6,    12e6],
        "Volume_SMA_50":   [40e6,    23e6,    25e6,    7.5e6],
    })


def _combined_df() -> pd.DataFrame:
    df = _fund_df().merge(_tech_df(), on="ticker")
    # Derive upside_to_target as app.py does — stored as decimal, not percentage points.
    df["upside_to_target"] = (
        (df["target_median_price"] - df["Close"]) / df["Close"]
    )
    return df


# ---------------------------------------------------------------------------
# is_any_fundamental_filter_active
# ---------------------------------------------------------------------------

def test_no_filters_active():
    spec = {"sector": {"enabled": False}, "market_cap": {"enabled": False}}
    assert not is_any_fundamental_filter_active(spec)


def test_one_filter_active():
    spec = {"sector": {"enabled": True, "values": ["Tech"]}}
    assert is_any_fundamental_filter_active(spec)


def test_new_fundamental_key_detected_as_active():
    spec = {"peg_ratio": {"enabled": True, "min": 0.0, "max": 2.0}}
    assert is_any_fundamental_filter_active(spec)


# ---------------------------------------------------------------------------
# Fundamental filters — existing
# ---------------------------------------------------------------------------

def test_sector_filter():
    df = _fund_df()
    spec = {"sector": {"enabled": True, "values": ["Tech"]}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "MSFT", "GOOG"}


def test_sector_filter_disabled():
    df = _fund_df()
    spec = {"sector": {"enabled": False, "values": ["Tech"]}}
    result = apply_fundamental_filters(df, spec)
    assert len(result) == len(df)


def test_industry_filter():
    df = _fund_df()
    spec = {"industry": {"enabled": True, "values": ["HW", "Oil"]}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "XOM"}


def test_market_cap_filter():
    df = _fund_df()
    spec = {"market_cap": {"enabled": True, "min": 1e12, "max": 2.5e12}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"MSFT", "GOOG"}


def test_trailing_pe_range():
    df = _fund_df()
    spec = {"trailing_pe": {"enabled": True, "min": 10.0, "max": 30.0, "exclude_negative": False}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "GOOG", "XOM"}


def test_pe_exclude_negative():
    df = _fund_df().copy()
    df.loc[df["ticker"] == "XOM", "trailing_pe"] = -5.0
    spec = {"trailing_pe": {"enabled": True, "min": -100.0, "max": 200.0, "exclude_negative": True}}
    result = apply_fundamental_filters(df, spec)
    assert "XOM" not in result["ticker"].values


def test_dividend_yield_filter_uses_percent():
    df = _fund_df()
    spec = {"dividend_yield": {"enabled": True, "min": 3.0, "max": 20.0}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"XOM"}


def test_multiple_fundamental_filters_combined():
    df = _fund_df()
    spec = {
        "sector":     {"enabled": True, "values": ["Tech"]},
        "market_cap": {"enabled": True, "min": 1.8e12, "max": 5e12},
    }
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# Fundamental filters — new fields
# ---------------------------------------------------------------------------

def test_peg_ratio_filter():
    df = _fund_df()
    spec = {"peg_ratio": {"enabled": True, "min": 0.0, "max": 2.0}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"GOOG", "XOM"}


def test_peg_ratio_filter_disabled():
    df = _fund_df()
    spec = {"peg_ratio": {"enabled": False, "min": 0.0, "max": 0.5}}
    result = apply_fundamental_filters(df, spec)
    assert len(result) == len(df)


def test_recommendation_key_filter():
    df = _fund_df()
    spec = {"recommendation_key": {"enabled": True, "values": ["buy"]}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "MSFT"}


def test_recommendation_key_hold():
    df = _fund_df()
    spec = {"recommendation_key": {"enabled": True, "values": ["hold"]}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"GOOG", "XOM"}


def test_recommendation_key_multiple_values():
    df = _fund_df()
    spec = {"recommendation_key": {"enabled": True, "values": ["buy", "hold"]}}
    result = apply_fundamental_filters(df, spec)
    assert len(result) == 4


def test_recommendation_mean_filter():
    df = _fund_df()
    # Mean <= 2.0: AAPL (2.0) and MSFT (1.8)
    spec = {"recommendation_mean": {"enabled": True, "min": 1.0, "max": 2.0}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "MSFT"}


def test_num_analyst_opinions_filter():
    df = _fund_df()
    # >= 40 analysts: AAPL (50), MSFT (45)
    spec = {"num_analyst_opinions": {"enabled": True, "min": 40, "max": 200}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "MSFT"}


def test_held_pct_insiders_filter():
    df = _fund_df()
    # MSFT insiders = 25%
    spec = {"held_pct_insiders": {"enabled": True, "min": 20.0, "max": 100.0}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"MSFT"}


def test_held_pct_institutions_filter():
    df = _fund_df()
    # >= 65%: MSFT (70%), GOOG (65%)
    spec = {"held_pct_institutions": {"enabled": True, "min": 65.0, "max": 100.0}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"MSFT", "GOOG"}


def test_short_pct_float_filter():
    df = _fund_df()
    # >= 2%: XOM (2.5%)
    spec = {"short_pct_float": {"enabled": True, "min": 2.0, "max": 100.0}}
    result = apply_fundamental_filters(df, spec)
    assert set(result["ticker"]) == {"XOM"}


# ---------------------------------------------------------------------------
# Technical filters — existing
# ---------------------------------------------------------------------------

def test_rsi_filter():
    df = _combined_df()
    spec = {"rsi_14": {"enabled": True, "min": 60.0, "max": 100.0}}
    result = apply_technical_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL"}


def test_close_above_sma200():
    df = _combined_df()
    spec = {"close_vs_sma200": {"enabled": True, "direction": "above"}}
    result = apply_technical_filters(df, spec)
    # GOOG: Close=140 < SMA_200=160 → excluded; AAPL, MSFT, XOM pass
    assert set(result["ticker"]) == {"AAPL", "MSFT", "XOM"}


def test_close_below_sma50():
    df = _combined_df()
    # GOOG: Close=140 < SMA_50=150
    spec = {"close_vs_sma50": {"enabled": True, "direction": "below"}}
    result = apply_technical_filters(df, spec)
    assert "GOOG" in result["ticker"].values


def test_close_either_direction_no_filter():
    df = _combined_df()
    spec = {"close_vs_sma50": {"enabled": True, "direction": "either"}}
    result = apply_technical_filters(df, spec)
    assert len(result) == len(df)


def test_rvol_filter():
    df = _combined_df()
    spec = {"rvol_50": {"enabled": True, "min": 1.4, "max": 10.0}}
    result = apply_technical_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "XOM"}


def test_pct_252_high_filter():
    df = _combined_df()
    spec = {"pct_252_high": {"enabled": True, "min": 90.0, "max": 100.0}}
    result = apply_technical_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "MSFT"}


def test_technical_filter_disabled():
    df = _combined_df()
    spec = {"rsi_14": {"enabled": False, "min": 70.0, "max": 100.0}}
    result = apply_technical_filters(df, spec)
    assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Technical filters — new (upside_to_target)
# ---------------------------------------------------------------------------

def test_target_upside_positive_only():
    df = _combined_df()
    # Stored as decimals: AAPL +0.105, MSFT +0.053, GOOG -0.143, XOM +0.182
    # Slider min=0 → lo = 0/100 = 0.0 → GOOG excluded
    spec = {"target_upside": {"enabled": True, "min": 0.0, "max": 200.0}}
    result = apply_technical_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL", "MSFT", "XOM"}


def test_target_upside_high_threshold():
    df = _combined_df()
    # Slider min=15 → lo = 15/100 = 0.15 → only XOM (0.182) passes
    spec = {"target_upside": {"enabled": True, "min": 15.0, "max": 200.0}}
    result = apply_technical_filters(df, spec)
    assert set(result["ticker"]) == {"XOM"}


def test_target_upside_disabled():
    df = _combined_df()
    spec = {"target_upside": {"enabled": False, "min": 50.0, "max": 200.0}}
    result = apply_technical_filters(df, spec)
    assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Combined AND logic
# ---------------------------------------------------------------------------

def test_and_combination_fundamental_and_technical():
    df = _combined_df()
    spec = {
        "sector":  {"enabled": True, "values": ["Tech"]},
        "rsi_14":  {"enabled": True, "min": 60.0, "max": 100.0},
    }
    result = apply_all_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL"}


def test_and_combination_new_fields():
    df = _combined_df()
    spec = {
        "recommendation_key": {"enabled": True, "values": ["buy"]},
        "target_upside":      {"enabled": True, "min": 8.0, "max": 200.0},
        # slider min=8 → lo=0.08; AAPL=0.105 ✓, MSFT=0.053 ✗
    }
    result = apply_all_filters(df, spec)
    assert set(result["ticker"]) == {"AAPL"}
