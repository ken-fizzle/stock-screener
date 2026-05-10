import numpy as np
import pandas as pd
import pytest
from core.indicators import _compute_for_ticker, build_technicals_df, INDICATOR_COLUMNS


def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    close = 100 + rng.normal(0, 1, n).cumsum()
    close = np.maximum(close, 1.0)
    high  = close * (1 + rng.uniform(0, 0.02, n))
    low   = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.01, n))
    vol   = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


class TestComputeForTicker:
    def test_returns_none_for_short_history(self):
        df = _make_ohlcv(15)
        assert _compute_for_ticker(df) is None

    def test_returns_series_for_adequate_history(self):
        df = _make_ohlcv(300)
        row = _compute_for_ticker(df)
        assert isinstance(row, pd.Series)

    def test_expected_indicator_columns_present(self):
        df = _make_ohlcv(300)
        row = _compute_for_ticker(df)
        for col in ["RSI_14", "BB_upper", "BB_lower", "SMA_50", "EMA_20",
                    "RVOL_50", "Pct_of_252_High", "Pct_of_252_Low"]:
            assert col in row.index, f"Missing: {col}"

    def test_rsi_in_valid_range(self):
        df = _make_ohlcv(300)
        row = _compute_for_ticker(df)
        rsi = row["RSI_14"]
        assert 0 <= rsi <= 100

    def test_nan_for_short_history_indicators(self):
        # 50 rows: not enough for SMA_200 or High_252
        df = _make_ohlcv(50)
        row = _compute_for_ticker(df)
        assert pd.isna(row["SMA_200"])
        assert pd.isna(row["High_252"])

    def test_pct_252_high_between_zero_and_one(self):
        df = _make_ohlcv(300)
        row = _compute_for_ticker(df)
        pct = row["Pct_of_252_High"]
        if not pd.isna(pct):
            assert 0 < pct <= 1.0

    def test_bb_bands_ordered(self):
        df = _make_ohlcv(300)
        row = _compute_for_ticker(df)
        if not any(pd.isna(row[c]) for c in ["BB_lower", "BB_middle", "BB_upper"]):
            assert row["BB_lower"] <= row["BB_middle"] <= row["BB_upper"]

    def test_rvol_positive(self):
        df = _make_ohlcv(300)
        row = _compute_for_ticker(df)
        if not pd.isna(row["RVOL_50"]):
            assert row["RVOL_50"] > 0


class TestBuildTechnicalsDF:
    def test_returns_dataframe_with_ticker_column(self):
        ohlcv = {"AAPL": _make_ohlcv(300), "MSFT": _make_ohlcv(300, seed=1)}
        df = build_technicals_df(ohlcv)
        assert "ticker" in df.columns
        assert set(df["ticker"]) == {"AAPL", "MSFT"}

    def test_skips_short_history_tickers(self):
        ohlcv = {"AAPL": _make_ohlcv(300), "NEW": _make_ohlcv(10)}
        df = build_technicals_df(ohlcv)
        assert "AAPL" in df["ticker"].values
        assert "NEW" not in df["ticker"].values

    def test_empty_input_returns_empty_df(self):
        df = build_technicals_df({})
        assert df.empty

    def test_all_short_returns_empty_df(self):
        ohlcv = {"X": _make_ohlcv(5), "Y": _make_ohlcv(10)}
        df = build_technicals_df(ohlcv)
        assert df.empty
