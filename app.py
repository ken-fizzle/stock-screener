"""
Stock Screener — main Streamlit page.

UI-only: no business logic lives here. All computation is delegated to core/.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core.fetcher import clear_cache, fetch_ohlcv
from core.filters import (
    apply_all_filters,
    apply_fundamental_filters,
    apply_technical_filters,
    is_any_fundamental_filter_active,
)
from core.fundamentals import (
    get_distinct_industries,
    get_distinct_recommendation_keys,
    get_distinct_sectors,
    is_db_populated,
    load_for_tickers,
)
from core.indicators import build_technicals_df
from core.input_parser import parse_csv_bytes, parse_csv_file, parse_text

DB_PATH = Path("data/fundamentals.db")

BUILTIN_LISTS: dict[str, Path] = {
    "S&P 500 (~500 tickers)":      Path("sp500_tickers.csv"),
    "US Broad (~2,700 tickers)":   Path("US2700_tickers.csv"),
}

DISPLAY_COLUMNS = [
    # Identifiers & company info
    "ticker", "company_name", "sector", "industry",
    # Valuation
    "market_cap", "trailing_pe", "forward_pe", "price_to_book", "peg_ratio",
    # Risk / income
    "beta", "dividend_yield",
    # Analyst ratings
    "recommendation_key", "recommendation_mean", "number_of_analyst_opinions",
    "target_median_price", "upside_to_target",
    # Ownership & short interest
    "held_percent_insiders", "held_percent_institutions", "short_percent_of_float",
    # Technical — price & volume
    "Close", "Volume",
    # Technical — momentum
    "RSI_14", "RVOL_50", "Pct_of_252_High", "Pct_of_252_Low",
    # Technical — moving averages
    "SMA_5", "SMA_50", "SMA_100", "SMA_200", "EMA_20",
    # Technical — Bollinger Bands
    "BB_upper", "BB_middle", "BB_lower",
]

# ---------------------------------------------------------------------------
# Display formatters
# ---------------------------------------------------------------------------

def _fmt_market_cap(v: float) -> str:
    if pd.isna(v):
        return "—"
    if abs(v) >= 1e12:
        return f"{v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.1f}M"
    return f"{v:,.0f}"


def _fmt_pct(v: float) -> str:
    return f"{v*100:.2f}%" if pd.notna(v) else "—"


def _fmt_signed_pct(v: float) -> str:
    if pd.isna(v):
        return "—"
    pct = v * 100
    return f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"


def _fmt_2dp(v: float) -> str:
    return f"{v:.2f}" if pd.notna(v) else "—"


def _fmt_int(v: float) -> str:
    return f"{int(v)}" if pd.notna(v) else "—"


def _build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready copy with human-friendly formatted columns."""
    d = df.copy()
    present = [c for c in DISPLAY_COLUMNS if c in d.columns]
    d = d[present]

    if "market_cap" in d.columns:
        d["market_cap"] = d["market_cap"].map(_fmt_market_cap)
    for col in ["dividend_yield", "held_percent_insiders",
                "held_percent_institutions", "short_percent_of_float"]:
        if col in d.columns:
            d[col] = d[col].map(_fmt_pct)
    for col in ["Pct_of_252_High", "Pct_of_252_Low"]:
        if col in d.columns:
            d[col] = d[col].map(_fmt_pct)
    if "upside_to_target" in d.columns:
        d["upside_to_target"] = d["upside_to_target"].map(_fmt_signed_pct)
    if "number_of_analyst_opinions" in d.columns:
        d["number_of_analyst_opinions"] = d["number_of_analyst_opinions"].map(_fmt_int)
    float_cols = [
        "trailing_pe", "forward_pe", "beta", "price_to_book", "peg_ratio",
        "recommendation_mean", "target_median_price",
        "Close", "RSI_14", "RVOL_50",
        "SMA_5", "SMA_50", "SMA_100", "SMA_200", "EMA_20",
        "BB_upper", "BB_middle", "BB_lower",
    ]
    for col in float_cols:
        if col in d.columns:
            d[col] = d[col].map(_fmt_2dp)
    return d


# ---------------------------------------------------------------------------
# Filter spec assembly
# ---------------------------------------------------------------------------

def _build_filter_spec(controls: dict) -> dict:
    c = controls
    return {
        # Fundamental — existing
        "sector":           {"enabled": c["sector_en"],        "values": c["sectors"]},
        "industry":         {"enabled": c["industry_en"],      "values": c["industries"]},
        "market_cap":       {"enabled": c["mc_en"],            "min": c["mc_min"],      "max": c["mc_max"]},
        "trailing_pe":      {"enabled": c["tpe_en"],           "min": c["tpe_min"],     "max": c["tpe_max"],
                             "exclude_negative": c["tpe_excl_neg"]},
        "forward_pe":       {"enabled": c["fpe_en"],           "min": c["fpe_min"],     "max": c["fpe_max"],
                             "exclude_negative": c["fpe_excl_neg"]},
        "beta":             {"enabled": c["beta_en"],          "min": c["beta_min"],    "max": c["beta_max"]},
        "dividend_yield":   {"enabled": c["div_en"],           "min": c["div_min"],     "max": c["div_max"]},
        "price_to_book":    {"enabled": c["pb_en"],            "min": c["pb_min"],      "max": c["pb_max"]},
        # Fundamental — new
        "peg_ratio":           {"enabled": c["peg_en"],           "min": c["peg_min"],        "max": c["peg_max"]},
        "recommendation_key":  {"enabled": c["rec_key_en"],       "values": c["rec_keys"]},
        "recommendation_mean": {"enabled": c["rec_mean_en"],      "min": c["rec_mean_min"],   "max": c["rec_mean_max"]},
        "num_analyst_opinions":{"enabled": c["num_analysts_en"],  "min": c["num_analysts_min"],"max": c["num_analysts_max"]},
        "held_pct_insiders":   {"enabled": c["insiders_en"],      "min": c["insiders_min"],   "max": c["insiders_max"]},
        "held_pct_institutions":{"enabled": c["institutions_en"], "min": c["instit_min"],     "max": c["instit_max"]},
        "short_pct_float":     {"enabled": c["short_float_en"],   "min": c["short_float_min"],"max": c["short_float_max"]},
        # Technical — existing
        "rsi_14":           {"enabled": c["rsi_en"],           "min": c["rsi_min"],     "max": c["rsi_max"]},
        "close_vs_sma5":    {"enabled": c["sma5_en"],          "direction": c["sma5_dir"]},
        "close_vs_sma50":   {"enabled": c["sma50_en"],         "direction": c["sma50_dir"]},
        "close_vs_sma100":  {"enabled": c["sma100_en"],        "direction": c["sma100_dir"]},
        "close_vs_sma200":  {"enabled": c["sma200_en"],        "direction": c["sma200_dir"]},
        "close_vs_ema20":   {"enabled": c["ema20_en"],         "direction": c["ema20_dir"]},
        "close_vs_bb_upper":{"enabled": c["bbu_en"],           "direction": c["bbu_dir"]},
        "close_vs_bb_lower":{"enabled": c["bbl_en"],           "direction": c["bbl_dir"]},
        "pct_252_high":     {"enabled": c["p252h_en"],         "min": c["p252h_min"],   "max": c["p252h_max"]},
        "pct_252_low":      {"enabled": c["p252l_en"],         "min": c["p252l_min"],   "max": c["p252l_max"]},
        "rvol_50":          {"enabled": c["rvol_en"],          "min": c["rvol_min"],    "max": c["rvol_max"]},
        # Technical — new (derived after join)
        "target_upside":    {"enabled": c["target_upside_en"], "min": c["target_upside_min"], "max": c["target_upside_max"]},
    }


# ---------------------------------------------------------------------------
# Sidebar filter widgets
# ---------------------------------------------------------------------------

def _render_sidebar_filters(db_populated: bool) -> dict:
    sectors      = get_distinct_sectors(DB_PATH)             if db_populated else []
    industries   = get_distinct_industries(DB_PATH)          if db_populated else []
    rec_keys     = get_distinct_recommendation_keys(DB_PATH) if db_populated else []

    controls: dict = {}

    st.sidebar.markdown("### Fundamental Filters")
    if not db_populated:
        st.sidebar.caption("⚠️ Fundamentals DB empty — fundamental filters unavailable.")

    with st.sidebar.expander("Sector / Industry", expanded=False):
        col1, col2 = st.columns([1, 3])
        controls["sector_en"] = col1.checkbox("Sector", key="f_sector_en", disabled=not db_populated)
        controls["sectors"]   = col2.multiselect("", sectors, key="f_sectors",
                                                  disabled=not controls["sector_en"])
        col1, col2 = st.columns([1, 3])
        controls["industry_en"]  = col1.checkbox("Industry", key="f_industry_en", disabled=not db_populated)
        controls["industries"]   = col2.multiselect("", industries, key="f_industries",
                                                     disabled=not controls["industry_en"])

    with st.sidebar.expander("Valuation", expanded=False):
        col1, col2 = st.columns([1, 3])
        controls["mc_en"] = col1.checkbox("Mkt Cap", key="f_mc_en", disabled=not db_populated)
        with col2:
            c1, c2 = st.columns(2)
            controls["mc_min"] = c1.number_input("Min ($B)", min_value=0.0, value=0.0,
                                                  step=1.0, key="f_mc_min",
                                                  disabled=not controls["mc_en"]) * 1e9
            controls["mc_max"] = c2.number_input("Max ($B)", min_value=0.0, value=5_000.0,
                                                  step=10.0, key="f_mc_max",
                                                  disabled=not controls["mc_en"]) * 1e9

        col1, col2 = st.columns([1, 3])
        controls["tpe_en"]       = col1.checkbox("Trail P/E", key="f_tpe_en", disabled=not db_populated)
        controls["tpe_excl_neg"] = col1.checkbox("excl. neg", key="f_tpe_excl_neg",
                                                  disabled=not controls["tpe_en"])
        tpe = col2.slider("", -100, 200, (-100, 200), key="f_tpe", disabled=not controls["tpe_en"])
        controls["tpe_min"], controls["tpe_max"] = tpe

        col1, col2 = st.columns([1, 3])
        controls["fpe_en"]       = col1.checkbox("Fwd P/E", key="f_fpe_en", disabled=not db_populated)
        controls["fpe_excl_neg"] = col1.checkbox("excl. neg", key="f_fpe_excl_neg",
                                                  disabled=not controls["fpe_en"])
        fpe = col2.slider("", -100, 200, (-100, 200), key="f_fpe", disabled=not controls["fpe_en"])
        controls["fpe_min"], controls["fpe_max"] = fpe

        col1, col2 = st.columns([1, 3])
        controls["pb_en"] = col1.checkbox("P/Book", key="f_pb_en", disabled=not db_populated)
        pb = col2.slider("", 0.0, 50.0, (0.0, 50.0), step=0.5, key="f_pb",
                         disabled=not controls["pb_en"])
        controls["pb_min"], controls["pb_max"] = pb

        col1, col2 = st.columns([1, 3])
        controls["peg_en"] = col1.checkbox("PEG Ratio", key="f_peg_en", disabled=not db_populated)
        peg = col2.slider("", 0.0, 10.0, (0.0, 10.0), step=0.1, key="f_peg",
                          disabled=not controls["peg_en"])
        controls["peg_min"], controls["peg_max"] = peg

    with st.sidebar.expander("Risk / Income", expanded=False):
        col1, col2 = st.columns([1, 3])
        controls["beta_en"] = col1.checkbox("Beta", key="f_beta_en", disabled=not db_populated)
        beta = col2.slider("", -2.0, 5.0, (-2.0, 5.0), step=0.1, key="f_beta",
                           disabled=not controls["beta_en"])
        controls["beta_min"], controls["beta_max"] = beta

        col1, col2 = st.columns([1, 3])
        controls["div_en"] = col1.checkbox("Div Yield %", key="f_div_en", disabled=not db_populated)
        div = col2.slider("", 0.0, 20.0, (0.0, 20.0), step=0.1, key="f_div",
                          disabled=not controls["div_en"])
        controls["div_min"], controls["div_max"] = div

    with st.sidebar.expander("Analyst Ratings", expanded=False):
        col1, col2 = st.columns([1, 3])
        controls["rec_key_en"] = col1.checkbox("Rating", key="f_rec_key_en", disabled=not db_populated)
        controls["rec_keys"]   = col2.multiselect("", rec_keys or ["buy", "hold", "sell",
                                                                    "strongBuy", "underperform"],
                                                   key="f_rec_keys",
                                                   disabled=not controls["rec_key_en"])

        col1, col2 = st.columns([1, 3])
        controls["rec_mean_en"] = col1.checkbox("Rating Mean", key="f_rec_mean_en",
                                                 disabled=not db_populated)
        col2.caption("1 = Strong Buy → 5 = Sell")
        rec_mean = col2.slider("", 1.0, 5.0, (1.0, 5.0), step=0.1, key="f_rec_mean",
                               disabled=not controls["rec_mean_en"])
        controls["rec_mean_min"], controls["rec_mean_max"] = rec_mean

        col1, col2 = st.columns([1, 3])
        controls["num_analysts_en"] = col1.checkbox("# Analysts", key="f_num_analysts_en",
                                                     disabled=not db_populated)
        num_analysts = col2.slider("", 0, 200, (0, 200), key="f_num_analysts",
                                   disabled=not controls["num_analysts_en"])
        controls["num_analysts_min"], controls["num_analysts_max"] = num_analysts

        col1, col2 = st.columns([1, 3])
        controls["target_upside_en"] = col1.checkbox("Upside to Target %", key="f_tu_en")
        target_upside = col2.slider("", -100, 200, (-100, 200), key="f_tu",
                                    disabled=not controls["target_upside_en"])
        controls["target_upside_min"], controls["target_upside_max"] = target_upside

    with st.sidebar.expander("Ownership & Short Interest", expanded=False):
        col1, col2 = st.columns([1, 3])
        controls["insiders_en"] = col1.checkbox("Insider Held %", key="f_insiders_en",
                                                 disabled=not db_populated)
        insiders = col2.slider("", 0.0, 100.0, (0.0, 100.0), step=0.5, key="f_insiders",
                               disabled=not controls["insiders_en"])
        controls["insiders_min"], controls["insiders_max"] = insiders

        col1, col2 = st.columns([1, 3])
        controls["institutions_en"] = col1.checkbox("Inst. Held %", key="f_instit_en",
                                                     disabled=not db_populated)
        institutions = col2.slider("", 0.0, 100.0, (0.0, 100.0), step=0.5, key="f_instit",
                                   disabled=not controls["institutions_en"])
        controls["instit_min"], controls["instit_max"] = institutions

        col1, col2 = st.columns([1, 3])
        controls["short_float_en"] = col1.checkbox("Short % Float", key="f_short_en",
                                                    disabled=not db_populated)
        short_float = col2.slider("", 0.0, 100.0, (0.0, 100.0), step=0.5, key="f_short",
                                  disabled=not controls["short_float_en"])
        controls["short_float_min"], controls["short_float_max"] = short_float

    st.sidebar.markdown("### Technical Filters")

    with st.sidebar.expander("Momentum / Volatility", expanded=False):
        col1, col2 = st.columns([1, 3])
        controls["rsi_en"] = col1.checkbox("RSI(14)", key="f_rsi_en")
        rsi = col2.slider("", 0, 100, (0, 100), key="f_rsi", disabled=not controls["rsi_en"])
        controls["rsi_min"], controls["rsi_max"] = rsi

        col1, col2 = st.columns([1, 3])
        controls["rvol_en"] = col1.checkbox("RVOL(50)", key="f_rvol_en")
        rvol = col2.slider("", 0.0, 10.0, (0.0, 10.0), step=0.1, key="f_rvol",
                           disabled=not controls["rvol_en"])
        controls["rvol_min"], controls["rvol_max"] = rvol

        col1, col2 = st.columns([1, 3])
        controls["p252h_en"] = col1.checkbox("% 52w High", key="f_p252h_en")
        p252h = col2.slider("", 0, 100, (0, 100), key="f_p252h", disabled=not controls["p252h_en"])
        controls["p252h_min"], controls["p252h_max"] = p252h

        col1, col2 = st.columns([1, 3])
        controls["p252l_en"] = col1.checkbox("% 52w Low", key="f_p252l_en")
        p252l = col2.slider("", 0, 100, (0, 100), key="f_p252l", disabled=not controls["p252l_en"])
        controls["p252l_min"], controls["p252l_max"] = p252l

    with st.sidebar.expander("Close vs. Moving Averages", expanded=False):
        _DIRECTIONS = ["either", "above", "below"]
        for label, key in [
            ("vs SMA(5)",   "sma5"),
            ("vs SMA(50)",  "sma50"),
            ("vs SMA(100)", "sma100"),
            ("vs SMA(200)", "sma200"),
            ("vs EMA(20)",  "ema20"),
        ]:
            col1, col2 = st.columns([1, 3])
            controls[f"{key}_en"]  = col1.checkbox(label, key=f"f_{key}_en")
            controls[f"{key}_dir"] = col2.selectbox("", _DIRECTIONS, key=f"f_{key}_dir",
                                                    disabled=not controls[f"{key}_en"])

    with st.sidebar.expander("Close vs. Bollinger Bands", expanded=False):
        _DIRECTIONS = ["either", "above", "below"]
        for label, key in [("vs BB Upper", "bbu"), ("vs BB Lower", "bbl")]:
            col1, col2 = st.columns([1, 3])
            controls[f"{key}_en"]  = col1.checkbox(label, key=f"f_{key}_en")
            controls[f"{key}_dir"] = col2.selectbox("", _DIRECTIONS, key=f"f_{key}_dir",
                                                    disabled=not controls[f"{key}_en"])

    return controls


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Stock Screener", layout="wide")
st.title("Stock Screener")

# --- Data dictionary download (always visible) ---
_dict_path = Path("field_dictionary.txt")
if _dict_path.exists():
    col_dd, col_txt = st.columns([1, 6])
    col_dd.download_button(
        label="⬇ Data Dictionary",
        data=_dict_path.read_bytes(),
        file_name="field_dictionary.txt",
        mime="text/plain",
    )
    col_txt.caption(
        "Download the complete data dictionary (TXT) for definitions of every "
        "field in the results table and exported CSV."
    )

# --- Sidebar input section ---
st.sidebar.header("Ticker Input")
input_mode = st.sidebar.radio(
    "Source", ["Built-in list", "CSV upload", "Text"],
    horizontal=True, key="input_mode",
)

tickers_raw: list[str] | None = None
rejected_symbols: list[str] = []

if input_mode == "Built-in list":
    selected_list = st.sidebar.selectbox(
        "Select list", list(BUILTIN_LISTS.keys()), key="builtin_list"
    )
    builtin_path = BUILTIN_LISTS[selected_list]
    if builtin_path.exists():
        tickers_raw, rejected_symbols = parse_csv_file(builtin_path)
    else:
        st.sidebar.error(f"List file not found: {builtin_path.name}")

elif input_mode == "CSV upload":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV (single column of tickers)", type=["csv", "txt"]
    )
    if uploaded:
        tickers_raw, rejected_symbols = parse_csv_bytes(uploaded.read())

else:  # Text
    text_input = st.sidebar.text_area(
        "Tickers (comma, space, semicolon, or newline separated)", height=120
    )
    if text_input.strip():
        tickers_raw, rejected_symbols = parse_text(text_input)

if rejected_symbols:
    st.sidebar.warning(f"Rejected {len(rejected_symbols)} invalid symbol(s): "
                       f"{', '.join(rejected_symbols[:10])}"
                       + (" …" if len(rejected_symbols) > 10 else ""))

if tickers_raw:
    st.sidebar.caption(f"{len(tickers_raw)} unique ticker(s) parsed.")

# --- Filter controls ---
st.sidebar.divider()
db_populated = is_db_populated(DB_PATH)
controls = _render_sidebar_filters(db_populated)

# --- Run / Advanced ---
st.sidebar.divider()
run_clicked = st.sidebar.button("▶ Run Screen", type="primary", use_container_width=True,
                                disabled=not tickers_raw)

with st.sidebar.expander("Advanced"):
    force_refresh = st.checkbox("Force refresh OHLCV cache", key="force_refresh")
    if st.button("Clear OHLCV cache"):
        n = clear_cache()
        st.success(f"Cleared {n} cached file(s).")


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if run_clicked and tickers_raw:
    filter_spec = _build_filter_spec(controls)

    status = st.empty()
    progress = st.progress(0)

    # Step 1 — load fundamentals for all input tickers
    status.info("Loading fundamentals from SQLite …")
    fund_df = load_for_tickers(tickers_raw, DB_PATH)

    # Step 2 — determine fetch targets
    use_fund_filter = db_populated and is_any_fundamental_filter_active(filter_spec)

    if use_fund_filter:
        status.info("Applying fundamental filters …")
        filtered_fund = apply_fundamental_filters(fund_df, filter_spec)
        fetch_targets = filtered_fund["ticker"].tolist()
        excl_by_filter = set(tickers_raw) - set(fetch_targets)
    else:
        filtered_fund = fund_df
        fetch_targets = list(tickers_raw)
        excl_by_filter = set()

    progress.progress(0.10)

    # Step 3 — fetch OHLCV
    def _ohlcv_progress(current, total, msg):
        frac = 0.10 + 0.60 * (current / max(total, 1))
        progress.progress(min(frac, 0.70))
        status.info(f"Fetching OHLCV: {msg}")

    status.info(f"Fetching OHLCV for {len(fetch_targets)} ticker(s) …")
    ohlcv = fetch_ohlcv(fetch_targets, force_refresh=force_refresh,
                        progress_cb=_ohlcv_progress)

    progress.progress(0.75)
    status.info("Computing indicators …")

    # Step 4 — compute indicators
    tech_df = build_technicals_df(ohlcv)

    progress.progress(0.90)

    # Step 5 — join fundamentals + technicals
    if db_populated and not fund_df.empty:
        combined = tech_df.merge(filtered_fund, on="ticker", how="inner")
        missing_from_db = set(fetch_targets) - set(fund_df["ticker"])
    else:
        combined = tech_df.copy()
        missing_from_db = set()

    # Step 6 — derive upside_to_target as a decimal (0.153 = 15.3% upside),
    # consistent with all other percentage-type fields in the output.
    if "target_median_price" in combined.columns and "Close" in combined.columns:
        combined["upside_to_target"] = (
            (combined["target_median_price"] - combined["Close"]) / combined["Close"]
        )

    progress.progress(1.0)
    status.empty()
    progress.empty()

    st.session_state["combined_df"]     = combined
    st.session_state["excl_by_filter"]  = excl_by_filter
    st.session_state["missing_from_db"] = missing_from_db
    st.session_state["input_count"]     = len(tickers_raw)
    st.session_state["run_ts"]          = datetime.now()


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if "combined_df" in st.session_state:
    combined: pd.DataFrame = st.session_state["combined_df"]
    excl_by_filter: set    = st.session_state["excl_by_filter"]
    missing_from_db: set   = st.session_state["missing_from_db"]
    input_count: int       = st.session_state["input_count"]
    run_ts: datetime       = st.session_state["run_ts"]

    filter_spec = _build_filter_spec(controls)
    filtered = apply_all_filters(combined, filter_spec)

    # Status banner
    cols = st.columns(4)
    cols[0].metric("Input tickers",      input_count)
    cols[1].metric("After fund. filter",  len(combined))
    cols[2].metric("After tech. filter",  len(filtered))
    cols[3].metric("Run at",             run_ts.strftime("%H:%M:%S"))

    # Remove the printed excl_by_filter tickers as requested by user
    if excl_by_filter:
        st.warning(
            f"{len(excl_by_filter)} ticker(s) excluded by fundamental filters "
            f"before OHLCV fetch."
        )#: {', '.join(sorted(excl_by_filter)[:20])}"
    #        + (" …" if len(excl_by_filter) > 20 else "")
    #    )
    if missing_from_db:
        st.warning(
            f"{len(missing_from_db)} ticker(s) fetched but absent from fundamentals DB "
            f"(excluded from results): {', '.join(sorted(missing_from_db)[:20])}"
            + (" …" if len(missing_from_db) > 20 else "")
        )

    if filtered.empty:
        st.info("No tickers match the current filters.")
    else:
        st.caption(
            f"Showing {len(filtered)} result(s). "
            "Adjust filters to re-screen without re-fetching. "
            "Click **▶ Run Screen** to re-fetch with updated fundamental filters."
        )
        st.dataframe(_build_display_df(filtered), use_container_width=True, height=500)

        # CSV export — raw numeric values, not the display-formatted strings
        present_cols = [c for c in DISPLAY_COLUMNS if c in filtered.columns]
        csv_bytes = filtered[present_cols].to_csv(index=False).encode()
        ts_str = run_ts.strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇ Download filtered results (CSV)",
            data=csv_bytes,
            file_name=f"screener_results_{ts_str}.csv",
            mime="text/csv",
        )
else:
    st.info("From the sidebar, select tickers (from built-in list by default), select any filters to apply, and click **▶ Run Screen** to begin.")
    st.info("NOTE - fundamental filters are not sourced live, but instead updated manually on a quarterly cadence by the developer.  This data is currently frozen in time as of May 10, 2026")
