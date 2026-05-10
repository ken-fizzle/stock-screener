# Stock Screener

Streamlit app for screening US-listed stocks against fundamental and technical criteria.

## Quick start

```bash
cd stock_screener
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Docker

```bash
docker build -t stock-screener .
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  stock-screener
```

Open http://localhost:8501.

---

## Architecture

### Filter-first pipeline

1. **Parse tickers** from a built-in list, CSV upload, or manual text entry.
2. **Load fundamentals** from `data/fundamentals.db` for all input tickers.
3. **If fundamental filters are active and the DB is populated:** apply them before fetching OHLCV — this avoids downloading price history for tickers that will be filtered out anyway.
4. **Fetch OHLCV** (5-year daily) via yfinance; cache to `cache/ohlcv/<TICKER>.parquet`.
5. **Compute indicators** with pandas-ta (RSI, Bollinger Bands, SMAs, EMA, RVOL, 52-week high/low %).
6. **Inner-join** technicals with fundamentals (tickers absent from DB are excluded from final results).
7. **Apply technical filters** and render the sortable results table.

Re-filtering after a run is instant — it re-applies filters to cached session data without re-fetching.

### Project layout

```
app.py                     Streamlit entry point (UI only)
core/
  input_parser.py          CSV + text ticker parsing
  fundamentals.py          SQLite read layer (latest snapshot per ticker)
  loader.py                SQLite write layer (quarterly refresh logic)
  fetcher.py               yfinance download + parquet cache
  indicators.py            pandas-ta indicator computation
  filters.py               Filter spec → boolean mask
load_fundamentals.py       CLI wrapper for headless/scheduled runs
sp500_tickers.csv          S&P 500 built-in ticker list (~500 tickers)
US2700_tickers.csv         US broad market built-in ticker list (~2,700 tickers)
cache/ohlcv/               Parquet files (gitignored)
data/fundamentals.db       SQLite fundamentals DB (committed to repo)
tests/                     pytest unit tests
field_dictionary.txt       Definitions for all output CSV columns
```

---

## Fundamentals database

The DB is included in the repo and updated quarterly. To refresh it locally:

```bash
python load_fundamentals.py --tickers sp500_tickers.csv --db data/fundamentals.db
python load_fundamentals.py --tickers sp500_tickers.csv --db data/fundamentals.db --resume
```

Re-run quarterly. Each run appends a new snapshot keyed by `(ticker, as_of_date)`,
so historical snapshots are preserved. The screener always reads the most-recent snapshot.

### Schema notes

- Table: `fundamentals`
- Primary key: `(ticker, as_of_date)` — supports multi-snapshot history.
- The screener reads the `MAX(as_of_date)` row per ticker at query time.

---

## OHLCV cache

- Location: `cache/ohlcv/<TICKER>.parquet`
- Staleness: files older than **24 hours** are re-fetched on the next run.
- **Force refresh:** checkbox in the Advanced sidebar expander.
- **Clear cache:** button in the Advanced sidebar expander.

### Performance targets (500 tickers)

| Stage | Target |
|---|---|
| Cold-cache fetch | < 5 min |
| Warm-cache load | < 10 s |
| Indicator computation | < 30 s |
| Filter + render | < 2 s |

For 3 000+ tickers cold-cache fetch may take 15–30 minutes due to yfinance rate limits.

---

## Indicator notes

All indicators use **unadjusted Close** (not Adj Close). This keeps computed
values consistent with the raw price shown in most charting applications, at the
cost of slight inaccuracy for long historical comparisons involving splits or
dividends.

`High_252` / `Low_252` / `SMA_200` / `Pct_of_252_*` return `NaN` for tickers
with fewer than 252 trading days of history. The filter step naturally excludes
these rows when a range comparison is applied.

---

## Tests

```bash
pytest tests/ -v
```

Mocks yfinance and SQLite — no network required.
