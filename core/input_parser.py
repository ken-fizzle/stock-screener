from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def _normalize(raw: list[str]) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    valid: list[str] = []
    rejected: list[str] = []
    for item in raw:
        sym = item.strip().upper()
        if not sym:
            continue
        if not TICKER_PATTERN.match(sym):
            rejected.append(sym)
        elif sym not in seen:
            seen.add(sym)
            valid.append(sym)
    return valid, rejected


def parse_text(text: str) -> tuple[list[str], list[str]]:
    """Parse tickers from a delimited string. Returns (valid, rejected)."""
    raw = re.split(r"[,\s;]+", text)
    return _normalize(raw)


def parse_csv_bytes(content: bytes) -> tuple[list[str], list[str]]:
    """Parse tickers from raw CSV bytes (e.g. from st.file_uploader)."""
    df = pd.read_csv(io.BytesIO(content), header=None, dtype=str)
    raw = df.iloc[:, 0].dropna().tolist()
    # Check the raw (non-uppercased) value — valid tickers are always uppercase,
    # so any lowercase letter unambiguously marks the first cell as a header.
    if raw and not TICKER_PATTERN.match(raw[0].strip()):
        raw = raw[1:]
    return _normalize(raw)


def parse_csv_file(path: Path) -> tuple[list[str], list[str]]:
    """Parse tickers from a CSV file on disk."""
    df = pd.read_csv(path, header=None, dtype=str)
    raw = df.iloc[:, 0].dropna().tolist()
    if raw and not TICKER_PATTERN.match(raw[0].strip()):
        raw = raw[1:]
    return _normalize(raw)
