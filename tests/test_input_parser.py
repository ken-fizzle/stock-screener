import io
import pytest
from core.input_parser import parse_text, parse_csv_bytes


# ---------------------------------------------------------------------------
# parse_text
# ---------------------------------------------------------------------------

def test_text_comma_separated():
    valid, rejected = parse_text("AAPL, MSFT, GOOG")
    assert valid == ["AAPL", "MSFT", "GOOG"]
    assert rejected == []


def test_text_mixed_delimiters():
    valid, rejected = parse_text("AAPL\nMSFT;GOOG TSLA,META")
    assert valid == ["AAPL", "MSFT", "GOOG", "TSLA", "META"]


def test_text_deduplication_preserves_order():
    valid, _ = parse_text("AAPL MSFT AAPL GOOG MSFT")
    assert valid == ["AAPL", "MSFT", "GOOG"]


def test_text_uppercases():
    valid, rejected = parse_text("aapl msft")
    assert valid == ["AAPL", "MSFT"]
    assert rejected == []


def test_text_rejects_invalid():
    valid, rejected = parse_text("AAPL 123BAD !@# MSFT")
    assert "AAPL" in valid
    assert "MSFT" in valid
    assert "123BAD" in rejected
    assert "!@#" in rejected


def test_text_empty_input():
    valid, rejected = parse_text("   ")
    assert valid == []
    assert rejected == []


def test_text_valid_patterns():
    symbols = ["A", "BRK.B", "BF-B", "GOOG1", "AAPL"]
    valid, rejected = parse_text(" ".join(symbols))
    assert set(valid) == set(symbols)
    assert rejected == []


def test_text_too_long_symbol_rejected():
    valid, rejected = parse_text("ABCDEFGHIJK")  # 11 chars, exceeds max
    assert "ABCDEFGHIJK" in rejected


# ---------------------------------------------------------------------------
# parse_csv_bytes
# ---------------------------------------------------------------------------

def _csv(rows: list[str]) -> bytes:
    return "\n".join(rows).encode()


def test_csv_without_header():
    content = _csv(["AAPL", "MSFT", "GOOG"])
    valid, rejected = parse_csv_bytes(content)
    assert valid == ["AAPL", "MSFT", "GOOG"]


def test_csv_with_header():
    content = _csv(["ticker", "AAPL", "MSFT"])
    valid, rejected = parse_csv_bytes(content)
    assert "AAPL" in valid
    assert "MSFT" in valid
    # "ticker" is lowercase so it must be detected as a header, not a symbol.
    assert "ticker" not in valid
    assert "TICKER" not in valid  # previously silently included due to pre-uppercase bug


def test_csv_with_non_ticker_header():
    content = _csv(["Symbol", "AAPL", "MSFT"])
    valid, rejected = parse_csv_bytes(content)
    assert "Symbol" not in valid
    assert "SYMBOL" not in valid
    assert "AAPL" in valid


def test_csv_deduplication():
    content = _csv(["AAPL", "AAPL", "MSFT"])
    valid, _ = parse_csv_bytes(content)
    assert valid.count("AAPL") == 1


def test_csv_strips_whitespace():
    content = _csv([" AAPL ", " MSFT"])
    valid, _ = parse_csv_bytes(content)
    assert valid == ["AAPL", "MSFT"]
