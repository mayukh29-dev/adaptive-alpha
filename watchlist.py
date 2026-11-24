"""Session-only watchlist helpers. No file, user account, or server database writes."""
from __future__ import annotations

import re

MAX_WATCHLIST = 12
TICKER_PATTERN = re.compile(r"[A-Z][A-Z0-9.-]{0,11}")


def normalize_ticker(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise ValueError("Enter a valid stock ticker, such as NVDA or BRK.B.")
    return ticker


def add_ticker(saved: list[str], raw: str, limit: int = MAX_WATCHLIST) -> list[str]:
    ticker = normalize_ticker(raw)
    result = list(saved)
    if ticker in result:
        return result
    if len(result) >= limit:
        raise ValueError(f"Watchlist limit reached ({limit} stocks). Remove one first.")
    return result + [ticker]


def remove_ticker(saved: list[str], raw: str) -> list[str]:
    ticker = normalize_ticker(raw)
    return [x for x in saved if x != ticker]


def clean_watchlist(saved: list[str], limit: int = MAX_WATCHLIST) -> list[str]:
    cleaned = []
    for raw in saved:
        ticker = normalize_ticker(raw)
        if ticker not in cleaned:
            cleaned.append(ticker)
    if len(cleaned) > limit:
        raise ValueError(f"Watchlist exceeds {limit} stocks.")
    return cleaned
