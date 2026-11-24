"""Pure comparison helpers for the public score dashboard.

Comparison ranks scores, not forecasts of future stock returns.
"""
import re

import pandas as pd

import signals


def parse_comparison_tickers(raw: str, max_tickers: int = 3) -> list[str]:
    """Parse comma-separated symbols; preserve order, deduplicate, validate."""
    items = [x.strip().upper() for x in raw.split(",") if x.strip()]
    unique = list(dict.fromkeys(items))
    if len(unique) < 2 or len(unique) > max_tickers:
        raise ValueError(f"Enter between 2 and {max_tickers} distinct stock tickers.")
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.\\-]{0,11}", t) for t in unique):
        raise ValueError("Invalid ticker. Use letters, numbers, periods, or hyphens.")
    return unique


def build_comparison(rows: list[dict], momentum_pct: int) -> pd.DataFrame:
    """Rank observed Momentum, Recovery and user-weighted Hybrid only.

    Rows have keys: ticker, momentum, recovery, date.
    """
    if not isinstance(momentum_pct, int) or not (0 <= momentum_pct <= 100):
        raise ValueError("Momentum weight must be a whole percentage from 0 to 100")
    if not rows:
        raise ValueError("No stock scores to compare.")
    records = []
    for row in rows:
        m = float(row["momentum"])
        r = float(row["recovery"])
        records.append({
            "Ticker": row["ticker"],
            "Momentum": round(m, 2),
            "Recovery": round(r, 2),
            "Standard Hybrid (60/40)": round(signals.weighted_hybrid(m, r, 0.6), 2),
            "Custom Hybrid": round(signals.weighted_hybrid(m, r, momentum_pct / 100), 2),
            "Score date": str(row["date"]),
        })
    out = pd.DataFrame(records).sort_values(
        ["Custom Hybrid", "Ticker"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out
