"""Curated sector discovery for the public score dashboard.

NOT an exhaustive securities universe or a vendor-verified GICS membership feed.
Quotes and score dates are fetched separately by the calling UI.
"""
from __future__ import annotations

import pandas as pd

import signals

# Small, transparent illustrative US equity lists to avoid vendor rate-limit
# bursts. Membership is illustrative/currently curated, not point-in-time.
SECTOR_UNIVERSE: dict[str, tuple[str, ...]] = {
    "Communication Services": ("GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ"),
    "Consumer Discretionary": ("AMZN", "TSLA", "HD", "MCD", "NKE", "LOW"),
    "Consumer Staples": ("WMT", "COST", "PG", "KO", "PEP", "CL"),
    "Energy": ("XOM", "CVX", "COP", "EOG", "SLB", "OXY"),
    "Financials": ("JPM", "BAC", "WFC", "GS", "MS", "BLK"),
    "Health Care": ("UNH", "JNJ", "ABBV", "MRK", "LLY", "AMGN"),
    "Industrials": ("CAT", "GE", "DE", "HON", "UPS", "RTX"),
    "Information Technology": ("AAPL", "MSFT", "NVDA", "AMD", "AVGO", "CRM"),
    "Materials": ("LIN", "APD", "SHW", "FCX", "NEM", "DOW"),
    "Real Estate": ("PLD", "AMT", "EQIX", "O", "SPG", "WELL"),
    "Utilities": ("NEE", "SO", "DUK", "AEP", "SRE", "EXC"),
}

SORT_CHOICES = ("Custom Hybrid", "Momentum", "Recovery", "Standard Hybrid")


def rank_sector(rows: list[dict], momentum_pct: int, sort_by: str) -> pd.DataFrame:
    """Rank successfully scored representatives; preserve missing-symbol audit elsewhere."""
    if not isinstance(momentum_pct, int) or not 0 <= momentum_pct <= 100:
        raise ValueError("Momentum weight must be a whole percentage in 0..100")
    if sort_by not in SORT_CHOICES:
        raise ValueError("Invalid ranking style")
    result = []
    for row in rows:
        m = float(row["momentum"])
        r = float(row["recovery"])
        result.append({
            "Ticker": row["ticker"],
            "Momentum": round(m, 2),
            "Recovery": round(r, 2),
            "Standard Hybrid": round(signals.weighted_hybrid(m, r, 0.6), 2),
            "Custom Hybrid": round(signals.weighted_hybrid(m, r, momentum_pct/100), 2),
            "As of": str(row["date"]),
        })
    columns = ["Rank", "Ticker", "Momentum", "Recovery",
               "Standard Hybrid", "Custom Hybrid", "As of"]
    if not result:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(result).sort_values(
        [sort_by, "Ticker"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    frame.insert(0, "Rank", range(1, len(frame)+1))
    return frame[columns]

def all_sector_tables(rows_by_sector: dict[str, list[dict]], momentum_pct: int,
                      sort_by: str) -> dict[str, pd.DataFrame]:
    """Display every sector independently; preserve even sectors with no valid scores."""
    if not isinstance(rows_by_sector, dict):
        raise ValueError("Expected grouped sector rows")
    return {
        sector: rank_sector(rows_by_sector.get(sector, []), momentum_pct, sort_by)
        for sector in SECTOR_UNIVERSE
    }


def sector_leaders(tables: dict[str, pd.DataFrame], sort_by: str) -> pd.DataFrame:
    """One row per scanned sector, with explicit empty/missing status."""
    if sort_by not in SORT_CHOICES:
        raise ValueError("Invalid ranking approach")
    rows = []
    for sector in SECTOR_UNIVERSE:
        table = tables.get(sector)
        if table is None or table.empty:
            rows.append({"Sector":sector, "Leader":"Unavailable", "Leader score":None,
                         "Scored":0, "Coverage":len(SECTOR_UNIVERSE[sector]),
                         "As of":"Unavailable"})
        else:
            first=table.iloc[0]
            rows.append({"Sector":sector, "Leader":first["Ticker"],
                         "Leader score":round(float(first[sort_by]),2),
                         "Scored":len(table), "Coverage":len(SECTOR_UNIVERSE[sector]),
                         "As of":first["As of"]})
    return pd.DataFrame(rows)
