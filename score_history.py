"""Historical descriptive signal-series utilities for the Streamlit score page.

Calculations reuse signals.indicators; not a backtest or return predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import signals


def score_history(history: pd.DataFrame, momentum_pct: int = 60,
                  months: int = 6) -> pd.DataFrame:
    """Return trailing calendar-month daily scores; never invent missing values.

    At each date, uses data no later than that date. Includes standard and
    user-weighted hybrids. Needs enough earlier input for indicator warmup.
    """
    if not isinstance(momentum_pct, int) or not 0 <= momentum_pct <= 100:
        raise ValueError("Momentum percentage must be an integer from 0 to 100")
    if months not in (1, 3, 6, 12):
        raise ValueError("Available score history is 1, 3, 6, or 12 months")
    if history is None or history.empty or "Close" not in history:
        return pd.DataFrame()
    prices = pd.to_numeric(history["Close"], errors="coerce")
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Historical series needs a DatetimeIndex")
    mom, rec, hyb = signals.indicators(pd.DataFrame({"Stock":prices}))
    weight = momentum_pct / 100.0
    result = pd.DataFrame({
        "Momentum":mom["Stock"],
        "Recovery":rec["Stock"],
        "Standard Hybrid":hyb["Stock"],
        "Custom Hybrid":weight*mom["Stock"]+(1-weight)*rec["Stock"],
    })
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    if result.empty:
        return result
    cutoff = result.index.max() - pd.DateOffset(months=months)
    return result.loc[result.index >= cutoff].copy()
