"""Adaptive Alpha price-only signal specification v1.0.
Same implementation for Streamlit and research/backtest. Not production V7.
Uses adjusted DAILY close only; no lookahead. Indicators need 127 closes.
"""
import numpy as np
import pandas as pd

SIGNAL_VERSION = "price-signals-1.0"
HORIZON = 21
WARMUP = 126

def indicators(prices):
    """Return (momentum, recovery, hybrid) DataFrames; input adjusted daily CLOSE DataFrame."""
    p = prices.astype(float).where(lambda x: x > 0)
    r63 = p.div(p.shift(63)).sub(1)
    mom = (50 + 200 * r63).clip(lower=0, upper=100)
    # At date t, 126 sessions ending t. No future data enters the window.
    lo = p.rolling(126, min_periods=126).min()
    hi = p.rolling(126, min_periods=126).max()
    rebound = (100 * p.div(lo).sub(1)).clip(lower=0, upper=100)
    proximity = (100 * p.div(hi)).clip(lower=0, upper=100)
    rec = 0.6 * rebound + 0.4 * proximity
    # Require a full 127 valid closing prices including price 63 sessions ago.
    valid = p.notna().rolling(127, min_periods=127).sum().eq(127)
    mom = mom.where(valid)
    rec = rec.where(valid)
    hybrid = (0.6 * mom + 0.4 * rec).where(valid)
    return mom, rec, hybrid

def latest_scores(history):
    """Shared live snapshot; returns None when history is insufficient."""
    if history is None or history.empty or "Close" not in history:
        return None
    close = pd.to_numeric(history["Close"], errors="coerce")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if len(close) < 127:
        return None
    m, r, h = indicators(pd.DataFrame({"TICKER": close}))
    vals = [x["TICKER"].iloc[-1] for x in (m, r, h)]
    if not all(np.isfinite(v) for v in vals):
        return None
    p = close.astype(float)
    window = p.iloc[-126:]
    daily = p.pct_change(fill_method=None).dropna()
    dd = window / window.cummax() - 1
    return dict(momentum=float(vals[0]), recovery=float(vals[1]),
                hybrid=float(vals[2]), return_63=float((p.iloc[-1]/p.iloc[-64]-1)*100),
                volatility_63=float(daily.iloc[-63:].std()*np.sqrt(252)*100),
                drawdown_126=float(dd.min()*100), date=str(p.index[-1].date()),
                close=float(p.iloc[-1]), version=SIGNAL_VERSION)

def weighted_hybrid(momentum: float, recovery: float, momentum_weight: float = 0.6) -> float:
    """Display-only user blend; NOT an evaluated ML forecast.

    momentum_weight is a fraction in [0, 1].  The research baseline is 0.6.
    """
    import math
    vals = (float(momentum), float(recovery), float(momentum_weight))
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("Momentum, Recovery, and weight must be finite")
    m, r, w = vals
    if not (0 <= m <= 100 and 0 <= r <= 100 and 0 <= w <= 1):
        raise ValueError("Scores must be 0–100 and weight must be 0–1")
    return w * m + (1 - w) * r
