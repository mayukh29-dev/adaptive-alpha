"""Final comparative price-only walk-forward evaluation for Adaptive Alpha.

Place next to backtest_v82.py in the local adaptive_alpha_v8_2 directory.
Requires the existing locally downloaded 'history/' and 'data/master_110.csv'.

Produces distinct Momentum, Recovery and Hybrid outputs. Does NOT backtest
point-in-time financial Quality or production V7 formulas. Historical universe
was selected retrospectively; survivorship and selection biases persist.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_v82 as bt

STRATEGIES = ("momentum", "recovery", "hybrid")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_prices(prices, tickers, threshold_pct=100.0):
    """Only flag large close-to-close moves; do not automatically discard returns."""
    alerts = []
    for t in tickers:
        p = pd.to_numeric(prices[t], errors="coerce")
        bad = p.notna() & (~np.isfinite(p) | (p <= 0))
        for date in p.index[bad]:
            alerts.append({"ticker": t, "date": str(date.date()),
                           "type": "invalid_price", "daily_move_pct": "",
                           "previous_close": "", "close": str(p.at[date])})
        ret = p.pct_change(fill_method=None)
        # A split or bad vendor adjustment can create very large positive outliers.
        selected = ret.notna() & (ret.abs() * 100 > threshold_pct)
        for date in ret.index[selected]:
            loc = p.index.get_loc(date)
            prior = p.iloc[loc - 1] if loc else np.nan
            alerts.append({"ticker": t, "date": str(date.date()),
                           "type": "extreme_daily_move",
                           "daily_move_pct": round(float(ret.at[date] * 100), 4),
                           "previous_close": round(float(prior), 6),
                           "close": round(float(p.at[date]), 6)})
    alerts.sort(key=lambda row: (row["ticker"], row["date"]))
    return alerts


def simulate(prices, signals, sectors, dates, top_per_sector, cost_bps):
    tickers = list(sectors)
    daily = prices.pct_change(fill_method=None)
    weights = {t: 0.0 for t in tickers}
    rebalance = []
    curve = []
    pending_cost = 0.0
    signal_set = set(dates)
    expected = len(set(sectors.values())) * top_per_sector
    for idx in range(bt.WARMUP, len(prices.index) - 1):
        date = prices.index[idx]
        if date in signal_set:
            selected = []
            for sector in sorted(set(sectors.values())):
                eligible = [
                    t for t in tickers
                    if sectors[t] == sector
                    and pd.notna(signals.at[date, t])
                    and pd.notna(prices.at[date, t])
                ]
                eligible.sort(key=lambda t: (-float(signals.at[date, t]), t))
                selected.extend(eligible[:top_per_sector])
            if len(selected) != expected:
                raise ValueError(
                    f"{date.date()}: {len(selected)}/{expected} positions; "
                    "strict full-portfolio coverage failed"
                )
            new = {t: (1.0 / expected if t in selected else 0.0) for t in tickers}
            turnover = sum(abs(new[t] - weights[t]) for t in tickers)
            weights = new
            pending_cost = turnover * cost_bps / 10000.0
            rebalance.append({
                "signal_date": date.date().isoformat(),
                "n_selected": len(selected),
                "turnover_sum_abs_weights": round(turnover, 8),
                "estimated_cost_pct_of_equity": round(100 * pending_cost, 8),
                "tickers": ";".join(selected),
            })
        following = prices.index[idx + 1]
        if pd.isna(daily.at[following, "SPY"]):
            raise ValueError(f"{following.date()}: SPY return is missing")
        missing = [
            t for t in tickers if weights[t] > 0
            and pd.isna(daily.at[following, t])
        ]
        if missing:
            raise ValueError(
                f"{following.date()}: missing held-stock returns: {missing}"
            )
        gross = sum(weights[t] * float(daily.at[following, t])
                    for t in tickers if weights[t] > 0)
        curve.append({
            "date": following.date().isoformat(),
            "strategy_return": gross - pending_cost,
            "spy_return": float(daily.at[following, "SPY"]),
            "n_held": sum(x > 0 for x in weights.values()),
            "cost_paid": pending_cost,
        })
        pending_cost = 0.0
    first = next((i for i, row in enumerate(curve) if row["n_held"]), None)
    if first is None:
        raise ValueError("No funded portfolio observations")
    curve = curve[first:]
    capital, spy_cap = 1.0, 1.0
    for row in curve:
        if row["strategy_return"] <= -1:
            raise ValueError(f"{row['date']}: portfolio lost >=100% in one day")
        capital *= 1 + row["strategy_return"]
        spy_cap *= 1 + row["spy_return"]
        row["strategy_equity"] = round(capital, 8)
        row["spy_equity"] = round(spy_cap, 8)
    return curve, rebalance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master", default="data/master_110.csv")
    p.add_argument("--history", default="history")
    p.add_argument("--output", default="final_backtest_output")
    p.add_argument("--reference-summary", default="backtest_output/summary.json")
    p.add_argument("--exclude-tickers", nargs="*", default=["NWE"])
    p.add_argument("--top-per-sector", type=int, default=2)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--outlier-threshold-pct", type=float, default=100.0)
    p.add_argument("--allow-unreviewed-outliers", action="store_true")
    args = p.parse_args()
    if not 1 <= args.top_per_sector <= 10:
        p.error("--top-per-sector must be 1..10")
    if not 0 <= args.cost_bps <= 1000:
        p.error("--cost-bps must be 0..1000")
    if args.outlier_threshold_pct <= 0:
        p.error("outlier threshold must be positive")

    members = bt.read_master(args.master)
    excluded = set(args.exclude_tickers)
    member_tickers = {r["ticker"] for r in members}
    if not excluded <= member_tickers:
        raise ValueError(f"Unknown excluded tickers: {sorted(excluded - member_tickers)}")
    sectors = {r["ticker"]: r["sector"] for r in members if r["ticker"] not in excluded}
    if len(set(sectors.values())) != 11:
        raise ValueError("Expected 11 represented sectors")
    tickers = list(sectors)
    prices = bt.load_prices(args.history, ["SPY"] + tickers)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    alerts = audit_prices(prices, ["SPY"] + tickers, args.outlier_threshold_pct)
    save_csv(out / "price_outlier_audit.csv", alerts)
    if alerts and not args.allow_unreviewed_outliers:
        print(f"PRICE AUDIT BLOCKED: {len(alerts)} suspicious observations.")
        print(f"Inspect {out / 'price_outlier_audit.csv'} before proceeding.")
        print("After examining them, you may run with --allow-unreviewed-outliers")
        print("to produce explicitly PROVISIONAL results without changing the data.")
        raise SystemExit(2)

    momentum, recovery, hybrid = bt.indicators(prices)
    frames = {"momentum": momentum, "recovery": recovery, "hybrid": hybrid}
    dates = bt.month_end_signal_dates(prices.index)
    summaries = {}
    curves = {}
    for name, signals in frames.items():
        curve, rebalance = simulate(
            prices, signals, sectors, dates, args.top_per_sector, args.cost_bps
        )
        metrics = bt.quality_metrics([r["strategy_return"] for r in curve])
        benchmark = bt.quality_metrics([r["spy_return"] for r in curve])
        summaries[name] = {
            "performance": metrics,
            "spy": benchmark,
            "rebalances": len(rebalance),
            "n_selected_min": min(x["n_selected"] for x in rebalance),
            "n_selected_max": max(x["n_selected"] for x in rebalance),
            "first": curve[0]["date"],
            "last": curve[-1]["date"],
            "sessions": len(curve),
            "total_gross_turnover": round(
                sum(x["turnover_sum_abs_weights"] for x in rebalance), 6),
        }
        curves[name] = curve
        save_csv(out / f"{name}_portfolio_curve.csv", curve)
        save_csv(out / f"{name}_rebalance_audit.csv", rebalance)

    reference = Path(args.reference_summary)
    hybrid_match = None
    if reference.exists():
        prior = json.loads(reference.read_text(encoding="utf-8"))
        if prior.get("ticker_count") == len(tickers) and set(prior.get("excluded_tickers", [])) == excluded:
            measured = summaries["hybrid"]["performance"]
            old = prior.get("strategy", {})
            hybrid_match = {
                key: {"reference": old.get(key), "recomputed": measured.get(key),
                      "absolute_difference": round(
                          abs(measured[key] - old[key]), 7)
                      if key in old and key in measured else None}
                for key in ("cagr_pct", "total_return_pct", "max_drawdown_pct")
            }
            if any(v["absolute_difference"] is not None and
                   v["absolute_difference"] > 0.05
                   for v in hybrid_match.values()):
                raise ValueError(
                    "Hybrid does not reproduce the baseline within 0.05 percentage points. "
                    "Compare implementation, files, parameters and signal dates."
                )

    out_summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Price-only historical PROXY, not production V7 fundamental backtest",
        "historical_universe": "Retrospectively selected; survivorship/selection bias",
        "selected_tickers": len(tickers),
        "excluded_tickers": sorted(excluded),
        "sector_count": len(set(sectors.values())),
        "cost_bps": args.cost_bps,
        "top_per_sector": args.top_per_sector,
        "outlier_alert_count": len(alerts),
        "outlier_review_status": "UNREVIEWED" if alerts else "NO_THRESHOLD_BREACHES",
        "historical_adjustments_status": "Vendor-adjusted prices not independently certified",
        "baseline_hybrid_reconciliation": hybrid_match,
        "strategies": summaries,
        "warnings": [
            "These are technical price-only signal proxies, not production V7 model replays.",
            "Current Quality fundamentals cannot be backtested without point-in-time data.",
            "Universe was selected with future knowledge: survivorship and selection bias.",
            "Vendor histories, delistings, splits, and outliers require source verification.",
            "Execution assumes same-close signal / next-close-to-close return; not guaranteed fills.",
            "Monthly equal-weight reset without intra-month weight drift.",
            "Transaction costs omit slippage, impact, spreads, taxes and financing.",
            "Comparisons do not establish a statistically validated predictive edge."
        ],
    }
    (out / "comparison.json").write_text(
        json.dumps(out_summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(out_summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
