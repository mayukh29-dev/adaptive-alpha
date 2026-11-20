"""Adaptive Alpha V10.1 fixed window-comparison and concentration audit.

Uses prior on-disk V10 simulation artifacts; makes no downloads and does NOT
refit models. All statistics are retrospective research diagnostics.
The 2023+ period is a repeatedly observed/selected test: NOT untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = {
    "expanding": "v10_walkforward_output",
    "rolling_36": "v10_rolling_36",
    "rolling_60": "v10_rolling_60",
}
STRATEGIES = ("momentum", "recovery", "hybrid")
SPLITS = ("validation", "test")


def load_monthly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"signal_date", "next_date", "split", "strategy_return",
                "spy_return", "rank_ic", "holdings"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame["signal_date"].duplicated().any():
        raise ValueError(f"{path}: duplicate signal dates")
    frame = frame.sort_values("signal_date").reset_index(drop=True)
    if frame["strategy_return"].isna().any() or frame["spy_return"].isna().any():
        raise ValueError(f"{path}: missing monthly returns")
    return frame


def growth(values) -> float:
    returns = np.asarray(values, dtype=float)
    if not len(returns) or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Missing or invalid monthly returns")
    return float(np.expm1(np.log1p(returns).sum()))


def stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"months": 0}
    start = pd.Timestamp(frame.iloc[0]["signal_date"])
    end = pd.Timestamp(frame.iloc[-1]["next_date"])
    years = (end-start).days / 365.25
    if years <= 0:
        raise ValueError("Invalid portfolio date range")
    annualized = lambda s: 100 * ((1+growth(s))**(1/years)-1)
    strategy = frame["strategy_return"].astype(float)
    spy = frame["spy_return"].astype(float)
    equity = np.cumprod(1+strategy.to_numpy())
    drawdown = equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1
    return {
        "months": int(len(frame)),
        "portfolio_cagr_pct": round(annualized(strategy), 4),
        "spy_cagr_pct": round(annualized(spy), 4),
        "portfolio_total_pct": round(growth(strategy)*100, 4),
        "max_drawdown_pct": round(float(drawdown.min()*100), 4),
        "mean_rank_ic": round(float(frame.rank_ic.dropna().mean()), 6)
            if frame.rank_ic.notna().any() else None,
    }


def analyze_contributors(monthly: pd.DataFrame, contribution_path: Path) -> dict:
    source = pd.read_csv(contribution_path)
    required = {"signal_date", "ticker", "gross_portfolio_contribution_pp"}
    if not required.issubset(source.columns):
        raise ValueError(f"{contribution_path}: missing required columns")
    if set(source.signal_date) != set(monthly.signal_date):
        raise ValueError("Contribution dates do not match Recovery rolling-60 test")
    counts = source.groupby("signal_date").size()
    if (counts != 22).any():
        raise ValueError("Expected exactly 22 holdings every month")
    sums = source.groupby("ticker").gross_portfolio_contribution_pp.sum().sort_values(ascending=False)
    pivot = source.pivot_table(index="signal_date", columns="ticker",
                               values="gross_portfolio_contribution_pp",
                               aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(monthly.signal_date).fillna(0)
    base = monthly.strategy_return.to_numpy(dtype=float)
    first = monthly.iloc[0]["signal_date"]
    last = monthly.iloc[-1]["next_date"]
    years = (pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25
    scenarios = []
    for n in (1, 3, 5):
        top = sums.head(n).index.tolist()
        # Diagnostic: zero the original profit/loss from top n ex-post winners
        # while keeping original stock selections and costs unchanged.
        adjusted = base - pivot[top].sum(axis=1).to_numpy(dtype=float)/100
        if np.any(adjusted <= -1):
            raise ValueError("Invalid neutralized contribution scenario")
        scenarios.append({
            "top_n": n, "tickers": top,
            "removed_gross_contribution_pp": round(float(sums.head(n).sum()), 4),
            "zero_contribution_cagr_pct":
                round(float(((1+growth(adjusted))**(1/years)-1)*100), 4)
        })
    return {
        "top_contributors": [
            {"ticker": t, "additive_gross_contribution_pp": round(float(v), 4)}
            for t, v in sums.head(10).items()
        ],
        "neutralization_scenarios": scenarios,
        "note": "Ex-post zero-contribution stress tests are not tradable portfolios."
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expanding", default=WINDOWS["expanding"])
    p.add_argument("--rolling-36", default=WINDOWS["rolling_36"])
    p.add_argument("--rolling-60", default=WINDOWS["rolling_60"])
    p.add_argument("--recovery-attribution",
                   default="v10_rolling_60/recovery_attribution/stock_month_contributions.csv")
    p.add_argument("--output", default="v10_model_review")
    args = p.parse_args()

    folders = {
        "expanding": Path(args.expanding),
        "rolling_36": Path(args.rolling_36),
        "rolling_60": Path(args.rolling_60),
    }
    datasets = {
        (window, strategy): load_monthly(path/f"{strategy}_monthly.csv")
        for window, path in folders.items() for strategy in STRATEGIES
    }
    reference = datasets["expanding", "recovery"]
    for (window, strategy), frame in datasets.items():
        if frame.signal_date.tolist() != reference.signal_date.tolist():
            raise ValueError(
                f"{window} {strategy}: rebalance dates differ from reference; "
                "do not compare unmatched periods.")

    rows = []
    for split in SPLITS:
        for window in folders:
            for strategy in STRATEGIES:
                part = datasets[window, strategy]
                part = part.loc[part.split == split].copy()
                row = {"split": split, "window": window, "strategy": strategy}
                row.update(stats(part))
                rows.append(row)
    comparison = pd.DataFrame(rows)

    recovery_test = datasets["rolling_60", "recovery"]
    recovery_test = recovery_test.loc[recovery_test.split == "test"].copy()
    concentration = None
    attrib = Path(args.recovery_attribution)
    if attrib.exists():
        concentration = analyze_contributors(recovery_test, attrib)

    validation = comparison.loc[comparison.split == "validation"].copy()
    test = comparison.loc[comparison.split == "test"].copy()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out/"window_comparison.csv", index=False)
    review = {
        "status": "UNVALIDATED - historical research and model selection",
        "validation_2021_2022": validation.to_dict(orient="records"),
        "previously_consulted_2023_onward": test.to_dict(orient="records"),
        "recovery_rolling60_concentration": concentration,
        "needs_recovery_attribution": not attrib.exists(),
        "warnings": [
            "2023 onward was used for selecting 60-month model; not untouched test",
            "Validation 2021-2022 is also part of iterative development",
            "Monthly rank IC near zero is not reliable evidence of forecast skill",
            "Retrospective stock universe/sector assignments, adjusted vendor prices and outliers remain",
            "A fresh forward period or independent point-in-time universe is required",
            "Do not enable Buy/Hold/Sell classifications from these results"
        ]
    }
    (out/"review_summary.json").write_text(json.dumps(review, indent=2, allow_nan=False))
    print(json.dumps({
        "result_file": str(out/"review_summary.json"),
        "validation": review["validation_2021_2022"],
        "recovery_rolling60_concentration": concentration,
        "needs_recovery_attribution": review["needs_recovery_attribution"]
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
