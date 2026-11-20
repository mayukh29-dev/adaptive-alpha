"""Train and evaluate time-separated Adaptive Alpha 21-session excess-return models.

Place in same folder as local backtest_v82.py, with sibling signals.py
and prediction.py downloaded from the repository. Price inputs are adjusted
daily closes. This is a price-only model trained on a retrospective universe;
performance is not a point-in-time investable strategy backtest.

Fixed split and alpha (NO tuning on the 2023+ holdout).
Training signal: 2016..2020; validation: 2021..2022; test: 2023 onward.
Labels MUST finish within their own split; 21-session label overlap purged
across split boundaries. Monthly samples reduce horizon overlap within split.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_v82 as bt
import signals as sig
import prediction as pred

HORIZON = 21
TRAIN_END = pd.Timestamp("2020-12-31")
VALID_START = pd.Timestamp("2021-01-01")
VALID_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")


def metrics(frame, model):
    """Diagnostics reported only for forecast-vs-benchmark labels."""
    if frame.empty:
        return {"observations": 0}
    x = np.vstack(frame["features"].values)
    actual = frame["target"].to_numpy(dtype=float)
    forecasts = pred.predict_model(model, x)
    result = frame[["signal_date", "sector", "ticker"]].copy()
    result["actual"], result["prediction"] = actual, forecasts
    daily_spreads = []
    for _, group in result.groupby("signal_date"):
        if len(group) < 10:
            continue
        sorted_g = group.sort_values("prediction")
        k = max(2, len(group)//5)
        daily_spreads.append(
            float(sorted_g["actual"].iloc[-k:].mean()
                  - sorted_g["actual"].iloc[:k].mean()))
    date_ics = []
    for _, g in result.groupby("signal_date"):
        if len(g) > 8 and g["prediction"].nunique() > 1 and g["actual"].nunique() > 1:
            date_ics.append(g["prediction"].corr(g["actual"], method="spearman"))
    base_mae = float(np.mean(abs(actual)))  # Zero excess return as benchmark
    return {
        "observations": int(len(actual)),
        "signal_months": int(result["signal_date"].nunique()),
        "mae_excess_return_pp": round(float(np.mean(abs(actual-forecasts)))*100, 4),
        "zero_excess_baseline_mae_pp": round(base_mae*100, 4),
        "mean_rank_ic": round(float(np.nanmean(date_ics)), 5) if date_ics else None,
        "top_minus_bottom_mean_pp": round(float(np.mean(daily_spreads))*100, 4) if daily_spreads else None,
        "direction_accuracy": round(float(np.mean(np.sign(forecasts)==np.sign(actual))), 4),
        "mean_actual_excess_pp": round(float(actual.mean())*100, 4),
        "mean_predicted_excess_pp": round(float(forecasts.mean())*100, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default="data/master_110.csv")
    parser.add_argument("--history", default="history")
    parser.add_argument("--output", default="predictive_output")
    parser.add_argument("--exclude-tickers", nargs="*", default=["NWE", "CHRD", "NCMI"])
    parser.add_argument("--alpha", type=float, default=20.0)
    args = parser.parse_args()
    if args.alpha <= 0:
        parser.error("alpha must be positive")
    members = bt.read_master(args.master)
    exclude = set(args.exclude_tickers)
    tickers = [r["ticker"] for r in members if r["ticker"] not in exclude]
    sectors = {r["ticker"]:r["sector"] for r in members if r["ticker"] not in exclude}
    if len(set(sectors.values()))!=11:
        raise ValueError("Need 11 represented sectors")

    prices = bt.load_prices(args.history, ["SPY"] + tickers)
    momentum, recovery, hybrid = sig.indicators(prices[tickers])
    dates = bt.month_end_signal_dates(prices.index)
    records=[]
    for date in dates:
        idx=prices.index.get_loc(date)
        if idx < sig.WARMUP or idx + HORIZON >= len(prices.index):
            continue
        end=prices.index[idx+HORIZON]
        start_close=prices.loc[date]
        end_close=prices.loc[end]
        spy0,spy1=start_close["SPY"],end_close["SPY"]
        if not np.isfinite(spy0) or not np.isfinite(spy1) or spy0<=0 or spy1<=0:
            continue
        spy_return=spy1/spy0-1
        # Prevent labels crossing the training/validation boundary.
        if date<=TRAIN_END and end<=TRAIN_END:
            split="train"
        elif VALID_START<=date<=VALID_END and end<=VALID_END:
            split="validation"
        elif date>=TEST_START:
            split="test"
        else:
            continue
        for t in tickers:
            m,r=momentum.at[date,t],recovery.at[date,t]
            p0,p1=start_close[t],end_close[t]
            if not all(np.isfinite(v) for v in (m,r,p0,p1)) or p0<=0 or p1<=0:
                continue
            records.append({
                "signal_date":date.date().isoformat(),
                "label_end":end.date().isoformat(),
                "ticker":t,"sector":sectors[t],
                "momentum":float(m),"recovery":float(r),
                "target":float((p1/p0-1)-spy_return),
                "split":split,
            })
    df=pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No eligible labeled observations.")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    artifacts={}
    for name in pred.FEATURES:
        subset=df.copy()
        subset["features"]=[
            pred.features(m,r,name) for m,r in zip(subset.momentum,subset.recovery)
        ]
        train=subset.loc[subset.split=="train"]
        valid=subset.loc[subset.split=="validation"]
        test=subset.loc[subset.split=="test"]
        if min(len(train),len(valid),len(test))<250:
            raise ValueError(f"Insufficient independent split coverage for {name}")
        X=np.vstack(train.features.to_numpy())
        model=pred.fit_model(X,train.target.to_numpy(),alpha=args.alpha)
        val_metrics=metrics(valid,model)
        test_metrics=metrics(test,model)
        # Gate protects against accidental inference of validated buy/sell ability.
        # It is a descriptive gate, NOT a statistical significance test.
        has_evidence = (
            test_metrics["observations"]>=250
            and test_metrics["mae_excess_return_pp"] <
                test_metrics["zero_excess_baseline_mae_pp"]
            and test_metrics["mean_rank_ic"] is not None
            and test_metrics["mean_rank_ic"]>0
            and test_metrics["top_minus_bottom_mean_pp"] is not None
            and test_metrics["top_minus_bottom_mean_pp"]>0
        )
        artifacts[name]={
            "feature_names":list(pred.FEATURES[name]),
            "model":model,
            "validation":val_metrics,
            "heldout_test":test_metrics,
            "research_gate_passed":bool(has_evidence),
        }

    payload={
        "model_version":pred.MODEL_VERSION,
        "signal_version":sig.SIGNAL_VERSION,
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "target":"21-session forward stock total return MINUS SPY total return (decimal)",
        "label_horizon_sessions":HORIZON,
        "train_through":"2020-12-31",
        "validation":"2021-01-01 through 2022-12-31",
        "test":"2023-01-01 onward, with fully observed 21-session labels",
        "excluded_tickers":sorted(exclude),
        "universe_size":len(tickers),
        "strategies":artifacts,
        "warnings":[
            "Prototype research model. Signals are NOT personalized buy/sell advice.",
            "Retrospectively selected universe; survivorship bias remains.",
            "Uses historical adjusted daily price, not point-in-time price or fundamentals.",
            "Heldout diagnostics are retrospective; multiple stocks/months correlated.",
            "No confidence intervals, calibration guarantees, or live predictive track record.",
            "Buy/sell candidate labels are disabled if the research gate fails.",
            "Historical backtest of this unified signal version must be run separately.",
        ]
    }
    (out/"predictive_model.json").write_text(
        json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    # Publish only aggregate diagnostics, not training rows (which can be large).
    (out/"evaluation.json").write_text(
        json.dumps({
            "model_version":pred.MODEL_VERSION,
            "signal_version":sig.SIGNAL_VERSION,
            "validation":{k:v["validation"] for k,v in artifacts.items()},
            "heldout_test":{k:v["heldout_test"] for k,v in artifacts.items()},
            "research_gate":{k:v["research_gate_passed"] for k,v in artifacts.items()},
        },indent=2),encoding="utf-8")
    print(json.dumps({
        "output":str(out/"predictive_model.json"),
        "research_gate":{k:v["research_gate_passed"] for k,v in artifacts.items()},
        "heldout":{k:v["heldout_test"] for k,v in artifacts.items()},
    },indent=2))


if __name__=="__main__":
    main()
