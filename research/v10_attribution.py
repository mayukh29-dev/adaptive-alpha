"""Stock-level contribution audit for Adaptive Alpha V10.

Reads existing V10 monthly holding logs and the SAME local adjusted-close panel.
No portfolio rerun or stock exclusion; reconciles each model's monthly returns.
Output: ticker-period contributions, ticker aggregated, annual returns,
largest gain/loss months, and best-month sensitivity.

Contributions are simple additive monthly return contributions, not compounded
standalone profit or causal alpha. Corporate actions/universe bias remain.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import backtest_v82 as bt

def annualized(d):
    if d.empty:return None
    grow=float(np.prod(1+d["strategy_return"].astype(float)))
    start=pd.Timestamp(d.iloc[0]["signal_date"])
    end=pd.Timestamp(d.iloc[-1]["next_date"])
    years=(end-start).days/365.25
    return round((grow**(1/years)-1)*100,4) if grow>0 and years>0 else None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--monthly",default="v10_walkforward_output/hybrid_monthly.csv")
    parser.add_argument("--history",default="history")
    parser.add_argument("--output",default="v10_walkforward_output/attribution")
    parser.add_argument("--cost-bps",type=float,default=10.0)
    parser.add_argument("--split",default="test",choices=["test","validation","all"])
    args=parser.parse_args()
    if args.cost_bps<0:parser.error("--cost-bps must be >=0")
    d=pd.read_csv(args.monthly)
    required={"signal_date","next_date","split","holdings","strategy_return","turnover"}
    missing=required-set(d.columns)
    if missing:raise ValueError(f"Missing columns: {sorted(missing)}")
    d=d.sort_values("signal_date")
    chosen=d if args.split=="all" else d.loc[d.split==args.split].copy()
    if chosen.empty:raise ValueError(f"No rows for split={args.split}")
    chosen=chosen.reset_index(drop=True)
    tickers=sorted(set(
        t for value in chosen["holdings"].fillna("") for t in str(value).split(";") if t
    ))
    prices=bt.load_prices(args.history,["SPY"]+tickers)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    monthly_reconciliations=[]
    for _,r in chosen.iterrows():
        start=pd.Timestamp(r["signal_date"])
        end=pd.Timestamp(r["next_date"])
        holdings=[x for x in str(r["holdings"]).split(";") if x]
        if len(holdings)!=22 or len(set(holdings))!=22:
            raise ValueError(f"{start.date()}: expected exactly 22 unique holdings, found {len(holdings)}")
        cost=float(r["turnover"])*args.cost_bps/10000
        contributions=[]
        for ticker in holdings:
            if start not in prices.index or end not in prices.index:
                raise ValueError(f"Date {start.date()} or {end.date()} missing from data")
            p0=float(prices.at[start,ticker])
            p1=float(prices.at[end,ticker])
            if not (np.isfinite(p0) and np.isfinite(p1) and p0>0 and p1>0):
                raise ValueError(f"Missing/invalid close for {ticker}, {start.date()} -> {end.date()}")
            holding_return=p1/p0-1
            contribution=holding_return/22
            contributions.append(contribution)
            rows.append({
                "signal_date":str(start.date()),"next_date":str(end.date()),
                "ticker":ticker,"holding_return_pct":holding_return*100,
                "gross_portfolio_contribution_pp":contribution*100,
                "equal_weight":1/22,
            })
        reconstructed=float(sum(contributions)-cost)
        reported=float(r["strategy_return"])
        gap=reconstructed-reported
        monthly_reconciliations.append({
            "signal_date":str(start.date()),"next_date":str(end.date()),
            "reported_net_return_pct":reported*100,
            "reconstructed_net_return_pct":reconstructed*100,
            "gross_return_pct":sum(contributions)*100,
            "turnover_cost_pp":cost*100,
            "reconciliation_gap_pp":gap*100,
        })
        if abs(gap)>1e-9:
            raise ValueError(f"Portfolio return cannot be reconciled {start.date()} delta={gap:.12g}")
    contributions=pd.DataFrame(rows)
    reconciliation=pd.DataFrame(monthly_reconciliations)
    summary=contributions.groupby("ticker").agg(
        months_held=("ticker","size"),
        sum_gross_contribution_pp=("gross_portfolio_contribution_pp","sum"),
        average_holding_return_pct=("holding_return_pct","mean"),
        largest_month_gain_pct=("holding_return_pct","max"),
        largest_month_loss_pct=("holding_return_pct","min"),
    ).sort_values("sum_gross_contribution_pp",ascending=False).reset_index()
    # Additive contribution sums are not exact compounded wealth attribution.
    by_year=chosen.copy()
    by_year["year"]=pd.to_datetime(by_year.signal_date).dt.year
    annual=[]
    for year,g in by_year.groupby("year"):
        annual.append({
            "year":int(year),"months":len(g),
            "strategy_total_return_pct":float((np.prod(1+g.strategy_return.astype(float))-1)*100),
            "spy_total_return_pct":float((np.prod(1+g.spy_return.astype(float))-1)*100) if "spy_return" in g else None,
        })
    actual_cagr=annualized(chosen)
    # Stress test removing the SINGLE strongest realized month (not a tradeable alternative).
    strongest_index=chosen.strategy_return.astype(float).idxmax()
    without_best=chosen.drop(index=strongest_index).copy()
    # Treat the month as zero return to preserve exactly the same elapsed calendar time.
    zero_best=chosen.copy()
    zero_best.loc[strongest_index,"strategy_return"]=0.0
    investment_return=float(np.prod(1+chosen.strategy_return.astype(float))-1)
    stressed_return=float(np.prod(1+zero_best.strategy_return.astype(float))-1)
    span_years=(pd.Timestamp(chosen.iloc[-1].next_date)-pd.Timestamp(chosen.iloc[0].signal_date)).days/365.25
    stressed_cagr=(1+stressed_return)**(1/span_years)-1 if span_years>0 else np.nan
    details={
        "scope":"Historical V10 stock-level attribution; NO inference of predictive validation",
        "selected_months":len(chosen),"split":args.split,
        "gross_portfolio_contributions_sum_pp":float(contributions.gross_portfolio_contribution_pp.sum()),
        "net_portfolio_return_compounded_pct":investment_return*100,
        "cagr_pct":actual_cagr,
        "best_month":str(chosen.loc[strongest_index,"signal_date"]),
        "best_month_return_pct":float(chosen.loc[strongest_index,"strategy_return"])*100,
        "best_month_set_to_zero_compounded_return_pct":stressed_return*100,
        "best_month_set_to_zero_cagr_pct":stressed_cagr*100,
        "max_abs_reconciliation_error_pp":float(reconciliation.reconciliation_gap_pp.abs().max()),
        "top_gross_contributor_tickers":summary.head(10)["ticker"].tolist(),
        "warnings":[
            "Contributions are additive monthly arithmetic percentage point effects; not causal or compounded ticker alpha",
            "Original portfolio reset each holding to 1/22 at every monthly close; shares do not drift",
            "Historical selection/survivorship bias remains and vendor adjusted prices have unresolved audits",
            "Best-month removal is a diagnostic scenario, not an executable rebalanced investment strategy",
            "Excess return versus SPY and risk-adjusted significance are not proved by this attribution"
        ]
    }
    contributions.to_csv(out/"stock_month_contributions.csv",index=False)
    summary.to_csv(out/"ticker_attribution.csv",index=False)
    reconciliation.to_csv(out/"reconciliation.csv",index=False)
    pd.DataFrame(annual).to_csv(out/"annual_returns.csv",index=False)
    (out/"attribution_summary.json").write_text(json.dumps(details,indent=2,allow_nan=False))
    print(json.dumps(details,indent=2,allow_nan=False))

if __name__=="__main__":
    main()
