"""V10 robustness audit: concentration, calendar regimes, matched benchmarks.

Requires outputs of v10_walkforward.py and v10_attribution.py; no new data
downloads.  This is a diagnostic analysis, not a trading strategy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

STRATEGIES=("momentum","recovery","hybrid",
            "momentum_baseline","recovery_baseline","hybrid_baseline")

def geometric_returns(arr):
    a=np.asarray(arr,dtype=float)
    if a.size == 0 or not np.isfinite(a).all() or np.any(a<=-1):
        raise ValueError("Invalid monthly return series")
    return float(np.expm1(np.log1p(a).sum()))

def cagr_from_dates(returns,dates,ends):
    if len(returns)==0:return None
    start=pd.Timestamp(dates.iloc[0]);end=pd.Timestamp(ends.iloc[-1])
    years=(end-start).days/365.25
    if years<=0:raise ValueError("Non-positive analysis date span")
    return float((1+geometric_returns(returns))**(1/years)-1)*100

def load_monthly(path):
    df=pd.read_csv(path,parse_dates=["signal_date","next_date"])
    required={"signal_date","next_date","strategy_return","spy_return","split","rank_ic","holdings"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(df.columns))}")
    df=df.sort_values("signal_date").reset_index(drop=True)
    if df.signal_date.duplicated().any():
        raise ValueError(f"Duplicate monthly signal dates in {path}")
    if df[["strategy_return","spy_return"]].isna().any().any():
        raise ValueError(f"Missing realized returns in {path}")
    return df

def robustness(frame, benchmark):
    frame=frame.copy()
    base=benchmark.set_index("signal_date")
    if not frame.signal_date.isin(base.index).all():
        raise ValueError("Benchmark dates differ")
    frame["baseline_return"]=frame.signal_date.map(base.strategy_return)
    rows=[]
    for year,g in frame.groupby(frame.signal_date.dt.year):
        rows.append({"year":int(year),"months":len(g),
                     "model_total_pct":round(geometric_returns(g.strategy_return)*100,4),
                     "benchmark_total_pct":round(geometric_returns(g.baseline_return)*100,4),
                     "spy_total_pct":round(geometric_returns(g.spy_return)*100,4),
                     "model_excess_spy_pp":round((geometric_returns(g.strategy_return)-geometric_returns(g.spy_return))*100,4),
                     "model_excess_baseline_pp":round((geometric_returns(g.strategy_return)-geometric_returns(g.baseline_return))*100,4)})
    win_months=(frame.strategy_return>frame.baseline_return).sum()
    rolling=[]
    for i in range(11,len(frame)):
        g=frame.iloc[i-11:i+1]
        rolling.append({
            "ending_month":str(g.iloc[-1].signal_date.date()),
            "model_12m_pct":round(geometric_returns(g.strategy_return)*100,4),
            "benchmark_12m_pct":round(geometric_returns(g.baseline_return)*100,4),
            "spy_12m_pct":round(geometric_returns(g.spy_return)*100,4),
        })
    rolling_positive=sum(row["model_12m_pct"]>row["benchmark_12m_pct"] for row in rolling)
    return {
        "annual":rows,"rolling_12m":rolling,
        "months_beating_rule":int(win_months),
        "months_total":len(frame),
        "rolling_years_beating_rule":rolling_positive,
        "rolling_years_tested":len(rolling),
        "model_cagr_pct":round(cagr_from_dates(frame.strategy_return,frame.signal_date,frame.next_date),4),
        "rule_cagr_pct":round(cagr_from_dates(frame.baseline_return,frame.signal_date,frame.next_date),4),
        "spy_cagr_pct":round(cagr_from_dates(frame.spy_return,frame.signal_date,frame.next_date),4),
        "mean_rank_ic":float(frame.rank_ic.dropna().mean()) if frame.rank_ic.notna().any() else None,
    }

def concentration(df, contributions, k_values=(1,3,5,10)):
    # Compute a diagnostic "remove ticker contribution" scenario by replacing
    # that ticker's periodic equal-weight profit/loss with cash return 0,
    # holding all original portfolio decisions fixed.
    stock=contributions.groupby("ticker").gross_portfolio_contribution_pp.sum().sort_values(ascending=False)
    ticker_history=contributions.pivot_table(index="signal_date",columns="ticker",
       values="gross_portfolio_contribution_pp",aggfunc="sum",fill_value=0)
    ticker_history.index=pd.to_datetime(ticker_history.index)
    ticker_history=ticker_history.reindex(df.signal_date).fillna(0)
    base=df.strategy_return.to_numpy(dtype=float)
    output=[]
    for k in k_values:
        top=stock.head(k).index.tolist()
        removed=ticker_history[top].sum(axis=1).to_numpy()/100
        adjusted=base-removed
        if np.any(adjusted<=-1):raise ValueError("Diagnostic returns invalid")
        output.append({
            "top_k":k,"tickers":top,
            "sum_removed_gross_contribution_pp":float(stock.head(k).sum()),
            "diagnostic_total_return_pct":round(geometric_returns(adjusted)*100,4),
            "diagnostic_cagr_pct":round(cagr_from_dates(adjusted,df.signal_date,df.next_date),4),
        })
    return {"contributors":stock.reset_index().rename(
                columns={"gross_portfolio_contribution_pp":"gross_contribution_pp"}).to_dict("records"),
            "remove_top_ticker_contributions_as_cash":output}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="v10_walkforward_output")
    parser.add_argument("--attribution",default="v10_walkforward_output/attribution/stock_month_contributions.csv")
    parser.add_argument("--output",default="v10_walkforward_output/robustness")
    parser.add_argument("--split",choices=["test","validation"],default="test")
    args=parser.parse_args()
    folder=Path(args.source)
    frames={name:load_monthly(folder/f"{name}_monthly.csv") for name in STRATEGIES}
    for name in frames:
        frames[name]=frames[name].loc[frames[name].split==args.split].copy().reset_index(drop=True)
    anchor=frames["hybrid"]
    if anchor.empty:raise ValueError("No results for selected split")
    expected=anchor.signal_date.tolist()
    for name,df in frames.items():
        if df.signal_date.tolist()!=expected:
            raise ValueError(f"{name} monthly dates are not aligned")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    assessments={}
    yearly=[]
    rolling=[]
    for name in ["momentum","recovery","hybrid"]:
        baseline=frames[f"{name}_baseline"]
        res=robustness(frames[name],baseline)
        assessments[name]={k:v for k,v in res.items() if k not in ("annual","rolling_12m")}
        yearly.extend({"strategy":name,**row} for row in res["annual"])
        rolling.extend({"strategy":name,**row} for row in res["rolling_12m"])
    contrib=pd.read_csv(args.attribution)
    if args.split!="test":
        # Attribution file defaults to test split; do not misapply results.
        raise ValueError("For validation split, rerun v10_attribution.py with --split validation and provide its contributions path")
    required={"signal_date","ticker","gross_portfolio_contribution_pp"}
    if not required.issubset(contrib.columns):
        raise ValueError("Attribution contributions missing required columns")
    if set(contrib.signal_date)!=set(anchor.signal_date.dt.strftime("%Y-%m-%d")):
        raise ValueError("Attribution dates do not exactly match the selected Hybrid portfolio months")
    conc=concentration(anchor,contrib)
    pd.DataFrame(yearly).to_csv(output/"annual_comparison.csv",index=False)
    pd.DataFrame(rolling).to_csv(output/"rolling_12m_comparison.csv",index=False)
    pd.DataFrame(conc["contributors"]).to_csv(output/"ticker_concentration.csv",index=False)
    doc={
        "status":"UNVALIDATED RESEARCH DIAGNOSTIC",
        "period":args.split,
        "matched_strategy_comparisons":assessments,
        "hybrid_concentration":conc["remove_top_ticker_contributions_as_cash"],
        "warnings":[
            "2023+ test period was already consulted in model development; NOT untouched",
            "Counterfactual zero ticker contributions assumes cash without rescreening or retraining, NOT tradable",
            "Contributions are gross additive return points; adjusted diagnostic retains original costs",
            "Retrospective stock universe/sector bias and unreviewed vendor prices persist",
            "12-month rolling windows overlap; proportions are descriptive, not independent significance tests",
            "No Buy/Hold/Sell research labels are authorized by this audit"
        ]
    }
    (output/"robustness_summary.json").write_text(json.dumps(doc,indent=2,allow_nan=False))
    print(json.dumps(doc,indent=2,allow_nan=False))

if __name__=="__main__":
    main()
