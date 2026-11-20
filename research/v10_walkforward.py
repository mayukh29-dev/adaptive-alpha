"""Adaptive Alpha V10 research experiment (not live trade recommendations).

Historical data: same locally stored adjusted price panel as V8.2.
Model: walk-forward ridge cross-sectional ranking, refit every rebalance.
Features: Momentum, Recovery, relative strength, trend, volatility,
drawdown, rebound and cross-sector percentile features.
Models: momentum, recovery, hybrid + fixed-score baselines.
Evaluation: held-out 2023 onward and validation 2021-2022.
All features at t or earlier; training labels fully mature by t.
Only adjusted closes available; volume, point-in-time fundamentals and
historic sector membership are NOT invented.
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

H = 21
EPS = 1e-12
TEST_START = pd.Timestamp("2023-01-01")
VALID_START = pd.Timestamp("2021-01-01")
WARMUP = 252
MODELS = {
    "momentum": ["momentum","return_21","return_63","return_126","rs_63","trend_50_200","vol_63","momentum_sector_pct"],
    "recovery": ["recovery","drawdown_126","rebound_126","distance_high_126","return_21","vol_63","recovery_sector_pct"],
    "hybrid": ["momentum","recovery","return_21","return_63","return_126","rs_63",
               "trend_50_200","vol_63","drawdown_126","rebound_126",
               "momentum_sector_pct","recovery_sector_pct"],
}
BASELINES = {"momentum_baseline":"momentum","recovery_baseline":"recovery","hybrid_baseline":"hybrid"}

def features_for_panel(prices: pd.DataFrame, sectors: dict):
    stocks = list(sectors)
    p = prices[stocks].astype(float).where(lambda x:x>0)
    spy = prices["SPY"].astype(float).where(lambda x:x>0)
    m,r,h = sig.indicators(p)
    ratio = lambda n: p.div(p.shift(n)).sub(1)
    ma50=p.rolling(50,min_periods=50).mean()
    ma200=p.rolling(200,min_periods=200).mean()
    rolling_low=p.rolling(126,min_periods=126).min()
    rolling_high=p.rolling(126,min_periods=126).max()
    d=p.pct_change(fill_method=None)
    items = {
        "momentum":m, "recovery":r, "hybrid":h,
        "return_21":ratio(21),
        "return_63":ratio(63),
        "return_126":ratio(126),
        "rs_63":ratio(63).sub(spy.div(spy.shift(63)).sub(1),axis=0),
        "trend_50_200":ma50.div(ma200).sub(1),
        "vol_63":d.rolling(63,min_periods=63).std()*np.sqrt(252),
        "drawdown_126":p.div(rolling_high).sub(1),
        "rebound_126":p.div(rolling_low).sub(1),
        "distance_high_126":p.div(rolling_high).sub(1)
    }
    # Across-sector percentile has meaningful peer comparisons with limited history.
    # Sector is supplied by retrospective master, NOT point-in-time sector membership.
    for signal, output in (("momentum","momentum_sector_pct"),
                           ("recovery","recovery_sector_pct")):
        ranked=pd.DataFrame(index=p.index,columns=stocks,dtype=float)
        for sector in sorted(set(sectors.values())):
            cols=[t for t in stocks if sectors[t]==sector]
            ranked[cols]=items[signal][cols].rank(axis=1,method="average",pct=True)
        items[output]=ranked
    # Require all features to be finite, at the exact signal date.
    return items

def model_fit(x,y,alpha=50.0):
    x=np.asarray(x,float);y=np.asarray(y,float)
    mu=x.mean(0)
    sd=x.std(0)
    sd=np.where(sd>1e-8,sd,1.)
    z=(x-mu)/sd
    weights=np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@(y-y.mean()))
    return mu,sd,weights,float(y.mean())

def model_predict(model,x):
    mu,sd,w,intercept=model
    return (np.asarray(x)-mu)/sd@w+intercept

def rank_ic(rows):
    if len(rows)<10:return None
    a=pd.Series([q["score"] for q in rows])
    b=pd.Series([q["actual_21"] for q in rows])
    if a.nunique()<2 or b.nunique()<2:return None
    # Pearson correlation of ranks (Spearman without scipy)
    val=a.rank().corr(b.rank(),method="pearson")
    return float(val) if np.isfinite(val) else None

def simulate_model(name, dates, panel, prices, sectors, min_training_months=24, cost_bps=10.,
                   alpha=50.,top_per_sector=2,train_window_months=0):
    stocks=list(sectors)
    all_feature_names=MODELS.get(name)
    results=[]
    prev={}
    equity=1.
    prev_spy_equity=1.
    all_rows=[]
    for j,when in enumerate(dates):
        next_when=dates[j+1] if j+1<len(dates) else None
        if next_when is None:break
        # Require next-holding-period close to exist; no incomplete last period.
        ix=prices.index.get_loc(when)
        if ix+H>=len(prices.index):break
        maturity=prices.index[ix+H]
        current_close=prices.loc[when]
        future21=prices.loc[maturity]
        next_close=prices.loc[next_when]
        if not np.isfinite(current_close["SPY"]) or not np.isfinite(next_close["SPY"]):continue
        peers=[]
        for ticker in stocks:
            # Determine eligibility using features available at the signal date,
            # NEVER by checking whether the stock has a future return.
            if not np.isfinite(current_close[ticker]) or current_close[ticker]<=0:
                continue
            vals={f:panel[f].at[when,ticker] for f in set(sum(MODELS.values(),[])+list(BASELINES.values()))}
            if not all(np.isfinite(v) for v in vals.values()):
                continue
            if not all(np.isfinite(x) and x>0 for x in (future21[ticker],next_close[ticker],future21["SPY"])):
                raise ValueError(
                    f"{ticker} on {when.date()}: eligible at signal date but missing subsequent price. "
                    "Do not silently remove this stock using future data; audit delisting/corporate actions.")
            peers.append({"ticker":ticker,"sector":sectors[ticker],
                          "features":vals,
                          "actual_21":float(future21[ticker]/current_close[ticker]-future21["SPY"]/current_close["SPY"]),
                          "holding_return":float(next_close[ticker]/current_close[ticker]-1)})
        # Fail rather than drop companies silently from the simulation.
        needed=len(set(sectors.values()))*top_per_sector
        sectors_count={sector:sum(x["sector"]==sector for x in peers) for sector in set(sectors.values())}
        if any(n<top_per_sector for n in sectors_count.values()):
            raise ValueError(f"{when.date()}: insufficient complete historical features: {sectors_count}")
        if name in BASELINES:
            for q in peers:q["score"]=q["features"][BASELINES[name]]
            train_count=None
        else:
            # Use past examples only when outcome observation ends by current signal.
            past=[]
            for previous in all_rows:
                if previous["label_end"]<=when:
                    past.append(previous)
            if train_window_months>0:
                old_threshold=when-pd.DateOffset(months=train_window_months)
                past=[q for q in past if q["signal_date"]>=old_threshold]
            months=len(set(q["signal_date"] for q in past))
            if months<min_training_months:
                # Skip untrained dates while preserving labelled rows for later.
                for q in peers:q.update({"signal_date":when,"label_end":maturity})
                all_rows.extend(peers)
                continue
            cols=all_feature_names
            x=np.asarray([[q["features"][c] for c in cols] for q in past])
            y=np.asarray([q["actual_21"] for q in past])
            params=model_fit(x,y,alpha)
            x_today=np.asarray([[q["features"][c] for c in cols] for q in peers])
            predicted=model_predict(params,x_today)
            for q, score in zip(peers,predicted):q["score"]=float(score)
            train_count=months
        # Label this month's observations only AFTER model fitted.
        for q in peers:q.update({"signal_date":when,"label_end":maturity})
        all_rows.extend(peers)

        chosen=[]
        for sector in sorted(set(sectors.values())):
            group=sorted((x for x in peers if x["sector"]==sector),
                         key=lambda z:(-z["score"],z["ticker"]))
            chosen.extend(group[:top_per_sector])
        equal_weight=1.0/len(chosen)
        weights={x["ticker"]:equal_weight for x in chosen}
        turnover=sum(abs(weights.get(t,0)-prev.get(t,0)) for t in set(weights)|set(prev))
        prev=weights
        gross=sum(x["holding_return"]*equal_weight for x in chosen)
        net=gross-turnover*cost_bps/10000.0
        spy_ret=float(next_close["SPY"]/current_close["SPY"]-1)
        if net<=-1:raise ValueError(f"Implausible one-period strategy bankruptcy: {when}")
        equity*=1+net
        prev_spy_equity*=1+spy_ret
        ic=rank_ic(peers)
        spread=np.nan
        if len(peers)>=10:
            ranked=sorted(peers,key=lambda q:q["score"])
            k=max(2,len(peers)//5)
            spread=float(np.mean([q["actual_21"] for q in ranked[-k:]])
                         -np.mean([q["actual_21"] for q in ranked[:k]]))
        results.append({
            "signal_date":str(when.date()),"next_date":str(next_when.date()),
            "split":"test" if when>=TEST_START else
                    "validation" if when>=VALID_START else "training",
            "n_eligible":len(peers),"n_selected":len(chosen),
            "training_months":train_count,
            "rank_ic":ic,"top_bottom_spread":spread,
            "strategy_return":net,"spy_return":spy_ret,
            "turnover":turnover,"equity":equity,"spy_equity":prev_spy_equity,
            "holdings":";".join(q["ticker"] for q in chosen),
        })
    frame=pd.DataFrame(results)
    frame.attrs["cost_bps"]=cost_bps
    return frame

def summary(d):
    if d.empty:return {"months":0}
    monthly=d.strategy_return.to_numpy(dtype=float)
    spy=d.spy_return.to_numpy(dtype=float)
    def stats(x):
        curve=np.cumprod(1+x)
        years=float((pd.Timestamp(d.iloc[-1].next_date)-pd.Timestamp(d.iloc[0].signal_date)).days/365.25)
        return {
            "total_return_pct":round(float((curve[-1]-1)*100),3),
            "cagr_pct":round(float((curve[-1]**(1/years)-1)*100),3),
            "volatility_pct":round(float(np.std(x,ddof=1)*np.sqrt(12)*100),3),
            "max_drawdown_pct":round(float((curve/np.maximum.accumulate(np.r_[1.,curve])[1:]-1).min()*100),3),
        }
    ics=d.rank_ic.dropna()
    return {
        "months":int(len(d)),"strategy":stats(monthly),"spy":stats(spy),
        "mean_rank_ic":round(float(ics.mean()),5) if len(ics) else None,
        "positive_ic_month_share":round(float((ics>0).mean()),4) if len(ics) else None,
        "mean_top_bottom_21d_spread_pp":round(float(d.top_bottom_spread.mean()*100),3),
        "total_turnover":round(float(d.turnover.sum()),3),
        "cost_bps":float(d.attrs.get("cost_bps",10)),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--master",default="data/master_110.csv")
    p.add_argument("--history",default="history")
    p.add_argument("--output",default="v10_walkforward_output")
    p.add_argument("--exclude-tickers",nargs="*",default=["NWE","CHRD","NCMI"])
    p.add_argument("--cost-bps",type=float,default=10.)
    p.add_argument("--alpha",type=float,default=50.)
    p.add_argument("--min-training-months",type=int,default=24)
    p.add_argument("--train-window-months",type=int,default=0,
                   help="0 means expanding window; >0 means trailing months")
    p.add_argument("--allow-unreviewed-outliers",action="store_true")
    args=p.parse_args()
    if args.alpha<=0 or args.cost_bps<0 or args.min_training_months<12:
        p.error("Use alpha>0, cost-bps>=0, min-training-months>=12.")
    master=bt.read_master(args.master)
    exclude=set(args.exclude_tickers)
    tickers={r["ticker"]:r["sector"] for r in master if r["ticker"] not in exclude}
    if len(set(tickers.values()))!=11:raise ValueError("Expected all eleven sectors")
    prices=bt.load_prices(args.history,["SPY"]+list(tickers))
    # Auditing is not optional for headline results.
    ret=prices.pct_change(fill_method=None)
    suspicious=(ret.abs()>1).stack()
    extreme=suspicious[suspicious]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    alerts=[{"date":str(idx[0].date()),"ticker":idx[1],
             "return_pct":float(ret.loc[idx])*100} for idx in extreme.index]
    (out/"price_audit.json").write_text(json.dumps(alerts,indent=2))
    if alerts and not args.allow_unreviewed_outliers:
        raise RuntimeError(f"{len(alerts)} large daily price movements flagged; review {out/'price_audit.json'}. To run a PROVISIONAL sensitivity experiment, pass --allow-unreviewed-outliers.")
    features=features_for_panel(prices,tickers)
    dates=[x for x in bt.month_end_signal_dates(prices.index)
           if x>=prices.index[WARMUP] and x<=prices.index[-H-1]]
    if len(dates)<50:raise ValueError("Too few historical rebalance dates")
    outputs={}
    for name in list(MODELS)+list(BASELINES):
        monthly=simulate_model(name,dates,features,prices,tickers,
                               min_training_months=args.min_training_months,
                               cost_bps=args.cost_bps,alpha=args.alpha,
                               train_window_months=args.train_window_months)
        monthly.to_csv(out/f"{name}_monthly.csv",index=False)
        outputs[name]={
            "validation_2021_2022":summary(monthly[(monthly.split=="validation")]),
            "heldout_2023_onward":summary(monthly[(monthly.split=="test")]),
        }
    doc={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "signal_version":sig.SIGNAL_VERSION,
        "experiment_version":"v10-walkforward-prototype-1",
        "universe_size":len(tickers),
        "excluded":sorted(exclude),
        "label_horizon_sessions":H,
        "training_policy":"expand or roll; labels mature <= date; monthly retrain",
        "model_features":MODELS,
        "outlier_count":len(alerts),
        "outlier_status":"UNREVIEWED" if alerts else "NO_THRESHOLD_BREACHES",
        "cost_bps":args.cost_bps,
        "alpha":args.alpha,
        "results":outputs,
        "limitations":[
          "Retrospectively selected universe and retrospective sector membership",
          "Adjusted vendor prices not verified as point-in-time or total-return certified",
          "Test period 2023 onward has informed previous modeling choices, not genuinely untouched",
          "Portfolio uses monthly rebalance close to next monthly close; ideal fills, no intramonth weight drift",
          "Simplified turnover costs; excludes slippage, market impact, taxes and financing",
          "No inference of probability calibration; return ranks and portfolio metrics only",
          "These are historical experiments, not trading recommendations."
        ]
    }
    (out/"evaluation.json").write_text(json.dumps(doc,indent=2,allow_nan=False))
    print(json.dumps({"results":str(out/"evaluation.json"),"strategies":outputs},indent=2))

if __name__=="__main__":
    main()
