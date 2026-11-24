import os
import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import yfinance as yf

import signals as signal_engine
import comparison as comparison_engine
import score_history as score_history_engine
import watchlist as watchlist_engine
import sector_discovery as sector_discovery_engine
import prediction as prediction_engine

st.set_page_config(page_title="Adaptive Alpha | Research Terminal", page_icon="📈", layout="wide")
st.markdown("""<style>
.stApp{background:#0a1220;color:#e9eff9}
[data-testid="stSidebar"]{background:#111e32}
[data-testid="stMetric"]{background:#15233b;border:1px solid #2a405e;border-radius:14px;padding:16px}
h1,h2,h3{letter-spacing:-.03em}
</style>""",unsafe_allow_html=True)

RESULTS = {
    "Adaptive Alpha proxy":{"CAGR":18.456,"Total return":469.402,"Volatility":21.22,"Sharpe":0.905,"Max drawdown":-42.944},
    "SPY":{"CAGR":14.698,"Total return":308.9,"Volatility":17.85,"Sharpe":0.858,"Max drawdown":-33.717},
}
RANDOM={"simulations":1000,"beaten":73.1,"median_cagr":15.3015,"p05_cagr":10.6098,"p95_cagr":48.5506,
        "equal_stock_cagr":31.723,"equal_stock_vol":77.64,"equal_sector_cagr":31.578,"equal_sector_vol":75.637}
CAVEAT = "Historical price-only proxy on a later-selected 109-stock universe, not a production Adaptive Alpha V7 signal replay. Survivorship/selection bias, possible price-data anomalies, and execution limitations remain. Not investment advice."
st.sidebar.markdown("## ◈ ADAPTIVE ALPHA")
st.sidebar.caption("Stock research beta · adjusted daily closes")
advanced_mode=st.sidebar.toggle("Advanced research",value=False,help="Show technical research tools and unaudited historical comparisons.")
if advanced_mode:
    page=st.sidebar.radio("Navigation",["Stock Scores","Overview","Live Market","Strategy Explorer",
        "Predictive Watchlist","Strategy benchmarks","Stock research agent","Methodology"])
    st.sidebar.warning("Historical research returns are unaudited and not independently predictive.")
else:
    page="Stock Scores"

@st.cache_data(ttl=900,show_spinner=False)
def load_stock(ticker):
    stock=yf.Ticker(ticker)
    info=stock.info or {}
    hist=stock.history(period="1y",auto_adjust=True)
    return info,hist

def nice(value,percent=False):
    try:
        n=float(value)
        if pd.isna(n): return "Not available"
        return f"{n*100:,.2f}%" if percent else f"{n:,.2f}"
    except (ValueError,TypeError):
        return "Not available"

def research_snapshot(ticker,info,hist):
    close=float(hist["Close"].iloc[-1]) if not hist.empty else None
    return {
        "ticker":ticker,
        "company":info.get("longName") or info.get("shortName") or ticker,
        "sector":info.get("sector"),
        "industry":info.get("industry"),
        "summary":(info.get("longBusinessSummary") or "")[:1800],
        "last_close":close,
        "last_price_date":str(hist.index[-1].date()) if not hist.empty else "Unavailable",
        "market_cap":info.get("marketCap"),
        "trailing_pe":info.get("trailingPE"),
        "forward_pe":info.get("forwardPE"),
        "revenue_growth":info.get("revenueGrowth"),
        "earnings_growth":info.get("earningsGrowth"),
        "profit_margin":info.get("profitMargins"),
        "operating_margin":info.get("operatingMargins"),
        "debt_to_equity":info.get("debtToEquity"),
        "free_cash_flow":info.get("freeCashflow"),
        "currency":info.get("currency"),
        "source":"Yahoo Finance through yfinance; retrieved at runtime and not independently audited"
    }

def basic_answer(question,s):
    q=question.lower()
    company=s["company"]
    if any(x in q for x in ["valuation","expensive","p/e","pe ratio","multiple"]):
        return f"{company} trailing P/E: {nice(s['trailing_pe'])}; forward P/E: {nice(s['forward_pe'])}. P/E alone cannot determine whether a stock is undervalued."
    if any(x in q for x in ["growth","revenue","sales","earnings"]):
        return f"Reported revenue growth: {nice(s['revenue_growth'],True)}; reported earnings growth: {nice(s['earnings_growth'],True)}. Verify reporting periods against filings."
    if any(x in q for x in ["profit","margin","cash flow","cashflow"]):
        return f"Reported profit margin: {nice(s['profit_margin'],True)}; operating margin: {nice(s['operating_margin'],True)}; free cash flow: {nice(s['free_cash_flow'])} in vendor-reported currency units."
    if any(x in q for x in ["debt","leverage","risk"]):
        return f"Vendor debt-to-equity ratio: {nice(s['debt_to_equity'])}. This alone is not a comprehensive risk assessment. Company-specific debt maturities, liquidity, competitive and regulatory risk need filings research."
    if any(x in q for x in ["price","close","trading","stock"]):
        return f"Most recent available adjusted close: {nice(s['last_close'])} {s['currency'] or ''} on {s['last_price_date']}. This is not a live trading quote."
    if any(x in q for x in ["what","business","company","do","overview"]):
        return s["summary"] or f"{company}: no business description available from the current data provider."
    return ("I can answer factual questions from this stock snapshot about valuation, revenue growth, margins, debt, price, and its business. "
            "For open-ended comparisons or an investment thesis, use AI analysis with your own key; missing evidence will not be invented.")

def ai_answer(question,s,key):
    from openai import OpenAI
    client=OpenAI(api_key=key,timeout=30)
    response=client.responses.create(
        model="gpt-4.1-mini",
        max_output_tokens=550,
        instructions="You are a cautious stock research assistant. Use ONLY the supplied vendor snapshot. Explicitly state missing data; distinguish facts from interpretations. Do not fabricate quotes, filings, catalysts, current news, price targets, or returns. No personalized investment advice. Be concise.",
        input="Stock snapshot:\n"+str(s)+"\n\nUser question: "+question
    )
    return response.output_text


def compute_strategy_signals(history, spy_history=None):
    """Exactly the shared v1.0 score implementation used by the new trainer."""
    return signal_engine.latest_scores(history)


@st.cache_data(ttl=300, show_spinner=False)
def prediction_artifact():
    try:
        return prediction_engine.load_artifact()
    except Exception:
        return None


def predictive_readout(scores, style):
    """Returns a readout only if model exists and research holdout gate passed."""
    artifact=prediction_artifact()
    key=style.lower()
    if not artifact:
        return None, "Prediction unavailable: run the time-separated trainer and publish its evaluated artifact."
    if artifact.get("signal_version") != signal_engine.SIGNAL_VERSION:
        return None, "Prediction unavailable: model and live signals have different versions."
    strategy=artifact.get("strategies",{}).get(key)
    if not strategy:
        return None, "No trained model for this strategy."
    if not strategy.get("research_gate_passed",False):
        return None, "Prediction withheld: this strategy did not meet its predefined holdout research checks."
    try:
        vector=prediction_engine.features(scores["momentum"],scores["recovery"],key)
        excess=float(prediction_engine.predict_model(strategy["model"],vector)[0])
        if not pd.notna(excess):
            return None, "Prediction unavailable because model output is invalid."
        return {"expected_excess":excess, "label":prediction_engine.research_signal(excess),
                "test":strategy.get("heldout_test",{})}, None
    except Exception:
        return None, "Prediction unavailable: model inference failed."


def show_prediction(scores, style):
    result,reason=predictive_readout(scores,style)
    st.subheader("21-trading-session predictive research")
    if result is None:
        st.info(reason)
        st.caption("This dashboard will not invent a forecast or Buy/Hold/Sell rating before passing held-out testing.")
        return
    st.metric("Forecast excess return vs SPY",f"{result['expected_excess']*100:+.2f} percentage points")
    st.markdown("**Research classification: "+result["label"]+"**")
    test=result["test"]
    st.caption("Model based on historical adjusted daily prices. Held-out mean monthly rank IC: "
               +str(test.get("mean_rank_ic","N/A"))
               +"; excess-return MAE: "+str(test.get("mae_excess_return_pp","N/A"))
               +" percentage points. No probability, guarantee, or personalized recommendation is implied.")


@st.cache_data(ttl=300, show_spinner=False)
def recent_daily_signal_history(symbol):
    # Daily adjusted close: use the same signal input as Strategy Explorer.
    # This is distinct from the intraday quote and may lag during the session.
    return yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=True)


@st.cache_data(ttl=60, show_spinner=False)
def market_intraday(symbol, period, interval):
    stock=yf.Ticker(symbol)
    return stock.history(period=period,interval=interval,auto_adjust=False,prepost=False)

@st.fragment(run_every="60s")
def market_live_panel(symbol, period, interval):
    st.caption("This panel checks for updated vendor quotes approximately every 60 seconds while open. Exchange quotes may be delayed; updates are not guaranteed.")
    try:
        data=market_intraday(symbol,period,interval)
        if data is None or data.empty or "Close" not in data:
            st.warning("No recent vendor prices available. Check the ticker or retry later.")
            return
        close=pd.to_numeric(data["Close"],errors="coerce").dropna()
        if close.empty:
            st.warning("Latest quote is unavailable.")
            return
        latest=float(close.iloc[-1])
        prior=float(close.iloc[-2]) if len(close)>1 else None
        change=(latest/prior-1)*100 if prior and prior>0 else None
        stamp=data.index[-1]
        currency=""
        try:
            currency=str(yf.Ticker(symbol).fast_info.get("currency") or "")
        except Exception:
            pass
        a,b,c=st.columns(3)
        a.metric(f"{symbol} last recorded price",f"{latest:,.2f} {currency}")
        b.metric("Change vs prior chart bar",f"{change:+.2f}%" if change is not None else "N/A")
        if "Volume" in data.columns:
            volume=pd.to_numeric(data["Volume"],errors="coerce")
            c.metric("Last bar volume",f"{int(volume.iloc[-1]):,}" if pd.notna(volume.iloc[-1]) else "N/A")
        else:
            c.metric("Last bar volume","N/A")
        st.caption(f"Latest recorded market bar: {stamp}. Data retrieved via Yahoo Finance/yfinance; quote delay and market session vary. Not a consolidated real-time feed.")
        st.subheader("Adaptive Alpha strategy reads")
        st.caption("Scores are calculated from recent adjusted DAILY closes. Refreshed from the vendor cache every 5 minutes, not tick-by-tick. These are exploratory 0–100 price-only proxy signals, not the full production V7 models.")
        try:
            signal_history=recent_daily_signal_history(symbol)
            signals=compute_strategy_signals(signal_history)
            if signals is None:
                st.info("Momentum, Recovery and Hybrid scores require at least 127 valid daily closes. No score is shown when history is insufficient.")
            else:
                m_col,r_col,h_col=st.columns(3)
                m_col.metric("Momentum",f"{signals['momentum']:.1f} / 100")
                r_col.metric("Recovery",f"{signals['recovery']:.1f} / 100")
                h_col.metric("Hybrid",f"{signals['hybrid']:.1f} / 100")
                st.caption(f"Signal data through {signals['date']} (daily adjusted close). Momentum: 63-session return proxy; Recovery: 126-session rebound/high proximity proxy; Hybrid: 60% Momentum + 40% Recovery.")
                selected_model=st.selectbox("Forecast strategy",["Momentum","Recovery","Hybrid"],index=2,key="live_predictive_style")
                show_prediction(signals,selected_model)
        except Exception:
            st.info("Strategy signals temporarily unavailable from the daily-price provider. The intraday price chart remains available.")

        st.line_chart(close.rename("Price"),height=290)
        if "Volume" in data.columns:
            st.bar_chart(pd.to_numeric(data["Volume"],errors="coerce").rename("Volume"),height=140)
        if len(close)>1:
            st.download_button("Export displayed prices",data.to_csv().encode("utf-8"),file_name=f"{symbol}_market_data.csv",mime="text/csv")
    except Exception:
        st.warning("Quote service temporarily unavailable. The rest of the dashboard remains usable.")

if page=="Stock Scores":
    st.title("Adaptive Alpha")
    st.caption("Three clear scores. One adjustable strategy. Research your stocks without the clutter.")
    st.caption("Price-only technical readings, not probabilities or Buy/Sell recommendations. Historical research is under Advanced research in the sidebar.")

    with st.form("simple_ticker_search", border=False):
        left,right=st.columns([4,1],vertical_alignment="bottom")
        with left:
            requested=st.text_input("Stock ticker",value=st.session_state.get("simple_current_ticker","NVDA"),
                                    key="simple_ticker_input",
                                    help="Enter a US-listed ticker, for example NVDA, AAPL, MSFT or SPY.")
        with right:
            submitted=st.form_submit_button("Analyze",type="primary",use_container_width=True)
    symbol=requested.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,11}",symbol):
        st.warning("Enter a valid ticker such as AAPL or NVDA.")
    else:
        st.session_state["simple_current_ticker"]=symbol
        try:
            history=recent_daily_signal_history(symbol)
            scores=compute_strategy_signals(history)
        except Exception:
            scores=None
            st.warning("Could not retrieve valid daily market data. Please retry later.")
        if scores is None:
            st.info("Scores require at least 127 valid adjusted daily closes. This stock may be too new or vendor history may be unavailable.")
        else:
            m=float(scores["momentum"])
            r=float(scores["recovery"])
            default_h=float(scores["hybrid"])
            st.subheader(symbol)
            st.caption(f"Daily adjusted price through {scores['date']} · Source: Yahoo Finance via yfinance · Not an intraday quote")
            st.markdown("#### Strategy scores")
            a,b,c=st.columns(3)
            a.metric("Momentum",f"{m:.1f} / 100",help="Based on the stock's return over the previous 63 trading sessions.")
            b.metric("Recovery",f"{r:.1f} / 100",help="Combines rebound from a 126-session low and closeness to the 126-session high.")
            c.metric("Standard Hybrid",f"{default_h:.1f} / 100",help="Fixed research formula: 60% Momentum and 40% Recovery.")

            st.divider()
            st.subheader("Customize your Hybrid score")
            st.caption("Adjust the mix below. Momentum and Recovery stay the same; only your custom Hybrid changes.")
            if "simple_hybrid_pct" not in st.session_state:
                st.session_state.simple_hybrid_pct=60
            st.slider("Momentum weight",min_value=0,max_value=100,step=5,
                      key="simple_hybrid_pct",help="Recovery automatically receives the remaining percentage.")
            w=st.session_state.simple_hybrid_pct
            st.write(f"**Momentum {w}%** + **Recovery {100-w}%**")
            personal_h=signal_engine.weighted_hybrid(m,r,w/100.0)
            x,y=st.columns([2,1],vertical_alignment="center")
            x.metric("Your custom Hybrid",f"{personal_h:.1f} / 100")
            with y:
                st.button("Reset to 60/40",use_container_width=True,
                          on_click=lambda: st.session_state.update(simple_hybrid_pct=60))
            st.progress(personal_h/100.0)
            if w==60:
                st.caption("Your custom score matches the standard 60/40 Hybrid.")
            else:
                st.caption(f"Your custom score differs from the standard Hybrid by {personal_h-default_h:+.1f} points.")
            st.warning("Custom weights are for exploration only. Their historical returns and predictive accuracy have not been backtested or validated.")
            with st.expander("How are these scores calculated?"):
                st.markdown("""
**Momentum:** The stock's 63-session return is mapped onto a 0–100 technical score.

**Recovery:** 60% rebound from the trailing 126-session low, plus 40% proximity to the trailing 126-session high. Each underlying component is bounded at 100.

**Standard Hybrid:** 60% Momentum + 40% Recovery. This is the reference formula used by the shared price-only scoring engine.

**Custom Hybrid:** Your chosen Momentum percentage times Momentum, plus the remainder times Recovery.

**What scores do not mean:** A score of 80 is not an 80% chance the stock will rise. Scores do not include business fundamentals, fair value, or complete investment risks.
""")
            with st.expander("Score history and price chart",expanded=False):
                st.caption("Explore how Momentum, Recovery, and Hybrid readings changed over time.")
                st.caption("Daily historical technical scores from past adjusted closes. This is a score history, not a historical investment-return chart.")
                months_label=st.segmented_control(
                    "Score history range",options=["1 month","3 months","6 months","12 months"],
                    default="3 months",key="simple_history_range")
                months_count={"1 month":1,"3 months":3,"6 months":6,"12 months":12}
                history_scores=score_history_engine.score_history(
                    history,momentum_pct=w,months=months_count[months_label or "3 months"])
                if history_scores.empty:
                    st.info("Not enough usable historical daily prices to plot scores over this range.")
                else:
                    st.line_chart(history_scores,y=["Momentum","Recovery","Standard Hybrid","Custom Hybrid"],
                                  height=290,y_label="Technical score (0–100)")
                    current=history_scores.iloc[-1]
                    first=history_scores.iloc[0]
                    st.caption(
                        f"Custom Hybrid: {first['Custom Hybrid']:.1f} → {current['Custom Hybrid']:.1f} "
                        f"({current['Custom Hybrid']-first['Custom Hybrid']:+.1f} score points across the displayed observations). "
                        "Changing the slider recomputes only the custom blend.")
                    st.download_button(
                        "Download score history CSV",
                        history_scores.rename_axis("Date").to_csv().encode("utf-8"),
                        file_name=f"{symbol}_score_history_{months_count[months_label or '3 months']}m.csv",
                        mime="text/csv")
                st.caption("A rising score indicates stronger recent technical readings, not a guaranteed future return.")

                st.caption("Recent adjusted closing prices")
                st.line_chart(history["Close"].dropna().tail(126),height=240)
    st.divider()
    st.markdown("### Explore more")
    st.caption("Choose one task at a time. The Momentum/Recovery slider above applies to every section.")
    compare_tab,discover_tab,watch_tab=st.tabs(["Compare stocks","Discover by sector","My watchlist"])
    with compare_tab:
        st.subheader("Compare stocks")
        st.caption("Enter two or three tickers to compare their technical scores using the same Hybrid weights shown above.")
        with st.form("compare_stock_form"):
            compare_raw=st.text_input("Tickers (comma separated)",value="NVDA, AMD, MSFT",
                                      help="Enter 2 or 3 distinct symbols, for example NVDA, AMD, MSFT.")
            compare_clicked=st.form_submit_button("Compare stocks",type="primary")
        if compare_clicked:
            try:
                symbols=comparison_engine.parse_comparison_tickers(compare_raw)
            except ValueError as exc:
                st.session_state.pop("compared_stocks",None)
                st.warning(str(exc))
            else:
                compared=[]
                problems=[]
                with st.spinner("Calculating comparable stock scores..."):
                    for ticker in symbols:
                        try:
                            history=recent_daily_signal_history(ticker)
                            observed=compute_strategy_signals(history)
                            if observed is None:
                                problems.append(f"{ticker}: insufficient daily price history")
                            else:
                                compared.append({"ticker":ticker,
                                                 "momentum":observed["momentum"],
                                                 "recovery":observed["recovery"],
                                                 "date":observed["date"]})
                        except Exception:
                            problems.append(f"{ticker}: market data unavailable")
                st.session_state["compared_stocks"]=compared
                st.session_state["comparison_errors"]=problems
        for problem in st.session_state.get("comparison_errors",[]):
            st.warning(problem)
        compared=st.session_state.get("compared_stocks",[])
        if len(compared)>=2:
            weight=st.session_state.get("simple_hybrid_pct",60)
            result=comparison_engine.build_comparison(compared,weight)
            dates={stock["date"] for stock in compared}
            if len(dates)>1:
                st.warning("Stocks have different latest price dates. Compare cautiously because signals are not as of the same market session.")
            st.caption(f"Ranked by Custom Hybrid · {weight}% Momentum / {100-weight}% Recovery · larger score ranks first")
            cards=st.columns(len(result))
            for col,(_,stock) in zip(cards,result.iterrows()):
                with col:
                    st.metric(f"#{int(stock['Rank'])}  {stock['Ticker']}",
                              f"{stock['Custom Hybrid']:.1f} / 100")
                    st.caption(f"Momentum {stock['Momentum']:.1f} · Recovery {stock['Recovery']:.1f}")
            st.dataframe(result,hide_index=True,use_container_width=True)
            st.caption("Ranking compares only the entered stocks; it is not a predicted-return ranking or an investment recommendation.")
            st.download_button("Download comparison CSV",
                               result.to_csv(index=False).encode("utf-8"),
                               file_name="adaptive_alpha_comparison.csv",mime="text/csv")
        elif compared:
            st.info("At least two stocks need valid prices to display a comparison.")
    with discover_tab:
        st.subheader("Discover stocks by sector")
        st.caption("Rank a curated set of US equities. View one sector or compare all 11 sectors at once.")
        sector_options=["All sectors (66 stocks)"]+list(sector_discovery_engine.SECTOR_UNIVERSE)
        sector_choice=st.selectbox("Sector view",sector_options,key="discovery_sector")
        rank_choice=st.selectbox("Rank by",list(sector_discovery_engine.SORT_CHOICES),
                                 key="discovery_rank",help="Custom Hybrid uses the Momentum/Recovery slider above.")
        all_sectors=sector_choice=="All sectors (66 stocks)"
        target_sectors=(list(sector_discovery_engine.SECTOR_UNIVERSE)
                        if all_sectors else [sector_choice])
        stocks_to_scan=list(dict.fromkeys(
            t for sector in target_sectors
            for t in sector_discovery_engine.SECTOR_UNIVERSE[sector]))
        st.caption(f"Coverage: {len(target_sectors)} sector(s); {len(stocks_to_scan)} curated tickers. "
                   "This is not a complete market or authoritative GICS sector-membership scan.")
        if all_sectors:
            st.info("The first full scan requests market history for up to 66 stocks and may be slower "
                    "or encounter vendor rate limits. Run it only when needed.")
        if st.button("Scan all sectors" if all_sectors else "Scan selected sector",
                     type="primary",key="discovery_scan"):
            scanned={}
            failed={}
            with st.spinner(f"Fetching scores for {len(stocks_to_scan)} curated stocks..."):
                for discovery_ticker in stocks_to_scan:
                    try:
                        observed=compute_strategy_signals(recent_daily_signal_history(discovery_ticker))
                        if observed is None:
                            failed[discovery_ticker]="Insufficient adjusted daily close history"
                        else:
                            scanned[discovery_ticker]={"ticker":discovery_ticker,
                                "momentum":observed["momentum"],
                                "recovery":observed["recovery"],
                                "date":observed["date"]}
                    except Exception:
                        failed[discovery_ticker]="Vendor market data unavailable"
            # Store the scored symbols and any failures without baking in
            # the slider setting; changing the slider only reranks cached rows.
            st.session_state["sector_discovery_snapshot"]={
                "tickers":stocks_to_scan,"rows":scanned,"errors":failed,
                "scanned_at":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            }
        snapshot=st.session_state.get("sector_discovery_snapshot",{})
        cached=snapshot.get("rows",{})
        scanned_tickers=set(snapshot.get("tickers",[]))
        if set(stocks_to_scan).issubset(scanned_tickers):
            weight=st.session_state.get("simple_hybrid_pct",60)
            grouped={sector:[cached[t] for t in sector_discovery_engine.SECTOR_UNIVERSE[sector]
                             if t in cached]
                     for sector in target_sectors}
            if all_sectors:
                tables=sector_discovery_engine.all_sector_tables(grouped,weight,rank_choice)
            else:
                tables={sector_choice:sector_discovery_engine.rank_sector(
                    grouped[sector_choice],weight,rank_choice)}
            failures=snapshot.get("errors",{})
            relevant_failures=[t for t in stocks_to_scan if t in failures]
            date_values={row["date"] for sec_rows in grouped.values() for row in sec_rows}
            st.caption(f"Snapshot fetched {snapshot.get('scanned_at','unknown time')} · "
                       f"{len(stocks_to_scan)-len(relevant_failures)} of {len(stocks_to_scan)} symbols scored")
            if relevant_failures:
                with st.expander(f"Unavailable tickers ({len(relevant_failures)})"):
                    for t in relevant_failures:
                        st.write(f"{t}: {failures[t]}")
            if len(date_values)>1:
                st.warning("Stocks have different latest price dates, so rankings are not fully synchronized.")
            st.caption(f"Custom Hybrid = {weight}% Momentum + {100-weight}% Recovery. "
                       "Changing the slider updates all displayed sector rankings without refetching.")
            if all_sectors:
                st.markdown("#### Sector leaders at a glance")
                leaders=sector_discovery_engine.sector_leaders(tables,rank_choice)
                st.dataframe(leaders,hide_index=True,use_container_width=True)
                st.caption("A leader is only the highest-scoring stock in this six-symbol sample, not the best investment in its sector.")
                st.markdown("#### All sector rankings")
                sector_names=list(sector_discovery_engine.SECTOR_UNIVERSE)
                # Two columns allow all eleven tables to be visible concurrently.
                cols=st.columns(2)
                for i,sector in enumerate(sector_names):
                    with cols[i%2]:
                        st.markdown(f"**{sector}**")
                        table=tables[sector]
                        if table.empty:
                            st.info("No eligible stock scores returned.")
                        else:
                            st.dataframe(table,hide_index=True,use_container_width=True,
                                         height=min(250,60+35*len(table)))
                st.download_button("Download all sector rankings CSV",
                    data=pd.concat([t.assign(Sector=sector) for sector,t in tables.items()],
                                   ignore_index=True).to_csv(index=False).encode("utf-8"),
                    file_name="adaptive_alpha_all_sector_rankings.csv",
                    mime="text/csv",key="discovery_export_all")
            else:
                table=tables[sector_choice]
                if table.empty:
                    st.info("No eligible stock scores were returned. Try scanning again.")
                else:
                    st.dataframe(table,hide_index=True,use_container_width=True)
                    st.caption(f"Highest {rank_choice} among this curated sample: {table.iloc[0]['Ticker']}. "
                               "This is not an investment recommendation.")
                    st.download_button("Download sector scores CSV",
                        data=table.to_csv(index=False).encode("utf-8"),
                        file_name="adaptive_alpha_sector_scores.csv",
                        mime="text/csv",key="discovery_export_single")
        else:
            st.info("Select the sector view and click Scan to fetch scores. "
                    "Previous results are only reused when all required tickers were scanned.")
        st.caption("This curated universe has six representative symbols per sector. "
                   "Scores use adjusted daily prices; cached price data may lag. "
                   "No complete sector coverage or predictive claim is implied.")

    with watch_tab:
        st.subheader("My watchlist")
        st.caption("Save up to 12 stock symbols for this browser session. No account or permanent storage.")
        if "simple_watchlist" not in st.session_state:
            st.session_state.simple_watchlist=[]
        if "watchlist_snapshots" not in st.session_state:
            st.session_state.watchlist_snapshots={}
        current_ticker=st.session_state.get("simple_current_ticker","NVDA")
        if re.fullmatch(r"[A-Z][A-Z0-9.-]{0,11}",current_ticker):
            if st.button(f"Save {current_ticker} to watchlist",key="watchlist_add_current"):
                try:
                    st.session_state.simple_watchlist=watchlist_engine.add_ticker(
                        st.session_state.simple_watchlist,current_ticker)
                except ValueError as exc:
                    st.warning(str(exc))
        if st.session_state.simple_watchlist:
            st.caption(f"{len(st.session_state.simple_watchlist)} / {watchlist_engine.MAX_WATCHLIST} saved")
            for saved_ticker in list(st.session_state.simple_watchlist):
                label_col,open_col,remove_col=st.columns([4,1,1],vertical_alignment="center")
                label_col.write(f"**{saved_ticker}**")
                open_col.button("View",key=f"watch_view_{saved_ticker}",use_container_width=True,
                                on_click=lambda t=saved_ticker: st.session_state.update(
                                    simple_current_ticker=t,simple_ticker_input=t))
                if remove_col.button("Remove",key=f"watch_remove_{saved_ticker}",use_container_width=True):
                    st.session_state.simple_watchlist=watchlist_engine.remove_ticker(
                        st.session_state.simple_watchlist,saved_ticker)
                    st.session_state.watchlist_snapshots.pop(saved_ticker,None)
                    st.rerun()
            if st.button("Refresh watchlist scores",type="primary"):
                snapshots={}
                errors=[]
                with st.spinner("Checking saved stocks with the market-data provider..."):
                    for t in st.session_state.simple_watchlist:
                        try:
                            h=recent_daily_signal_history(t)
                            z=compute_strategy_signals(h)
                            if z is None:
                                errors.append(f"{t}: not enough valid daily closes")
                            else:
                                snapshots[t]={"ticker":t,"momentum":z["momentum"],
                                              "recovery":z["recovery"],"date":z["date"]}
                        except Exception:
                            errors.append(f"{t}: vendor market data unavailable")
                st.session_state.watchlist_snapshots=snapshots
                st.session_state.watchlist_errors=errors
            for problem in st.session_state.get("watchlist_errors",[]):
                st.warning(problem)
            saved_rows=[st.session_state.watchlist_snapshots[t]
                        for t in st.session_state.simple_watchlist
                        if t in st.session_state.watchlist_snapshots]
            if saved_rows:
                weight=st.session_state.get("simple_hybrid_pct",60)
                frame=[]
                for record in saved_rows:
                    m=float(record["momentum"])
                    r=float(record["recovery"])
                    frame.append({
                        "Ticker":record["ticker"],
                        "Momentum":round(m,1),
                        "Recovery":round(r,1),
                        "Standard Hybrid":round(signal_engine.weighted_hybrid(m,r,0.6),1),
                        "Custom Hybrid":round(signal_engine.weighted_hybrid(m,r,weight/100),1),
                        "As of":record["date"],
                    })
                watch_df=pd.DataFrame(frame).sort_values(
                    ["Custom Hybrid","Ticker"],ascending=[False,True]).reset_index(drop=True)
                st.dataframe(watch_df,hide_index=True,use_container_width=True)
                st.caption(f"Custom Hybrid uses the current {weight}% Momentum / {100-weight}% Recovery weights. "
                           "Saved scores update only when you click Refresh; changing the slider changes the blend, not prices.")
                st.download_button("Download watchlist CSV",data=watch_df.to_csv(index=False).encode("utf-8"),
                                   file_name="adaptive_alpha_session_watchlist.csv",mime="text/csv")
            else:
                st.info("Click Refresh watchlist scores to load the latest available technical readings.")
        else:
            st.info("No stocks saved yet. Analyze a ticker above and choose Save to watchlist.")
        st.caption("Watchlists remain in Streamlit session memory only; they may disappear after refresh, disconnect, or app restart. No cross-device sync.")

    st.caption("Adaptive Alpha · Research beta · Not investment advice. Advanced tools are in the sidebar.")

elif page=="Overview":
    st.title("Adaptive Alpha")
    st.caption("QUANTITATIVE MARKET INTELLIGENCE • INTERACTIVE BETA")
    st.info(CAVEAT)
    left,right=st.columns(2)
    with left: st.metric("Strategy CAGR","18.46%","3.76 percentage points vs SPY")
    with right: st.metric("Strategy Sharpe (zero RF)","0.905","SPY 0.858")
    a,b,c=st.columns(3)
    a.metric("Monthly rebalances","124")
    b.metric("Positions per rebalance","22")
    c.metric("Eligible historical stocks","109 / 110")
    st.subheader("Portfolio performance summary")
    st.dataframe(pd.DataFrame(RESULTS).T,use_container_width=True)
    st.warning("The return figures are from a retrospectively selected universe. They do not validate live trading or the production quality-scoring model.")


elif page=="Live Market":
    st.title("Live Market")
    st.caption("Auto-updating vendor market data · for research and demonstration, not live order execution.")
    symbol=st.text_input("Search stock or ETF ticker",value="NVDA",max_chars=12,key="live_symbol").strip().upper()
    preset=st.selectbox("Chart range",["1 day (5 min bars)","5 days (15 min bars)","1 month (daily)","3 months (daily)"],key="live_range")
    settings={"1 day (5 min bars)":("1d","5m"),"5 days (15 min bars)":("5d","15m"),
              "1 month (daily)":("1mo","1d"),"3 months (daily)":("3mo","1d")}
    if not re.fullmatch(r"[A-Z][A-Z0-9.\\-]{0,11}",symbol):
        st.warning("Enter a valid ticker such as AAPL, SPY or NVDA.")
    else:
        p,i=settings[preset]
        market_live_panel(symbol,p,i)
    st.info("Important: This is a refreshed vendor feed, not guaranteed real-time exchange-licensed data. Prices and volume may arrive late, especially outside regular US market hours.")

elif page=="Strategy Explorer":
    st.title("Strategy Explorer")
    st.caption("Explore Momentum, Recovery and Hybrid using an illustrative recent-price signal snapshot.")
    st.warning("These are single-stock current-snapshot calculations, NOT validated historical strategy returns or production Adaptive Alpha scores.")
    style=st.radio("View strategy",["Momentum","Recovery","Hybrid"],horizontal=True)
    ticker_explorer=st.text_input("Stock ticker",value="NVDA",key="explorer_ticker",max_chars=12).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\\-]{0,11}",ticker_explorer):
        st.error("Enter a ticker such as NVDA or MSFT.")
    else:
        try:
            # Require at least 126 trading sessions plus a prior close.
            ex_history=yf.Ticker(ticker_explorer).history(period="2y",auto_adjust=True)
            signal=compute_strategy_signals(ex_history)
            if signal is None:
                st.info("Not enough valid market history for this signal (127+ valid daily closes required).")
            else:
                st.caption(f"Vendor adjusted-close series · as of {signal['date']} · not a point-in-time certified dataset")
                st.line_chart(ex_history["Close"].tail(126),height=240)
                if style=="Momentum":
                    st.subheader("Momentum · 63-trading-session price strength")
                    a,b=st.columns(2)
                    a.metric("Momentum proxy score",f"{signal['momentum']:.1f} / 100")
                    b.metric("63-session price return",f"{signal['return_63']:+.2f}%")
                    st.write("Price-only research proxy: 63-session return mapped to 0–100 using ±25% bounds, clipped to the score range.")
                    st.info("Momentum is a directional technical signal. It does not capture company quality, valuation, liquidity or financial risk.")
                elif style=="Hybrid":
                    st.subheader("Hybrid · Momentum + Recovery")
                    a,b,c=st.columns(3)
                    a.metric("Hybrid proxy",f"{signal['hybrid']:.1f} / 100")
                    b.metric("Momentum",f"{signal['momentum']:.1f}")
                    c.metric("Recovery",f"{signal['recovery']:.1f}")
                    st.progress(min(100,max(0,int(round(signal["hybrid"]))))/100)
                    st.write("This explorer uses the historical pilot's 60% Momentum / 40% Recovery blend. Recovery is approximated from the trailing 126-day low and proximity to the high; it is not a replay of production V7.")
                else:
                    st.subheader("Recovery · Rebound and price stabilization")
                    a,b,c=st.columns(3)
                    a.metric("Recovery proxy score",f"{signal['recovery']:.1f} / 100")
                    b.metric("126-day maximum drawdown",f"{signal['drawdown_126']:.1f}%")
                    c.metric("126-day current close",f"{signal['close']:.2f}")
                    st.progress(min(100,max(0,int(round(signal["recovery"]))))/100)
                    st.write("This illustrative Recovery proxy combines a rebound from the trailing 126-session low (60%) and proximity to the trailing 126-session high (40%). A recovery signal does not establish business quality or a favorable investment outcome.")
                    st.caption("This is an exploratory price-only approximation; it is not the certified production V7 Recovery calculation.")
                show_prediction(signal,style)
                st.markdown("**Research note:** To compare actual strategy performance, we must run separate point-in-time backtests for Momentum, Recovery, and Hybrid.")
        except Exception:
            st.error("Could not fetch or process sufficient vendor market data. Please retry later.")


elif page=="Predictive Watchlist":
    st.title("Predictive Watchlist")
    st.caption("Rank a user-defined stock watchlist using consistent Momentum, Recovery and Hybrid signals. This is not a full-market screen.")
    watch_raw=st.text_input("Tickers, separated by commas (up to 8)",value="NVDA,MSFT,AAPL,AMZN,GOOGL,META",key="watchlist_tickers")
    selected_style=st.radio("Ranking approach",["Momentum","Recovery","Hybrid"],horizontal=True,key="watchlist_style")
    tickers=list(dict.fromkeys(x.strip().upper() for x in watch_raw.split(",") if x.strip()))
    if len(tickers)>8:
        st.warning("Limit the watchlist to 8 stocks to avoid market data provider rate limits.")
    elif any(not re.fullmatch(r"[A-Z][A-Z0-9.\\-]{0,11}",t) for t in tickers):
        st.warning("One or more ticker symbols are invalid.")
    elif st.button("Analyze watchlist",type="primary"):
        rows=[]
        with st.spinner("Retrieving recent daily closes and calculating shared strategy scores..."):
            for t in tickers:
                try:
                    history=recent_daily_signal_history(t)
                    scores=compute_strategy_signals(history)
                    if scores is None:
                        rows.append({"Ticker":t,"Status":"Insufficient data"})
                        continue
                    prediction,reason=predictive_readout(scores,selected_style)
                    rows.append({
                        "Ticker":t,
                        "Status":"Evaluated" if prediction else "Score only",
                        "Momentum":round(scores["momentum"],1),
                        "Recovery":round(scores["recovery"],1),
                        "Hybrid":round(scores["hybrid"],1),
                        "Forecast excess vs SPY (%)":round(prediction["expected_excess"]*100,2) if prediction else None,
                        "Research label":prediction["label"] if prediction else "Not validated",
                        "As of":scores["date"],
                    })
                except Exception:
                    rows.append({"Ticker":t,"Status":"Market data unavailable"})
        table=pd.DataFrame(rows)
        sort_key="Forecast excess vs SPY (%)" if "Forecast excess vs SPY (%)" in table and table["Forecast excess vs SPY (%)"].notna().any() else selected_style
        if sort_key in table.columns:
            table=table.sort_values(sort_key,ascending=False,na_position="last")
        st.dataframe(table,use_container_width=True,hide_index=True)
        st.caption("Watchlist ranks are relative to tickers entered, not sector ranks. Signals are descriptive until evaluated predictive artifacts are available. Buy/Hold/Sell research labels are withheld when model validation fails.")
        st.download_button("Download watchlist research",table.to_csv(index=False).encode("utf-8"),file_name="adaptive_alpha_watchlist.csv",mime="text/csv")

elif page=="Strategy benchmarks":
    st.title("Strategy benchmarks")
    st.caption("1,000 seeded randomized selections, same historical universe and dates")
    a,b,c=st.columns(3)
    a.metric("Random portfolios beaten","73.1%")
    b.metric("Random median CAGR","15.30%")
    c.metric("Strategy CAGR","18.46%")
    data=pd.DataFrame({"CAGR (%)":[10.6098,15.3015,18.456,48.5506,31.723,31.578]},
                      index=["Random p05","Random median","Strategy","Random p95","Equal-stock universe","Equal-sector universe"])
    st.bar_chart(data)
    st.caption("The random 95th-percentile CAGR and unusually high equal-universe volatility require historical price auditing. This is not statistical proof of stock-selection skill.")
    st.dataframe(pd.DataFrame({"Strategy":[ "Equal-stock universe","Equal-sector universe"],
        "CAGR (%)":[31.723,31.578],"Volatility (%)":[77.64,75.637]}),hide_index=True)
elif page=="Stock research agent":
    st.title("Stock Research Agent")
    st.caption("Ask company-specific questions using a public vendor snapshot, not a verified fundamental filings database.")
    ticker=st.text_input("Ticker",value="NVDA",max_chars=12).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,11}",ticker):
        st.error("Enter a valid ticker, such as NVDA or MSFT.")
        st.stop()
    try:
        info,hist=load_stock(ticker)
    except Exception as exc:
        st.error("The market-data provider could not be reached. Please retry. No values were fabricated.")
        st.stop()
    if not info and hist.empty:
        st.warning("No market data available for this ticker.")
        st.stop()
    s=research_snapshot(ticker,info,hist)
    st.subheader(s["company"])
    st.caption(f"{s['sector'] or 'Sector unavailable'} · {s['industry'] or 'Industry unavailable'}")
    a,b,c=st.columns(3)
    a.metric("Latest available close",nice(s["last_close"]))
    b.metric("Trailing P/E",nice(s["trailing_pe"]))
    c.metric("Revenue growth",nice(s["revenue_growth"],True))
    if not hist.empty:
        st.line_chart(hist["Close"],height=230)
    st.caption(f"Latest vendor price date: {s['last_price_date']}. Not a real-time or verified quote.")
    mode=st.radio("Answer mode",["Snapshot facts (free)","AI interpretation (requires your own API key)"],horizontal=True)
    key=""
    if mode.startswith("AI"):
        key=st.text_input("OpenAI API key (session use only)",type="password",help="Provide your own API key. Do not paste credentials into the public chat. This beta does not store your key.")
        st.caption("AI requests may incur OpenAI API costs. This demo does not provide site-wide metering or rate limiting.")
    with st.form("research"):
        question=st.text_area("Ask a question about this stock",value="What are the main financial risks?")
        asked=st.form_submit_button("Research stock",type="primary")
    if asked:
        if not question.strip():
            st.warning("Enter a research question.")
        elif mode.startswith("AI") and not key:
            st.error("Enter your API key to use AI interpretation, or choose Snapshot facts.")
        else:
            try:
                answer=ai_answer(question,s,key) if mode.startswith("AI") else basic_answer(question,s)
                st.markdown("### Research answer")
                st.write(answer)
                st.caption(s["source"])
            except Exception:
                st.error("The requested AI response failed; the app did not substitute fabricated research.")
else:
    st.title("Methodology and limitations")
    st.write(CAVEAT)
    st.markdown("""
**Backtest:** price-only technical proxy; 2016-08-01 to 2026-11-13; 2,588 sessions, monthly top-two selection in each of 11 sectors, 10 bps per dollar traded, excludes NWE.

**Key limitations:** survivorship and retrospective selection bias; no historical point-in-time fundamentals; adjusted-price revisions; execution friction; no untouched holdout; random portfolios may have different turnover; return outliers require auditing.

**Research agent:** Yahoo Finance vendor snapshot. Free question answering is deterministic and limited to supported fields. Optional AI interpretation uses a visitor-supplied API key, and does not verify external filings or live news.

**Not investment advice.** Do not rely on displayed figures to execute trades.
""")
st.divider()
st.caption("Adaptive Alpha · Research beta · Demo data and unaudited methodology")
