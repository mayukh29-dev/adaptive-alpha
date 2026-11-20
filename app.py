import os
import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import yfinance as yf

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
page=st.sidebar.radio("Navigation",["Overview","Strategy Explorer","Strategy benchmarks","Stock research agent","Methodology"])
st.sidebar.caption("BETA • Historical data through 2026-11-13")
st.sidebar.warning("Research prototype. Historical returns have not passed the final price audit.")

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
    """Exploratory current-date signals; not the historical backtest engine."""
    if history.empty or "Close" not in history or len(history) < 127:
        return None
    p = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(p) < 127 or (p <= 0).any():
        return None
    current = float(p.iloc[-1])
    r63 = current / float(p.iloc[-64]) - 1
    momentum = max(0.0, min(100.0, 50.0 + 50.0 * r63 / 0.25))
    window = p.iloc[-126:]
    low, high = float(window.min()), float(window.max())
    rebound = 100.0 * (current - low) / low if low > 0 else 0.0
    rebound_score = max(0.0, min(100.0, rebound))
    proximity = 100.0 * current / high if high > 0 else 0.0
    proximity_score = max(0.0, min(100.0, proximity))
    recovery = 0.6 * rebound_score + 0.4 * proximity_score
    daily = p.pct_change(fill_method=None).dropna()
    vol = float(daily.iloc[-63:].std() * (252 ** 0.5) * 100) if len(daily) >= 63 else None
    peak = p.iloc[-126:].cummax()
    drawdown = float(((p.iloc[-126:] / peak) - 1).min() * 100)
    return {"momentum":momentum,"recovery":recovery,"hybrid":0.6*momentum+0.4*recovery,
            "return_63":r63*100,"volatility_63":vol,"drawdown_126":drawdown,
            "date":str(p.index[-1].date()),"close":current}

if page=="Overview":
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

elif page=="Strategy Explorer":
    st.title("Strategy Explorer")
    st.caption("Switch between historical-proxy strategies and an illustrative risk-screen preview.")
    st.warning("These are single-stock current-snapshot calculations, NOT validated historical strategy returns or production Adaptive Alpha scores.")
    style=st.radio("View strategy",["Momentum","Hybrid","Conservative (risk-screen preview)"],horizontal=True)
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
                    st.subheader("Conservative · Risk-screen preview")
                    st.info("The historical project files do not establish a separately validated Conservative scoring formula. This preview applies user-selected risk guardrails to the same current-stock data and makes no historical performance claim.")
                    max_vol=st.slider("Maximum annualized 63-day volatility (%)",10,100,35,5,key="max_vol")
                    max_dd=st.slider("Maximum 126-day peak-to-trough drawdown (%)",5,80,25,5,key="max_dd")
                    min_mom=st.slider("Minimum Momentum proxy score",0,100,40,5,key="min_mom")
                    a,b,c=st.columns(3)
                    a.metric("Annualized 63-day volatility",f"{signal['volatility_63']:.1f}%" if signal["volatility_63"] is not None else "N/A")
                    b.metric("126-day max drawdown",f"{signal['drawdown_126']:.1f}%")
                    c.metric("Momentum",f"{signal['momentum']:.1f}")
                    meets=(signal["volatility_63"] is not None and signal["volatility_63"]<=max_vol
                           and abs(signal["drawdown_126"])<=max_dd and signal["momentum"]>=min_mom)
                    if meets:
                        st.success("Passes the selected illustrative risk-screen guardrails. This is NOT a buy recommendation.")
                    else:
                        st.warning("Does not pass the selected illustrative risk-screen guardrails.")
                    st.caption("These thresholds are user-adjustable prototype controls, not the original Adaptive Alpha Conservative strategy.")
                st.markdown("**Research note:** To compare actual strategy performance, we must run separate point-in-time backtests for Momentum, Hybrid, and a formally specified Conservative strategy.")
        except Exception:
            st.error("Could not fetch or process sufficient vendor market data. Please retry later.")
\nelif page=="Strategy benchmarks":
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
