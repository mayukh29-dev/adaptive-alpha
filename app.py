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
page=st.sidebar.radio("Navigation",["Overview","Live Market","Strategy Explorer","Strategy benchmarks","Stock research agent","Methodology"])
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
        except Exception:
            st.info("Strategy signals temporarily unavailable from the daily-price provider. The intraday price chart remains available.")

        st.line_chart(close.rename("Price"),height=290)
        if "Volume" in data.columns:
            st.bar_chart(pd.to_numeric(data["Volume"],errors="coerce").rename("Volume"),height=140)
        if len(close)>1:
            st.download_button("Export displayed prices",data.to_csv().encode("utf-8"),file_name=f"{symbol}_market_data.csv",mime="text/csv")
    except Exception:
        st.warning("Quote service temporarily unavailable. The rest of the dashboard remains usable.")

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
                st.markdown("**Research note:** To compare actual strategy performance, we must run separate point-in-time backtests for Momentum, Recovery, and Hybrid.")
        except Exception:
            st.error("Could not fetch or process sufficient vendor market data. Please retry later.")

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
