# Adaptive Alpha — AI Stock Research Beta

A Streamlit research dashboard demonstrating a **price-only historical proxy** of the Adaptive Alpha investing research project, plus a question-answering Stock Research Agent.

## Simplified stock-scoring homepage (Phase 1)

The app opens by default to a single-stock research screen. Enter a ticker to see the current 0–100 Momentum, Recovery and standard Hybrid scores. The standard Hybrid remains fixed at 60% Momentum / 40% Recovery. Visitors can change a separate **custom Hybrid** with a 0–100% Momentum slider; the Recovery weight is automatically 100% minus Momentum, and a reset button restores 60/40. These custom weights do **not** change the historical research models, imply a validated return forecast, or authorize Buy/Hold/Sell ratings.

The homepage now includes **historical score trends** (1, 3, 6, or 12 calendar months) from the same daily adjusted close signals used in the current score cards. Changing the user Hybrid weight updates its entire historical custom-score line, while the baseline 60/40 line stays fixed. Users can download chart data as CSV. A score chart is not a return backtest and cannot validate customized strategy performance.\n\nThe homepage also supports **side-by-side comparison of 2–3 stocks**. Enter comma-separated tickers, click **Compare stocks**, and see each stock's Momentum, Recovery, original 60/40 Hybrid, and user-weighted Hybrid scores. The same custom-weight slider reranks comparisons without refetching market data; results can be exported as CSV. Market scores for each ticker show the adjusted-close date, and differing dates are flagged. Comparison ranks are descriptive, not predicted returns.

### Simultaneous sector rankings

In **Discover by sector**, choose **All sectors (66 stocks)** and click **Scan all sectors**. The page displays a leader overview plus rankings for all 11 curated sectors in two columns simultaneously. Scores use the same live adjusted-daily-close signal engine; the Momentum/Recovery slider and ranking-method selector rerank all 11 tables without new data requests. The full scan is on-demand rather than automatic because 66 individual vendor requests may be slow or rate-limited. Failed symbols and differing price dates are disclosed, and a CSV export includes the sector for every scored ticker. A single-sector scanning option remains.

### Curated sector discovery

Choose among **11 curated sectors** on the simplified homepage, then click **Scan selected sector** to score **six representative US-listed stocks** in that sector using the same adjusted daily close calculations as the main stock page. Rank by Momentum, Recovery, Standard Hybrid, or the current Custom Hybrid slider weighting. Results can be exported as CSV. The scanner shows data dates, records failed symbols, and explicitly warns when the stocks' latest adjusted-close dates differ.

**Coverage is not exhaustive**; sector assignments are manually curated and illustrative, rather than independently verified or point-in-time GICS membership. This is not a complete US-market screen, and the highest score is not a prediction or investment recommendation. Scores are cached by the app and vendor; rerun the scan to refresh the snapshot.

### Session watchlists

Visitors can save up to **12 tickers per Streamlit browser session** on the home page. Use **Save to watchlist** to add the selected ticker, **View** to switch the analysis ticker, **Remove** to delete it, and **Refresh watchlist scores** to fetch recent adjusted-close scores for saved tickers. The table ranks the saved symbols by the current custom Hybrid mix and supports CSV export. This is session memory only: no login, persistent storage, cross-device sync, or price alerts. Watchlist scores are refreshed only on request; moving the weight slider recomputes the blend without downloading quotes.

Use the sidebar **Advanced research** toggle for the original historical views, market data, watchlists and company research tools. Daily adjusted signals require 127 valid closes.

## Homepage navigation refresh

The default screen keeps the ticker search, Momentum and Recovery readings, fixed 60/40 Hybrid, and adjustable Custom Hybrid in view. Historical score trends and adjusted-close prices sit in a collapsed **Score history and price chart** expander. Below the main stock analysis, three tabs group secondary workflows: **Compare stocks**, **Discover by sector**, and **My watchlist**. These are presentation changes only: the standard score definitions, custom slider, manually curated sector sample, session-only watchlist and research limitations are unchanged. Advanced research remains accessible through the sidebar toggle.

## Features

- Strategy overview with historic Adaptive Alpha vs SPY results.
- Benchmark results from 1,000 randomized portfolios.
- Stock Research Agent with vendor market-data snapshots; optional conversational AI using a visitor-supplied OpenAI API key.
- Explicit methodology limitations and bias disclosures.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Visit https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → **Deploy from GitHub**.
3. Select **mayukh29-dev/adaptive-alpha**, branch **main**, main file **app.py**.
4. Deploy and use the actual URL assigned by Streamlit.

## Research and disclosure

The historical performance (2016-08-01 to 2026-11-13) is an **unaudited price-only proxy**, not a historical replay of production Adaptive Alpha V7 fundamentals. The 109-stock universe was selected retrospectively, introducing survivorship/selection bias. Some benchmark return outliers and vendor adjusted prices remain to be audited. No point-in-time fundamentals were used. Execution and transaction costs are approximate.

The Research Agent uses Yahoo Finance vendor snapshots. Its free mode answers only snapshot-supported factual questions. AI interpretation is optional and requires the viewer's own API key; never commit a key to this public repository. No financial advice.

## Development

The beta originated in a separate branch of the Insight Agent project and has been copied to this dedicated repository.

## Unified signals and predictive research (new development track)

The public dashboard now uses `signals.py` as the **single scoring source** for live Momentum, Recovery and Hybrid. Historical V8.2 results visible under Strategy Benchmarks are **legacy** and NOT comparable to `signals.py` until the unified backtest is run.

- `signals.py`: versioned, fully historical-data-only signal calculations.
- `prediction.py`: linear ridge model inference and research classification.
- `research/backtest_shared_v1.py`: matching unified v1.0 backtest.
- `research/train_predictor.py`: fixed chronological train/validation/heldout evaluation for 21-trading-day **excess return relative to SPY**.
- `tests/test_unified_signals.py`: unit tests for live/backtest consistency and look-ahead safety.

### Run training locally (your history files are not on GitHub)

In PowerShell, from `$HOME\Downloads\adaptive_alpha_v8_2`:

```powershell
$root = "https://raw.githubusercontent.com/mayukh29-dev/adaptive-alpha/main"
Invoke-WebRequest "$root/signals.py" -OutFile ".\signals.py"
Invoke-WebRequest "$root/prediction.py" -OutFile ".\prediction.py"
Invoke-WebRequest "$root/research/train_predictor.py" -OutFile ".\train_predictor.py"
Invoke-WebRequest "$root/research/backtest_shared_v1.py" -OutFile ".\backtest_shared_v1.py"

python .\train_predictor.py --exclude-tickers NWE CHRD NCMI
python .\backtest_shared_v1.py --exclude-tickers NWE CHRD NCMI
```

The backtest intentionally stops on unreviewed price moves. Do not suppress data audit errors just to obtain performance claims. The training outputs `predictive_output/evaluation.json` and `predictive_output/predictive_model.json`. The Streamlit app requires the latter at `models/predictive_model.json` to display predictions; without it, no Buy/Hold/Sell labels will be produced. Only publish after inspecting results, data quality and the holdout report.

The model trains on labels ending by 2020, validates on 2021–2022, and tests on 2023 onward, with 21-trading-session label maturity at split edges. It never uses future labels to form training features, but the retrospective universe bias remains. A passing descriptive research gate is not a significance test or a trading recommendation. **Do not describe forecasts as reliable until independent out-of-sample testing and price integrity checks succeed.**
