# Adaptive Alpha — AI Stock Research Beta

A Streamlit research dashboard demonstrating a **price-only historical proxy** of the Adaptive Alpha investing research project, plus a question-answering Stock Research Agent.

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
