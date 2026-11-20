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
