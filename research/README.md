# Final Adaptive Alpha Price-Only Backtest

This folder contains `final_backtest.py`, a reproducible **three-strategy historical comparison** using the existing V8.2 local data and indicator engine.

## Important scope

This is **not** a point-in-time replay of the full production V7 fundamentals model. It compares historical **price-only Momentum, Recovery, and Hybrid proxies** and uses a retrospective 109-company stock universe. Selection/survivorship bias remains.

## Install locally on the machine holding V8.2 data

From PowerShell in `$HOME\Downloads\adaptive_alpha_v8_2`:

```powershell
Invoke-WebRequest "https://raw.githubusercontent.com/mayukh29-dev/adaptive-alpha/main/research/final_backtest.py" -OutFile ".\final_backtest.py"
python .\final_backtest.py --exclude-tickers NWE
```

The script requires local `backtest_v82.py`, `data/master_110.csv`, and historical `history/*.csv` (including SPY).

### If the price audit blocks execution

Inspect `final_backtest_output/price_outlier_audit.csv` to determine whether flagged returns are legitimate corporate actions or provider errors. To produce a **provisional** comparison without modifying prices:

```powershell
python .\final_backtest.py --exclude-tickers NWE --allow-unreviewed-outliers
```

This override **does not validate or repair** the data; it only allows explicitly provisional results. Never label them final and audited until reviewed.

## Output

- `final_backtest_output/comparison.json` with each strategy's return, CAGR, volatility, Sharpe and drawdown, plus SPY.
- `final_backtest_output/{momentum,recovery,hybrid}_portfolio_curve.csv`
- `final_backtest_output/{momentum,recovery,hybrid}_rebalance_audit.csv`
- `final_backtest_output/price_outlier_audit.csv` when price anomalies are found.

The script checks that Hybrid agrees with the prior V8.2 backtest summary when the reference universe and exclusions match. A mismatch stops execution.

## Suggested final validation

Review every flagged price observation using a second vendor or exchange/issuer adjustment record, verify benchmark accounting, report trade turnover, and then assess the dependence on the retrospectively selected stock universe. Actual historical fundamental strategy evaluation requires archived point-in-time financial inputs.
