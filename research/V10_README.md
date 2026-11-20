# Adaptive Alpha V10 — walk-forward ranking experiment

**Status: research prototype; not live trade advice or a validated predictive model.**

This experiment extends the shared price-signals-1.0 scoring engine with additional price-only predictors. It deliberately does not claim historical fundamental or volume features are available.

### What is evaluated

- Learned monthly refit ridge models: `momentum`, `recovery`, `hybrid`.
- Fixed-rule sector-relative comparison: `momentum_baseline`, `recovery_baseline`, `hybrid_baseline`.
- 21-session forward stock excess return over SPY as a *training label*, with only labels matured on/before each decision date allowed for training.
- Investable-like 22-position (top 2 in each of 11 sectors) monthly close-to-close portfolios, with assumed 10 bps per unit gross turnover.
- Rank IC, top-minus-bottom future-return spread, turnover and period-specific portfolio CAGR/volatility/drawdown.
- Reporting splits: validation 2021–2022, test 2023 onwards. **The test period is not genuinely untouched:** prior modeling work has already consulted 2023–2026 results.

### Run from local Windows V8.2 folder

```powershell
cd "$HOME\Downloads\adaptive_alpha_v8_2"
$root="https://raw.githubusercontent.com/mayukh29-dev/adaptive-alpha/main"
Invoke-WebRequest "$root/signals.py" -OutFile ".\signals.py"
Invoke-WebRequest "$root/research/v10_walkforward.py" -OutFile ".\v10_walkforward.py"
python .\v10_walkforward.py --exclude-tickers NWE CHRD NCMI
```

If the audit blocks on the three previously flagged, news-linked large price jumps, first inspect `v10_walkforward_output/price_audit.json`. To run a **provisional sensitivity experiment**, without changing any prices:

```powershell
python .\v10_walkforward.py --exclude-tickers NWE CHRD NCMI --allow-unreviewed-outliers
Get-Content .\v10_walkforward_output\evaluation.json -Raw
```

The override does not certify price continuity. It must not be used to call performance final.

### Important known limitations

The project still uses a *retrospectively selected universe* and retrospective sectors. This risks material survivorship/selection bias and means the strategy results are not true point-in-time investable results. Corporate actions, delistings, adjusted-price continuity, and trading execution need further validation. Months with holdings missing future prices are blocked rather than silently removed using future information. All current results are research diagnostics, not a recommendation to buy, hold, or sell any security.

No forecast artifact is trained or published by this script. The Streamlit dashboard retains its existing holdout research gates and does not auto-activate Buy/Hold/Sell labels from this experiment.
