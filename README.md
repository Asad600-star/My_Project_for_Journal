# Risk-aware conformal control in an auditable production-rule expert system

Code and data for the paper *"Risk-aware conformal control in an auditable
production-rule expert system for stock direction and volatility forecasting"*.

The system forecasts the 5-day direction and realised volatility of eight
instruments (AAPL, TSLA, MSFT, GLD, S&P 500, NASDAQ Composite, Dow Jones,
Russell 2000) from a 56-dimensional causal feature vector, and turns the
forecasts into auditable decisions:

- **Direction:** HYBRID_VOTING, a soft-voting ensemble of five learners, with
  isotonic (PAV) calibration fitted on the most recent block the ensemble has
  not been trained on.
- **Volatility:** H-ECON-ET, an ExtraTrees correction of the equal-weight
  average of GARCH(1,1), GJR-GARCH(1,1)-t and a range-based HAR model, with the
  blend weight chosen on validation data.
- **Rules:** 66 production rules read off the paths of a CART tree (training block purged at its end).
- **Risk module:** Value-at-Risk with an asymmetric multiplier and a banded
  position size (10% / 5% / 0% of capital).
- **RACC:** Risk-Aware Conformal Control, a decision layer that abstains unless
  the expected volatility-scaled loss can be kept below a chosen level. The
  models are fitted on a training block, the calibrator and fusion weight on a
  separate tuning block and the threshold on a calibration block. The static
  form has a finite-sample bound under exchangeability (Proposition 1); the
  online form bounds the long-run risk for any data sequence (Proposition 2).
- **Interfaces:** a Streamlit dashboard and a Telegram bot, refreshed daily.

## Main results (22 expanding walk-forward folds per instrument)

| Result | Value | Artefact |
|---|---|---|
| Volatility: mean test RMSE | H-ECON-ET 0.00604; ECON 0.00606; GJR-GARCH 0.00612; GARCH 0.00625; ExtraTrees 0.00678 | `artifacts/metrics_walk_volatility_k5.csv` |
| Volatility: mean QLIKE loss | H-ECON-ET 0.431; ECON 0.457; GARCH 0.504 | `artifacts/volatility_models_summary_k5.csv` |
| 90% model confidence set (squared error / QLIKE) | H-ECON-ET 8/8 and 7/8 instruments, the only model with at least 7 under both | `artifacts/volatility_mcs_k5.csv` |
| Daily Diebold-Mariano, H-ECON-ET vs GARCH | better on 6/8 (squared error) and 7/8 (QLIKE) instruments, never worse | `artifacts/volatility_dm_daily_k5.csv` |
| Out-of-sample R² of H-ECON-ET vs historical mean | positive on 8/8 instruments (0.18 to 0.39) | `artifacts/metrics_walk_volatility_k5.csv` |
| Direction: mean test AUC of every model | 0.48 to 0.55 (no exploitable skill) | `artifacts/metrics_walk_direction_k5.csv` |
| Static RACC, 2024-2026 test period | bound holds at 62% of the risk levels; fails for α = 0.35-0.60 after a shift of the signal | `artifacts/racc_results.csv` |
| Static RACC, 2022 bear market | worst excess over the target 0.32 | `artifacts/racc_regime_stress.csv` |
| Online RACC, 2024-2026 / 2022 / 2020 | worst excess 0.009 / 0.014 / 0.051 (Proposition 2 allows 0.077 / 0.18 / 0.44) | `artifacts/racc_regime_stress.csv` |
| RACC back-test (λ = 0.5), maximum drawdown | 23.9% ungated, 18.2% static and 12.5% online (α = 0.30); online return 3.7% vs 17.0% ungated | `artifacts/racc_backtest_portfolio.csv` |
| Deflated Sharpe ratio of the best of 20 configurations | 0.205 (below the usual 0.95 level) | `artifacts/robustness_dsr.csv` |
| Decision pipeline vs buy-and-hold (2022-2026) | 0.22% per year vs 0.89% to 22.9% | `artifacts/backtest_summary.csv` |

Daily direction is close to unpredictable for every model tested, and no
active policy beats buy-and-hold. The contribution of the system is the
risk-controlled, auditable decision layer, not a trading edge.

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions used for the paper
cp .env.example .env                   # defaults work with the bundled database
docker-compose up -d                   # PostgreSQL
```

## Reproducing the paper

```bash
bash reproduce_results.sh
```

The script loads the frozen market-data snapshot (`data/market_ohlcv_snapshot.csv`),
rebuilds the features and regenerates every table and figure, with all settings
pinned (seed 42, data up to 2026-06-01, k = 5 with training data from 2015, k = 10
and 20 from 2017). `REFRESH_DATA=1 bash reproduce_results.sh` downloads the data
from Yahoo Finance instead. Figures are written to `figures/` with the numbers of
the paper (`fig01_architecture.png` ... `fig14_pr_rules.png`); the run takes about
10 hours on an Apple M4, mostly for the walk-forward models and the horizon sensitivity.

| Paper element | Script |
|---|---|
| Tables 5-6, walk-forward models | `jobs/train_baseline.py`, `jobs/train_deep_baselines.py` |
| GARCH and HAR baselines | `jobs/econometric_baselines.py` |
| GJR-GARCH, range-based HAR, H-ECON-ET, QLIKE, MCS, daily DM (Tables 6-7) | `jobs/volatility_models.py` |
| Figs. 3-4 (Friedman-Nemenyi), fold-level Diebold-Mariano | `jobs/statistical_tests.py`, `jobs/diebold_mariano_test.py` |
| Table 8 (rule stability), Table 3 and Appendix A (rules) | `jobs/rule_stability.py`, `jobs/extract_rules_from_tree.py` |
| Table 9 (horizons), Table 10 (ablation), Fig. 5 (SHAP) | `jobs/sensitivity_analysis.py`, `jobs/ablation_study.py`, `jobs/shap_walkforward.py` |
| Table 11 (back-test of the decision pipeline) | `jobs/backtest.py` |
| Tables 12-15, Figs. 7-9 (RACC) | `jobs/racc.py`, `jobs/racc_backtest.py`, `jobs/racc_regime_stress.py` |
| Tables 15-17, deflated Sharpe ratio | `jobs/robustness_checks.py` |
| Figs. 1, 2, 6 and 10-12 | `jobs/make_diagram_figures.py`, `jobs/make_paper_figures.py` |

## Tests

```bash
python -m pytest -q tests
```

The suite checks that every feature and the GARCH and HAR forecasts use only
past data, that the targets are purged inside each walk-forward block, that the
risk module implements Equations 25-28 and the banded sizing rule, the conformal
threshold of Proposition 1 on its fixed grid and its risk guarantee under
exchangeability, that the RACC score does not depend on the calibration labels, the
long-run bound of Proposition 2 on a sequence with abrupt regime switches, the
Diebold-Mariano test, the deflated Sharpe ratio and the SHAP attribution of the
voting ensemble.

## Daily operation

```bash
python -m jobs.daily_update          # prices -> features -> calibrated forecasts
streamlit run apps/web/main.py       # dashboard at http://localhost:8501
python -m jobs.bot                   # Telegram bot (set TELEGRAM_BOT_TOKEN in .env)
```

A step-by-step guide in Russian is in `GUIDE_RU.md`.

## Repository layout

```
jobs/        ingestion, features, models, baselines, RACC, evaluation, figures
core/        risk manager and SHAP explainer
services/    prediction service used by the user interfaces
apps/web/    Streamlit dashboard
data/        frozen market-data snapshot
artifacts/   metrics and result tables used in the paper
tests/       unit tests
```

`jobs/train_arima_garch_baseline.py` and `jobs/train_lstm_baseline.py` are kept
for the record only; they are superseded by `jobs/econometric_baselines.py` and
`jobs/train_deep_baselines.py` and are not used in the paper.

## License

MIT (see `LICENSE`).
