#!/usr/bin/env bash
#
# reproduce_results.sh
# =====================================================================
# Reproduces every table and figure of the paper, end to end, from the
# frozen market-data snapshot in data/ to the final artefacts.
#
# Usage:
#     bash reproduce_results.sh              # use the frozen snapshot (exact)
#     REFRESH_DATA=1 bash reproduce_results.sh   # download from Yahoo Finance
#
# Requirements:
#     - Docker (PostgreSQL, see docker-compose.yml)
#     - Python 3.13 with the packages in requirements.txt
#     - DATABASE_URL in .env
#
# Reproducibility pins:
#     - SEED = 42, END_DATE = 2026-06-01
#     - k = 5: TRAIN_START_DATE = 2015-01-01 (22 walk-forward folds)
#     - k = 10, 20: TRAIN_START_DATE = 2017-01-01 (14 walk-forward folds)
#     - walk-forward: min_train = 1200, val = 126, test = 126, step = 63
#
# Running time: about 10 hours on an Apple M4 (the walk-forward models, the
# deep baselines and the horizon sensitivity take most of it).
# =====================================================================

set -euo pipefail
cd "$(dirname "$0")"

export SEED=42
export END_DATE=2026-06-01
export TRAIN_START_DATE=2015-01-01
export HORIZON_DAYS=5
# economic back-test of Table 11 (set here so that a local .env cannot change it)
export BACKTEST_START_DATE=2022-01-01
export BACKTEST_FEE_BPS=5
export BACKTEST_PROB_THRESHOLD=0.55
export BACKTEST_INITIAL_CAPITAL=100000
export BACKTEST_MODEL=HYBRID_VOTING
export FIG_DIR="${FIG_DIR:-$(pwd)/figures}"
export FIG_DPI="${FIG_DPI:-300}"
mkdir -p "$FIG_DIR"

PY="${PYTHON:-python}"
step() { echo; echo "=== [$1] $2"; }

step 0 "PostgreSQL"
docker-compose up -d
sleep 5

step 1 "Market data (8 instruments + ^VIX, ^IRX, ^TNX)"
if [[ "${REFRESH_DATA:-0}" == "1" ]]; then
    SYMBOLS="AAPL,TSLA,MSFT,GLD,^GSPC,^IXIC,^DJI,^RUT,^VIX,^IRX,^TNX" START_DATE=2015-01-01 \
        INGEST_FULL_REFRESH=1 $PY -m jobs.ingest_prices
else
    $PY -m jobs.load_snapshot
fi

step 2 "Features (56 causal columns)"
$PY -m jobs.build_features

step 3 "Walk-forward models, k = 5 (Tables 5, 6; HYBRID_VOTING, HYBRID_STACK, tree regressors)"
TASK=direction  $PY -m jobs.train_baseline
TASK=volatility $PY -m jobs.train_baseline

step 4 "Deep baselines (LSTM, GRU, Transformer; N-BEATSx for volatility)"
# single-threaded CPU training makes the deep baselines bit-for-bit reproducible
DEEP_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m jobs.train_deep_baselines
DEEP_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TASK=volatility MODELS=NBEATSX $PY -m jobs.train_deep_baselines

step 5 "Volatility: econometric baselines, hybrids, QLIKE, model confidence set (Tables 6-7)"
$PY -m jobs.econometric_baselines
$PY -m jobs.volatility_hybrids          # exploratory variants of the GARCH-based hybrid
$PY -m jobs.volatility_models           # GJR-GARCH, HAR-RG, ECON and H-ECON-ET (the system model)

step 6 "Significance tests and result figures (Figs. 3-4, 10-12)"
$PY -m jobs.statistical_tests
$PY -m jobs.diebold_mariano_test
$PY -m jobs.make_paper_figures

step 7 "Horizon sensitivity, k = 10 and 20 (Table 9)"
TRAIN_START_DATE=2017-01-01 $PY -m jobs.sensitivity_analysis

step 8 "Feature-group ablation (Table 10) and SHAP attributions (Fig. 5)"
$PY -m jobs.ablation_study
$PY -m jobs.shap_walkforward

step 9 "Production rules (Table 3, Appendix A, Figs. 13-14) and their stability (Table 8)"
$PY -m jobs.extract_rules_from_tree
$PY -m jobs.rule_stability

step 10 "Economic back-test of the decision pipeline (Table 11)"
$PY -m jobs.backtest

step 11 "RACC: risk control, back-test, regime stress, robustness (Tables 12-17, Figs. 7-9)"
$PY -m jobs.racc
$PY -m jobs.racc_backtest
RACC_LAMBDA=1.0 $PY -m jobs.racc_backtest
$PY -m jobs.racc_regime_stress
for g in 0.01 0.02 0.1 0.2; do RACC_GAMMA=$g $PY -m jobs.racc_regime_stress; done
RACC_SIGMA=garch $PY -m jobs.racc_regime_stress
RACC_WINDOW_DAYS=90 $PY -m jobs.racc_regime_stress
$PY -m jobs.robustness_checks

step 12 "Diagram figures (Figs. 1 and 6)"
$PY -m jobs.make_diagram_figures

step 13 "Unit tests (causality, purging, risk module, conformal guarantees)"
$PY -m pytest -q tests

echo
echo "Done. Tables: artifacts/*.csv   Figures: $FIG_DIR"
