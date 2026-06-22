#!/usr/bin/env bash
#
# reproduce_results.sh
# =====================================================================
# Воспроизводит ВСЕ результаты статьи end-to-end, от сырых данных до
# финальных таблиц. Один скрипт — полная воспроизводимость для рецензентов.
#
# Reproduces every number reported in the paper, end to end, from raw
# market data to the final tables. Single-script reproducibility.
#
# Использование / Usage:
#     bash reproduce_results.sh
#
# Требования / Requirements:
#     - Docker (PostgreSQL)
#     - Python 3.13 + зависимости из requirements.txt
#     - Переменная DATABASE_URL в .env
#
# Фиксация для воспроизводимости / Reproducibility pins:
#     - SEED = 42
#     - END_DATE = 2026-06-01
#     - TRAIN_START_DATE = 2015-01-01
#     - Walk-forward: min_train=1200, val=126, test=126, step=63 (~22 фолда)
# =====================================================================

set -euo pipefail

# ─── Конфигурация (зафиксирована для воспроизводимости) ───
export SEED=42
export END_DATE=2026-06-01
export TRAIN_START_DATE=2015-01-01
export HORIZON_DAYS=5
export SYMBOLS="AAPL,TSLA,MSFT,GLD,^GSPC,^IXIC,^DJI,^RUT"

echo "======================================================================"
echo " ВОСПРОИЗВЕДЕНИЕ РЕЗУЛЬТАТОВ СТАТЬИ / REPRODUCING PAPER RESULTS"
echo "======================================================================"
echo " SEED=$SEED  END_DATE=$END_DATE  START=$TRAIN_START_DATE"
echo " Инструменты / Instruments: $SYMBOLS (+ макро ^VIX,^IRX,^TNX)"
echo "======================================================================"

# ─── Шаг 0: PostgreSQL через Docker ───
echo ""
echo "[0/8] Запуск PostgreSQL (Docker)..."
docker-compose up -d
sleep 5

# ─── Шаг 1: Загрузка рыночных данных (8 активов + 3 макро) ───
echo ""
echo "[1/8] Загрузка OHLCV из yfinance..."
SYMBOLS="AAPL TSLA MSFT ^GSPC ^IXIC ^VIX ^IRX ^TNX" \
    START_DATE=2015-01-01 python -m jobs.ingest_prices
python -m jobs.ingest_additional_tickers   # ^DJI, ^RUT, GLD, MSFT

# ─── Шаг 2: Построение признаков (56-мерный причинный вектор) ───
echo ""
echo "[2/8] Построение features_daily (56 признаков)..."
python -m jobs.build_features

# ─── Шаг 3: Обучение HYBRID + базовых моделей (direction) ───
echo ""
echo "[3/8] Обучение direction-моделей (HYBRID_VOTING, HYBRID_STACK, baselines)..."
TASK=direction python -m jobs.train_baseline

# ─── Шаг 4: Обучение volatility-моделей ───
echo ""
echo "[4/8] Обучение volatility-моделей (ExtraTrees, HYBRID_STACK_REG, baselines)..."
TASK=volatility python -m jobs.train_baseline

# ─── Шаг 5: Baselines (LSTM, ARIMA, GARCH) ───
echo ""
echo "[5/8] LSTM baseline (direction)..."
python -m jobs.train_lstm_baseline
echo "      ARIMA + GARCH baselines (volatility)..."
python -m jobs.train_arima_garch_baseline

# ─── Шаг 6: Sensitivity (K=10, K=20) ───
echo ""
echo "[6/8] Sensitivity analysis (горизонты K=10, K=20)..."
for K in 10 20; do
    HORIZON_DAYS=$K TASK=direction python -m jobs.train_baseline
    HORIZON_DAYS=$K TASK=volatility python -m jobs.train_baseline
done

# ─── Шаг 7: Ablation study ───
echo ""
echo "[7/8] Ablation study (вклад групп признаков)..."
python -m jobs.ablation_study

# ─── Шаг 8: Diebold-Mariano + финальные таблицы ───
echo ""
echo "[8/8] Diebold-Mariano test + генерация таблиц..."
python -m jobs.diebold_mariano_test
python -m jobs.generate_paper_tables

echo ""
echo "======================================================================"
echo " ГОТОВО / DONE"
echo " Все таблицы в artifacts/paper_tables.txt"
echo " Метрики в artifacts/metrics_walk_*.csv"
echo " DM-тест в artifacts/dm_test_results.csv"
echo "======================================================================"
