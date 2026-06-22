# Master of Degree: Прогнозирование направления и волатильности акций

**Тема дипломной работы:**  
Разработка программы для прогнозирования направления и волатильности цен акций и фондовых индексов с использованием гибридных моделей машинного обучения.

---

## 📋 О проекте

Это **полноценная production-система** машинного обучения, которая:
- Прогнозирует **направление** цены на 5 дней вперёд (рост/падение)
- Прогнозирует **волатильность** на 5 дней вперёд
- Выдаёт **рекомендацию** инвестору: «Покупать», «Не покупать», «Задуматься о покупке»
- Рассчитывает **риск-менеджмент** (VaR, размер позиции, уровень риска)
- Объясняет решение модели с помощью **SHAP**
- Имеет удобный **веб-сайт** и **Telegram-бот**

**Анализируемые инструменты:** AAPL, TSLA, ^GSPC (S&P 500), ^IXIC (NASDAQ)

---

## 🛠 Технологический стек

- **Язык:** Python 3.13
- **База данных:** PostgreSQL + Docker
- **Модели:** Гибридный ensemble (LightGBM + XGBoost + Logistic Regression + VotingClassifier)
- **Объяснимость:** SHAP
- **Веб-интерфейс:** Streamlit
- **Telegram-бот:** aiogram + APScheduler
- **Оркестрация:** Docker Compose

---

## 📁 Структура проекта

Master_of_Degree/
├── .env                          ← Основные настройки
├── docker-compose.yml
├── requirements.txt
├── README.md
├── jobs/                         ← Основные скрипты
│   ├── ingest_prices.py          ← Загрузка цен
│   ├── build_features.py         ← Создание признаков
│   ├── train_baseline.py         ← Обучение и inference
│   ├── daily_update.py           ← Ежедневное обновление (главная команда)
│   └── bot.py                    ← Telegram-бот
├── services/
│   └── predict.py                ← Основная функция предсказания
├── core/                         ← Ядро проекта (SHAP + Risk Manager)
├── apps/web/main.py              ← Веб-сайт на Streamlit
├── artifacts/                    ← Все результаты (модели, метрики, предсказания)
├── users.json                    ← Информация о пользователях
└── subscribers.json              ← Подписчики на уведомления


---

## 🚀 Установка и первый запуск

### 1. Запуск базы данных
```bash
docker-compose up -d

2. Активация виртуального окружения
source .venv/bin/activate

3. Установка зависимостей
pip install -r requirements.txt

4. Первый запуск проекта (выполнить один раз)
# 1. Загрузка исторических данных
python -m jobs.ingest_prices

# 2. Создание признаков
python -m jobs.build_features

# 3. Обучение и обновление предсказаний
env ACTION=infer TASK=direction python -m jobs.train_baseline
env ACTION=infer TASK=volatility python -m jobs.train_baseline

📅 Ежедневная работа с проектом
Самая важная команда (запускай каждый день после 19:00):
python -m jobs.daily_update

🔄 Переобучение модели
Переобучать модель рекомендуется 1 раз в 2–4 недели или при значительном изменении рынка.
# Полное переобучение (walk-forward)
env MODE=walk TASK=direction python -m jobs.train_baseline
env MODE=walk TASK=volatility python -m jobs.train_baseline

После переобучения обязательно обнови предсказания:
env ACTION=infer TASK=direction python -m jobs.train_baseline
env ACTION=infer TASK=volatility python -m jobs.train_baseline

🌐 Запуск веб-сайта (Streamlit)
streamlit run apps/web/main.py

🤖 Запуск Telegram-бота
cd /Users/asadbekikromov/Documents/GitHub/Master_of_Degree
source .venv/bin/activate
python jobs/bot.py

🔧 Полезные команды
Команда,                                                            Описание
python -m jobs.ingest_prices,                                       Загрузить/обновить цены
python -m jobs.build_features,                                      Пересчитать признаки
python -m jobs.daily_update,                                        Главное ежедневное обновление
env ACTION=infer TASK=direction python -m jobs.train_baseline,      Обновить предсказания направления
env ACTION=infer TASK=volatility python -m jobs.train_baseline,     Обновить предсказания волатильности

📊 Как интерпретировать результаты
p_up — вероятность роста за 5 дней
vol_pred — ожидаемая волатильность
recommendation_ru — рекомендация модели
confidence — уровень уверенности
risk_summary_ru — рекомендации по риску

Хороший сигнал на покупку:
p_up > 0.60 + риск: низкий + position_size: 8-12%

⚙️ Настройка параметров
Все основные настройки находятся в файле .env:

TRAIN_START_DATE=2022-01-01 — с какой даты обучать модель
HORIZON_DAYS=5 — горизонт прогнозирования
Другие параметры можно менять по необходимости

Автор: Асадбек Икромов
Год: 2026


---

## 🔬 Воспроизведение результатов статьи (Q1 paper)

Эта система лежит в основе научной статьи "Mathematical Representation of
Decision-Making for Hybrid Forecasting of Stock Price Direction and Volatility
Based on Production Logic".

### Полное воспроизведение одной командой

```bash
bash reproduce_results.sh
```

Скрипт воспроизводит **все** числа из статьи end-to-end: от загрузки сырых
данных через yfinance до финальных таблиц. Зафиксировано для воспроизводимости:
`SEED=42`, `END_DATE=2026-06-01`, `TRAIN_START_DATE=2015-01-01`.

### Экспериментальный протокол

- **8 инструментов:** AAPL, TSLA, MSFT, GLD, S&P 500, NASDAQ, Dow Jones, Russell 2000
- **3 макро-индикатора (признаки):** ^VIX, ^IRX, ^TNX
- **Walk-forward CV:** ~22 фолда (min_train=1200, val=126, test=126, step=63)
- **Baselines:** LSTM (deep learning), ARIMA + GARCH (эконометрика), vanilla XGB/LGBM
- **Статистика:** Diebold-Mariano test (HLN-корректировка) + bootstrap 95% CI (B=1000)

### Ключевые скрипты экспериментов

| Скрипт | Назначение |
|--------|-----------|
| `jobs/train_baseline.py` | HYBRID_VOTING, HYBRID_STACK, ExtraTrees, vanilla baselines |
| `jobs/train_lstm_baseline.py` | LSTM baseline (PyTorch, MPS/CUDA/CPU) |
| `jobs/train_arima_garch_baseline.py` | ARIMA + GARCH эконометрические baselines |
| `jobs/diebold_mariano_test.py` | Тест статистической значимости (HLN) |
| `jobs/ablation_study.py` | Вклад групп признаков |
| `jobs/generate_paper_tables.py` | Генерация всех таблиц статьи |

### Основные результаты

- **Volatility:** ExtraTrees / HYBRID_STACK_REG превосходят ARIMA на **40%** и
  GARCH на **33%** по среднему RMSE (значимо по Diebold-Mariano на большинстве
  инструментов).
- **Direction:** прогноз направления на дневных данных близок к случайному
  (mean AUC 0.48–0.55 у всех моделей, включая LSTM), что согласуется с
  гипотезой эффективного рынка. Главный вклад работы — интерпретируемая,
  воспроизводимая система с 150 продукционными правилами и риск-модулем.

---

# 🧑‍🔬 Quick Start for Reviewers

This section gives a step-by-step guide to run the system and verify the
claims made in the paper. Two independent paths are provided: a **live demo**
(web app + Telegram bot) and a **full reproduction** of every reported number.

## Prerequisites

- **Python 3.13**
- **Docker** (for the bundled PostgreSQL database)
- macOS / Linux (the LSTM baseline auto-detects Apple MPS, NVIDIA CUDA, or CPU)

## Installation

```bash
# 1. Clone the repository
git clone <REPO_URL>
cd hybrid-stock-forecasting

# 2. Create environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# (the defaults work out of the box; edit only if you change DB credentials
#  or want to run the Telegram bot — then add your own BotFather token)

# 4. Start the PostgreSQL database
docker-compose up -d
```

## Option A — Reproduce the paper results (recommended for verification)

A single script regenerates every number, table, and metric reported in the
paper, end to end, from raw market data:

```bash
bash reproduce_results.sh
```

This will:
1. Ingest OHLCV data from Yahoo Finance for 8 instruments + 3 macro indicators.
2. Build the 56-dimensional causal feature matrix.
3. Train the hybrid models, ExtraTrees, and all baselines (LSTM, ARIMA, GARCH,
   vanilla XGB/LGBM) under walk-forward cross-validation.
4. Run the sensitivity analysis (K = 5, 10, 20), the ablation study, and the
   Diebold-Mariano significance test.
5. Write all comparison tables to `artifacts/paper_tables.txt`.

**Reproducibility is pinned** (`SEED=42`, `END_DATE=2026-06-01`,
`TRAIN_START_DATE=2015-01-01`), so the output matches the paper. A static data
snapshot is included in `data/market_ohlcv_snapshot.csv` in case the live API
is unavailable.

Expected total runtime: ~3-5 hours on a modern laptop (the walk-forward
training of hybrid ensembles across 8 instruments and 3 horizons is the
dominant cost).

### How to verify the key claims

| Claim in paper | Where to check |
|----------------|----------------|
| ExtraTrees beats ARIMA/GARCH by ~40%/33% in mean RMSE | `artifacts/paper_tables.txt`, Table B2 |
| Difference is statistically significant | `artifacts/dm_test_results.csv` (p-values) |
| Direction AUC is near the no-skill boundary for all models | `artifacts/paper_tables.txt`, Table B1 |
| Feature groups each contribute (ablation) | `artifacts/ablation_pivot.csv` |
| Robustness across horizons | `artifacts/metrics_walk_volatility_k{5,10,20}.csv` |

## Option B — Live demo (interactive)

To see the system in action on current market data:

```bash
source .venv/bin/activate

# Web dashboard (opens in browser at http://localhost:8501)
streamlit run apps/web/main.py
```

In the dashboard you can select any of the 8 instruments and any of the 3
forecast horizons (5, 10, 20 days). For each request the system returns the
calibrated up-probability, the forecast volatility, a Value-at-Risk based
recommendation, and the SHAP attribution explaining the decision.

> Note: the first request for a given (symbol, horizon) trains the selected
> model on the fly (1-2 minutes); subsequent requests are instant.

Optionally, the Telegram bot:

```bash
# add your BotFather token to .env first (TELEGRAM_BOT_TOKEN)
python -m jobs.bot
```

## Repository layout

```
jobs/        data ingestion, feature building, training, baselines, evaluation
core/        risk manager, SHAP explainer
services/    high-level prediction service used by the UI
apps/web/    Streamlit dashboard
data/        static market-data snapshot for offline reproduction
artifacts/   metrics, SHAP values, comparison tables (results)
reproduce_results.sh   single-command full reproduction
```

## Notes on honest reporting

The paper reports results as they are. Directional prediction on daily data is
close to the no-skill boundary for **every** model tested, including the
deep-learning LSTM baseline; this is consistent with weak-form market
efficiency and is discussed in the Limitations section. The system's
quantitative strength is in volatility forecasting and in providing an
auditable, rule-based, fully reproducible decision pipeline.

## Economic back-test (honest reporting)

An event-driven back-test (`jobs/backtest.py`, ~3.2 years, 5 bps per trade) shows
that the rule-gated trading strategy **does not beat passive buy-and-hold**: its
annualised return is near zero on every instrument while buy-and-hold returned
4-47% per year over the same bull-market window. This is the expected outcome
given that daily directional accuracy is near the no-skill boundary, and it is a
direct empirical confirmation of weak-form market efficiency. We report it
openly: the value of the system is in calibrated volatility forecasting,
interpretability, and reproducibility, not in a trading edge.

```bash
python -m jobs.backtest   # regenerates the back-test table
```
