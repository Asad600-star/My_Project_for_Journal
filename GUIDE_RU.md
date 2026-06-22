# 📘 Личное руководство (для автора)

Это твоя шпаргалка: что куда нажимать, что запускать, как всё проверить.
Файл **только для тебя** — в публичный репозиторий для рецензента идёт `README.md` (на английском).

---

## ⚡ САМОЕ ГЛАВНОЕ ПРАВИЛО

**Всегда работай из основной папки проекта**, не из `.claude/worktrees/...`:

```bash
cd ~/Documents/GitHub/Master_of_Degree
```

Если запустить Docker из другой папки — создастся пустая база (мы это уже проходили).

---

## 1️⃣ Первый запуск (один раз)

```bash
# Зайти в папку проекта
cd ~/Documents/GitHub/Master_of_Degree

# Создать виртуальное окружение и поставить зависимости
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Создать .env из шаблона (если ещё нет)
cp .env.example .env
# затем открой .env и впиши свой TELEGRAM_BOT_TOKEN (от @BotFather)
```

---

## 2️⃣ Каждый раз перед работой

```bash
cd ~/Documents/GitHub/Master_of_Degree
source .venv/bin/activate          # включить окружение
docker-compose up -d               # запустить базу PostgreSQL
```

Проверить что база жива и с данными:

```bash
docker exec stock_db psql -U stock -d stockdb -c "SELECT symbol, COUNT(*) FROM market_ohlcv GROUP BY symbol ORDER BY symbol;"
```

Должно показать 11 символов (8 инструментов + 3 макро).

---

## 3️⃣ Запустить САЙТ (Streamlit)

```bash
streamlit run apps/web/main.py
```

Откроется браузер на `http://localhost:8501`. Что там:
- **Слева вверху** — выбор языка (Русский / English)
- **Выбери инструмент** — 8 вариантов (AAPL, TSLA, MSFT, GLD, S&P 500, NASDAQ, Dow Jones, Russell 2000)
- **Выбери горизонт** — 5, 10 или 20 дней
- **Кнопка "🔄 Обновить"** — подтянуть свежие данные и пересчитать
- Внизу — график цены с прогнозным коридором + SHAP (почему модель так решила)

> ⚠️ Первый запрос на новый горизонт обучает модель на лету (1-2 минуты). Дальше мгновенно.

Остановить сайт: `Ctrl+C` в терминале.

---

## 4️⃣ Запустить ТЕЛЕГРАМ-БОТА

```bash
python -m jobs.bot
```

В Telegram у бота:
- Кнопки символов (8 штук)
- Кнопки горизонта (📅 5 / 10 / 20 дней) — сначала жми горизонт, он запомнится
- Кнопка "Все прогнозы" — все 8 инструментов сразу

Остановить бота: `Ctrl+C`.

---

## 5️⃣ Воспроизвести результаты статьи (полностью с нуля)

```bash
bash reproduce_results.sh
```

Это пройдёт все 8 шагов: загрузка данных → признаки → обучение → baselines → sensitivity → ablation → Diebold-Mariano → таблицы. Займёт ~3-5 часов. Все числа сохранятся в `artifacts/`.

Зафиксировано для воспроизводимости: `SEED=42`, `END_DATE=2026-06-01`, `START=2015-01-01`.

---

## 6️⃣ Тест прибыльности (backtest)

```bash
python -m jobs.backtest
```

Покажет по портфелю и каждому символу: CAGR (% в год), Sharpe, Max Drawdown, Win-rate, сравнение с Buy&Hold. Результаты в `artifacts/backtest_summary.csv`.

> Честный вывод: стратегия не бьёт Buy&Hold (CAGR ~0%) — это нормально, прогноз направления близок к случайному, и мы это честно описали в статье.

---

## 7️⃣ Сгенерировать таблицы для статьи

```bash
python -m jobs.generate_paper_tables
```

Все таблицы (direction, volatility, DM-тест, sensitivity, ablation) в `artifacts/paper_tables.txt`.

---

## 🧩 Отдельные операции (если нужно)

```bash
# Только загрузить свежие данные
python -m jobs.ingest_prices
python -m jobs.ingest_additional_tickers

# Пересчитать признаки
python -m jobs.build_features

# Обучить direction-модели (горизонт 5)
TASK=direction HORIZON_DAYS=5 python -m jobs.train_baseline

# Обучить volatility-модели
TASK=volatility HORIZON_DAYS=5 python -m jobs.train_baseline

# LSTM baseline (для статьи)
python -m jobs.train_lstm_baseline

# ARIMA + GARCH baselines (для статьи)
python -m jobs.train_arima_garch_baseline

# Diebold-Mariano тест значимости
python -m jobs.diebold_mariano_test
```

---

## 🆘 Частые проблемы

**Docker: "container name already in use"**
```bash
docker rm -f stock_db && docker-compose up -d
```

**База пустая после запуска**
Ты запустил Docker НЕ из основной папки. Останови и запусти правильно:
```bash
docker rm -f stock_db
cd ~/Documents/GitHub/Master_of_Degree && docker-compose up -d
```

**"role postgres does not exist"**
Правильный пользователь — `stock`, не `postgres`:
```bash
docker exec stock_db psql -U stock -d stockdb -c "..."
```

**Прервался процесс (Ctrl+C случайно)**
Запускай долгие задачи в фоне:
```bash
nohup bash -c 'команда' > лог.log 2>&1 &
tail -f лог.log    # смотреть прогресс
```

**SHAP не показывается для символа**
Прогрей его:
```bash
ACTION=infer HORIZON_DAYS=5 ONLY_SYMBOL="СИМВОЛ" python -m jobs.train_baseline
```

---

## 📂 Где что лежит

| Папка/файл | Что внутри |
|---|---|
| `jobs/` | загрузка данных, обучение, baselines, бот, backtest |
| `apps/web/main.py` | сайт (Streamlit) |
| `services/predict.py` | логика прогноза для сайта/бота |
| `core/` | риск-менеджер, SHAP |
| `artifacts/` | результаты: метрики, таблицы, SHAP |
| `data/` | снимок рыночных данных |
| `reproduce_results.sh` | воспроизведение всей статьи одной командой |
| `README.md` | руководство для рецензента (английский) |
| `GUIDE_RU.md` | этот файл (для тебя) |
