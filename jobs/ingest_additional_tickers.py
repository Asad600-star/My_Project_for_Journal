"""
Загрузка дополнительных тикеров для расширения универсума в Q1 статье.
=====================================================================

Добавляем 4 новых символа к существующим (AAPL, TSLA, ^GSPC, ^IXIC):

- **^DJI**  — Dow Jones Industrial Average (крупные cap, classic blue-chip index)
- **^RUT**  — Russell 2000 (small-cap, другой market segment)
- **GLD**   — SPDR Gold Trust (другой asset class — золото, защитный)
- **MSFT**  — Microsoft (ещё одна крупная tech акция)

Цель: показать в статье обобщение метода на разные классы активов и капитализации.

Запуск:
    python -m jobs.ingest_additional_tickers
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


# Дополнительные тикеры (расширение универсума для Q1)
ADDITIONAL_SYMBOLS = ["^DJI", "^RUT", "GLD", "MSFT"]
START_DATE = os.environ.get("START_DATE", "2015-01-01")
END_DATE = os.environ.get("END_DATE", "2026-06-01")


def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL env var не задан")
    return create_engine(db_url, pool_pre_ping=True)


def ensure_market_table(engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS market_ohlcv (
        symbol      text NOT NULL,
        date        date NOT NULL,
        open        double precision,
        high        double precision,
        low         double precision,
        close       double precision,
        adj_close   double precision,
        volume      bigint,
        source      text NOT NULL DEFAULT 'yfinance',
        ingested_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (symbol, date)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def download_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Загружает OHLCV для одного символа из yfinance."""
    print(f"[INGEST] {symbol}: загрузка с {start} по {end}...")
    df = (
        yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        .reset_index()
    )

    if df.empty:
        print(f"[WARN] {symbol}: пустые данные")
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [(c[0] if c[0] else c[1]) if isinstance(c, tuple) else c for c in df.columns]

    rename = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = symbol
    print(f"[INGEST] {symbol}: {len(df)} строк ({df['date'].min()} … {df['date'].max()})")
    return df


def upsert_rows(engine, rows: list[dict]) -> int:
    """UPSERT строк в market_ohlcv."""
    if not rows:
        return 0

    upsert_sql = text("""
        INSERT INTO market_ohlcv(symbol, date, open, high, low, close, adj_close, volume, source)
        VALUES (:symbol, :date, :open, :high, :low, :close, :adj_close, :volume, :source)
        ON CONFLICT (symbol, date) DO UPDATE SET
            open        = EXCLUDED.open,
            high        = EXCLUDED.high,
            low         = EXCLUDED.low,
            close       = EXCLUDED.close,
            adj_close   = EXCLUDED.adj_close,
            volume      = EXCLUDED.volume,
            source      = EXCLUDED.source,
            ingested_at = now();
    """)

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    return len(rows)


def main() -> None:
    print("=" * 70)
    print(" Загрузка дополнительных тикеров для Q1 статьи")
    print("=" * 70)
    print(f" Символы: {ADDITIONAL_SYMBOLS}")
    print(f" Период:  {START_DATE} … {END_DATE}")
    print("=" * 70)

    engine = get_engine()
    ensure_market_table(engine)

    total = 0
    for sym in ADDITIONAL_SYMBOLS:
        try:
            df = download_one(sym, START_DATE, END_DATE)
            if df.empty:
                continue
            rows = df.to_dict(orient="records")
            for r in rows:
                r["source"] = "yfinance"
            n = upsert_rows(engine, rows)
            total += n
            print(f"[OK] {sym}: upserted {n} строк")
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[DONE] Загружено всего: {total} строк")
    print("\n[NEXT] После загрузки нужно:")
    print("  1. Пересчитать features_daily через jobs/build_features.py")
    print("  2. Обновить SYMBOLS в train_lstm_baseline.py и train_arima_garch_baseline.py")
    print("  3. Запустить полное обучение на 8 символах")


if __name__ == "__main__":
    main()
