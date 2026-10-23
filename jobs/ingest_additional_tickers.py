"""
Ingestion of additional tickers to expand the instrument universe for the paper.
=====================================================================

Adds 4 new instruments to the existing ones (AAPL, TSLA, ^GSPC, ^IXIC):

- **^DJI**  : Dow Jones Industrial Average (large-cap, classic blue-chip index)
- **^RUT**  : Russell 2000 (small-cap, a different market segment)
- **GLD**   : SPDR Gold Trust (a different asset class: gold, defensive)
- **MSFT**  : Microsoft (another large-cap tech equity)

Goal: demonstrate generalisation across asset classes and market caps.

Run:
    python -m jobs.ingest_additional_tickers
"""
from __future__ import annotations

import os

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


# Additional tickers (universe expansion)
ADDITIONAL_SYMBOLS = ["^DJI", "^RUT", "GLD", "MSFT"]
START_DATE = os.environ.get("START_DATE", "2015-01-01")
END_DATE = os.environ.get("END_DATE", "2026-06-01")


def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL env var is not set")
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
    """Loads OHLCV for one instrument from yfinance."""
    print(f"[INGEST] {symbol}: downloading from {start} to {end}...")
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
        print(f"[WARN] {symbol}: empty data")
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
    print(f"[INGEST] {symbol}: {len(df)} rows ({df['date'].min()} .. {df['date'].max()})")
    return df


def upsert_rows(engine, rows: list[dict]) -> int:
    """UPSERTs rows into market_ohlcv."""
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
    print(" Ingesting additional tickers for the paper")
    print("=" * 70)
    print(f" Instruments: {ADDITIONAL_SYMBOLS}")
    print(f" Period:  {START_DATE} .. {END_DATE}")
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
            print(f"[OK] {sym}: upserted {n} rows")
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[DONE] Total ingested: {total} rows")
    print("\n[NEXT] After ingestion:")
    print("  1. Rebuild features_daily via jobs/build_features.py")
    print("  2. Update SYMBOLS in train_lstm_baseline.py and train_arima_garch_baseline.py")
    print("  3. Run full training on the 8 instruments")


if __name__ == "__main__":
    main()
