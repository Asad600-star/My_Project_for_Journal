"""
Loads the frozen market-data snapshot into PostgreSQL.
======================================================

Yahoo Finance revises and occasionally removes historical rows, so a fresh
download may not reproduce the paper exactly. data/market_ohlcv_snapshot.csv
holds the daily OHLCV rows used in the paper (8 instruments and the ^VIX, ^IRX
and ^TNX context series). This script writes them to the market_ohlcv table,
after which jobs.build_features rebuilds the 56 features from the same data.

It is meant for a fresh database: on an existing one it overwrites the rows of
the snapshot period with the snapshot values.

Run:
    python -m jobs.load_snapshot
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from jobs.ingest_prices import ensure_market_table

load_dotenv()
SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "market_ohlcv_snapshot.csv"

UPSERT = text("""
    INSERT INTO market_ohlcv(symbol, date, open, high, low, close, adj_close, volume, source)
    VALUES (:symbol, :date, :open, :high, :low, :close, :adj_close, :volume, 'snapshot')
    ON CONFLICT (symbol, date) DO UPDATE SET
        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
        close = EXCLUDED.close, adj_close = EXCLUDED.adj_close,
        volume = EXCLUDED.volume, source = EXCLUDED.source, ingested_at = now();
""")


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    ensure_market_table(engine)
    df = pd.read_csv(SNAPSHOT, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df["volume"] = df["volume"].fillna(0).astype("int64")
    rows = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(UPSERT, rows)
    for sym, g in df.groupby("symbol"):
        print(f"[OK] {sym}: {len(g)} rows ({g['date'].min()} -> {g['date'].max()})")
    print(f"[DONE] {len(df)} rows loaded from {SNAPSHOT.name}")


if __name__ == "__main__":
    main()
