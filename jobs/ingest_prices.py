import os
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v).strip()


def parse_symbols(s: str) -> list[str]:
    parts = [p.strip() for p in s.replace(" ", ",").split(",")]
    return [p for p in parts if p]


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


def _get_last_date(engine, symbol: str) -> date | None:
    q = text("SELECT MAX(date) AS max_date FROM market_ohlcv WHERE symbol = :symbol")
    with engine.begin() as conn:
        r = conn.execute(q, {"symbol": symbol}).mappings().one()
    return r["max_date"]


def compute_start_for_symbol(engine, symbol: str, fallback_start: str, lookback_days: int) -> str:
    """
    If data exists — start from (last_date - lookback_days) to safely re-upsert recent rows.
    If no data — use fallback_start.

    You can force a full backfill (ignore last_date) via:
      - INGEST_FULL_REFRESH=1  (for all symbols)
      - INGEST_FORCE_SYMBOLS=AAPL,TSLA,^GSPC (comma-separated)
    """
    # Force full backfill for all symbols
    full_refresh = (os.environ.get("INGEST_FULL_REFRESH", "0") or "0").strip() == "1"

    # Force full backfill for selected symbols
    force_symbols_raw = (os.environ.get("INGEST_FORCE_SYMBOLS", "") or "").strip()
    force_symbols = set(parse_symbols(force_symbols_raw)) if force_symbols_raw else set()

    if full_refresh or (symbol in force_symbols):
        return fallback_start

    last = _get_last_date(engine, symbol)
    if last is None:
        return fallback_start

    start_dt = last - timedelta(days=max(0, lookback_days))
    return start_dt.isoformat()


def download_one(symbol: str, start: str) -> pd.DataFrame:
    df = (
        yf.download(
            symbol,
            start=start,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        .reset_index()
    )

    if df.empty:
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
    return df


def main() -> None:
    db_url = get_env("DATABASE_URL")
    symbols = parse_symbols(get_env("SYMBOLS"))
    start_date = get_env("START_DATE")
    source = (os.environ.get("SOURCE", "yfinance") or "yfinance").strip() or "yfinance"

    lookback_days = int(os.environ.get("LOOKBACK_DAYS", "7"))
    if lookback_days < 0:
        raise ValueError("LOOKBACK_DAYS must be >= 0")
    full_refresh = (os.environ.get("INGEST_FULL_REFRESH", "0") or "0").strip() == "1"
    force_symbols_raw = (os.environ.get("INGEST_FORCE_SYMBOLS", "") or "").strip()

    engine = create_engine(db_url, pool_pre_ping=True)
    ensure_market_table(engine)

    upsert_sql = text(
        """
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
        """
    )

    print(f"[DB] Using DATABASE_URL={db_url}")
    print(
        f"[INFO] Symbols={symbols}, START_DATE={start_date}, LOOKBACK_DAYS={lookback_days}, SOURCE={source}, "
        f"INGEST_FULL_REFRESH={int(full_refresh)}, INGEST_FORCE_SYMBOLS='{force_symbols_raw}'"
    )

    total = 0
    for sym in symbols:
        sym_start = compute_start_for_symbol(engine, sym, start_date, lookback_days)
        df = download_one(sym, sym_start)

        if df.empty:
            print(f"[WARN] {sym}: no data for start={sym_start}, skipping")
            continue

        rows = df.to_dict(orient="records")
        for r in rows:
            r["source"] = source

        with engine.begin() as conn:
            conn.execute(upsert_sql, rows)

        total += len(df)
        print(f"[OK] {sym}: upserted {len(df)} rows ({df['date'].min()} -> {df['date'].max()}) start={sym_start}")

    print(f"[DONE] total upserted rows: {total}")


if __name__ == "__main__":
    main()