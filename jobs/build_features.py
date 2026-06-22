import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

WINDOWS = [5, 10, 20]
MAX_LAG = 5
ONLY_SYMBOL: str | None = (os.environ.get("ONLY_SYMBOL") or "").strip() or None


def get_env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v).strip()


def load_macro_frame(engine) -> pd.DataFrame:
    """Build macro/market context features by date (causal, no leakage)."""
    macro_symbols = ["^GSPC", "^VIX", "^IRX", "^TNX"]

    m = pd.read_sql_query(
        text(
            """
            SELECT date, symbol, close
            FROM market_ohlcv
            WHERE symbol = ANY(:symbols)
            ORDER BY date
            """
        ),
        con=engine,
        params={"symbols": macro_symbols},
        parse_dates=["date"],
    )

    if m.empty:
        raise RuntimeError(
            "Macro symbols not found in market_ohlcv. "
            "Run ingest_prices.py with ^GSPC,^VIX,^IRX,^TNX in SYMBOLS."
        )

    m["date"] = pd.to_datetime(m["date"]).dt.normalize()

    piv = (
        m.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        .sort_index()
        .ffill()
    )

    out = pd.DataFrame({"date": pd.to_datetime(piv.index).normalize()})

    # Market (^GSPC)
    if "^GSPC" in piv.columns:
        sp = piv["^GSPC"].astype(float)
        out["mkt_close"] = sp.to_numpy()
        out["mkt_return_1d"] = sp.pct_change().to_numpy()
        out["mkt_log_return"] = np.log(sp).diff().to_numpy()
        out["mkt_mom_5"] = sp.pct_change(5).to_numpy()
        out["mkt_mom_10"] = sp.pct_change(10).to_numpy()
        out["mkt_mom_20"] = sp.pct_change(20).to_numpy()
        out["mkt_vol_20"] = out["mkt_return_1d"].rolling(20, min_periods=20).std(ddof=0)

    # VIX (^VIX)
    if "^VIX" in piv.columns:
        vx = piv["^VIX"].astype(float)
        out["vix_level"] = vx.to_numpy()
        out["vix_return_1d"] = vx.pct_change().to_numpy()
        out["vix_change_1d"] = vx.diff().to_numpy()

    # Rates (^IRX, ^TNX)
    if "^IRX" in piv.columns:
        irx = piv["^IRX"].astype(float)
        out["irx_level"] = irx.to_numpy()
        out["irx_change_1d"] = irx.diff().to_numpy()

    if "^TNX" in piv.columns:
        tnx = piv["^TNX"].astype(float)
        out["tnx_level"] = tnx.to_numpy()
        out["tnx_change_1d"] = tnx.diff().to_numpy()

    out = out.reset_index(drop=True)
    return out


def build_features_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    df["return_1d"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()

    for w in WINDOWS:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"volatility_{w}"] = df["return_1d"].rolling(w).std()

    for lag in range(1, MAX_LAG + 1):
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)

    df["target_return_1d"] = df["return_1d"].shift(-1)
    df["symbol"] = symbol

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


def ensure_features_table(engine):
    """Создаём/обновляем таблицу features_daily + гарантируем наличие всех макро-колонок"""
    ddl = """
    CREATE TABLE IF NOT EXISTS features_daily (
        symbol text NOT NULL,
        date date NOT NULL,
        open double precision,
        high double precision,
        low double precision,
        close double precision,
        volume bigint,
        return_1d double precision,
        log_return double precision,
        sma_5 double precision,
        volatility_5 double precision,
        sma_10 double precision,
        volatility_10 double precision,
        sma_20 double precision,
        volatility_20 double precision,
        return_lag_1 double precision,
        return_lag_2 double precision,
        return_lag_3 double precision,
        return_lag_4 double precision,
        return_lag_5 double precision,
        PRIMARY KEY (symbol, date)
    );
    """

    with engine.begin() as conn:
        conn.execute(text(ddl))

        # Добавляем макро-колонки, если их ещё нет
        macro_cols = [
            ("mkt_return_1d", "double precision"),
            ("mkt_log_return", "double precision"),
            ("mkt_mom_5", "double precision"),
            ("mkt_mom_10", "double precision"),
            ("mkt_mom_20", "double precision"),
            ("mkt_vol_20", "double precision"),
            ("vix_level", "double precision"),
            ("vix_return_1d", "double precision"),
            ("vix_change_1d", "double precision"),
            ("irx_level", "double precision"),
            ("irx_change_1d", "double precision"),
            ("tnx_level", "double precision"),
            ("tnx_change_1d", "double precision"),
        ]

        for col, typ in macro_cols:
            conn.execute(text(f"ALTER TABLE features_daily ADD COLUMN IF NOT EXISTS {col} {typ}"))

        # Создаём индексы
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS features_daily_symbol_date_uq ON features_daily(symbol, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS features_daily_date_idx ON features_daily(date)"))

    print("[OK] Таблица features_daily проверена и обновлена")


def main() -> None:
    db_url = get_env("DATABASE_URL")
    engine = create_engine(db_url, pool_pre_ping=True)
    ensure_features_table(engine)

    if ONLY_SYMBOL:
        symbols = [ONLY_SYMBOL]
    else:
        symbols_df = pd.read_sql_query(
            """
            SELECT DISTINCT symbol
            FROM market_ohlcv
            WHERE symbol NOT IN ('^VIX','^IRX','^TNX')
            ORDER BY symbol
            """,
            con=engine,
        )
        symbols = symbols_df["symbol"].tolist()

    if not symbols:
        raise RuntimeError("No symbols found in market_ohlcv. Run ingest_prices.py first.")

    print(f"[INFO] Building features for symbols: {symbols}")

    macro = load_macro_frame(engine)
    print(f"[INFO] Loaded macro frame rows={len(macro)}")

    q = text(
        """
        SELECT date, open, high, low, close, volume
        FROM market_ohlcv
        WHERE symbol = :symbol
        ORDER BY date
        """
    )

    cols = [
        "symbol", "date", "open", "high", "low", "close", "volume",
        "return_1d", "log_return",
        "sma_5", "volatility_5",
        "sma_10", "volatility_10",
        "sma_20", "volatility_20",
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
        "target_return_1d",
        "mkt_close", "mkt_return_1d", "mkt_log_return", "mkt_mom_5", "mkt_mom_10", "mkt_mom_20", "mkt_vol_20",
        "vix_level", "vix_return_1d", "vix_change_1d",
        "irx_level", "irx_change_1d",
        "tnx_level", "tnx_change_1d",
    ]

    upsert_sql = text(
        """
        INSERT INTO features_daily(
            symbol, date, open, high, low, close, volume,
            return_1d, log_return,
            sma_5, volatility_5,
            sma_10, volatility_10,
            sma_20, volatility_20,
            return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
            target_return_1d
            , mkt_close, mkt_return_1d, mkt_log_return, mkt_mom_5, mkt_mom_10, mkt_mom_20, mkt_vol_20
            , vix_level, vix_return_1d, vix_change_1d
            , irx_level, irx_change_1d
            , tnx_level, tnx_change_1d
        )
        VALUES (
            :symbol, :date, :open, :high, :low, :close, :volume,
            :return_1d, :log_return,
            :sma_5, :volatility_5,
            :sma_10, :volatility_10,
            :sma_20, :volatility_20,
            :return_lag_1, :return_lag_2, :return_lag_3, :return_lag_4, :return_lag_5,
            :target_return_1d
            , :mkt_close, :mkt_return_1d, :mkt_log_return, :mkt_mom_5, :mkt_mom_10, :mkt_mom_20, :mkt_vol_20
            , :vix_level, :vix_return_1d, :vix_change_1d
            , :irx_level, :irx_change_1d
            , :tnx_level, :tnx_change_1d
        )
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            return_1d = EXCLUDED.return_1d,
            log_return = EXCLUDED.log_return,
            sma_5 = EXCLUDED.sma_5,
            volatility_5 = EXCLUDED.volatility_5,
            sma_10 = EXCLUDED.sma_10,
            volatility_10 = EXCLUDED.volatility_10,
            sma_20 = EXCLUDED.sma_20,
            volatility_20 = EXCLUDED.volatility_20,
            return_lag_1 = EXCLUDED.return_lag_1,
            return_lag_2 = EXCLUDED.return_lag_2,
            return_lag_3 = EXCLUDED.return_lag_3,
            return_lag_4 = EXCLUDED.return_lag_4,
            return_lag_5 = EXCLUDED.return_lag_5,
            target_return_1d = EXCLUDED.target_return_1d,
            mkt_close = EXCLUDED.mkt_close,
            mkt_return_1d = EXCLUDED.mkt_return_1d,
            mkt_log_return = EXCLUDED.mkt_log_return,
            mkt_mom_5 = EXCLUDED.mkt_mom_5,
            mkt_mom_10 = EXCLUDED.mkt_mom_10,
            mkt_mom_20 = EXCLUDED.mkt_mom_20,
            mkt_vol_20 = EXCLUDED.mkt_vol_20,
            vix_level = EXCLUDED.vix_level,
            vix_return_1d = EXCLUDED.vix_return_1d,
            vix_change_1d = EXCLUDED.vix_change_1d,
            irx_level = EXCLUDED.irx_level,
            irx_change_1d = EXCLUDED.irx_change_1d,
            tnx_level = EXCLUDED.tnx_level,
            tnx_change_1d = EXCLUDED.tnx_change_1d;
        """
    )

    total_rows = 0
    for sym in symbols:
        raw = pd.read_sql_query(q, con=engine, params={"symbol": sym}, parse_dates=["date"])
        # Need enough history for rolling windows + lags
        min_rows = max(WINDOWS) + MAX_LAG + 5
        if len(raw) < min_rows:
            print(f"[WARN] {sym}: too few rows in market_ohlcv ({len(raw)}<{min_rows}), skipping")
            continue
        if raw.empty:
            print(f"[WARN] No rows in market_ohlcv for symbol={sym}, skipping")
            continue

        feats = build_features_for_symbol(raw, sym)

        # join macro context by date (left join keeps asset dates)
        feats = feats.merge(macro, on="date", how="left")

        # Fill only the macro columns that are actually stored in features_daily.
        macro_fill_cols = [
            "mkt_close", "mkt_return_1d", "mkt_log_return", "mkt_mom_5", "mkt_mom_10", "mkt_mom_20", "mkt_vol_20",
            "vix_level", "vix_return_1d", "vix_change_1d",
            "irx_level", "irx_change_1d",
            "tnx_level", "tnx_change_1d",
        ]
        present_macro_fill_cols = [c for c in macro_fill_cols if c in feats.columns]
        if present_macro_fill_cols:
            feats = feats.sort_values("date")
            feats[present_macro_fill_cols] = (
                feats[present_macro_fill_cols]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .bfill()
            )

        if feats.empty:
            print(f"[WARN] Features empty after base feature engineering for symbol={sym}, skipping")
            continue

        # Keep macro columns in DB even if some rows still contain NaN.
        feats = feats.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

        # Keep required base columns; macro columns may remain NaN
        feats = feats[cols]

        # Sanity: ensure key columns are present
        if feats["symbol"].isna().any() or feats["date"].isna().any():
            print(f"[WARN] {sym}: produced NULLs in symbol/date, skipping")
            continue

        # Drop rows only if required causal core columns are broken.
        core_cols = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return_1d",
            "log_return",
        ]
        feats = feats.dropna(subset=core_cols).reset_index(drop=True)

        macro_check_cols = [
            "mkt_return_1d", "mkt_log_return", "mkt_mom_5", "mkt_vol_20",
            "vix_level", "vix_return_1d", "irx_level", "tnx_level",
        ]
        present_macro_check_cols = [c for c in macro_check_cols if c in feats.columns]
        if present_macro_check_cols:
            non_null_counts = feats[present_macro_check_cols].notna().sum().to_dict()
            print(f"[INFO] {sym}: macro non-null counts = {non_null_counts}")

        if feats.empty:
            print(f"[WARN] {sym}: empty after core-only dropna, skipping")
            continue



        rows = feats.to_dict(orient="records")

        if not rows:
            print(f"[WARN] {sym}: no rows to upsert, skipping")
            continue

        with engine.begin() as conn:
            conn.execute(upsert_sql, rows)

        total_rows += len(feats)
        print(f"[OK] {sym}: features rows={len(feats)}")

    print(f"[DONE] features_daily built. total_rows={total_rows}")


if __name__ == "__main__":
    main()