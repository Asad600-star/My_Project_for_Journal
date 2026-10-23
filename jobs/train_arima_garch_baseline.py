"""
DEPRECATED - superseded by jobs/econometric_baselines.py; kept only for the record.
==================================================================================
This script evaluated the econometric models on its own fold grid and held one
forecast constant over each 126-day test block, and it used the dispersion of
ARIMA mean forecasts as a volatility forecast. Both choices make the comparison
with daily-updated machine-learning forecasts unfair, so its results are not
used in the paper. Running it would also overwrite the GARCH rows written by
jobs/econometric_baselines.py.

ARIMA + GARCH baselines for volatility forecasting (original description).

Classical econometric baselines for the paper:
- ARIMA(p, d, q): AIC-based order search, k-day realised-volatility forecast
- GARCH(1, 1): standard conditional-variance forecast

Goal: show that the proposed ExtraTrees / HYBRID_STACK_REG
outperform the classical econometric methods (Engle 1982, Bollerslev 1986).

Run:
    python -m jobs.train_arima_garch_baseline

Results are appended to artifacts/metrics_walk_volatility_k5.csv
with model="ARIMA" and model="GARCH" for direct comparison.

Reproducibility:
- Random seed = 42
- END_DATE = 2026-06-01 (pinned for the paper)
"""
from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine, text

# Statsmodels for ARIMA
from statsmodels.tsa.arima.model import ARIMA

# arch for GARCH
from arch import arch_model

warnings.filterwarnings("ignore")
load_dotenv()


# ─────────────────────────────────────────────────────────────────────
#  Configuration (synced with train_baseline.py and LSTM)
# ─────────────────────────────────────────────────────────────────────
SEED = int(os.environ.get("SEED", "42"))
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))

WF_MIN_TRAIN_ROWS = int(os.environ.get("WF_MIN_TRAIN_ROWS", "1200"))
WF_VAL_DAYS = int(os.environ.get("WF_VAL_DAYS", "126"))
WF_TEST_DAYS = int(os.environ.get("WF_TEST_DAYS", "126"))
WF_STEP_DAYS = int(os.environ.get("WF_STEP_DAYS", "63"))

SYMBOLS_DEFAULT = ["AAPL", "TSLA", "^GSPC", "^IXIC", "^DJI", "^RUT", "GLD", "MSFT"]
SYMBOLS_ENV = os.environ.get("SYMBOLS", "")
SYMBOLS = (
    [s.strip() for s in SYMBOLS_ENV.split(",") if s.strip()]
    if SYMBOLS_ENV
    else SYMBOLS_DEFAULT
)

ARTIFACTS_DIR = Path(os.environ.get("METRICS_DIR", "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ARTIFACTS_DIR / f"metrics_walk_volatility_k{HORIZON_DAYS}.csv"

# ARIMA order-search parameters
ARIMA_P_RANGE = range(0, 4)
ARIMA_D_RANGE = [0, 1]
ARIMA_Q_RANGE = range(0, 4)


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────
#  Load OHLCV from PostgreSQL
# ─────────────────────────────────────────────────────────────────────
def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL env var is not set")
    return create_engine(db_url, pool_pre_ping=True)


def load_close(engine, symbol: str, end_date: str) -> pd.DataFrame:
    q = text("""
        SELECT date, close
        FROM market_ohlcv
        WHERE symbol = :sym AND date <= :end_date
        ORDER BY date ASC
    """)
    df = pd.read_sql(q, engine, params={"sym": symbol, "end_date": end_date})
    df["date"] = pd.to_datetime(df["date"])
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["simple_return"] = df["close"].pct_change()
    return df.dropna().reset_index(drop=True)


def compute_target_vol(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Target k-day realised volatility = std(r_{t+1}, ..., r_{t+k})."""
    df = df.copy()
    fut = pd.concat(
        [df["simple_return"].shift(-i) for i in range(1, horizon + 1)],
        axis=1,
    )
    df["target_vol_kd"] = fut.std(axis=1, ddof=0)
    return df


# ─────────────────────────────────────────────────────────────────────
#  ARIMA baseline
# ─────────────────────────────────────────────────────────────────────
def find_best_arima_order(returns: np.ndarray, max_p: int = 3, max_d: int = 1, max_q: int = 3) -> tuple[int, int, int]:
    """ARIMA(p, d, q) order search by AIC."""
    best_aic = np.inf
    best_order = (1, 0, 1)
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(returns, order=(p, d, q))
                    res = model.fit(method="statespace", disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order


def forecast_arima_volatility(returns_train: np.ndarray, horizon: int) -> float:
    """Volatility forecast via ARIMA:
    1) Fit ARIMA on returns
    2) Forecast k steps ahead
    3) Volatility = std of forecast returns
    """
    if len(returns_train) < 100:
        return float(np.std(returns_train, ddof=0))

    try:
        order = find_best_arima_order(returns_train)
        model = ARIMA(returns_train, order=order)
        res = model.fit(method="statespace", disp=False)
        forecast = res.forecast(steps=horizon)
        # Volatility = std of forecast returns
        return float(np.std(forecast, ddof=0)) if len(forecast) > 1 else float(np.abs(forecast.iloc[0]))
    except Exception:
        # Fallback: historical volatility
        return float(np.std(returns_train[-horizon * 5 :], ddof=0))


# ─────────────────────────────────────────────────────────────────────
#  GARCH(1,1) baseline
# ─────────────────────────────────────────────────────────────────────
def forecast_garch_volatility(returns_train: np.ndarray, horizon: int) -> float:
    """Volatility forecast via GARCH(1, 1):
    1) Fit GARCH(1, 1) on returns
    2) Forecast conditional variance k steps ahead
    3) Volatility = sqrt of mean variance
    """
    if len(returns_train) < 100:
        return float(np.std(returns_train, ddof=0))

    try:
        # Scale returns to % for numerical stability
        ret_pct = returns_train * 100.0
        am = arch_model(ret_pct, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=horizon, reindex=False)
        variances = forecast.variance.values[0]
        # Mean volatility over the horizon
        avg_vol_pct = np.sqrt(np.mean(variances))
        # Convert back to a fraction
        return float(avg_vol_pct / 100.0)
    except Exception:
        return float(np.std(returns_train[-horizon * 5 :], ddof=0))


# ─────────────────────────────────────────────────────────────────────
#  Walk-forward for one model and one instrument
# ─────────────────────────────────────────────────────────────────────
@dataclass
class RowReg:
    symbol: str
    mode: str
    fold: int
    split: str
    task: str
    horizon_days: int
    model: str
    n_rows: int
    mae: float
    rmse: float
    r2: float
    extra: str


def evaluate_split(
    sym: str, fold: int, split_name: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, extra: str = ""
) -> RowReg:
    return RowReg(
        symbol=sym,
        mode="walk",
        fold=fold,
        split=split_name,
        task="volatility",
        horizon_days=HORIZON_DAYS,
        model=model_name,
        n_rows=len(y_true),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
        extra=extra,
    )


def run_arima_garch_walk_forward(engine, symbol: str) -> list[RowReg]:
    """Walk-forward for ARIMA and GARCH jointly (same folds)."""
    print(f"\n[ARIMA/GARCH] === Instrument: {symbol} ===")
    df = load_close(engine, symbol, END_DATE)
    if len(df) < WF_MIN_TRAIN_ROWS + WF_VAL_DAYS + WF_TEST_DAYS:
        print(f"[ARIMA/GARCH] {symbol}: not enough data, skipping")
        return []

    n = len(df)
    start_idx = WF_MIN_TRAIN_ROWS
    last_start = n - (WF_VAL_DAYS + WF_TEST_DAYS)

    results: list[RowReg] = []
    fold = 0
    i = start_idx

    while i <= last_start:
        tr = df.iloc[:i].copy()
        va = df.iloc[i : i + WF_VAL_DAYS].copy()
        te = df.iloc[i + WF_VAL_DAYS : i + WF_VAL_DAYS + WF_TEST_DAYS].copy()

        tr = compute_target_vol(tr, HORIZON_DAYS).dropna(subset=["target_vol_kd"])
        va = compute_target_vol(va, HORIZON_DAYS).dropna(subset=["target_vol_kd"])
        te = compute_target_vol(te, HORIZON_DAYS).dropna(subset=["target_vol_kd"])

        if len(tr) < 500 or len(va) < 80 or len(te) < 80:
            i += WF_STEP_DAYS
            continue

        print(f"[ARIMA/GARCH] {symbol} fold={fold}: train={len(tr)} val={len(va)} test={len(te)}")

        # Rolling forecast: each val/test day uses only past data
        # For speed: one ARIMA/GARCH per fold (on train), constant forecast over val/test
        # Standard baseline approach (finer rolling is expensive)

        returns_train = tr["simple_return"].values

        # --- ARIMA forecast ---
        arima_vol = forecast_arima_volatility(returns_train, HORIZON_DAYS)
        arima_pred_val = np.full(len(va), arima_vol)
        arima_pred_test = np.full(len(te), arima_vol)

        # --- GARCH forecast ---
        garch_vol = forecast_garch_volatility(returns_train, HORIZON_DAYS)
        garch_pred_val = np.full(len(va), garch_vol)
        garch_pred_test = np.full(len(te), garch_vol)

        y_va = va["target_vol_kd"].values
        y_te = te["target_vol_kd"].values

        # Store ARIMA results
        results.append(evaluate_split(symbol, fold, "val", "ARIMA", y_va, arima_pred_val, f"vol={arima_vol:.6f}"))
        results.append(evaluate_split(symbol, fold, "test", "ARIMA", y_te, arima_pred_test, f"vol={arima_vol:.6f}"))

        # Store GARCH results
        results.append(evaluate_split(symbol, fold, "val", "GARCH", y_va, garch_pred_val, f"vol={garch_vol:.6f}"))
        results.append(evaluate_split(symbol, fold, "test", "GARCH", y_te, garch_pred_test, f"vol={garch_vol:.6f}"))

        last_arima = results[-3]  # test ARIMA
        last_garch = results[-1]  # test GARCH
        print(
            f"[ARIMA/GARCH] {symbol} fold={fold} TEST: "
            f"ARIMA RMSE={last_arima.rmse:.6f}  |  GARCH RMSE={last_garch.rmse:.6f}"
        )

        fold += 1
        i += WF_STEP_DAYS

    return results


# ─────────────────────────────────────────────────────────────────────
#  Saving
# ─────────────────────────────────────────────────────────────────────
def append_to_metrics_csv(rows: list[RowReg]) -> None:
    if not rows:
        print("[ARIMA/GARCH] No results to save")
        return

    new_df = pd.DataFrame([asdict(r) for r in rows])

    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        # Drop only ARIMA/GARCH rows for the symbols being recomputed
        # (so adding new tickers does not erase earlier results).
        recomputed_symbols = set(new_df["symbol"].unique())
        mask_drop = old["model"].isin(["ARIMA", "GARCH"]) & old["symbol"].isin(recomputed_symbols)
        old = old[~mask_drop]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(OUT_CSV, index=False)
    print(f"\n[ARIMA/GARCH] Saved {len(rows)} rows to {OUT_CSV}")
    print(f"[ARIMA/GARCH] Total rows in file: {len(combined)}")


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print(" ARIMA + GARCH baselines for volatility forecasting")
    print("=" * 70)
    print(f" Random seed:   {SEED}")
    print(f" END_DATE:      {END_DATE} (pinned for the paper)")
    print(f" Horizon:       k = {HORIZON_DAYS} days")
    print(f" Instruments:   {SYMBOLS}")
    print("=" * 70)

    set_seeds(SEED)
    engine = get_engine()

    all_results: list[RowReg] = []
    for sym in SYMBOLS:
        try:
            sym_results = run_arima_garch_walk_forward(engine, sym)
            all_results.extend(sym_results)
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")
            import traceback
            traceback.print_exc()

    append_to_metrics_csv(all_results)

    # Summary
    if all_results:
        df = pd.DataFrame([asdict(r) for r in all_results])
        test_df = df[df["split"] == "test"]
        if not test_df.empty:
            print("\n[ARIMA/GARCH] Test-split summary:")
            for model in ["ARIMA", "GARCH"]:
                sub = test_df[test_df["model"] == model]
                if not sub.empty:
                    print(f"\n  {model}:")
                    summary = sub.groupby("symbol").agg(
                        rmse_mean=("rmse", "mean"),
                        rmse_std=("rmse", "std"),
                        mae_mean=("mae", "mean"),
                        n_folds=("fold", "count"),
                    )
                    print(summary.to_string())


if __name__ == "__main__":
    main()
