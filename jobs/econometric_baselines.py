"""
Econometric volatility baselines evaluated on exactly the walk-forward folds of
the machine-learning models.
=============================================================================

An earlier ARIMA/GARCH baseline, since removed from the repository, evaluated the
econometric models on their own fold grid and held a single forecast constant
over each 126-day test block. That makes the comparison with daily-updated
machine-learning forecasts unfair. This script replaces it with:

  GARCH        GARCH(1,1) fitted on the training block of each fold; the
               conditional variance is then filtered day by day through the
               validation and test blocks with the fitted parameters, so every
               forecast uses information up to day t only. The k-step forecast
               is the square root of the mean predicted variance over t+1..t+k,
               mapped to the target scale by an OLS fit on the training block.
  HAR          Heterogeneous autoregressive model of realised volatility
               (Corsi, 2009): OLS of the k-day-ahead volatility on daily,
               weekly (5-day) and monthly (22-day) realised volatility,
               fitted on the training block.
  PERSIST      k-day backward realised volatility, recomputed here only to
               check that the folds coincide with those of train_baseline.py.

Data preparation, fold boundaries and targets are taken from train_baseline.py,
so the validation and test blocks are identical to those of ExtraTrees and the
deep baselines. ARIMA is not used as a volatility baseline: it models the
conditional mean, and the dispersion of its point forecasts is not a volatility
forecast.

Run:
    TRAIN_START_DATE=2015-01-01 python -m jobs.econometric_baselines
Outputs:
    artifacts/econometric_baselines_k{K}.csv (one row per symbol, fold, split, model)
    artifacts/metrics_walk_volatility_k{K}.csv: the ARIMA/GARCH rows written by the
    earlier script are replaced by the GARCH and HAR rows computed here
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from jobs.racc import _SQL
from jobs.train_baseline import (
    add_technical_features, build_feature_matrix, compute_targets,
    WF_MIN_TRAIN_ROWS, WF_VAL_DAYS, WF_TEST_DAYS, WF_STEP_DAYS, HORIZON_DAYS,
)

warnings.filterwarnings("ignore")
load_dotenv()
START = os.environ.get("TRAIN_START_DATE", "2015-01-01")
END = os.environ.get("END_DATE", "2026-06-01")
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]
A = Path(__file__).resolve().parents[1] / "artifacts"
K = HORIZON_DAYS


def prepare(engine, sym: str) -> pd.DataFrame:
    """Same steps as train_baseline.main() followed by run_walk()."""
    df = pd.read_sql(_SQL, engine, params={"s": sym}, parse_dates=["date"])
    df = df[(df["date"] >= pd.to_datetime(START)) & (df["date"] <= pd.to_datetime(END))]
    df = df.replace([np.inf, -np.inf], np.nan)
    core = ["date", "open", "high", "low", "close", "volume", "return_1d", "log_return"]
    df = df.dropna(subset=core).reset_index(drop=True)
    df = add_technical_features(df).replace([np.inf, -np.inf], np.nan)
    df2, _ = build_feature_matrix(df)
    df2 = df2.copy()
    df2["persist_vol_kd"] = df2["log_return"].rolling(K, min_periods=K).std(ddof=0)
    return df2.reset_index(drop=True)


def garch_daily(r_train: np.ndarray, r_all: np.ndarray, n_train: int) -> np.ndarray:
    """k-step volatility forecast for every day, using parameters fitted on the training returns."""
    am = arch_model(r_train * 100.0, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    mu, om, al, be = (res.params[x] for x in ("mu", "omega", "alpha[1]", "beta[1]"))
    x = r_all * 100.0 - mu
    s2 = np.empty(len(x))
    s2[0] = np.var(x[:n_train])
    for t in range(1, len(x)):
        s2[t] = om + al * x[t - 1] ** 2 + be * s2[t - 1]
    # variance for day t+1 given information up to t, then the k-step mean
    nxt = om + al * x ** 2 + be * s2
    pers = al + be
    lr = om / (1 - pers) if pers < 1 else np.mean(s2[:n_train])
    hs = np.arange(K)
    mean_var = lr + (nxt[:, None] - lr) * (pers ** hs)[None, :]
    return np.sqrt(np.clip(mean_var.mean(axis=1), 1e-12, None)) / 100.0


def har_features(d: pd.DataFrame) -> np.ndarray:
    r = d["log_return"]
    return np.column_stack([r.abs(), r.rolling(5, min_periods=5).std(ddof=0),
                            r.rolling(22, min_periods=22).std(ddof=0)])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def metric_row(sym, fold, split, model, y, pred):
    return dict(symbol=sym, mode="walk", fold=fold, split=split, task="volatility",
                horizon_days=K, model=model, n_rows=len(y),
                mae=float(mean_absolute_error(y, pred)), rmse=rmse(y, pred),
                r2=float(r2_score(y, pred)), extra="")


def main():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    rows = []
    for sym in SYMBOLS:
        df2 = prepare(engine, sym)
        n = len(df2)
        X_har = har_features(df2)
        i, fold = WF_MIN_TRAIN_ROWS, 0
        last_start = n - (WF_VAL_DAYS + WF_TEST_DAYS)
        while i <= last_start:
            tr = compute_targets(df2.iloc[:i].copy(), K)
            va = compute_targets(df2.iloc[i:i + WF_VAL_DAYS].copy(), K)
            te = compute_targets(df2.iloc[i + WF_VAL_DAYS:i + WF_VAL_DAYS + WF_TEST_DAYS].copy(), K)
            if len(tr) < 500 or len(va) < 80 or len(te) < 80:
                i += WF_STEP_DAYS
                continue
            # positions of the retained rows inside df2 (compute_targets keeps order, drops the tail)
            pos = {"train": np.arange(len(tr)),
                   "val": np.arange(i, i + len(va)),
                   "test": np.arange(i + WF_VAL_DAYS, i + WF_VAL_DAYS + len(te))}
            y = {"train": tr["target_vol_kd"].values, "val": va["target_vol_kd"].values,
                 "test": te["target_vol_kd"].values}

            # GARCH: parameters from the training returns, daily filtering afterwards
            end = i + WF_VAL_DAYS + WF_TEST_DAYS
            g_all = garch_daily(df2["log_return"].values[:i], df2["log_return"].values[:end], i)
            cal = LinearRegression().fit(g_all[pos["train"]].reshape(-1, 1), y["train"])

            # HAR on daily, weekly and monthly realised volatility
            ok = np.isfinite(X_har[pos["train"]]).all(axis=1)
            har = LinearRegression().fit(X_har[pos["train"]][ok], y["train"][ok])
            fill = np.nanmean(X_har[pos["train"]], axis=0)

            for split, d in (("val", va), ("test", te)):
                g = cal.predict(g_all[pos[split]].reshape(-1, 1))
                Xs = X_har[pos[split]]
                h = har.predict(np.where(np.isfinite(Xs), Xs, fill))
                pers = d["persist_vol_kd"].values
                pers = np.where(np.isfinite(pers), pers, y["train"].mean())
                rows.append(metric_row(sym, fold, split, "GARCH", y[split], g))
                rows.append(metric_row(sym, fold, split, "HAR", y[split], h))
                rows.append(metric_row(sym, fold, split, "PERSIST_CHECK", y[split], pers))
            fold += 1
            i += WF_STEP_DAYS
        print(f"{sym}: {fold} folds", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(A / f"econometric_baselines_k{K}.csv", index=False)
    print(f"[ARTIFACT] {A / f'econometric_baselines_k{K}.csv'}")

    # Replace the econometric rows of the walk-forward metric file, so that the
    # significance tests and figures use these baselines on the common folds.
    mpath = A / f"metrics_walk_volatility_k{K}.csv"
    if mpath.exists():
        m = pd.read_csv(mpath)
        m = m[~m["model"].isin(["ARIMA", "GARCH", "HAR"])]
        m = pd.concat([m, out[out["model"] != "PERSIST_CHECK"]], ignore_index=True)
        m.to_csv(mpath, index=False)
        print(f"[ARTIFACT] GARCH and HAR rows written to {mpath}")


if __name__ == "__main__":
    main()
