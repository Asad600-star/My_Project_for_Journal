"""
Volatility models that combine econometric forecasts with tree ensembles.
=========================================================================

With correctly implemented, daily-updated econometric baselines (see
jobs/econometric_baselines.py), GARCH(1,1) and HAR are more accurate than a
tree ensemble trained on the raw volatility target. This script tests the
standard remedies, all fitted on the training block of each walk-forward fold
and evaluated on exactly the same folds as every other model:

  ET            ExtraTrees on the raw target (replicates train_baseline.py)
  ET_RATIO      ExtraTrees on y / v22, where v22 is the trailing 22-day realised
                volatility; the forecast is multiplied back by v22. A scale-free
                target lets the trees transfer across volatility regimes.
  HAR_ET        HAR forecast plus an ExtraTrees correction of the HAR residual
  GARCH_ET      GARCH forecast plus an ExtraTrees correction of its residual
  ET_FEAT       ExtraTrees with the HAR and GARCH forecasts as extra features
  ET_GSCALE     ExtraTrees on y / g, where g is the calibrated GARCH forecast; the
                forecast is g times the tree prediction, so GARCH supplies the
                volatility level and the trees a multiplicative correction
  BLEND         w * GARCH + (1 - w) * ET_GSCALE, with w in {0, 0.1, ..., 1}
                chosen on the validation block of the same fold
  COMBO         equal-weight average of the GARCH and HAR forecasts

Model choice is made on the validation blocks; the test blocks are only used for
reporting, so the comparison is not tuned on the test data.

Run:
    TRAIN_START_DATE=2015-01-01 python -m jobs.volatility_hybrids
Outputs:
    artifacts/volatility_hybrids_k{K}.csv (symbol, fold, split, model, mae, rmse, r2)
    artifacts/metrics_walk_volatility_k{K}.csv: BLEND rows added as HYBRID_GARCH_ET
    artifacts/volatility_blend_weights_k{K}.csv (validation-selected GARCH weight per fold)
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

from jobs.econometric_baselines import prepare, garch_daily, har_features, metric_row, rmse, SYMBOLS, K, A
from jobs.train_baseline import (
    build_feature_matrix, compute_targets,
    WF_MIN_TRAIN_ROWS, WF_VAL_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
)

warnings.filterwarnings("ignore")
load_dotenv()
SEED = 42


def et():
    return ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                               random_state=SEED, n_jobs=-1)


def main():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    rows, weights = [], []
    for sym in SYMBOLS:
        df2 = prepare(engine, sym)
        _, feat = build_feature_matrix(df2.copy())
        feat = [c for c in feat if c in df2.columns]
        X = df2[feat].values
        H = har_features(df2)
        v22 = df2["log_return"].rolling(22, min_periods=22).std(ddof=0).values
        n = len(df2)
        i, fold = WF_MIN_TRAIN_ROWS, 0
        while i <= n - (WF_VAL_DAYS + WF_TEST_DAYS):
            tr = compute_targets(df2.iloc[:i].copy(), K)
            va = compute_targets(df2.iloc[i:i + WF_VAL_DAYS].copy(), K)
            te = compute_targets(df2.iloc[i + WF_VAL_DAYS:i + WF_VAL_DAYS + WF_TEST_DAYS].copy(), K)
            if len(tr) < 500 or len(va) < 80 or len(te) < 80:
                i += WF_STEP_DAYS
                continue
            P = {"train": np.arange(len(tr)),
                 "val": np.arange(i, i + len(va)),
                 "test": np.arange(i + WF_VAL_DAYS, i + WF_VAL_DAYS + len(te))}
            Y = {"train": tr["target_vol_kd"].values, "val": va["target_vol_kd"].values,
                 "test": te["target_vol_kd"].values}
            ptr = P["train"]

            g = garch_daily(df2["log_return"].values[:i], df2["log_return"].values[:i + WF_VAL_DAYS + WF_TEST_DAYS], i)
            gcal = LinearRegression().fit(g[ptr].reshape(-1, 1), Y["train"])
            G = {s: gcal.predict(g[P[s]].reshape(-1, 1)) for s in P}
            Hf = np.nan_to_num(H, nan=0.0)
            ok = np.isfinite(H[ptr]).all(axis=1)
            har = LinearRegression().fit(Hf[ptr][ok], Y["train"][ok])
            R = {s: har.predict(Hf[P[s]]) for s in P}
            Xs = {s: X[P[s]] for s in P}

            preds = {}
            m = et().fit(Xs["train"], Y["train"]); preds["ET"] = {s: m.predict(Xs[s]) for s in ("val", "test")}
            sc = {s: np.where(np.isfinite(v22[P[s]]) & (v22[P[s]] > 0), v22[P[s]], np.nanmean(v22[ptr])) for s in P}
            m = et().fit(Xs["train"], Y["train"] / sc["train"])
            preds["ET_RATIO"] = {s: m.predict(Xs[s]) * sc[s] for s in ("val", "test")}
            m = et().fit(Xs["train"], Y["train"] - R["train"])
            preds["HAR_ET"] = {s: R[s] + m.predict(Xs[s]) for s in ("val", "test")}
            m = et().fit(Xs["train"], Y["train"] - G["train"])
            preds["GARCH_ET"] = {s: G[s] + m.predict(Xs[s]) for s in ("val", "test")}
            Xa = {s: np.column_stack([Xs[s], G[s], R[s]]) for s in P}
            m = et().fit(Xa["train"], Y["train"])
            preds["ET_FEAT"] = {s: m.predict(Xa[s]) for s in ("val", "test")}
            gs = {s: np.clip(G[s], 1e-6, None) for s in P}
            m = et().fit(Xs["train"], Y["train"] / gs["train"])
            preds["ET_GSCALE"] = {s: m.predict(Xs[s]) * gs[s] for s in ("val", "test")}
            grid = np.linspace(0.0, 1.0, 11)
            w = grid[int(np.argmin([rmse(Y["val"], a * G["val"] + (1 - a) * preds["ET_GSCALE"]["val"]) for a in grid]))]
            preds["BLEND"] = {s: w * G[s] + (1 - w) * preds["ET_GSCALE"][s] for s in ("val", "test")}
            weights.append(dict(symbol=sym, fold=fold, w_garch=w))
            preds["COMBO"] = {s: 0.5 * (G[s] + R[s]) for s in ("val", "test")}
            preds["GARCH"] = {s: G[s] for s in ("val", "test")}
            preds["HAR"] = {s: R[s] for s in ("val", "test")}

            for name, pr in preds.items():
                for s in ("val", "test"):
                    rows.append(metric_row(sym, fold, s, name, Y[s], pr[s]))
            fold += 1
            i += WF_STEP_DAYS
        print(f"{sym}: {fold} folds", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(A / f"volatility_hybrids_k{K}.csv", index=False)
    print(f"[ARTIFACT] {A / f'volatility_hybrids_k{K}.csv'}")

    # The validation-weighted blend is the earlier H-GARCH-ET hybrid, kept for comparison
    # (the volatility model of the system is H-ECON-ET, jobs/volatility_models.py); its
    # rows are added to the walk-forward metric file under the name HYBRID_GARCH_ET.
    mpath = A / f"metrics_walk_volatility_k{K}.csv"
    if mpath.exists():
        m = pd.read_csv(mpath)
        m = m[m["model"] != "HYBRID_GARCH_ET"]
        b = out[out["model"] == "BLEND"].assign(model="HYBRID_GARCH_ET")
        pd.concat([m, b], ignore_index=True).to_csv(mpath, index=False)
        print(f"[ARTIFACT] HYBRID_GARCH_ET rows written to {mpath}")
    pd.DataFrame(weights).to_csv(A / f"volatility_blend_weights_k{K}.csv", index=False)


if __name__ == "__main__":
    main()
