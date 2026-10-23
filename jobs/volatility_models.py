"""
Extended volatility study with daily forecasts and standard volatility losses.
==============================================================================

Extends jobs/volatility_hybrids.py in three ways:

1. Stronger econometric baselines, all fitted on the training block of each
   walk-forward fold and updated daily without refitting:
     GARCH   GARCH(1,1) with normal innovations (as in econometric_baselines.py)
     GJR     GJR-GARCH(1,1) with Student-t innovations (leverage effect;
             Glosten, Jagannathan & Runkle, 1993)
     HAR     HAR on absolute returns (Corsi, 2009), as in econometric_baselines.py
     HAR_RG  HAR on a range-based daily variance: the Garman-Klass estimator
             plus the squared overnight return, which uses open, high, low and
             close prices and is more efficient than squared returns
     ECON    equal-weight average of GARCH, GJR and HAR_RG
2. A tree-based correction on top of each econometric base b:
     H_<b>   w * b + (1 - w) * b * ET(f), where ET is trained on y / b and the
             weight w in {0, 0.1, ..., 1} is chosen on the validation block.
   H_GARCH is the earlier H-GARCH-ET hybrid; H_ECON (H-ECON-ET) is the volatility
   model of the system.
3. Daily forecasts of every model are stored, so that the models can be compared
   with RMSE, with the QLIKE loss on variances (Patton, 2011), which is robust to
   noise in the volatility proxy, and with the model confidence set (Hansen,
   Lunde & Nason, 2011).

The model used by the system is chosen on the validation blocks only (lowest
mean validation RMSE over the eight instruments among the hybrids); the test
blocks are used for reporting.

Run:
    TRAIN_START_DATE=2015-01-01 python -m jobs.volatility_models           # fit
    python -m jobs.volatility_models evaluate                              # tables
Outputs:
    artifacts/volatility_daily_k{K}.csv.gz    one row per (symbol, fold, split, date)
    artifacts/volatility_models_summary_k{K}.csv
    artifacts/volatility_mcs_k{K}.csv, artifacts/volatility_dm_daily_k{K}.csv
    artifacts/metrics_walk_volatility_k{K}.csv: rows for GJR_GARCH, HAR_RG, ECON and
    HYBRID_ECON_ET (the volatility model of the system)
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from arch.bootstrap import MCS
from dotenv import load_dotenv
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine

from jobs.diebold_mariano_test import dm_test
from jobs.econometric_baselines import prepare, garch_daily, har_features, rmse, SYMBOLS, K, A
from jobs.train_baseline import (
    build_feature_matrix, compute_targets,
    WF_MIN_TRAIN_ROWS, WF_VAL_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
)

warnings.filterwarnings("ignore")
load_dotenv()
SEED = 42
GRID = np.linspace(0.0, 1.0, 11)
BASES = ["GARCH", "GJR", "HAR", "HAR_RG", "ECON"]
HYBRIDS = ["H_GARCH", "H_GJR", "H_HAR_RG", "H_ECON"]
MODELS = BASES + ["ET"] + HYBRIDS
NAIVE = ["MEAN", "PERSIST"]           # training-sample mean and backward k-day realised volatility
OUT = A / f"volatility_daily_k{K}.csv.gz"


def et():
    return ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                               random_state=SEED, n_jobs=-1)


def gjr_daily(r_train: np.ndarray, r_all: np.ndarray, n_train: int) -> np.ndarray:
    """k-step volatility forecast of a GJR-GARCH(1,1)-t fitted on the training returns."""
    res = arch_model(r_train * 100.0, vol="GARCH", p=1, o=1, q=1, dist="t",
                     rescale=False).fit(disp="off", show_warning=False)
    mu, om, al, ga, be = (res.params[x] for x in ("mu", "omega", "alpha[1]", "gamma[1]", "beta[1]"))
    x = r_all * 100.0 - mu
    neg = (x < 0).astype(float)
    s2 = np.empty(len(x))
    s2[0] = np.var(x[:n_train])
    for t in range(1, len(x)):
        s2[t] = om + (al + ga * neg[t - 1]) * x[t - 1] ** 2 + be * s2[t - 1]
    nxt = om + (al + ga * neg) * x ** 2 + be * s2          # variance of day t+1 given day t
    pers = al + ga / 2.0 + be                              # symmetric innovations
    lr = om / (1.0 - pers) if pers < 1 else np.mean(s2[:n_train])
    mean_var = lr + (nxt[:, None] - lr) * (pers ** np.arange(K))[None, :]
    return np.sqrt(np.clip(mean_var.mean(axis=1), 1e-12, None)) / 100.0


def range_features(d: pd.DataFrame) -> np.ndarray:
    """HAR regressors from a range-based daily variance (Garman-Klass + overnight)."""
    o, h, lo, c = (d[k].astype(float).values for k in ("open", "high", "low", "close"))
    prev_c = np.r_[np.nan, c[:-1]]
    overnight = np.log(o / prev_c) ** 2
    gk = 0.5 * np.log(h / lo) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(c / o) ** 2
    v = pd.Series(np.clip(np.nan_to_num(overnight) + np.clip(gk, 0.0, None), 0.0, None))
    return np.column_stack([np.sqrt(v), np.sqrt(v.rolling(5, min_periods=5).mean()),
                            np.sqrt(v.rolling(22, min_periods=22).mean())])


def naive_forecasts(df2: pd.DataFrame, pos: dict, y_tr: np.ndarray) -> dict:
    """Training-sample mean of the target and the backward k-day realised volatility."""
    pers = df2["log_return"].rolling(K, min_periods=K).std(ddof=0).values
    return {"MEAN": {s: np.full(len(pos[s]), y_tr.mean()) for s in pos},
            "PERSIST": {s: np.where(np.isfinite(pers[pos[s]]), pers[pos[s]], y_tr.mean()) for s in pos}}


def add_naive():
    """Adds the naive benchmarks to an existing daily-forecast file without refitting."""
    d = pd.read_csv(OUT, parse_dates=["date"])
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    parts = []
    for sym in SYMBOLS:
        df2 = prepare(engine, sym)
        n = len(df2)
        i, fold = WF_MIN_TRAIN_ROWS, 0
        while i <= n - (WF_VAL_DAYS + WF_TEST_DAYS):
            tr = compute_targets(df2.iloc[:i].copy(), K)
            va = compute_targets(df2.iloc[i:i + WF_VAL_DAYS].copy(), K)
            te = compute_targets(df2.iloc[i + WF_VAL_DAYS:i + WF_VAL_DAYS + WF_TEST_DAYS].copy(), K)
            if len(tr) < 500 or len(va) < 80 or len(te) < 80:
                i += WF_STEP_DAYS
                continue
            pos = {"train": np.arange(len(tr)), "val": np.arange(i, i + len(va)),
                   "test": np.arange(i + WF_VAL_DAYS, i + WF_VAL_DAYS + len(te))}
            nv = naive_forecasts(df2, pos, tr["target_vol_kd"].values)
            for s in ("val", "test"):
                parts.append(pd.DataFrame({"symbol": sym, "fold": fold, "split": s,
                                           "date": df2["date"].values[pos[s]],
                                           "MEAN": nv["MEAN"][s], "PERSIST": nv["PERSIST"][s]}))
            fold += 1
            i += WF_STEP_DAYS
    nv = pd.concat(parts, ignore_index=True)
    d = d.drop(columns=[c for c in NAIVE if c in d.columns]).merge(nv, on=["symbol", "fold", "split", "date"],
                                                                     how="left", validate="one_to_one")
    assert d[NAIVE].notna().all().all()
    d.to_csv(OUT, index=False, compression="gzip")
    print(f"[ARTIFACT] naive benchmarks added to {OUT}")


def ols_forecast(F: np.ndarray, pos: dict, y_tr: np.ndarray) -> dict:
    """OLS of the target on F over the training rows; NaN regressors filled with training means."""
    Ftr = F[pos["train"]]
    ok = np.isfinite(Ftr).all(axis=1)
    fill = np.nanmean(Ftr, axis=0)
    m = LinearRegression().fit(Ftr[ok], y_tr[ok])
    return {s: m.predict(np.where(np.isfinite(F[pos[s]]), F[pos[s]], fill)) for s in pos}


def fit_all():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    frames = []
    for sym in SYMBOLS:
        df2 = prepare(engine, sym)
        _, feat = build_feature_matrix(df2.copy())
        feat = [c for c in feat if c in df2.columns]
        X, r = df2[feat].values, df2["log_return"].values
        H, RG = har_features(df2), range_features(df2)
        n = len(df2)
        i, fold = WF_MIN_TRAIN_ROWS, 0
        while i <= n - (WF_VAL_DAYS + WF_TEST_DAYS):
            tr = compute_targets(df2.iloc[:i].copy(), K)
            va = compute_targets(df2.iloc[i:i + WF_VAL_DAYS].copy(), K)
            te = compute_targets(df2.iloc[i + WF_VAL_DAYS:i + WF_VAL_DAYS + WF_TEST_DAYS].copy(), K)
            if len(tr) < 500 or len(va) < 80 or len(te) < 80:
                i += WF_STEP_DAYS
                continue
            pos = {"train": np.arange(len(tr)), "val": np.arange(i, i + len(va)),
                   "test": np.arange(i + WF_VAL_DAYS, i + WF_VAL_DAYS + len(te))}
            Y = {"train": tr["target_vol_kd"].values, "val": va["target_vol_kd"].values,
                 "test": te["target_vol_kd"].values}
            end = i + WF_VAL_DAYS + WF_TEST_DAYS

            P = {}
            P["GARCH"] = ols_forecast(garch_daily(r[:i], r[:end], i).reshape(-1, 1), pos, Y["train"])
            P["GJR"] = ols_forecast(gjr_daily(r[:i], r[:end], i).reshape(-1, 1), pos, Y["train"])
            P["HAR"] = ols_forecast(H, pos, Y["train"])
            P["HAR_RG"] = ols_forecast(RG, pos, Y["train"])
            P["ECON"] = {s: (P["GARCH"][s] + P["GJR"][s] + P["HAR_RG"][s]) / 3.0 for s in pos}
            P.update(naive_forecasts(df2, pos, Y["train"]))

            Xs = {s: X[pos[s]] for s in pos}
            m = et().fit(Xs["train"], Y["train"])
            P["ET"] = {s: m.predict(Xs[s]) for s in pos}
            weights = {}
            for base in ("GARCH", "GJR", "HAR_RG", "ECON"):
                b = {s: np.clip(P[base][s], 1e-6, None) for s in pos}
                m = et().fit(Xs["train"], Y["train"] / b["train"])
                ml = {s: m.predict(Xs[s]) * b[s] for s in ("val", "test")}
                w = GRID[int(np.argmin([rmse(Y["val"], a * b["val"] + (1 - a) * ml["val"]) for a in GRID]))]
                P["H_" + base] = {s: w * b[s] + (1 - w) * ml[s] for s in ("val", "test")}
                weights["w_" + base] = w

            for s in ("val", "test"):
                f = pd.DataFrame({"symbol": sym, "fold": fold, "split": s,
                                  "date": df2["date"].values[pos[s]], "y": Y[s]})
                for name in MODELS + NAIVE:
                    f[name] = P[name][s]
                for k, v in weights.items():
                    f[k] = v
                frames.append(f)
            fold += 1
            i += WF_STEP_DAYS
        print(f"{sym}: {fold} folds", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False, compression="gzip")
    print(f"[ARTIFACT] {OUT}")


def hybrid_forecast(df2: pd.DataFrame, feat: list[str], base: str = "ECON", k: int = K,
                    val_days: int = WF_VAL_DAYS) -> float:
    """Live forecast of H_<base> for the last row of df2.

    The blend weight is chosen on the most recent val_days labelled rows with all
    components fitted on the rows before them (purged by k rows); every component
    is then refitted on all labelled rows and the forecast of the last row is returned.
    """
    lab = compute_targets(df2.copy(), k)              # labelled rows are df2.iloc[:len(lab)]
    n_lab = len(lab)
    y = lab["target_vol_kd"].values
    X, r = df2[feat].values, df2["log_return"].values
    H, RG = har_features(df2), range_features(df2)
    n = len(df2)

    def components(n_fit):
        pos = {"train": np.arange(n_fit), "all": np.arange(n)}
        P = {"GARCH": ols_forecast(garch_daily(r[:n_fit], r, n_fit).reshape(-1, 1), pos, y[:n_fit])["all"],
             "GJR": ols_forecast(gjr_daily(r[:n_fit], r, n_fit).reshape(-1, 1), pos, y[:n_fit])["all"],
             "HAR_RG": ols_forecast(RG, pos, y[:n_fit])["all"],
             "HAR": ols_forecast(H, pos, y[:n_fit])["all"]}
        P["ECON"] = (P["GARCH"] + P["GJR"] + P["HAR_RG"]) / 3.0
        b = np.clip(P[base], 1e-6, None)
        m = et().fit(X[:n_fit], y[:n_fit] / b[:n_fit])
        return b, m

    n_fit = n_lab - val_days - k
    b, m = components(n_fit)
    v = np.arange(n_lab - val_days, n_lab)
    ml = m.predict(X[v]) * b[v]
    w = GRID[int(np.argmin([rmse(y[v], a * b[v] + (1 - a) * ml) for a in GRID]))]
    b, m = components(n_lab)
    last = n - 1
    return float(w * b[last] + (1 - w) * m.predict(X[[last]])[0] * b[last])


def qlike(y, f):
    """QLIKE loss on variances (Patton, 2011): s/h - log(s/h) - 1, with s = y^2 and h = f^2."""
    s = np.maximum(np.asarray(y) ** 2, 1e-12)
    h = np.maximum(np.asarray(f) ** 2, 1e-12)
    return s / h - np.log(s / h) - 1.0


def evaluate():
    d = pd.read_csv(OUT)
    rows = []
    for (sym, split), g in d.groupby(["symbol", "split"]):
        for mname in MODELS:
            fold_rmse = g.groupby("fold").apply(lambda x: rmse(x["y"], x[mname]), include_groups=False)
            rows.append(dict(symbol=sym, split=split, model=mname, rmse=float(fold_rmse.mean()),
                             qlike=float(qlike(g["y"], g[mname]).mean())))
    summ = pd.DataFrame(rows)
    summ.to_csv(A / f"volatility_models_summary_k{K}.csv", index=False)

    val = summ[summ.split == "val"].pivot(index="symbol", columns="model", values="rmse")
    test_rmse = summ[summ.split == "test"].pivot(index="symbol", columns="model", values="rmse")
    test_ql = summ[summ.split == "test"].pivot(index="symbol", columns="model", values="qlike")
    chosen = val[HYBRIDS].mean().idxmin()
    print("validation RMSE (mean over instruments):", val[MODELS].mean().round(6).to_dict())
    print(f"model chosen on validation: {chosen}")
    print("\ntest RMSE:\n", test_rmse[MODELS].round(5).to_string())
    print("mean:", test_rmse[MODELS].mean().round(6).to_dict())
    print("rank:", test_rmse[MODELS].rank(axis=1).mean().round(2).to_dict())
    print("\ntest QLIKE mean:", test_ql[MODELS].mean().round(4).to_dict())
    print("QLIKE rank:", test_ql[MODELS].rank(axis=1).mean().round(2).to_dict())

    # Daily test series without duplicates: consecutive test blocks overlap by half,
    # so each date keeps the forecast of the most recent fold, as a system refitted
    # every quarter would use.
    test = (d[d.split == "test"].sort_values(["symbol", "date", "fold"])
            .drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True))

    # Diebold-Mariano tests on the daily losses (HLN correction, h = k for the
    # overlapping k-day targets): the chosen model against every other model
    dm_rows = []
    for loss_name, fn in (("squared error", lambda y, f: (y - f) ** 2), ("QLIKE", qlike)):
        for sym, g in test.groupby("symbol"):
            la = fn(g["y"].values, g[chosen].values)
            for m in MODELS + NAIVE:
                if m == chosen:
                    continue
                lb = fn(g["y"].values, g[m].values)
                better = dm_test(la, lb, horizon=K, alternative="less")
                worse = dm_test(lb, la, horizon=K, alternative="less")
                dm_rows.append(dict(loss=loss_name, symbol=sym, model_a=chosen, model_b=m, T=better["T"],
                                    p_a_better=better["p_value"], p_b_better=worse["p_value"]))
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(A / f"volatility_dm_daily_k{K}.csv", index=False)
    print(f"\ndaily DM, {chosen} vs B: instruments where {chosen} is significantly better / worse (p < 0.05)")
    print(dm.groupby(["loss", "model_b"]).agg(better=("p_a_better", lambda p: int((p < 0.05).sum())),
                                              worse=("p_b_better", lambda p: int((p < 0.05).sum()))).to_string())

    # model confidence set per instrument on the daily test losses (90% level)
    mcs_rows = []
    for loss_name, fn in (("squared error", lambda y, f: (y - f) ** 2), ("QLIKE", qlike)):
        for sym, g in test.groupby("symbol"):
            L = pd.DataFrame({m: fn(g["y"].values, g[m].values) for m in MODELS})
            mcs = MCS(L, size=0.10, reps=2000, block_size=10, bootstrap="stationary", seed=SEED)
            mcs.compute()
            inc = set(mcs.included)
            for m in MODELS:
                mcs_rows.append(dict(loss=loss_name, symbol=sym, model=m, in_mcs=m in inc,
                                     pvalue=float(mcs.pvalues.loc[m].iloc[0])))
    mcs = pd.DataFrame(mcs_rows)
    mcs.to_csv(A / f"volatility_mcs_k{K}.csv", index=False)
    print("\ninstruments for which a model is in the 90% MCS:")
    print(mcs.groupby(["loss", "model"]).in_mcs.sum().unstack("loss").loc[MODELS].to_string())
    export_metrics(d)


# names of the new models in the walk-forward metric file
EXPORT = {"GJR": "GJR_GARCH", "HAR_RG": "HAR_RG", "ECON": "ECON", "H_ECON": "HYBRID_ECON_ET"}


def export_metrics(d: pd.DataFrame):
    """Fold-level MAE, RMSE and R² of the new models, written to the walk-forward metric
    file in the format of train_baseline.py, so that the significance tests and the
    figures treat them like every other model."""
    from sklearn.metrics import mean_absolute_error, r2_score
    rows = []
    for (sym, fold, split), g in d.groupby(["symbol", "fold", "split"]):
        for src, name in EXPORT.items():
            rows.append(dict(symbol=sym, mode="walk", fold=fold, split=split, task="volatility", horizon_days=K,
                             model=name, n_rows=len(g), mae=float(mean_absolute_error(g["y"], g[src])),
                             rmse=rmse(g["y"], g[src]), r2=float(r2_score(g["y"], g[src])), extra=""))
    mpath = A / f"metrics_walk_volatility_k{K}.csv"
    m = pd.read_csv(mpath)
    m = pd.concat([m[~m["model"].isin(EXPORT.values())], pd.DataFrame(rows)], ignore_index=True)
    m.to_csv(mpath, index=False)
    print(f"[ARTIFACT] {', '.join(EXPORT.values())} rows written to {mpath}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate()
    elif len(sys.argv) > 1 and sys.argv[1] == "add-naive":
        add_naive()
    else:
        fit_all()
        evaluate()
