"""
Robustness checks: transaction-cost sensitivity and market-regime stress slices.
===============================================================================

Two checks that address the standard reviewer objections to back-tested
financial ML, namely "your conclusion depends on the cost assumption" and
"your walk-forward may still hide regime overfitting":

  A. COST SENSITIVITY. The RACC-gated and ungated strategies are re-run over a
     grid of transaction costs (0, 5, 10, 20 bps per position change) on the
     held-out block, so the reader can see exactly where each conclusion breaks.

  B. REGIME STRESS SLICES. Models are fit on pre-2020 data only and then
     evaluated on three disjoint out-of-sample regimes:
         - COVID crash        (2020-02-01 .. 2020-06-30)
         - 2022 bear market   (2022-01-01 .. 2022-12-31)
         - Calm / bull        (all remaining post-2020 days)
     Because realised volatility differs by an order of magnitude across
     regimes, raw RMSE is not comparable; we therefore also report the ratio of
     the ExtraTrees error to a persistence baseline fitted on the same slice.

Run:
    python -m jobs.robustness_checks
Outputs:
    artifacts/robustness_costs.csv, artifacts/robustness_regimes.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

# one definition of the data, the RACC loss and the threshold (jobs/racc.py)
from jobs.racc import block, build_pool, crc_threshold, fit_racc, get_engine, online_thresholds, split

load_dotenv()
SEED = 42
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON = 5
TAU = 0.01
# cost grid in basis points per position change
COST_GRID = [0.0, 5.0, 10.0, 20.0]
ALPHAS = [0.30, 0.45]
GAMMA = 0.05                     # step size of the online threshold (as in racc_backtest)
LAG = 6                          # trading days until a 5-day label has resolved
# back-test protocol (same as Table 14)
BT_TRAIN_END, BT_TUNE_END, BT_CALIB_END = "2022-12-31", "2023-06-30", "2023-12-31"
# regime protocol: fit before 2020 so every regime below is out-of-sample
RG_TRAIN_END = "2019-12-31"
REGIMES = [
    ("COVID crash (2020)",   "2020-02-01", "2020-06-30"),
    ("Bear market (2022)",   "2022-01-01", "2022-12-31"),
]
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]
A = Path(__file__).resolve().parents[1] / "artifacts"

def perf(pnl):
    pnl = np.asarray(pnl, dtype=float)
    sharpe = 0.0 if pnl.std() < 1e-12 else pnl.mean() / pnl.std() * np.sqrt(252)
    eq = np.cumprod(1.0 + pnl)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return sharpe, mdd, float(eq[-1] - 1.0)


# ─────────────────────────────────────────────────────────────────────
#  A. Cost sensitivity
# ─────────────────────────────────────────────────────────────────────
def cost_sensitivity(data, feat):
    tr, tu, ca, te = split(data, BT_TRAIN_END, BT_TUNE_END, BT_CALIB_END)   # as in racc_backtest.py
    racc_model = fit_racc(tr, tu, feat, lam=0.5)

    def signal(part):
        s, sig, _ = racc_model.score(part)
        return s, sig

    s_ca, _, loss_ca = racc_model.score(ca)
    thr = {al: crc_threshold(np.abs(s_ca), loss_ca, al) for al in ALPHAS}

    # online RACC thresholds (one path over the pooled test days, as in racc_backtest)
    s_te, _, loss_te = racc_model.score(te)
    days = np.unique(te["date"].values)
    day_te = np.searchsorted(days, te["date"].values)
    thr_on = {al: pd.Series(online_thresholds(day_te, np.abs(s_te), loss_te, al, thr[al], GAMMA, LAG)[1], index=days)
              for al in ALPHAS}

    rows = []
    trials: dict[str, np.ndarray] = {}      # each (policy, cost) pair is one trial for the DSR
    for bps in COST_GRID:
        fee = bps / 10000.0
        port = {}
        for sym in SYMBOLS:
            dpart = te[te["symbol"] == sym].sort_values("date")
            if len(dpart) < 50:
                continue
            s, _ = signal(dpart)
            ret = dpart["return_1d"].values
            fwd = np.roll(ret, -1); fwd[-1] = 0.0
            pol = {"Ungated": np.sign(s), "Buy & hold": np.ones_like(s)}
            for al in ALPHAS:
                pol[f"RACC (a={al})"] = np.sign(s) * (np.abs(s) >= thr[al])
            for al in ALPHAS:
                pol[f"Online RACC (a={al})"] = np.sign(s) * (np.abs(s) >= thr_on[al].loc[dpart["date"].values].values)
            for name, pos in pol.items():
                pos = pos.astype(float)
                cost = fee * np.abs(np.diff(np.concatenate([[0.0], pos])))
                port.setdefault(name, []).append(pd.Series(pos * fwd - cost, index=dpart["date"].values))
        for name, lst in port.items():
            m = pd.concat(lst, axis=1, join="inner").mean(axis=1).values   # aligned by date
            sh, mdd, tot = perf(m)
            rows.append(dict(cost_bps=bps, policy=name, sharpe=round(sh, 2),
                             max_drawdown=round(mdd, 3), total_return=round(tot, 3)))
            if name != "Buy & hold":          # the passive benchmark is not a searched trial
                trials[f"{name} @ {bps:.0f}bps"] = m
    df = pd.DataFrame(rows)
    df.to_csv(A / "robustness_costs.csv", index=False)
    print("\n=== A. COST SENSITIVITY (equal-weight portfolio, held-out 2024-2026) ===")
    print(df.pivot(index="policy", columns="cost_bps", values="sharpe").to_string())
    print("\n-- max drawdown --")
    print(df.pivot(index="policy", columns="cost_bps", values="max_drawdown").to_string())
    return df, trials


# ─────────────────────────────────────────────────────────────────────
#  B. Regime stress slices
# ─────────────────────────────────────────────────────────────────────
def regime_slices(data, feat):
    tr = block(data, upto=RG_TRAIN_END)
    post = block(data, after=RG_TRAIN_END, purge=False).copy()
    print(f"\n[regimes] train rows (<= {RG_TRAIN_END}): {len(tr)}   post-2019 rows: {len(post)}")

    et = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                             random_state=SEED, n_jobs=-1).fit(tr[feat].values, tr["target_vol_kd"].values)
    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=SEED))])
    clf.fit(tr[feat].values, tr["target_direction"].values)

    post["vol_pred"] = et.predict(post[feat].values)
    post["p_up"] = clf.predict_proba(post[feat].values)[:, 1]
    # persistence baseline: trailing 20-day realised vol as the forecast
    post["vol_persist"] = post.groupby("symbol")["log_return"].transform(
        lambda s: s.rolling(20, min_periods=20).std(ddof=0))

    covid = (post["date"] >= "2020-02-01") & (post["date"] <= "2020-06-30")
    bear = (post["date"] >= "2022-01-01") & (post["date"] <= "2022-12-31")
    slices = [("COVID crash (2020)", post[covid]),
              ("Bear market (2022)", post[bear]),
              ("Calm / bull (rest)", post[~covid & ~bear])]

    rows = []
    for name, sl in slices:
        sl = sl.dropna(subset=["vol_persist"])
        if sl.empty:
            continue
        rmse_et = float(np.sqrt(mean_squared_error(sl["target_vol_kd"], sl["vol_pred"])))
        rmse_pe = float(np.sqrt(mean_squared_error(sl["target_vol_kd"], sl["vol_persist"])))
        try:
            auc = float(roc_auc_score(sl["target_direction"], sl["p_up"]))
        except Exception:
            auc = float("nan")
        rows.append(dict(regime=name, days=int(sl["date"].nunique()), n_obs=len(sl),
                         mean_realised_vol=round(float(sl["target_vol_kd"].mean()), 5),
                         extratrees_rmse=round(rmse_et, 5),
                         persistence_rmse=round(rmse_pe, 5),
                         ratio_et_vs_persist=round(rmse_et / rmse_pe, 3),
                         direction_auc=round(auc, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(A / "robustness_regimes.csv", index=False)
    print("\n=== B. REGIME STRESS SLICES (models fit on pre-2020 data only) ===")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────
#  C. Training-window-length sensitivity
# ─────────────────────────────────────────────────────────────────────
# Expanding walk-forward uses all available history; a common objection is that
# the result depends on how much history the model is given. We therefore hold
# the evaluation block fixed and vary only the length of the training window.
WINDOWS = [("Expanding (all history)", None), ("Last 1500 days", 1500),
           ("Last 1000 days", 1000), ("Last 500 days", 500)]


def window_sensitivity(data, feat):
    tr_all = block(data, upto=BT_TRAIN_END)
    te = block(data, after=BT_CALIB_END, purge=False)                # fixed 2024-2026 block
    rows = []
    for label, n_days in WINDOWS:
        if n_days is None:
            tr = tr_all
        else:                                                        # last n trading days per instrument
            tr = tr_all.groupby("symbol", group_keys=False).apply(lambda g: g.tail(n_days))
        et = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                                 random_state=SEED, n_jobs=-1).fit(tr[feat].values, tr["target_vol_kd"].values)
        clf = Pipeline([("sc", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                                                  random_state=SEED))]).fit(tr[feat].values, tr["target_direction"].values)
        rmse = float(np.sqrt(mean_squared_error(te["target_vol_kd"], et.predict(te[feat].values))))
        try:
            auc = float(roc_auc_score(te["target_direction"], clf.predict_proba(te[feat].values)[:, 1]))
        except Exception:
            auc = float("nan")
        rows.append(dict(train_window=label, train_rows=int(len(tr)),
                         years=round(len(tr) / len(SYMBOLS) / 252.0, 1),
                         volatility_rmse=round(rmse, 5), direction_auc=round(auc, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(A / "robustness_windows.csv", index=False)
    print("\n=== C. TRAINING-WINDOW SENSITIVITY (evaluation block fixed at 2024-2026) ===")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────
#  D. Deflated Sharpe Ratio (Bailey & Lopez de Prado)
# ─────────────────────────────────────────────────────────────────────
def deflated_sharpe(pnl_by_trial: dict[str, np.ndarray]):
    """Deflates the best observed Sharpe ratio for the number of trials run.

    Because several strategy configurations were evaluated on the same data, the
    best observed Sharpe ratio is upward biased. The deflated Sharpe ratio gives
    the probability that the true Sharpe ratio of the selected configuration is
    positive once that selection bias, the sample length, and the non-normality
    of the returns are taken into account.
    """
    from scipy.stats import norm, skew, kurtosis
    GAMMA = 0.5772156649                       # Euler-Mascheroni constant
    srs = {k: (v.mean() / v.std() if v.std() > 1e-12 else 0.0) for k, v in pnl_by_trial.items()}
    N = len(srs)
    best = max(srs, key=srs.get)
    sr_hat = srs[best]                          # per-period Sharpe of the selected trial
    v = np.asarray(pnl_by_trial[best], dtype=float)
    T = len(v)
    g3, g4 = float(skew(v)), float(kurtosis(v, fisher=False))
    var_sr = float(np.var(list(srs.values()), ddof=1)) if N > 1 else 0.0
    # expected maximum Sharpe under the null of zero true skill across N trials
    sr0 = np.sqrt(var_sr) * ((1 - GAMMA) * norm.ppf(1 - 1.0 / N) +
                             GAMMA * norm.ppf(1 - 1.0 / (N * np.e)))
    denom = np.sqrt(max(1e-12, 1 - g3 * sr_hat + ((g4 - 1) / 4.0) * sr_hat ** 2))
    dsr = float(norm.cdf(((sr_hat - sr0) * np.sqrt(T - 1)) / denom))
    rows = [dict(selected_trial=best, n_trials=N, n_obs=T,
                 sharpe_annualised=round(sr_hat * np.sqrt(252), 3),
                 expected_max_sharpe_null_ann=round(float(sr0) * np.sqrt(252), 3),
                 skewness=round(g3, 3), kurtosis=round(g4, 3),
                 deflated_sharpe_prob=round(dsr, 3))]
    df = pd.DataFrame(rows)
    df.to_csv(A / "robustness_dsr.csv", index=False)
    print("\n=== D. DEFLATED SHARPE RATIO (selection-bias adjustment) ===")
    print(df.to_string(index=False))
    return df


def main():
    engine = get_engine()
    data, feat = build_pool(engine)
    print(f"pool={len(data)} rows, {len(feat)} features")
    _, trials = cost_sensitivity(data, feat)
    regime_slices(data, feat)
    window_sensitivity(data, feat)
    deflated_sharpe(trials)
    print(f"\n[ARTIFACT] {A/'robustness_costs.csv'}, {A/'robustness_regimes.csv'}, "
          f"{A/'robustness_windows.csv'}, {A/'robustness_dsr.csv'}")


if __name__ == "__main__":
    main()
