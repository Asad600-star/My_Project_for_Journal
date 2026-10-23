"""
Stress test of the RACC risk guarantee across market regimes.
=============================================================

Proposition 1 is a finite-sample bound that holds under exchangeability of the
calibration and test points. Financial series are not exchangeable, so the
relevant question is empirical: how much does the realised risk move above the
target when the test period is a structural break?

For each scenario the models are fit on data up to the year before the
calibration year; the first half of that year fits the PAV calibrator and lambda,
the conformal threshold is calibrated on its second half, and the realised
volatility-scaled risk is measured on the test window. The protocol, the loss and the fusion weight
lambda are identical to jobs/racc.py.

Three threshold policies are compared on every scenario:
  - static   : one conformal threshold from the calibration year (Proposition 1);
  - adaptive : the threshold is re-estimated every REFRESH trading days from the
               most recent WINDOW calendar days of resolved (score, loss) pairs,
               following the adaptive-conformal idea of Gibbs & Candes (2021).
               Only information available at the decision date is used: a pair
               enters the window once its k-day label has resolved (its label_end
               date, k trading days later, precedes the decision date).
               The models themselves are not refitted; only the threshold moves.
  - online   : the threshold is updated every trading day by GAMMA times the gap
               between the realised risk of the latest resolved day and alpha
               (online risk control, Feldman et al., 2023), starting from the
               static threshold. Its long-run guarantee holds for any sequence.

    scenario              train <=      calibration     test window
    Reference (Table 13)  2022-12-31    2023            2024-01-01 .. END_DATE
    COVID crash           2018-12-31    2019            2020-02-01 .. 2020-06-30
    2022 bear market      2020-12-31    2021            2022-01-01 .. 2022-12-31

Run:
    python -m jobs.racc_regime_stress                   # loss scaled by ExtraTrees
    RACC_SIGMA=garch python -m jobs.racc_regime_stress  # loss scaled by GARCH(1,1)
Output: artifacts/racc_regime_stress.csv; with RACC_SIGMA or RACC_GAMMA set,
        racc_regime_stress_garch.csv or racc_regime_stress_gamma<value>.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from jobs.econometric_baselines import garch_daily
from jobs.racc import build_pool, block, split, fit_racc, get_engine, crc_threshold, online_thresholds, B, END_DATE, A

# Volatility forecast used to scale the loss: "et" (ExtraTrees on the 56 features,
# as in jobs/racc.py) or "garch" (per-instrument GARCH(1,1) fitted on the training
# returns and filtered day by day, mapped to the target scale on the training rows).
SIGMA = os.environ.get("RACC_SIGMA", "et").lower()

SCENARIOS = [
    ("Reference 2024-2026 (Table 13)", "2022-12-31", "2023-12-31", "2024-01-01", END_DATE),
    ("COVID crash 2020",              "2018-12-31", "2019-12-31", "2020-02-01", "2020-06-30"),
    ("Bear market 2022",              "2020-12-31", "2021-12-31", "2022-01-01", "2022-12-31"),
]
ALPHAS = np.round(np.arange(0.05, 0.81, 0.05), 2)
REPORT = (0.30, 0.45)
WINDOW_DAYS = int(os.environ.get("RACC_WINDOW_DAYS", "365"))   # history of the adaptive threshold (calendar days)
REFRESH = 21               # trading days between threshold updates (about one month)
GAMMA = float(os.environ.get("RACC_GAMMA", "0.05"))   # step size of the online threshold
LAG = 6                    # trading days until the 5-day label of a decision is known


def garch_sigma(data, train_end, ca, te):
    """Per-instrument GARCH(1,1) forecasts of the k-day volatility for the calibration
    and test rows. Parameters and the linear map to the target scale use only rows
    dated on or before train_end."""
    sig = pd.Series(np.nan, index=data.index)
    for _, g in data.groupby("symbol"):
        g = g.sort_values("date")
        trn = (g["date"] <= pd.to_datetime(train_end)).values
        lab = trn & (g["label_end"] <= pd.to_datetime(train_end)).values      # purged labels
        r = g["log_return"].values
        n_tr = int(trn.sum())
        f = garch_daily(r[:n_tr], r, n_tr)
        cal = LinearRegression().fit(f[lab].reshape(-1, 1), g["target_vol_kd"].values[lab])
        sig.loc[g.index] = cal.predict(f.reshape(-1, 1))
    return sig.loc[ca.index].values, sig.loc[te.index].values


def run_scenario(data, feat, train_end, calib_end, test_start, test_end):
    # the calibration year is split: first half to fit the PAV calibrator and lambda,
    # second half for the conformal threshold (same protocol as jobs/racc.py)
    tune_end = f"{pd.to_datetime(calib_end).year}-06-30"
    tr, tu, ca, _ = split(data, train_end, tune_end, calib_end)
    te = block(data, after=pd.to_datetime(test_start) - pd.Timedelta(days=1), upto=test_end, purge=False)
    m = fit_racc(tr, tu, feat)
    lam = m.lam
    sig_ca = sig_te = None
    if SIGMA == "garch":
        sig_ca, sig_te = garch_sigma(data, train_end, ca, te)
    s_ca, _, l_ca = m.score(ca, sig_ca)
    s_te, _, l_te = m.score(te, sig_te)

    # history pool for the adaptive policy: calibration year followed by the test window
    hist = pd.DataFrame({"date": np.concatenate([ca["date"].values, te["date"].values]),
                         "label_end": np.concatenate([ca["label_end"].values, te["label_end"].values]),
                         "s": np.abs(np.concatenate([s_ca, s_te])),
                         "loss": np.concatenate([l_ca, l_te])}).sort_values("date").reset_index(drop=True)
    te_dates = te["date"].values
    refresh_dates = np.unique(te_dates)[::REFRESH]

    def adaptive_act(al):
        act = np.zeros(len(te), dtype=bool)
        for j, d0 in enumerate(refresh_dates):
            d1 = refresh_dates[j + 1] if j + 1 < len(refresh_dates) else np.datetime64("2100-01-01")
            lo = d0 - np.timedelta64(WINDOW_DAYS, "D")
            w = hist[(hist["date"] > lo) & (hist["label_end"] < d0)]   # labels resolved before d0
            thr = crc_threshold(w["s"].values, w["loss"].values, al)
            blk = (te_dates >= d0) & (te_dates < d1)
            act[blk] = np.abs(s_te[blk]) >= thr
        return act

    # online policy (jobs.racc.online_thresholds): the threshold moves every trading
    # day by GAMMA times the gap between the realised risk of a resolved day and alpha
    day_of = np.searchsorted(np.unique(te_dates), te_dates)

    def online_act(al, t0):
        return online_thresholds(day_of, np.abs(s_te), l_te, al, t0, GAMMA, LAG)[0]

    rows = []
    for al in ALPHAS:
        t = crc_threshold(np.abs(s_ca), l_ca, al)
        a_st = np.abs(s_te) >= t
        a_ad = adaptive_act(al)
        a_on = online_act(al, t)
        daily_on = np.bincount(day_of, weights=l_te * a_on) / np.bincount(day_of)
        n_days = len(daily_on)
        bound = al + ((0.5 - min(t, 0.5)) / GAMMA + 2 * LAG * B) / n_days   # Proposition 2, M = 1/2
        assert daily_on.mean() <= bound + 1e-12
        rows.append(dict(alpha=float(al),
                         coverage=float(a_st.mean()), realised=float(np.mean(l_te * a_st)),
                         coverage_ad=float(a_ad.mean()), realised_ad=float(np.mean(l_te * a_ad)),
                         coverage_on=float(a_on.mean()), realised_on=float(np.mean(l_te * a_on)),
                         realised_on_daily=float(daily_on.mean()), bound_on=float(bound), n_days=n_days))
    g = pd.DataFrame(rows)
    g["excess"] = g["realised"] - g["alpha"]
    g["excess_ad"] = g["realised_ad"] - g["alpha"]
    g["excess_on"] = g["realised_on"] - g["alpha"]
    return lam, len(tr), len(ca), len(te), float(te["target_vol_kd"].mean()), g


def main():
    data, feat = build_pool(get_engine())
    out = []
    for name, tr_end, ca_end, te_s, te_e in SCENARIOS:
        lam, ntr, nca, nte, vol, g = run_scenario(data, feat, tr_end, ca_end, te_s, te_e)
        row = dict(scenario=name, lam=lam, n_train=ntr, n_calib=nca, n_test=nte,
                   mean_realised_vol=round(vol, 5),
                   held_static=round(float((g["excess"] <= 0).mean()), 2),
                   held_adaptive=round(float((g["excess_ad"] <= 0).mean()), 2),
                   held_online=round(float((g["excess_on"] <= 0).mean()), 2),
                   max_excess_static=round(float(g["excess"].max()), 4),
                   max_excess_adaptive=round(float(g["excess_ad"].max()), 4),
                   max_excess_online=round(float(g["excess_on"].max()), 4),
                   mean_abs_gap_online=round(float(g["excess_on"].abs().mean()), 4),
                   n_days=int(g["n_days"].iloc[0]),
                   max_bound_excess_online=round(float((g["bound_on"] - g["alpha"]).max()), 4),
                   max_excess_online_daily=round(float((g["realised_on_daily"] - g["alpha"]).max()), 4))
        for al in REPORT:
            r = g[np.isclose(g["alpha"], al)].iloc[0]
            row[f"risk_static@{al}"] = round(r["realised"], 4)
            row[f"risk_adaptive@{al}"] = round(r["realised_ad"], 4)
            row[f"cov_static@{al}"] = round(r["coverage"], 3)
            row[f"cov_adaptive@{al}"] = round(r["coverage_ad"], 3)
            row[f"risk_online@{al}"] = round(r["realised_on"], 4)
            row[f"cov_online@{al}"] = round(r["coverage_on"], 3)
        out.append(row)
        print(f"\n--- {name} (lambda={lam}, train={ntr}, calib={nca}, test={nte}) ---")
        print(g.round(4).to_string(index=False))
    res = pd.DataFrame(out)
    name = "racc_regime_stress.csv" if SIGMA == "et" else f"racc_regime_stress_{SIGMA}.csv"
    if "RACC_GAMMA" in os.environ:
        name = name.replace(".csv", f"_gamma{GAMMA:g}.csv")
    if "RACC_WINDOW_DAYS" in os.environ:
        name = name.replace(".csv", f"_window{WINDOW_DAYS}.csv")
    res.to_csv(A / name, index=False)
    print("\n=== RACC GUARANTEE UNDER REGIME BREAKS ===")
    print(res.to_string(index=False))
    print(f"\n[ARTIFACT] {A / name}")


if __name__ == "__main__":
    main()
