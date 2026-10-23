"""
RACC — Risk-Aware Conformal Control for the production-rule decision layer.
==========================================================================

A risk-controlled, abstaining decision layer that unifies the system's two
forecasts and gives a distribution-free guarantee on the *monetary* risk of the
trades it chooses to take (rather than on classification error).

Pipeline (all on the canonical 56 features, pooled over the 8 instruments,
chronological train / calibration / test split):

  1. Calibrated direction probability  p_cal = PAV( LogReg P(up) ).
  2. Production-rule belief             q_up  = weighted aggregation of fired
                                                CART rules (purity * log(1+support));
                                                for the single-tree layer this is the
                                                leaf class-probability for a1 (Buy).
  3. Fusion                             pi = lambda * p_cal + (1 - lambda) * q_up.
  4. Volatility-scaled loss            l_i = clip( max(0, -d_i * r_i) / sigma_hat_i, 0, B ),
                                                d_i = sign(pi_i - 0.5), only when acting.
  5. Conformal Risk Control (Angelopoulos et al., 2024): choose the abstention
     threshold t so that  E[L_test] <= alpha  holds under exchangeability.
     online_thresholds() gives the online variant used in the regime stress test
     and the back-test, whose long-run guarantee needs no exchangeability.

The system ACTS (Buy/Sell) only when |pi - 0.5| >= t and ABSTAINS (no-trade,
i.e. the rule action a2) otherwise.

Run:
    python -m jobs.racc
Outputs:
    artifacts/racc_results.csv, artifacts/racc_summary.csv, artifacts/racc_aggregation.csv
    /tmp/figs/fig07_risk_control.png, /tmp/figs/fig08_risk_coverage.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, accuracy_score

from jobs.train_baseline import add_technical_features, build_feature_matrix, compute_targets

load_dotenv()

SEED = 42
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON = 5
TAU = 0.01                      # +/-1% threshold for the a1/a2/a3 rule classes
TRAIN_END = "2022-12-31"        # <= : train the models
TUNE_END = "2023-06-30"         # (TRAIN_END, TUNE_END] : PAV calibrator and fusion weight lambda
CALIB_END = "2023-12-31"        # (TUNE_END, CALIB_END] : conformal threshold only
                                # > CALIB_END : held-out test
B = 3.0                          # loss bound (volatility-normalised downside)
EPS = 1e-4
GRID = np.append(np.linspace(0.0, 0.5, 1001), np.inf)   # fixed CRC threshold grid; |s| <= 1/2
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
FIG = Path(os.environ.get("FIG_DIR", "/tmp/figs"))

_SQL = text("""
    SELECT symbol, date, open, high, low, close, volume,
           return_1d, log_return,
           sma_5, volatility_5, sma_10, volatility_10, sma_20, volatility_20,
           return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
           mkt_return_1d, mkt_log_return, mkt_mom_5, mkt_mom_10, mkt_mom_20, mkt_vol_20,
           vix_level, vix_return_1d, vix_change_1d,
           irx_level, irx_change_1d, tnx_level, tnx_change_1d
    FROM features_daily WHERE symbol = :s ORDER BY date ASC
""")


def get_engine():
    return create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)


def build_pool(engine):
    """Pooled dataset on the canonical 56 features with the targets used by RACC, the
    rule layer and the robustness checks (the single loader of these scripts).

    label_end is the date on which the k-day label of a row becomes known (the date
    k trading days later); block() uses it to purge rows whose label would reach
    into the next block of a chronological split."""
    frames, feat = [], None
    for s in SYMBOLS:
        df = pd.read_sql(_SQL, engine, params={"s": s}, parse_dates=["date"])
        df = df[df["date"] <= pd.to_datetime(END_DATE)].reset_index(drop=True)
        if len(df) < 400:
            continue
        df = add_technical_features(df)
        df, cols = build_feature_matrix(df)
        if df.empty:
            continue
        df["label_end"] = df["date"].shift(-HORIZON)
        df = compute_targets(df, HORIZON)            # target_return_kd, target_direction, target_vol_kd
        feat = feat or cols
        frames.append(df)
    data = (pd.concat(frames, ignore_index=True)
            .sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True))
    # 3-class rule target
    r = data["target_return_kd"].values
    data["cls3"] = np.where(r > TAU, 0, np.where(r < -TAU, 2, 1))   # 0=a1 up,1=a2,2=a3 down
    return data, feat


def block(data, after=None, upto=None, purge=True):
    """Rows dated in (after, upto]. With purge, rows whose label is not yet known on
    `upto` are dropped, so that no label of a training or calibration block uses
    prices from the block that follows it."""
    d = data["date"]
    m = pd.Series(True, index=data.index)
    if after is not None:
        m &= d > pd.to_datetime(after)
    if upto is not None:
        m &= d <= pd.to_datetime(upto)
        if purge:
            m &= data["label_end"] <= pd.to_datetime(upto)
    return data[m]                     # the original index is kept so that rows can be matched


def split(data, train_end=None, tune_end=None, calib_end=None):
    """Training block, tuning block (PAV calibrator and lambda), calibration block
    (conformal threshold only) and held-out test block; every block except the test
    block is purged at its end."""
    train_end, tune_end, calib_end = train_end or TRAIN_END, tune_end or TUNE_END, calib_end or CALIB_END
    return (block(data, upto=train_end), block(data, after=train_end, upto=tune_end),
            block(data, after=tune_end, upto=calib_end), block(data, after=calib_end, purge=False))


def fit_racc(tr, tu, feat, lam=None):
    """The RACC scoring function, fitted without the calibration block.

    The logistic model, the rule tree (q = leaf probability of a1) and the ExtraTrees
    volatility model are fitted on the training block; the isotonic PAV calibrator
    and, when lam is None, the fusion weight are fitted on the tuning block. The
    conformal threshold is then computed on a separate calibration block, so the
    score is fixed before the calibration data are seen, as Proposition 1 requires.
    Returns an object with .lam, .parts(part) -> (p_cal, q, sigma) and
    .score(part) -> (s, sigma, loss)."""
    from types import SimpleNamespace
    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                                              random_state=SEED))]).fit(tr[feat].values, tr["target_direction"].values)
    iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(tu[feat].values)[:, 1],
                                                       tu["target_direction"].values)
    tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=80, random_state=SEED).fit(
        tr[feat].values, tr["cls3"].values)
    a1 = list(tree.classes_).index(0)
    et = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                             random_state=SEED, n_jobs=-1).fit(tr[feat].values, tr["target_vol_kd"].values)

    def parts(part):
        X = part[feat].values
        return iso.predict(clf.predict_proba(X)[:, 1]), tree.predict_proba(X)[:, a1], et.predict(X)

    tune_auc = None
    if lam is None:                                   # fusion weight: best AUC on the tuning block
        p_tu, q_tu, _ = parts(tu)
        lam, tune_auc = 0.5, -1.0
        for cand in (0.0, 0.25, 0.5, 0.75, 1.0):
            try:
                au = roc_auc_score(tu["target_direction"].values, cand * p_tu + (1 - cand) * q_tu)
            except ValueError:
                au = 0.5
            if au > tune_auc:
                lam, tune_auc = cand, au

    def score(part, sig=None):
        p, q, s_et = parts(part)
        sig = s_et if sig is None else sig
        s = lam * p + (1 - lam) * q - 0.5
        d = np.sign(s); d[d == 0] = 1
        loss = np.clip(np.maximum(0.0, -d * part["target_return_kd"].values) / np.maximum(sig, EPS), 0, B)
        return s, sig, loss

    return SimpleNamespace(lam=lam, tune_auc=tune_auc, parts=parts, score=score, tree=tree, a1=a1)


def crc_threshold(s_abs, loss, alpha, grid=None):
    """Conformal Risk Control: smallest threshold t of a fixed grid with the
    finite-sample upper bound (sum_i loss_i 1[|s_i| >= t] + B) / (n + 1) <= alpha.

    The loss is non-increasing in t. The grid does not depend on the data (GRID by
    default) and ends at +inf, the threshold that abstains on every decision and has
    zero loss; these are the conditions of the conformal risk control theorem
    (Angelopoulos et al., 2024) used in Proposition 1."""
    grid = GRID if grid is None else np.asarray(grid, dtype=float)
    s_abs, loss = np.asarray(s_abs, dtype=float), np.asarray(loss, dtype=float)
    n = len(loss)
    order = np.argsort(s_abs, kind="mergesort")
    s_sorted = s_abs[order]
    top_sum = np.concatenate([[0.0], np.cumsum(loss[order][::-1])])   # loss sum of the j largest scores
    n_act = n - np.searchsorted(s_sorted, grid, side="left")         # number of scores >= t
    feasible = (top_sum[n_act] + B) / (n + 1) <= alpha
    return float(grid[np.argmax(feasible)]) if feasible.any() else np.inf


def online_thresholds(day, s_abs, loss, alpha, t0, gamma=0.05, lag=6):
    """Online RACC: the threshold is updated after every trading day.

    day[i] is the trading-day index (0, 1, ...) of decision i, s_abs its |pi - 0.5|
    and loss its loss if the system acts. Before the decisions of day j are made,
    the threshold moves by gamma * (realised risk of day j - lag - alpha), where the
    realised risk of a day is the mean over its decisions of loss * 1[act]. With
    lag = 6 the 5-day label of day j - lag has resolved before day j, so only past
    information is used. For any sequence of scores and losses in [0, B], the
    average daily risk over T days is at most alpha + ((M - t0) / gamma + 2 lag B) / T,
    where M = 1/2 >= max |s| and t0 <= M (Proposition 2 of the paper, following
    Feldman et al., 2023); no exchangeability is assumed. Returns the act mask and
    the threshold of each day.
    """
    n_days = int(day.max()) + 1
    rows_of = [np.flatnonzero(day == j) for j in range(n_days)]
    act = np.zeros(len(s_abs), dtype=bool)
    thr_day = np.empty(n_days)
    thr = min(float(t0), 0.5)                       # t0 <= M = 1/2, as Proposition 2 requires
    for j in range(n_days):
        jr = j - lag
        if jr >= 0 and len(rows_of[jr]):
            rb = rows_of[jr]
            thr += gamma * (float(np.mean(loss[rb] * act[rb])) - alpha)
        thr_day[j] = thr
        act[rows_of[j]] = s_abs[rows_of[j]] >= thr
    return act, thr_day


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    data, feat = build_pool(engine)
    tr, tu, ca, te = split(data)
    print(f"pool={len(data)}  train={len(tr)}  tune={len(tu)}  calib={len(ca)}  test={len(te)}  features={len(feat)}")

    m = fit_racc(tr, tu, feat)
    lam = m.lam
    print(f"fusion lambda={lam} (tuning-block AUC={m.tune_auc:.3f})")
    _, qw_te, _ = m.parts(te)
    leaf_cls = m.tree.classes_[np.argmax(m.tree.predict_proba(te[feat].values), axis=1)]
    qm_te = (leaf_cls == 0).astype(float)                                          # majority (hard)
    pcal_te = m.parts(te)[0]

    s_ca, _, loss_ca = m.score(ca)
    s_te, _, loss_te = m.score(te)
    d_te = np.sign(s_te); d_te[d_te == 0] = 1
    up_te = te["target_direction"].values

    # --- sweep target risk alpha: CRC on calib, evaluate on test ---
    rows = []
    for alpha in np.round(np.arange(0.05, 0.81, 0.05), 2):
        t = crc_threshold(np.abs(s_ca), loss_ca, alpha)
        act = np.abs(s_te) >= t
        cov = float(np.mean(act))
        realized = float(np.mean(loss_te * act))
        # directional accuracy on the acted subset (predicted up/down vs realised)
        if act.sum() > 0:
            pred_up = (d_te[act] > 0).astype(int)
            acc = float(accuracy_score(up_te[act], pred_up))
        else:
            acc = float("nan")
        # the same threshold on the calibration block, and the share of short positions among
        # the decisions taken: a shift between the two blocks breaks exchangeability
        act_ca = np.abs(s_ca) >= t
        short = lambda s_, a_: round(float(np.mean(s_[a_] < 0)), 3) if a_.any() else None
        rows.append(dict(alpha=alpha, threshold=round(float(t), 4), coverage=round(cov, 3),
                         realized_risk=round(realized, 4), guarantee_holds=bool(realized <= alpha),
                         acc_acted=round(acc, 3) if act.sum() else None,
                         coverage_calib=round(float(act_ca.mean()), 3),
                         risk_calib=round(float(np.mean(loss_ca * act_ca)), 4),
                         short_share_calib=short(s_ca, act_ca), short_share_test=short(s_te, act)))
    res = pd.DataFrame(rows)
    res.to_csv(A / "racc_results.csv", index=False)
    acc_all = accuracy_score(up_te, (d_te > 0).astype(int))
    pd.DataFrame([dict(lam=lam, tune_auc=round(float(m.tune_auc), 4) if m.tune_auc is not None else None,
                       n_train=len(tr), n_tune=len(tu), n_calib=len(ca), n_test=len(te),
                       acc_unconditional=round(float(acc_all), 4),
                       score_values_calib=int(len(np.unique(np.round(np.abs(s_ca), 10)))),
                       up_rate_calib=round(float(ca["target_direction"].mean()), 3),
                       up_rate_test=round(float(te["target_direction"].mean()), 3))]).to_csv(A / "racc_summary.csv", index=False)
    print(res.to_string(index=False))
    print(f"\nUnconditional directional accuracy (act always): {acc_all:.3f}")
    print(f"Guarantee held on test for {res.guarantee_holds.mean()*100:.0f}% of alpha levels")

    # --- weighted vs majority rule aggregation (selective AUC of pi for up) ---
    agg = []
    for name, q in [("weighted", qw_te), ("majority", qm_te)]:
        pi = lam * pcal_te + (1 - lam) * q
        try:
            au = roc_auc_score(up_te, pi)
        except Exception:
            au = 0.5
        agg.append(dict(aggregation=name, fused_auc=round(float(au), 4)))
    pd.DataFrame(agg).to_csv(A / "racc_aggregation.csv", index=False)
    print("\nRule aggregation (fused AUC):", agg)

    # --- Fig 12: risk-control curve (target vs realised; must lie on/below diagonal) ---
    plt.figure(figsize=(5.6, 5.2))
    plt.plot([0, 0.8], [0, 0.8], "--", color="gray", lw=1, label="target = realised")
    plt.plot(res.alpha, res.realized_risk, "o-", color="#1f77b4", lw=1.8, label="RACC realised risk")
    plt.fill_between(res.alpha, 0, res.alpha, color="#2ca02c", alpha=0.08, label="guarantee satisfied")
    plt.xlabel("Target risk level α (volatility-scaled loss)")
    plt.ylabel("Realised risk on held-out test")
    plt.legend(fontsize=8, loc="upper left"); plt.tight_layout()
    plt.savefig(FIG / "fig07_risk_control.png", dpi=int(os.environ.get("FIG_DPI", "300")), facecolor="white"); plt.close()

    # --- Fig 13: risk-coverage frontier (monotone; the dialable risk budget) ---
    rc = res.sort_values("coverage")
    fig, ax1 = plt.subplots(figsize=(6.0, 5.0))
    ax1.plot(rc.coverage, rc.realized_risk, "o-", color="#1f77b4", lw=1.9)
    # one label per distinct operating point, naming the range of alpha that selects it;
    # points crowded near zero coverage are labelled from the empty lower-right corner
    pts = []
    for (cov, risk), grp in rc.groupby(["coverage", "realized_risk"], sort=False):
        lo, hi = grp["alpha"].min(), grp["alpha"].max()
        pts.append((cov, risk, f"α = {lo:.2f}" if lo == hi else f"α = {lo:.2f}–{hi:.2f}"))
    pts.sort()
    x_max = max(c for c, _, _ in pts)
    crowded = [q for q in pts if q[0] < 0.15 * x_max]
    for k, (cov, risk, lab) in enumerate(crowded):
        ax1.annotate(lab, (cov, risk), xytext=(0.35, 0.05 + 0.07 * k), textcoords="axes fraction",
                     fontsize=7.5, color="#444", va="center",
                     arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    for cov, risk, lab in pts[len(crowded):]:
        right = cov > 0.8 * x_max
        ax1.annotate(lab, (cov, risk), textcoords="offset points", xytext=(-8, 8) if right else (6, -10),
                     ha="right" if right else "left", fontsize=7.5, color="#444")
    ax1.set_xlabel("Coverage (fraction of days the system trades)")
    ax1.set_ylabel("Realised volatility-scaled risk on test")
    ax1.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(FIG / "fig08_risk_coverage.png", dpi=int(os.environ.get("FIG_DPI", "300")), facecolor="white"); plt.close()

    print(f"\n[ARTIFACT] {A/'racc_results.csv'}, {A/'racc_aggregation.csv'}")
    print(f"[FIG] {FIG/'fig07_risk_control.png'}, {FIG/'fig08_risk_coverage.png'}")


if __name__ == "__main__":
    main()
