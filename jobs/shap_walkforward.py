"""
SHAP attributions of the direction ensemble over the walk-forward folds.
========================================================================

For every instrument and every walk-forward fold, the three gradient-boosting
members of HYBRID_VOTING (HGB, XGB and LGBM, which carry 7 of the 9 voting
weights) are refitted on the training block exactly as in train_baseline.py.
Exact TreeSHAP values on the log-odds scale are computed for the rows of the
test block and combined with the voting weights (2, 2 and 3). The importance of
a feature is its mean absolute combined SHAP value over the test rows, averaged
over the folds of an instrument and then over the eight instruments.

The logistic-regression and random-forest members are left out because their
SHAP values are on a different scale (linear log-odds and probability), which
would make the combination inconsistent.

Run:
    TRAIN_START_DATE=2015-01-01 python -m jobs.shap_walkforward
Outputs:
    artifacts/shap_walkforward_direction.csv (feature, per-instrument and mean |SHAP|)
    FIG_DIR/fig05_shap_top15.png
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sqlalchemy import create_engine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jobs.econometric_baselines import prepare, SYMBOLS, K, A
from jobs.train_baseline import (
    build_feature_matrix, compute_targets, make_models,
    WF_MIN_TRAIN_ROWS, WF_VAL_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
)

warnings.filterwarnings("ignore")
FIG = Path(os.environ.get("FIG_DIR", "/tmp/figs"))
WEIGHTS = {"HGB": 2.0, "XGB": 2.0, "LGBM": 3.0}     # voting weights of these members
TOP = 15


def main():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    members = {name: est for name, est, _ in make_models("direction") if name in WEIGHTS}
    wsum = sum(WEIGHTS.values())
    per_symbol = {}
    for sym in SYMBOLS:
        df2 = prepare(engine, sym)
        _, feat = build_feature_matrix(df2.copy())
        feat = [c for c in feat if c in df2.columns]
        n = len(df2)
        i, folds = WF_MIN_TRAIN_ROWS, []
        while i <= n - (WF_VAL_DAYS + WF_TEST_DAYS):
            tr = compute_targets(df2.iloc[:i].copy(), K)
            va = compute_targets(df2.iloc[i:i + WF_VAL_DAYS].copy(), K)
            te = compute_targets(df2.iloc[i + WF_VAL_DAYS:i + WF_VAL_DAYS + WF_TEST_DAYS].copy(), K)
            if len(tr) < 500 or len(va) < 80 or len(te) < 80:
                i += WF_STEP_DAYS
                continue
            Xtr, ytr, Xte = tr[feat].values, tr["target_direction"].values, te[feat].values
            combined = np.zeros_like(Xte, dtype=float)
            for name, est in members.items():
                m = clone(est).fit(Xtr, ytr)
                sv = shap.TreeExplainer(m).shap_values(Xte)
                sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
                if sv.ndim == 3:                       # (rows, features, classes)
                    sv = sv[:, :, 1]
                combined += WEIGHTS[name] / wsum * sv
            folds.append(np.abs(combined).mean(axis=0))
            i += WF_STEP_DAYS
        per_symbol[sym] = pd.Series(np.mean(folds, axis=0), index=feat)
        print(f"{sym}: {len(folds)} folds", flush=True)

    tab = pd.DataFrame(per_symbol)
    tab["mean_abs_shap"] = tab.mean(axis=1)
    tab = tab.sort_values("mean_abs_shap", ascending=False)
    tab.index.name = "feature"
    tab.to_csv(A / "shap_walkforward_direction.csv")
    print(tab["mean_abs_shap"].head(TOP).round(4).to_string())

    top = tab["mean_abs_shap"].head(TOP)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.barh(top.index, top.values, color="#1f77b4")
    ax.set_xlabel("Mean |SHAP| (log-odds, weighted boosting members)")
    for y, v in enumerate(top.values):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8)
    ax.margins(x=0.12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig05_shap_top15.png", dpi=int(os.environ.get("FIG_DPI", "300")), facecolor="white")
    plt.close(fig)
    print(f"[ARTIFACT] {A / 'shap_walkforward_direction.csv'}  [FIG] {FIG / 'fig05_shap_top15.png'}")


if __name__ == "__main__":
    main()
