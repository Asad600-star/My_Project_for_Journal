"""
Per-instrument result figures and their bootstrap confidence intervals.
=======================================================================

Regenerates, from the walk-forward metric files, the three result figures that
summarise Section 5 of the paper:

  fig10_auc_by_symbol.png    test AUC of the seven classical direction models per instrument
  fig11_rmse_by_symbol.png   test RMSE of the eight volatility models per instrument
  fig12_bootstrap_ci.png     95% bootstrap intervals for (a) the best direction model and
                           (b) the volatility model of the system, per instrument

The intervals are non-parametric percentile bootstrap intervals of the mean over the
walk-forward folds (B = 1000 resamples of the folds, seed 42). For panel (a) the best
model is chosen on the same test folds, so its interval is optimistic and is reported
as such in the paper.

Run:
    python -m jobs.make_paper_figures
Outputs:
    artifacts/bootstrap_ci.csv and three PNG files in FIG_DIR (default /tmp/figs)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
B = 1000
DPI = int(os.environ.get("FIG_DPI", "300"))
A = Path(__file__).resolve().parents[1] / "artifacts"
FIG = Path(os.environ.get("FIG_DIR", "/tmp/figs"))
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]
LABEL = {"^GSPC": "S&P500", "^IXIC": "NASDAQ", "^DJI": "DJI", "^RUT": "RUT"}
DIR_MODELS = [("HYBRID_VOTING", "H-VOTING"), ("HYBRID_STACK", "H-STACK"), ("LOGREG", "LOGREG"),
              ("RF", "RF"), ("XGB", "XGB"), ("LGBM", "LGBM"), ("HGB", "HGB")]
VOL_MODELS = [("HYBRID_ECON_ET", "H-ECON-ET"), ("ECON", "ECON"), ("GJR_GARCH", "GJR-GARCH"), ("GARCH", "GARCH"),
              ("HAR_RG", "HAR-RG"), ("HAR", "HAR"), ("EXTRATREES", "ExtraTrees"), ("HYBRID_STACK_REG", "H-STK-R")]
VOL_SYSTEM = "HYBRID_ECON_ET"      # volatility model of the system (jobs/volatility_models.py)
COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#2ca02c"]
plt.rcParams.update({"font.family": "serif", "font.size": 11})


def load(task: str) -> pd.DataFrame:
    df = pd.read_csv(A / f"metrics_walk_{task}_k5.csv")
    return df[(df["split"] == "test") & (df["symbol"].isin(SYMBOLS))]


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(B, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def grouped_bars(table: pd.DataFrame, models, ylabel: str, fname: str, ylim=None, hline=None):
    x = np.arange(len(SYMBOLS)); w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    if hline is not None:
        ax.axhline(hline, ls="--", color="gray", lw=1.1, label="no-skill (0.5)")
    for k, (code, lab) in enumerate(models):
        ax.bar(x - 0.4 + w * (k + 0.5), table[code].values, w, color=COLORS[k], label=lab)
    ax.set_xticks(x, [LABEL.get(s, s) for s in SYMBOLS])
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=DPI, facecolor="white"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dr, vo = load("direction"), load("volatility")
    auc = dr.groupby(["symbol", "model"])["auc"].mean().unstack().reindex(SYMBOLS)
    rmse = vo.groupby(["symbol", "model"])["rmse"].mean().unstack().reindex(SYMBOLS)

    grouped_bars(auc, DIR_MODELS, "AUC (test, mean over 22 folds)", "fig10_auc_by_symbol.png",
                 ylim=(0.40, 0.65), hline=0.5)
    grouped_bars(rmse, VOL_MODELS, "RMSE (test, mean over 22 folds)", "fig11_rmse_by_symbol.png")

    rows = []
    for s in SYMBOLS:
        best = auc.loc[s, [c for c, _ in DIR_MODELS]].idxmax()
        a = dr[(dr["symbol"] == s) & (dr["model"] == best)]["auc"].to_numpy()
        r = vo[(vo["symbol"] == s) & (vo["model"] == VOL_SYSTEM)]["rmse"].to_numpy()
        a_lo, a_hi = boot_ci(a, rng)
        r_lo, r_hi = boot_ci(r, rng)
        rows.append(dict(symbol=s, best_direction_model=best, n_folds=len(a),
                         auc_mean=round(a.mean(), 4), auc_ci_low=round(a_lo, 4), auc_ci_high=round(a_hi, 4),
                         vol_model=VOL_SYSTEM, vol_rmse=round(r.mean(), 5),
                         rmse_ci_low=round(r_lo, 5), rmse_ci_high=round(r_hi, 5)))
    ci = pd.DataFrame(rows)
    ci.to_csv(A / "bootstrap_ci.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = np.arange(len(SYMBOLS)); labels = [LABEL.get(s, s) for s in SYMBOLS]
    ax1.errorbar(x, ci.auc_mean, yerr=[ci.auc_mean - ci.auc_ci_low, ci.auc_ci_high - ci.auc_mean],
                 fmt="o", color="#1f77b4", capsize=4, lw=1.6)
    ax1.axhline(0.5, ls="--", color="gray", lw=1)
    ax1.set_ylabel("AUC"); ax1.set_xticks(x, labels, rotation=45, ha="right")
    ax1.text(0.02, 0.97, "(a) best direction model", transform=ax1.transAxes, va="top")
    ax2.errorbar(x, ci.vol_rmse, yerr=[ci.vol_rmse - ci.rmse_ci_low, ci.rmse_ci_high - ci.vol_rmse],
                 fmt="s", color="#2ca02c", capsize=4, lw=1.6)
    ax2.set_ylabel("RMSE"); ax2.set_xticks(x, labels, rotation=45, ha="right")
    ax2.text(0.98, 0.97, "(b) H-ECON-ET volatility", transform=ax2.transAxes, va="top", ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig12_bootstrap_ci.png", dpi=DPI, facecolor="white"); plt.close(fig)

    print(ci.to_string(index=False))
    print(f"\nAUC intervals containing 0.5: {int(((ci.auc_ci_low <= 0.5) & (ci.auc_ci_high >= 0.5)).sum())}/8")
    print(f"[ARTIFACT] {A/'bootstrap_ci.csv'}  [FIG] {FIG}")


if __name__ == "__main__":
    main()
