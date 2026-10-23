"""
Cross-model statistical significance tests (Demsar 2006 protocol).
==================================================================

Treats each instrument as a "dataset" and each model as a "method", then runs:
  - Friedman test          : is there any significant difference among models?
  - Nemenyi post-hoc        : critical-difference (CD) ranking diagram.
  - Wilcoxon signed-rank     : pairwise comparison of the top model vs each baseline.

Direction task  : metric = AUC  (higher is better) over 8 instruments.
Volatility task : metric = RMSE (lower is better)  over 8 instruments.

Run:
    python -m jobs.statistical_tests

Outputs:
    artifacts/statistical_tests.csv
    /tmp/figs/fig03_cd_direction.png
    /tmp/figs/fig04_cd_volatility.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path("artifacts")
FIG_DIR = Path(os.environ.get("FIG_DIR", "/tmp/figs"))
TARGET = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]

# Nemenyi critical values q_alpha (alpha = 0.05), infinite degrees of freedom,
# indexed by the number of compared models k: k <= 10 from Demsar (2006, Table 5),
# k = 11..15 computed in the same way, as the studentized range quantile q(k, inf) / sqrt(2).
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
       7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}

DIR_MODELS = ["HYBRID_VOTING", "HYBRID_STACK", "LSTM", "GRU", "TRANSFORMER",
              "LOGREG", "RF", "XGB", "LGBM", "HGB"]
VOL_MODELS = ["HYBRID_ECON_ET", "ECON", "GJR_GARCH", "GARCH", "HAR_RG", "HAR",
              "EXTRATREES", "HYBRID_STACK_REG", "XGB", "LGBM", "HGB",
              "LSTM", "GRU", "TRANSFORMER", "NBEATSX"]

# names used in the paper
DISPLAY = {"HYBRID_ECON_ET": "H-ECON-ET", "HYBRID_GARCH_ET": "H-GARCH-ET", "GJR_GARCH": "GJR-GARCH",
           "HAR_RG": "HAR-RG", "EXTRATREES": "ExtraTrees", "NBEATSX": "N-BEATSx",
           "TRANSFORMER": "Transformer", "LOGREG": "LogReg"}

rows_out = []


def build_matrix(csv: str, models: list[str], metric: str) -> pd.DataFrame:
    """Returns an (instrument x model) matrix of mean-over-folds metric values."""
    d = pd.read_csv(A / csv)
    d = d[(d["split"] == "test") & (d["symbol"].isin(TARGET)) & (d["model"].isin(models))]
    m = d.groupby(["symbol", "model"])[metric].mean().unstack("model")
    m = m.reindex(index=[s for s in TARGET if s in m.index], columns=models)
    return m.dropna(axis=0, how="any")


def nemenyi_cd(k: int, n: int) -> float:
    q = Q05[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def cd_diagram(avg_ranks: pd.Series, cd: float, title: str, out: Path, lower_is_better_note: str):
    """Draws a critical-difference diagram (lower average rank = better)."""
    models = list(avg_ranks.index)
    ranks = avg_ranks.values
    m = len(models)
    lo, hi = 1, m
    fig, ax = plt.subplots(figsize=(9.0, 0.62 * m + 2.6))
    ax.set_xlim(lo - 3.2, hi + 0.6)
    ax.set_ylim(0, m + 4.2)
    ax.axis("off")

    yaxis = m + 1.2                      # rank axis line
    # title (top), then CD bar, then axis — no overlaps
    ax.text((lo + hi) / 2, m + 3.4, f"Average rank ({lower_is_better_note})",
            ha="center", fontsize=8.5, color="#444")

    # axis with ticks
    ax.plot([lo, hi], [yaxis, yaxis], "k-", lw=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [yaxis, yaxis + 0.15], "k-", lw=1.0)
        ax.text(r, yaxis + 0.42, str(r), ha="center", va="center", fontsize=8)

    # CD bar (between axis and title)
    cy = m + 2.6
    ax.plot([lo, lo + cd], [cy, cy], "k-", lw=2.0)
    ax.plot([lo, lo], [cy - 0.12, cy + 0.12], "k-", lw=1.2)
    ax.plot([lo + cd, lo + cd], [cy - 0.12, cy + 0.12], "k-", lw=1.2)
    ax.text(lo + cd / 2, cy + 0.22, f"CD = {cd:.2f}", ha="center", fontsize=8.5)

    # methods hanging from the axis
    order = np.argsort(ranks)
    for i, idx in enumerate(order):
        y = m - i
        rk = ranks[idx]
        ax.plot([rk, rk], [yaxis, y], "k-", lw=0.8)
        ax.plot([rk, lo - 0.3], [y, y], "k-", lw=0.8)
        ax.text(lo - 0.4, y, f"{DISPLAY.get(models[idx], models[idx])}  ({rk:.2f})",
                ha="right", va="center", fontsize=8.5)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=int(os.environ.get("FIG_DPI", "300")), facecolor="white")
    plt.close(fig)


def run_task(name: str, csv: str, models: list[str], metric: str, higher_better: bool):
    print("=" * 70)
    print(f" {name.upper()} — Friedman + Nemenyi + Wilcoxon")
    print("=" * 70)
    M = build_matrix(csv, models, metric)
    n, k = M.shape
    print(f" matrix: {n} instruments x {k} models (metric={metric})")

    # Friedman
    stat, p = friedmanchisquare(*[M[c].values for c in M.columns])
    print(f" Friedman: chi2={stat:.3f}, p={p:.4g}")

    # ranks per instrument (1 = best)
    sign = -1.0 if higher_better else 1.0   # rankdata ascending => for higher-better, negate
    ranks = np.vstack([rankdata(sign * M.loc[s].values) for s in M.index])
    avg = pd.Series(ranks.mean(axis=0), index=M.columns).sort_values()
    print(" average ranks (lower=better):")
    for mdl, r in avg.items():
        print(f"   {mdl:18s} {r:.3f}")

    cd = nemenyi_cd(k, n)
    print(f" Nemenyi CD (alpha=0.05) = {cd:.3f}")

    best = avg.index[0]
    # Wilcoxon: best vs each other
    wilcox = {}
    for mdl in models:
        if mdl == best:
            continue
        try:
            w, pw = wilcoxon(M[best].values, M[mdl].values)
            wilcox[mdl] = pw
        except Exception:
            wilcox[mdl] = float("nan")

    for mdl in models:
        rows_out.append({
            "task": name, "model": mdl, "avg_rank": round(float(avg[mdl]), 3),
            "friedman_chi2": round(float(stat), 3), "friedman_p": float(p),
            "nemenyi_cd": round(float(cd), 3),
            "best_model": best,
            "wilcoxon_p_vs_best": (round(float(wilcox[mdl]), 4) if mdl in wilcox else None),
        })

    # CD diagram
    fig_name = "fig03_cd_direction.png" if name == "direction" else "fig04_cd_volatility.png"
    note = "AUC, higher better" if higher_better else "RMSE, lower better"
    cd_diagram(avg, cd, f"Critical-difference diagram — {name} ({metric})",
               FIG_DIR / fig_name, note)
    print(f" saved {FIG_DIR/fig_name}")
    print()
    return stat, p, avg, cd, best, wilcox


def main():
    run_task("direction", "metrics_walk_direction_k5.csv", DIR_MODELS, "auc", higher_better=True)
    run_task("volatility", "metrics_walk_volatility_k5.csv", VOL_MODELS, "rmse", higher_better=False)
    out = pd.DataFrame(rows_out)
    out.to_csv(A / "statistical_tests.csv", index=False)
    print(f"[ARTIFACT] saved {A/'statistical_tests.csv'}")


if __name__ == "__main__":
    main()
