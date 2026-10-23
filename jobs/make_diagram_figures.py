"""
Regenerates the diagram figures used in the paper.
==================================================

- Fig. 1: end-to-end system architecture (eight modules).
- Fig. 2: three-step data pipeline (ingest, features, targets).
- Fig. 6: illustrative multi-instrument daily output of the system for eight
          instruments at the 5-day horizon.

These are diagram/illustrative figures (not data plots). In the snapshot only
p_up and the forecast volatility are set by hand; the recommendation, VaR,
Sharpe ratio and position size are computed by core.risk.RiskManager. Run:

    python -m jobs.make_diagram_figures

Output: fig01_architecture.png, fig02_pipeline.png and fig06_snapshot.png in FIG_DIR
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(os.environ.get("FIG_DIR", "/tmp/figs"))
# number of production rules, read from the knowledge base written by extract_rules_from_tree.py
N_RULES = json.loads((Path(__file__).resolve().parents[1] / "production_rules_knowledge_base.json")
                     .read_text(encoding="utf-8"))["total_rules"]
N_FEATURES = 56

BG = "#f7f8fa"
INK = "#1f2d3d"


# ─────────────────────────────────────────────────────────────────────
#  Fig. 1 — architecture
# ─────────────────────────────────────────────────────────────────────
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    boxes = [
        # (x, y, text, facecolor)
        (0.6, 5.6, "Data Ingest\n(Yahoo Finance)\nPostgreSQL", "#a8d5a2"),
        (4.4, 5.6, f"Feature\nEngineering\n|F| = {N_FEATURES}", "#9ec5e8"),
        (8.2, 5.6, "Hybrid Models\nVOTING (direction)\nH-ECON-ET (vol.)", "#f5c98a"),
        (12.0, 5.6, "Isotonic\nCalibration\n(PAV)", "#c9a8e0"),
        (0.6, 1.4, f"Production\nRules ({N_RULES})\nP : F → A", "#f3a6a6"),
        (4.4, 1.4, "Risk Manager\n+ RACC\nVaR / position", "#f2d98a"),
        (8.2, 1.4, "SHAP\nExplainer\n(Top-15)", "#a8d5a2"),
        (12.0, 1.4, "User Interfaces\nStreamlit +\nTelegram Bot", "#b6b6e8"),
    ]
    bw, bh = 3.4, 2.0
    centers = []
    for x, y, text, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), bw, bh,
                     boxstyle="round,pad=0.04,rounding_size=0.18",
                     fc=fc, ec="#5a6b7b", lw=1.2))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=INK)
        centers.append((x + bw / 2, y, x, y + bh / 2, x + bw, y + bh / 2))

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                     lw=1.4, color="#5a6b7b"))

    # top row left -> right
    for i in range(3):
        arrow((boxes[i][0] + bw, boxes[i][1] + bh / 2),
              (boxes[i + 1][0], boxes[i + 1][1] + bh / 2))
    # bottom row left -> right
    for i in range(4, 7):
        arrow((boxes[i][0] + bw, boxes[i][1] + bh / 2),
              (boxes[i + 1][0], boxes[i + 1][1] + bh / 2))
    # data-ingest -> production-rules (vertical)
    arrow((boxes[0][0] + bw / 2, boxes[0][1]),
          (boxes[4][0] + bw / 2, boxes[4][1] + bh))

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig01_architecture.png", dpi=int(os.environ.get("FIG_DPI", "300")), facecolor="white")
    plt.close(fig)
    print(f"[OK] {FIG_DIR}/fig01_architecture.png")


# ─────────────────────────────────────────────────────────────────────
#  Data pipeline: ingest -> features -> targets
# ─────────────────────────────────────────────────────────────────────
def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 10)
    ax.axis("off")
    steps = [
        ("Step 1. Ingest", "Daily OHLCV from Yahoo Finance\nupserted into PostgreSQL\n(key: instrument, date; 7-day overlap)", "#a8d5a2"),
        ("Step 2. Features", f"Causal feature vector\n$f_t \\in \\mathbb{{R}}^{{{N_FEATURES}}}$ from data up to day $t$\n(Theorem 1)", "#b3d7f7"),
        ("Step 3. Targets", "$d_{t+5} = 1[C_{t+5} \\geq C_t]$\n$\\sigma_{t,5} = \\mathrm{std}(r_{t+1}, \\ldots, r_{t+5})$\nrecomputed inside every fold", "#fde3a0"),
    ]
    bw, bh, y0 = 8.6, 5.6, 3.4
    for k, (head, body, fc) in enumerate(steps):
        x0 = 0.6 + k * 9.9
        ax.add_patch(FancyBboxPatch((x0, y0), bw, bh, boxstyle="round,pad=0.05,rounding_size=0.35",
                                    fc=fc, ec="#333", lw=1.4))
        ax.text(x0 + bw / 2, y0 + bh - 1.0, head, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=INK)
        ax.text(x0 + bw / 2, y0 + bh / 2 - 0.6, body, ha="center", va="center", fontsize=9.5,
                color=INK, linespacing=1.5)
        if k < 2:
            ax.add_patch(FancyArrowPatch((x0 + bw + 0.05, y0 + bh / 2), (x0 + 9.9 - 0.05, y0 + bh / 2),
                                         arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#444"))
    ax.text(15, 1.4, "Labels are computed inside each walk-forward block after the split, "
                     "so no label uses prices from a later block.",
            ha="center", va="center", fontsize=9, style="italic", color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f6f8", ec="#888"))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig02_pipeline.png", dpi=int(os.environ.get("FIG_DPI", "300")), facecolor="white")
    plt.close(fig)
    print(f"[OK] {FIG_DIR}/fig02_pipeline.png")


# ─────────────────────────────────────────────────────────────────────
#  Fig. 6 — illustrative multi-instrument snapshot (8 instruments)
# ─────────────────────────────────────────────────────────────────────
def fig_snapshot() -> None:
    # One illustrative trading day. Only the calibrated up-probability and the
    # forecast volatility are set by hand; the recommendation, VaR, Sharpe ratio
    # and position size are computed by the production RiskManager, so the
    # figure shows exactly what the system would report for these inputs.
    from core.risk.risk_manager import RiskManager
    rm = RiskManager()
    inputs = [
        # ticker, name, p_up, daily forecast volatility (%)
        ("AAPL",  "Apple Inc.",        0.642, 1.48),
        ("TSLA",  "Tesla Inc.",        0.574, 2.18),
        ("MSFT",  "Microsoft Corp.",   0.611, 1.36),
        ("GLD",   "SPDR Gold Trust",   0.558, 0.84),
        ("^GSPC", "S&P 500",           0.492, 0.71),
        ("^IXIC", "NASDAQ Comp.",      0.560, 1.23),
        ("^DJI",  "Dow Jones",         0.470, 0.68),
        ("^RUT",  "Russell 2000",      0.515, 1.02),
    ]
    label_of = {"Buy": "Buy", "Consider Buying": "Consider", "Do Not Buy": "Do not buy"}
    colour_of = {"Buy": "#2e9e5b", "Consider": "#e08a1e", "Do not buy": "#cf4040"}
    cards = []
    for ticker, name, p, sig in inputs:
        m = rm.evaluate(p, sig / 100.0)
        cards.append((ticker, name, p, sig, -100.0 * m.var_5d_approx, m.sharpe_approx,
                      round(100.0 * rm.position_size_pct(p, sig / 100.0)), label_of[m.recommendation_en]))

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 9.0))
    fig.patch.set_facecolor(BG)

    for ax, c in zip(axes.ravel(), cards):
        ticker, name, p, sig, var, sharpe, pos, label = c
        col = colour_of[label]
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                     boxstyle="round,pad=0.01,rounding_size=0.03",
                     transform=ax.transAxes, fc="white", ec=col, lw=2.2))
        # header bar
        ax.add_patch(FancyBboxPatch((0.02, 0.80), 0.96, 0.18,
                     boxstyle="round,pad=0.01,rounding_size=0.03",
                     transform=ax.transAxes, fc=col, ec=col, lw=0))
        ax.text(0.07, 0.89, ticker, transform=ax.transAxes, fontsize=14,
                fontweight="bold", color="white", va="center")
        ax.text(0.07, 0.835, name, transform=ax.transAxes, fontsize=8,
                color="white", va="center")
        # recommendation
        ax.text(0.5, 0.70, label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=col, ha="center")
        # probability
        ax.text(0.5, 0.55, f"{p:.3f}", transform=ax.transAxes, fontsize=20,
                fontweight="bold", color=INK, ha="center")
        ax.text(0.5, 0.46, "p̂ — prob. up move (5d)", transform=ax.transAxes,
                fontsize=7.5, color="#6b7280", ha="center")
        # risk row
        ax.text(0.5, 0.35, f"σ̂ (1d): {sig:.2f}%     VaR (5d): {var:.2f}%".replace("-", "\u2212"),
                transform=ax.transAxes, fontsize=9, color=INK, ha="center")
        ax.text(0.5, 0.28, f"Sharpe (annualised): {sharpe:+.2f}".replace("-", "\u2212"),
                transform=ax.transAxes, fontsize=9, color=INK, ha="center")
        # position
        ax.text(0.5, 0.18, f"Position size: {pos}%", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=col, ha="center")

    fig.tight_layout(rect=[0, 0.01, 1, 0.99])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig06_snapshot.png", dpi=int(os.environ.get("FIG_DPI", "300")), facecolor=BG)
    plt.close(fig)
    print(f"[OK] {FIG_DIR}/fig06_snapshot.png")


def main() -> None:
    fig_architecture()
    fig_pipeline()
    fig_snapshot()


if __name__ == "__main__":
    main()
