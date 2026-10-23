"""
RACC-gated economic back-test.
==============================

Tests whether abstaining when the conformal risk bound cannot be met (RACC)
produces better *risk-adjusted* trading than acting on every signal.

For each instrument, on the held-out test block (2024-2026), a daily directional
strategy is simulated under four policies:
  - UNGATED     : take the signed signal every day.
  - RACC@alpha  : take the signal only when |s| >= t_alpha (else stay in cash).
  - ONLINE@alpha: as RACC, with the threshold updated every day by online risk
                  control (jobs.racc.online_thresholds), using resolved labels only.
  - BUY&HOLD    : passive long.

Metrics: annualised Sharpe, max drawdown, total return, coverage (days in market),
number of position changes. Models are fit on data up to 2022, the PAV calibrator
on the first half of 2023 and the conformal threshold on its second half (no test
peeking).

Run:
    python -m jobs.racc_backtest
Outputs: artifacts/racc_backtest.csv, /tmp/figs/fig09_racc_equity.png
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

# one definition of the data, the RACC loss and the threshold (jobs/racc.py)
from jobs.racc import build_pool, crc_threshold, fit_racc, get_engine, online_thresholds, split

load_dotenv()
SEED = 42
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON = 5
TAU = 0.01
TRAIN_END = "2022-12-31"
TUNE_END = "2023-06-30"
CALIB_END = "2023-12-31"
FEE = 5e-4                       # 5 bps per position change
ALPHAS = [0.30, 0.45]            # RACC risk budgets to report
GAMMA = 0.05                     # step size of the online threshold (fixed a priori)
LAG = 6                          # trading days until a 5-day label has resolved
# fusion weight of the calibrated probability against the rule belief; 0.5 weights
# both equally (published back-test), 1.0 is the value selected on the tuning block in racc.py
LAMBDA = float(os.environ.get("RACC_LAMBDA", "0.5"))
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]
LABEL = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones", "^RUT": "Russell 2k"}
A = Path(__file__).resolve().parents[1] / "artifacts"
FIG = Path(os.environ.get("FIG_DIR", "/tmp/figs"))

def perf(pnl):
    """Annualised Sharpe, max drawdown, total return from a daily-pnl series."""
    pnl = np.asarray(pnl)
    if pnl.std() < 1e-12:
        sharpe = 0.0
    else:
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
    eq = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    tot = float(eq[-1] - 1.0)
    return sharpe, mdd, tot, eq


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    data, feat = build_pool(engine)
    tr, tu, ca, te = split(data, TRAIN_END, TUNE_END, CALIB_END)   # purged blocks (jobs/racc.py)
    print(f"train={len(tr)} tune={len(tu)} calib={len(ca)} test={len(te)}  lambda={LAMBDA}")
    racc_model = fit_racc(tr, tu, feat, lam=LAMBDA)       # PAV on the tuning block; lambda fixed in advance

    def signal(part):
        s, sig, _ = racc_model.score(part)
        return s, sig

    # conformal thresholds from the calibration block
    s_ca, _, loss_ca = racc_model.score(ca)
    thr = {al: crc_threshold(np.abs(s_ca), loss_ca, al) for al in ALPHAS}
    print("RACC thresholds:", {a: round(float(t), 4) for a, t in thr.items()})

    # online thresholds: one path over the pooled test days, started from the static
    # threshold and driven by the realised volatility-scaled loss of resolved days
    s_te, _, loss_te = racc_model.score(te)
    days = np.unique(te["date"].values)
    day_te = np.searchsorted(days, te["date"].values)
    thr_on = {al: pd.Series(online_thresholds(day_te, np.abs(s_te), loss_te, al, thr[al], GAMMA, LAG)[1], index=days)
              for al in ALPHAS}

    # per-instrument daily back-test on test block
    rows = []
    port = {"UNGATED": [], "BUY&HOLD": [], **{f"RACC@{a}": [] for a in ALPHAS},
            **{f"ONLINE@{a}": [] for a in ALPHAS}}
    for sym in SYMBOLS:
        d = te[te["symbol"] == sym].sort_values("date")
        if len(d) < 50:
            continue
        s, _ = signal(d)
        ret = d["return_1d"].values
        fwd = np.roll(ret, -1); fwd[-1] = 0.0          # next-day return
        pol = {"UNGATED": np.sign(s)}
        for al in ALPHAS:
            pol[f"RACC@{al}"] = np.sign(s) * (np.abs(s) >= thr[al])
        for al in ALPHAS:
            pol[f"ONLINE@{al}"] = np.sign(s) * (np.abs(s) >= thr_on[al].loc[d["date"].values].values)
        pol["BUY&HOLD"] = np.ones_like(s)
        for name, pos in pol.items():
            pos = pos.astype(float); pos[pos == 0] = 0.0 if name != "BUY&HOLD" else 1.0
            cost = FEE * np.abs(np.diff(np.concatenate([[0.0], pos])))
            pnl = pos * fwd - cost
            sh, mdd, tot, _ = perf(pnl)
            cov = float(np.mean(pos != 0))
            nch = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], pos]))) > 0))
            rows.append(dict(symbol=LABEL.get(sym, sym), policy=name, sharpe=round(sh, 2),
                             max_drawdown=round(mdd, 3), total_return=round(tot, 3),
                             coverage=round(cov, 2), n_changes=nch))
            port[name].append(pd.Series(pnl, index=d["date"].values))

    res = pd.DataFrame(rows)
    tag = "" if LAMBDA == 0.5 else f"_lam{LAMBDA:g}"
    res.to_csv(A / f"racc_backtest{tag}.csv", index=False)

    # portfolio = equal-weight average daily pnl across instruments, aligned by date
    # on the trading days common to all instruments
    print("\n=== PORTFOLIO (equal-weight) ===")
    eqs = {}
    psum = []
    for name, lst in port.items():
        m = pd.concat(lst, axis=1, join="inner").mean(axis=1).values
        sh, mdd, tot, eq = perf(m)
        eqs[name] = eq
        cov = np.mean([np.mean(np.asarray(p) != 0) for p in lst]) if name != "BUY&HOLD" else 1.0
        psum.append(dict(policy=name, sharpe=round(sh, 2), max_drawdown=round(mdd, 3),
                         total_return=round(tot, 3), coverage=round(float(cov), 2)))
    pdf = pd.DataFrame(psum)
    print(pdf.to_string(index=False))
    pdf.to_csv(A / f"racc_backtest_portfolio{tag}.csv", index=False)

    # equity figure
    plt.figure(figsize=(7.2, 4.6))
    colour = {"UNGATED": "#d62728", "BUY&HOLD": "#777", "RACC@0.3": "#2ca02c", "RACC@0.45": "#1f77b4",
              "ONLINE@0.3": "#9467bd", "ONLINE@0.45": "#ff7f0e"}
    style = {"BUY&HOLD": "--", "ONLINE@0.3": "-.", "ONLINE@0.45": "-."}
    for name, eq in eqs.items():
        plt.plot(eq, style.get(name, "-"), lw=1.9, label=name, color=colour.get(name))
    plt.axhline(1.0, color="gray", lw=0.6)
    plt.xlabel("Trading day (held-out test 2024–2026)")
    plt.ylabel("Equity (start = 1.0)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIG / f"fig09_racc_equity{tag}.png", dpi=int(os.environ.get("FIG_DPI", "300")), facecolor="white"); plt.close()
    print(f"\n[ARTIFACT] {A/'racc_backtest.csv'}, {A/'racc_backtest_portfolio.csv'}")
    print(f"[FIG] {FIG/'fig09_racc_equity.png'}")


if __name__ == "__main__":
    main()
