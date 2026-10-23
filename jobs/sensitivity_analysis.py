"""
Sensitivity analysis: robustness of the method to the forecast horizon K.
========================================================================

Repeats the walk-forward evaluation of the machine-learning models of
train_baseline.py at K = 10 and K = 20 (K = 5 is the main run) and summarises
the mean test metric per instrument, model and horizon (Table 9 reports the best
machine-learning volatility RMSE per instrument and horizon). The econometric,
hybrid and deep models are evaluated at K = 5 only.

A full run takes about 3-5 hours.

Run:
    python -m jobs.sensitivity_analysis              # train K = 5, 10, 20 and summarise
    python -m jobs.sensitivity_analysis summarise    # summary from existing metric files

Output:
    artifacts/sensitivity_K_summary.csv  — summary table for the paper
    artifacts/metrics_walk_direction_k{K}.csv (per K)
    artifacts/metrics_walk_volatility_k{K}.csv (per K)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Horizons tested for robustness
HORIZONS = [5, 10, 20]
ML_MODELS = {"EXTRATREES", "HYBRID_STACK_REG", "XGB", "LGBM", "HGB",
             "LOGREG", "RF", "HYBRID_VOTING", "HYBRID_STACK"}
TASKS = ["direction", "volatility"]
END_DATE = os.environ.get("END_DATE", "2026-06-01")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_OUT = ARTIFACTS_DIR / "sensitivity_K_summary.csv"


def run_one_config(horizon: int, task: str) -> int:
    """Runs train_baseline.py for the given K and task."""
    env = os.environ.copy()
    env["HORIZON_DAYS"] = str(horizon)
    env["TASK"] = task
    env["END_DATE"] = END_DATE

    print("\n" + "=" * 70)
    print(f" Sensitivity: K={horizon}  task={task}")
    print("=" * 70)

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "jobs.train_baseline"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    dt = time.time() - t0
    print(f"\n[INFO] K={horizon} task={task} done in {dt/60:.1f} min")
    return proc.returncode


def aggregate_results() -> pd.DataFrame:
    """Aggregates the sensitivity summary table from the per-K CSV files."""
    all_rows = []

    for k in HORIZONS:
        for task in TASKS:
            csv = ARTIFACTS_DIR / f"metrics_walk_{task}_k{k}.csv"
            if not csv.exists():
                print(f"[WARN] Not found: {csv}")
                continue
            df = pd.read_csv(csv)
            # only the walk-forward models of train_baseline.py, which are run at every K
            # (Table 9); econometric, hybrid and deep models exist only for K = 5
            df = df[(df["split"] == "test") & (df["model"].isin(ML_MODELS))].copy()
            df["K"] = k

            # direction: AUC, volatility: RMSE
            metric = "auc" if task == "direction" else "rmse"
            if metric not in df.columns:
                continue

            # Summary by (symbol, model)
            summary = (
                df.groupby(["symbol", "model"], as_index=False)
                .agg(
                    metric_mean=(metric, "mean"),
                    metric_std=(metric, "std"),
                    n_folds=("fold", "nunique"),
                )
            )
            summary["task"] = task
            summary["metric"] = metric.upper()
            summary["K"] = k
            all_rows.append(summary)

    if not all_rows:
        print("[WARN] No data to aggregate")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[["task", "K", "symbol", "model", "metric", "metric_mean", "metric_std", "n_folds"]]
    combined = combined.sort_values(["task", "symbol", "K", "model"]).reset_index(drop=True)
    return combined


def make_summary_pivot(df: pd.DataFrame) -> dict:
    """Builds paper pivot tables: models x K for each instrument."""
    pivots = {}

    for task in TASKS:
        sub = df[df["task"] == task].copy()
        if sub.empty:
            continue
        for sym in sub["symbol"].unique():
            sym_df = sub[sub["symbol"] == sym].copy()
            pivot = sym_df.pivot_table(
                index="model",
                columns="K",
                values="metric_mean",
                aggfunc="mean",
            )
            # Best model per K
            best_row = pivot.idxmax() if task == "direction" else pivot.idxmin()
            pivots[(task, sym)] = {
                "pivot": pivot,
                "best_per_K": best_row,
            }

    return pivots


def main() -> None:
    print("=" * 70)
    print(" SENSITIVITY ANALYSIS — robustness to horizon K")
    print("=" * 70)
    print(f" Horizons:  K = {HORIZONS}")
    print(f" Tasks:     {TASKS}")
    print(f" END_DATE:  {END_DATE}")
    print(" Tickers:   taken from features_daily")
    print("=" * 70)
    print(" WARNING: a full run takes ~3-5 hours on an M4!")
    print(" Can be run overnight.")
    print("=" * 70)

    # --- Stage 1: run train_baseline for each K and task (skipped with "summarise") ---
    overall_start = time.time()
    for k in ([] if "summarise" in sys.argv[1:] else HORIZONS):
        # Skip K=5 if files already exist (already generated)
        skip_k5 = (
            k == 5
            and (ARTIFACTS_DIR / "metrics_walk_direction_k5.csv").exists()
            and (ARTIFACTS_DIR / "metrics_walk_volatility_k5.csv").exists()
        )
        if skip_k5:
            print(f"\n[SKIP] K={k}: files already exist, skipping re-run")
            continue

        for task in TASKS:
            rc = run_one_config(k, task)
            if rc != 0:
                print(f"[ERROR] K={k} task={task} returned code {rc}")

    elapsed_total = time.time() - overall_start
    print(f"\n[INFO] All runs finished in {elapsed_total/60:.1f} min")

    # --- Stage 2: aggregate into one summary table ---
    print("\n" + "=" * 70)
    print(" Aggregating results")
    print("=" * 70)

    combined = aggregate_results()
    if combined.empty:
        print("[ERROR] No data to aggregate")
        return

    combined.to_csv(SUMMARY_OUT, index=False)
    print(f"[ARTIFACT] Saved: {SUMMARY_OUT}")
    print(f"[INFO] Total rows: {len(combined)}")

    # --- Stage 3: pivot tables for the paper ---
    pivots = make_summary_pivot(combined)

    print("\n" + "=" * 70)
    print(" PIVOTS for the paper (model x K)")
    print("=" * 70)

    for (task, sym), data in sorted(pivots.items()):
        if not isinstance(data, dict) or "pivot" not in data:
            continue
        pivot = data["pivot"]
        if pivot.empty:
            continue
        print(f"\n── {task.upper()} — {sym} ──")
        print(pivot.round(5).to_string())
        print(f"  Best per K: {data['best_per_K'].to_dict()}")

    # --- Stage 4: winner-stability summary ---
    print("\n" + "=" * 70)
    print(" Winner stability across K")
    print("=" * 70)

    stable_winners = []
    for (task, sym), data in pivots.items():
        if "best_per_K" not in data:
            continue
        winners = data["best_per_K"].values
        unique_winners = set(winners)
        is_stable = len(unique_winners) == 1
        stable_winners.append({
            "task": task,
            "symbol": sym,
            "stable": is_stable,
            "winners_by_K": dict(data["best_per_K"]),
        })

    n_stable = sum(1 for w in stable_winners if w["stable"])
    print(f"\n[STATS] Stable winners: {n_stable} / {len(stable_winners)}")
    for w in stable_winners:
        mark = "✅" if w["stable"] else "⚠️"
        print(f"  {mark} {w['task']:<10} {w['symbol']:<8}: {w['winners_by_K']}")


if __name__ == "__main__":
    main()
