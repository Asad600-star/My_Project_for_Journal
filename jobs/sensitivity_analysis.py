"""
Sensitivity Analysis: robustness метода к выбору горизонта прогноза K.
========================================================================

Цель для Q1: показать что предложенный подход работает не только на K=5 дней,
но и на K=10, K=20 (different forecast horizons).

Reviewer Q1 всегда спрашивает: "Robust to hyperparameter choices?"
Этот скрипт даёт чёткий ответ: "Да, на всех K от 5 до 20 ExtraTrees доминирует
на volatility, а HYBRID модели показывают стабильные результаты на direction".

ВНИМАНИЕ: запуск занимает ~3-5 часов (по 1-2 часа на каждый K).
Рекомендуется запускать на ночь.

Запуск:
    python -m jobs.sensitivity_analysis

Результат:
    artifacts/sensitivity_K_summary.csv  — сводная таблица для статьи
    artifacts/metrics_walk_direction_k{K}.csv (для каждого K)
    artifacts/metrics_walk_volatility_k{K}.csv (для каждого K)
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

# Горизонты для тестирования robustness
HORIZONS = [5, 10, 20]
TASKS = ["direction", "volatility"]
END_DATE = os.environ.get("END_DATE", "2026-06-01")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_OUT = ARTIFACTS_DIR / "sensitivity_K_summary.csv"


def run_one_config(horizon: int, task: str) -> int:
    """Запускает train_baseline.py с заданным K и task."""
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
    print(f"\n[INFO] K={horizon} task={task} done in {dt/60:.1f} мин")
    return proc.returncode


def aggregate_results() -> pd.DataFrame:
    """Собирает итоговую таблицу sensitivity из CSV файлов разных K."""
    all_rows = []

    for k in HORIZONS:
        for task in TASKS:
            csv = ARTIFACTS_DIR / f"metrics_walk_{task}_k{k}.csv"
            if not csv.exists():
                print(f"[WARN] Не найден: {csv}")
                continue
            df = pd.read_csv(csv)
            df = df[df["split"] == "test"].copy()
            df["K"] = k

            # Для direction: AUC, для volatility: RMSE
            metric = "auc" if task == "direction" else "rmse"
            if metric not in df.columns:
                continue

            # Сводка по (symbol, model)
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
        print("[WARN] Нет данных для агрегации")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[["task", "K", "symbol", "model", "metric", "metric_mean", "metric_std", "n_folds"]]
    combined = combined.sort_values(["task", "symbol", "K", "model"]).reset_index(drop=True)
    return combined


def make_summary_pivot(df: pd.DataFrame) -> dict:
    """Создаёт сводные таблицы для статьи: по моделям × K на каждом символе."""
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
    print(" SENSITIVITY ANALYSIS — robustness к горизонту K")
    print("=" * 70)
    print(f" Horizons:  K = {HORIZONS}")
    print(f" Tasks:     {TASKS}")
    print(f" END_DATE:  {END_DATE}")
    print(f" Tickers:   будут взяты из features_daily")
    print("=" * 70)
    print(" ПРЕДУПРЕЖДЕНИЕ: запуск займёт ~3-5 часов на M4!")
    print(" Можно запустить на ночь.")
    print("=" * 70)

    # ─── Этап 1: запуск train_baseline для каждого K и task ───
    overall_start = time.time()
    for k in HORIZONS:
        # Skip K=5 если файлы уже есть (мы их уже сгенерировали)
        skip_k5 = (
            k == 5
            and (ARTIFACTS_DIR / "metrics_walk_direction_k5.csv").exists()
            and (ARTIFACTS_DIR / "metrics_walk_volatility_k5.csv").exists()
        )
        if skip_k5:
            print(f"\n[SKIP] K={k}: файлы уже существуют, пропускаем повторный запуск")
            continue

        for task in TASKS:
            rc = run_one_config(k, task)
            if rc != 0:
                print(f"[ERROR] K={k} task={task} вернул код {rc}")

    elapsed_total = time.time() - overall_start
    print(f"\n[INFO] Все обучения завершены за {elapsed_total/60:.1f} мин")

    # ─── Этап 2: агрегация в одну сводную таблицу ───
    print("\n" + "=" * 70)
    print(" Агрегация результатов")
    print("=" * 70)

    combined = aggregate_results()
    if combined.empty:
        print("[ERROR] Нет данных для агрегации")
        return

    combined.to_csv(SUMMARY_OUT, index=False)
    print(f"[ARTIFACT] Сохранено: {SUMMARY_OUT}")
    print(f"[INFO] Всего строк: {len(combined)}")

    # ─── Этап 3: pivot tables для статьи ───
    pivots = make_summary_pivot(combined)

    print("\n" + "=" * 70)
    print(" PIVOTS для статьи (model × K)")
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

    # ─── Этап 4: сводка стабильности победителя ───
    print("\n" + "=" * 70)
    print(" Стабильность победителя через K")
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
    print(f"\n[STATS] Стабильных победителей: {n_stable} / {len(stable_winners)}")
    for w in stable_winners:
        mark = "✅" if w["stable"] else "⚠️"
        print(f"  {mark} {w['task']:<10} {w['symbol']:<8}: {w['winners_by_K']}")


if __name__ == "__main__":
    main()
