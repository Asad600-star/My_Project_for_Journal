"""
Генератор всех таблиц для Q1 статьи из собранных метрик.
==========================================================

Читает CSV из artifacts/ и формирует готовые таблицы для вставки в статью:
- Table B1: Direction baseline comparison (AUC, 8 активов)
- Table B2: Volatility baseline comparison (RMSE, 8 активов)
- Table B3: Diebold-Mariano значимость (сводка)
- Table B4: Sensitivity по горизонту K=5/10/20
- Table B5: Ablation study (вклад групп признаков)

Запуск:
    python -m jobs.generate_paper_tables

Все числа — точные из реальных walk-forward экспериментов.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

A = Path("artifacts")
OUT = A / "paper_tables.txt"

# 8 целевых активов (макро ^VIX/^IRX/^TNX исключены — они признаки, не цели)
TARGET = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]
ASSET_LABEL = {
    "AAPL": "AAPL (Apple)",
    "TSLA": "TSLA (Tesla)",
    "MSFT": "MSFT (Microsoft)",
    "GLD": "GLD (Gold ETF)",
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000",
}

lines: list[str] = []


def w(s: str = "") -> None:
    lines.append(s)
    print(s)


# ─────────────────────────────────────────────────────────────────────
#  TABLE B1 — Direction baseline comparison
# ─────────────────────────────────────────────────────────────────────
def table_direction():
    w("=" * 90)
    w(" TABLE B1 — DIRECTION FORECASTING: AUC (test, K=5, mean over walk-forward folds)")
    w("=" * 90)
    d = pd.read_csv(A / "metrics_walk_direction_k5.csv")
    d = d[(d["split"] == "test") & (d["symbol"].isin(TARGET))]
    models = ["HYBRID_VOTING", "HYBRID_STACK", "LSTM", "LOGREG", "RF", "XGB", "LGBM", "HGB"]

    agg = d[d["model"].isin(models)].groupby(["symbol", "model"]).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std")
    ).reset_index()

    w(f"\n{'Asset':<16}", )
    header = f"{'Asset':<16}" + "".join(f"{m:>16}" for m in models)
    w(header)
    w("-" * len(header))
    for sym in TARGET:
        row = f"{ASSET_LABEL[sym]:<16}"
        for m in models:
            sub = agg[(agg["symbol"] == sym) & (agg["model"] == m)]
            if sub.empty:
                row += f"{'—':>16}"
            else:
                row += f"{sub['auc_mean'].iloc[0]:>10.4f}±{sub['auc_std'].iloc[0]:.3f}"
        w(row)

    # Среднее
    w("-" * len(header))
    row = f"{'MEAN':<16}"
    for m in models:
        vals = agg[agg["model"] == m]["auc_mean"]
        row += f"{vals.mean():>16.4f}" if len(vals) else f"{'—':>16}"
    w(row)
    w("")
    # Лучшая модель на каждом активе
    w("Лучшая модель по активу:")
    for sym in TARGET:
        sub = agg[agg["symbol"] == sym].sort_values("auc_mean", ascending=False)
        if not sub.empty:
            best = sub.iloc[0]
            w(f"  {ASSET_LABEL[sym]:<16} → {best['model']:<15} AUC={best['auc_mean']:.4f}")
    w("")


# ─────────────────────────────────────────────────────────────────────
#  TABLE B2 — Volatility baseline comparison
# ─────────────────────────────────────────────────────────────────────
def table_volatility():
    w("=" * 90)
    w(" TABLE B2 — VOLATILITY FORECASTING: RMSE (test, K=5, mean over walk-forward folds)")
    w("=" * 90)
    v = pd.read_csv(A / "metrics_walk_volatility_k5.csv")
    v = v[(v["split"] == "test") & (v["symbol"].isin(TARGET))]
    models = ["EXTRATREES", "HYBRID_STACK_REG", "XGB", "LGBM", "HGB", "ARIMA", "GARCH"]

    agg = v[v["model"].isin(models)].groupby(["symbol", "model"]).agg(
        rmse_mean=("rmse", "mean")
    ).reset_index()

    header = f"{'Asset':<16}" + "".join(f"{m:>14}" for m in models)
    w("")
    w(header)
    w("-" * len(header))
    for sym in TARGET:
        row = f"{ASSET_LABEL[sym]:<16}"
        for m in models:
            sub = agg[(agg["symbol"] == sym) & (agg["model"] == m)]
            row += f"{sub['rmse_mean'].iloc[0]:>14.6f}" if not sub.empty else f"{'—':>14}"
        w(row)
    w("-" * len(header))
    row = f"{'MEAN':<16}"
    for m in models:
        vals = agg[agg["model"] == m]["rmse_mean"]
        row += f"{vals.mean():>14.6f}" if len(vals) else f"{'—':>14}"
    w(row)
    w("")
    # ExtraTrees vs эконометрика
    et = agg[agg["model"] == "EXTRATREES"]["rmse_mean"].mean()
    arima = agg[agg["model"] == "ARIMA"]["rmse_mean"].mean()
    garch = agg[agg["model"] == "GARCH"]["rmse_mean"].mean()
    w(f"ExtraTrees средний RMSE:  {et:.6f}")
    w(f"ARIMA средний RMSE:       {arima:.6f}  (ExtraTrees лучше на {100*(arima-et)/arima:.1f}%)")
    w(f"GARCH средний RMSE:       {garch:.6f}  (ExtraTrees лучше на {100*(garch-et)/garch:.1f}%)")
    w("")


# ─────────────────────────────────────────────────────────────────────
#  TABLE B3 — Diebold-Mariano сводка
# ─────────────────────────────────────────────────────────────────────
def table_dm():
    w("=" * 90)
    w(" TABLE B3 — DIEBOLD-MARIANO TEST (HLN-corrected): сводка значимости")
    w("=" * 90)
    if not (A / "dm_test_results.csv").exists():
        w("  (файл dm_test_results.csv не найден)")
        return
    dm = pd.read_csv(A / "dm_test_results.csv")
    w(f"\nВсего сравнений: {len(dm)}")
    w(f"Значимых (p<0.05): {sum(dm['p_value'] < 0.05)}")
    w(f"Очень значимых (p<0.01): {sum(dm['p_value'] < 0.01)}")
    w("")
    w("По задачам:")
    for task in dm["task"].unique():
        sub = dm[dm["task"] == task]
        sig = sum(sub["p_value"] < 0.05)
        w(f"  {task:<12}: {sig}/{len(sub)} значимых (p<0.05)")
    w("")
    # ExtraTrees vs ARIMA/GARCH — ключевой результат
    w("Ключевые сравнения (ExtraTrees против эконометрики):")
    key = dm[(dm["model_A"] == "EXTRATREES") & (dm["model_B"].isin(["ARIMA", "GARCH"]))]
    sig_key = sum(key["p_value"] < 0.05)
    w(f"  ExtraTrees vs ARIMA/GARCH: {sig_key}/{len(key)} значимых (p<0.05)")
    w("")


# ─────────────────────────────────────────────────────────────────────
#  TABLE B4 — Sensitivity по горизонту
# ─────────────────────────────────────────────────────────────────────
def table_sensitivity():
    w("=" * 90)
    w(" TABLE B4 — SENSITIVITY: лучшая volatility-модель RMSE по горизонту K")
    w("=" * 90)
    rows = []
    for k in [5, 10, 20]:
        f = A / f"metrics_walk_volatility_k{k}.csv"
        if not f.exists():
            continue
        v = pd.read_csv(f)
        v = v[(v["split"] == "test") & (v["symbol"].isin(TARGET))]
        v = v[~v["model"].str.startswith("BASELINE")]
        for sym in TARGET:
            sub = v[v["symbol"] == sym]
            if sub.empty:
                continue
            best = sub.groupby("model")["rmse"].mean().sort_values()
            rows.append({"K": k, "symbol": sym, "best_model": best.index[0], "best_rmse": best.iloc[0]})
    if not rows:
        w("  (нет данных sensitivity)")
        return
    df = pd.DataFrame(rows)
    w("")
    # Pivot: symbol × K с RMSE лучшей модели
    piv = df.pivot(index="symbol", columns="K", values="best_rmse")
    piv = piv.reindex([s for s in TARGET if s in piv.index])
    w("Лучший RMSE по (актив × горизонт K):")
    w(piv.round(6).to_string())
    w("")
    # ExtraTrees стабильность
    et_wins = df[df["best_model"] == "EXTRATREES"].groupby("K").size()
    w("Сколько активов ExtraTrees побеждает на каждом K:")
    for k in [5, 10, 20]:
        if k in et_wins.index:
            w(f"  K={k}: {et_wins[k]}/8 активов")
    w("")


# ─────────────────────────────────────────────────────────────────────
#  TABLE B5 — Ablation study
# ─────────────────────────────────────────────────────────────────────
def table_ablation():
    w("=" * 90)
    w(" TABLE B5 — ABLATION STUDY: RMSE по группам признаков (volatility, ExtraTrees)")
    w("=" * 90)
    f = A / "ablation_pivot.csv"
    if not f.exists():
        w("  (нет ablation_pivot.csv)")
        return
    piv = pd.read_csv(f, index_col=0)
    piv = piv[piv.index.isin(TARGET)]
    w("")
    w(piv.round(6).to_string())
    w("")
    # Улучшение PRICE_ONLY → ALL_FULL
    if "PRICE_ONLY" in piv.columns and "ALL_FULL" in piv.columns:
        w("Улучшение от PRICE_ONLY (5 призн.) до ALL_FULL (56 призн.):")
        for sym in piv.index:
            p0 = piv.loc[sym, "PRICE_ONLY"]
            pf = piv.loc[sym, "ALL_FULL"]
            impr = 100 * (p0 - pf) / p0
            mark = "✅" if impr > 0 else "⚠️"
            w(f"  {mark} {ASSET_LABEL.get(sym, sym):<16}: {p0:.6f} → {pf:.6f}  ({impr:+.1f}%)")
    w("")


def main():
    w("\n")
    w("#" * 90)
    w("#  ТАБЛИЦЫ ДЛЯ Q1 СТАТЬИ — сгенерированы из реальных walk-forward экспериментов")
    w("#  8 активов | K=5/10/20 | walk-forward CV | bootstrap CI | Diebold-Mariano")
    w("#" * 90)
    w("")
    table_direction()
    table_volatility()
    table_dm()
    table_sensitivity()
    table_ablation()

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ARTIFACT] Все таблицы сохранены в {OUT}")


if __name__ == "__main__":
    main()
