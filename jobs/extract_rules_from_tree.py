"""
Production-rule extraction from a trained decision tree.
========================================================

A data-grounded replacement for random generation: rules are extracted from the
root-to-leaf paths of a CART decision tree trained on the 56 causal features
across 8 instruments. Each rule corresponds to a real branch of the tree and is
therefore grounded in the data.

Target — three classes by 5-day forward return:
    a1 (Strong Bullish):  r_{t+5} > +tau
    a3 (Bearish):         r_{t+5} < -tau
    a2 (Consider/neutral): otherwise

Reproducibility: random_state = 42.

Run:
    python -m jobs.extract_rules_from_tree
Output: production_rules_knowledge_base.json
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from sqlalchemy import create_engine
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# Reuse the SAME 56-feature pipeline as the production models (train_baseline),
# so the rule tree operates on exactly the canonical 56-dimensional feature vector.
from jobs.racc import block, build_pool

SEED = 42
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON = 5
TAU = 0.01  # +/-1% threshold for classes a1/a3
MAX_DEPTH = int(os.environ.get("TREE_MAX_DEPTH", "7"))
MIN_LEAF = int(os.environ.get("TREE_MIN_LEAF", "80"))
# Chronological split date: rows on/before TRAIN_END train the tree (and the
# knowledge base); later rows form the held-out test set for the ROC/PR figures.
TRAIN_END = os.environ.get("RULES_TRAIN_END", "2023-12-31")
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(os.environ.get("FIG_DIR", "/tmp/figs"))

def get_engine():
    cfg = dotenv_values(ROOT / ".env")
    url = os.environ.get("DATABASE_URL") or cfg.get("DATABASE_URL")
    return create_engine(url, pool_pre_ping=True)


def build_dataset(engine) -> tuple[pd.DataFrame, list[str]]:
    """Pooled dataset of the 8 instruments on the canonical 56 features (the loader
    of jobs/racc.py) with the 3-class target of the rules."""
    data, feat_cols = build_pool(engine)
    data["ret_fwd"] = data["target_return_kd"]
    data["target"] = data["cls3"].map({0: "a1", 1: "a2", 2: "a3"})
    return data, feat_cols


def fmt_threshold(v):
    """A rule threshold with six significant digits in positional notation (no exponent).
    With fewer digits, rows close to a threshold of a small-scale feature (e.g. atrp_14
    around 0.008) would fall into another rule."""
    return np.format_float_positional(float(v), precision=6, unique=False, fractional=False, trim="-")

def tree_to_rules(tree: DecisionTreeClassifier, feat_names: list[str]) -> list[dict]:
    """Extracts rules from the tree paths (root -> leaf)."""
    t = tree.tree_
    classes = tree.classes_
    rules = []

    def recurse(node, conds):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            name = feat_names[t.feature[node]]
            thr = t.threshold[node]
            # left branch: feature <= thr
            recurse(t.children_left[node], conds + [(name, "<=", thr)])
            # right branch: feature > thr
            recurse(t.children_right[node], conds + [(name, ">", thr)])
        else:
            # leaf: majority class
            counts = t.value[node][0]
            cls = classes[int(np.argmax(counts))]
            support = int(t.n_node_samples[node])   # number of training samples in the leaf
            purity = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
            rules.append({
                "conditions": conds,
                "consequent": cls,
                "support": support,
                "purity": round(purity, 3),
            })

    recurse(0, [])
    return rules


def main():
    print("=" * 80)
    print(" PRODUCTION-RULE EXTRACTION FROM DECISION TREE")
    print("=" * 80)
    print(f" seed={SEED}  END_DATE={END_DATE}  max_depth={MAX_DEPTH}  min_leaf={MIN_LEAF}  tau=+/-{TAU}")

    engine = get_engine()
    data, feat_cols = build_dataset(engine)

    # Chronological split: the tree (and therefore the knowledge base) is fit on
    # the training block only; the later block is held out for the ROC/PR figures.
    # the training block is purged: rows whose 5-day label ends after TRAIN_END are dropped
    train = block(data, upto=TRAIN_END).reset_index(drop=True)
    test = block(data, after=TRAIN_END, purge=False).reset_index(drop=True)
    X = train[feat_cols].values
    y = train["target"].values
    print(f" Dataset: {len(data)} rows ({len(feat_cols)} features); "
          f"train={len(train)} (<= {TRAIN_END}), test={len(test)}")
    print(f" Train classes: {dict(pd.Series(y).value_counts())}")

    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
        criterion="gini", random_state=SEED,
    )
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f" Tree trained: train accuracy={acc:.3f}, leaves={clf.get_n_leaves()}")

    # ---- Held-out ROC / PR of the rule system (Buy = class a1) ----------------
    X_te = test[feat_cols].values
    a1_idx = list(clf.classes_).index("a1")
    p_buy = clf.predict_proba(X_te)[:, a1_idx]
    y_buy = (test["ret_fwd"].values > TAU).astype(int)   # actual "strong-bullish" outcome
    test_auc = float(roc_auc_score(y_buy, p_buy))
    print(f" Held-out rule-system ROC AUC (Buy vs rest): {test_auc:.3f} on {len(test)} samples")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Fig. 13: ROC
    fpr, tpr, _ = roc_curve(y_buy, p_buy)
    plt.figure(figsize=(5.2, 5.0))
    plt.plot(fpr, tpr, color="#d62728", lw=2.0,
             label=f"Production-rule system (AUC = {test_auc:.3f})")
    plt.fill_between(fpr, fpr, tpr, color="#d62728", alpha=0.10)
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1.0, label="Random (AUC = 0.5)")
    plt.xlabel("False-Positive Rate (1 − Specificity)")
    plt.ylabel("True-Positive Rate (Sensitivity)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_roc_rules.png", dpi=int(os.environ.get("FIG_DPI", "300")), facecolor="#f7f8fa")
    plt.close()
    # Fig. 14: precision-recall
    prec, rec, _ = precision_recall_curve(y_buy, p_buy)
    base = y_buy.mean()
    plt.figure(figsize=(5.2, 5.0))
    plt.plot(rec, prec, color="#2ca02c", lw=2.0, label="Production-rule system")
    plt.fill_between(rec, base, prec, color="#2ca02c", alpha=0.10)
    plt.axhline(base, ls="--", color="gray", lw=1.0,
                label=f"Random precision (≈ {base:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig14_pr_rules.png", dpi=int(os.environ.get("FIG_DPI", "300")), facecolor="#f7f8fa")
    plt.close()
    print(f" Saved figures -> {FIG_DIR}/fig13_roc_rules.png, fig14_pr_rules.png")

    rules_raw = tree_to_rules(clf, feat_cols)
    print(f" Extracted rules (= number of leaves): {len(rules_raw)}")

    # Convert to p_i format with f-indices
    fidx = {nm: (i + 1) for i, nm in enumerate(feat_cols)}
    def simplify_conditions(conditions):
        """Collapses repeated splits on one feature into a minimal interval.
        A tree path may bound a feature several times; we keep the tightest
        lower (>) and upper (<=) bounds."""
        bounds = {}  # name -> [lower, upper]
        for nm, op, thr in conditions:
            lo, hi = bounds.get(nm, [None, None])
            if op == ">":
                lo = thr if lo is None else max(lo, thr)
            else:  # "<="
                hi = thr if hi is None else min(hi, thr)
            bounds[nm] = [lo, hi]
        fmt = fmt_threshold
        parts = []
        for nm in sorted(bounds):
            lo, hi = bounds[nm]
            if lo is not None and hi is not None:
                parts.append(f"{fmt(lo)} < {nm} ≤ {fmt(hi)}")
            elif lo is not None:
                parts.append(f"{nm} > {fmt(lo)}")
            else:
                parts.append(f"{nm} ≤ {fmt(hi)}")
        return parts, sorted(bounds.keys()), bounds

    rules = []
    for k, r in enumerate(rules_raw, 1):
        parts, ante_names, bounds = simplify_conditions(r["conditions"])
        ante = sorted({fidx[nm] for nm in ante_names})
        cond = " ∧ ".join(parts)
        rules.append({
            "id": f"p{k}",
            "antecedent_features": ante_names,
            "ante": ante,
            "cons": r["consequent"],
            "condition": cond,
            # exact interval of every feature (lower, upper; None = unbounded): lower < f <= upper
            "bounds": {nm: [float(lo) if lo is not None else None, float(hi) if hi is not None else None]
                       for nm, (lo, hi) in bounds.items()},
            "support": r["support"],
            "purity": r["purity"],
        })

    # The published rules must reproduce the tree: with the printed (rounded)
    # thresholds every training row has to fire exactly the rule of its own leaf.
    fpos = {nm: i for i, nm in enumerate(feat_cols)}
    leaf_nodes = [i for i in range(clf.tree_.node_count) if clf.tree_.children_left[i] == -1]
    own = np.searchsorted(leaf_nodes, clf.apply(X))
    fired = np.zeros((len(X), len(rules)), dtype=bool)
    for j, r in enumerate(rules):
        m = np.ones(len(X), dtype=bool)
        for nm, (lo, hi) in r["bounds"].items():
            col = X[:, fpos[nm]]
            if lo is not None:
                m &= col > float(fmt_threshold(lo))
            if hi is not None:
                m &= col <= float(fmt_threshold(hi))
        fired[:, j] = m
    assert (fired.sum(axis=1) == 1).all(), "the published rules do not partition the training data"
    assert fired[np.arange(len(X)), own].all(), "a published rule disagrees with the tree"
    print(f" Check: all {len(X)} training rows fire exactly the rule of their own leaf")

    EVENTS = {"a1": "Strong Bullish", "a2": "Consider / neutral", "a3": "Bearish"}
    out = {
        "generated_at": datetime.now().isoformat(),
        "method": "decision-tree path extraction (CART)",
        "seed": SEED,
        "tree_max_depth": MAX_DEPTH,
        "tree_min_samples_leaf": MIN_LEAF,
        "train_accuracy": round(acc, 3),
        "train_end": TRAIN_END,
        "test_size": int(len(test)),
        "test_roc_auc": round(test_auc, 3),
        "total_rules": len(rules),
        "features_count": len(feat_cols),
        "events": EVENTS,
        "rules": rules,
    }
    out_path = ROOT / "production_rules_knowledge_base.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary
    cls_counts = pd.Series([r["cons"] for r in rules]).value_counts()
    print(f"\n Rules by class: {dict(cls_counts)}")
    print(f" Mean purity: {np.mean([r['purity'] for r in rules]):.3f}")
    print("\n Sample rules (top-5 by support):")
    for r in sorted(rules, key=lambda x: -x["support"])[:5]:
        ante = " ∧ ".join(f"f{i}" for i in r["ante"])
        print(f"   {r['id']}: {ante} → {r['cons']} (support={r['support']}, purity={r['purity']})")
    print(f"\n[OK] Saved: {out_path} ({len(rules)} rules)")


if __name__ == "__main__":
    main()
