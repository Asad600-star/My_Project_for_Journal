"""
Stability analysis of the production-rule layer.
================================================

The production rules are read off a single CART tree, and a single tree is a
high-variance estimator. Rather than leaving that as a caveat, we quantify it:
the tree is refitted on B bootstrap resamples of the training block and we
measure how much the induced rule system actually moves.

Reported quantities
  - rules per run           : how stable the size of the knowledge base is;
  - feature-selection freq. : how often each feature is used by the tree, i.e.
                              whether the rules keep talking about the same drivers;
  - pairwise agreement      : fraction of held-out days on which two independently
                              bootstrapped rule systems return the same action;
  - agreement with the base : same, measured against the published rule base.

Run:
    python -m jobs.rule_stability
Output: artifacts/rule_stability.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.tree import DecisionTreeClassifier

from jobs.racc import block, build_pool          # the same data as the rule base and RACC

load_dotenv()
SEED = 42
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON = 5
TAU = 0.01
TRAIN_END = os.environ.get("RULES_TRAIN_END", "2023-12-31")   # identical to extract_rules_from_tree.py
MAX_DEPTH, MIN_LEAF = 7, 80
B = int(os.environ.get("STABILITY_B", "20"))
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]
A = Path(__file__).resolve().parents[1] / "artifacts"

def main():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    data, feat = build_pool(engine)
    tr = block(data, upto=TRAIN_END)                 # purged, as in extract_rules_from_tree.py
    te = block(data, after=TRAIN_END, purge=False)
    X, y = tr[feat].values, tr["cls3"].values
    Xte = te[feat].values
    print(f"train={len(tr)}  held-out={len(te)}  features={len(feat)}  bootstraps={B}")

    base = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                                  random_state=SEED).fit(X, y)
    base_pred = base.predict(Xte)
    base_feats = {feat[i] for i in base.tree_.feature if i >= 0}

    rng = np.random.default_rng(SEED)
    preds, n_rules, featsets = [], [], []
    for b in range(B):
        idx = rng.integers(0, len(X), len(X))                 # bootstrap resample
        t = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                                   random_state=SEED).fit(X[idx], y[idx])
        preds.append(t.predict(Xte))
        n_rules.append(int(t.get_n_leaves()))
        featsets.append({feat[i] for i in t.tree_.feature if i >= 0})

    P = np.vstack(preds)
    pair = [float(np.mean(P[i] == P[j])) for i in range(B) for j in range(i + 1, B)]
    vs_base = [float(np.mean(p == base_pred)) for p in preds]
    jac = [len(s & base_feats) / len(s | base_feats) for s in featsets]

    # how often each feature is picked across bootstraps
    cnt = pd.Series([f for s in featsets for f in s]).value_counts() / B
    top = cnt.head(10)

    summary = pd.DataFrame([dict(
        bootstraps=B,
        rules_published=int(base.get_n_leaves()),
        rules_mean=round(float(np.mean(n_rules)), 1),
        rules_sd=round(float(np.std(n_rules)), 1),
        pairwise_action_agreement=round(float(np.mean(pair)), 3),
        agreement_with_published=round(float(np.mean(vs_base)), 3),
        feature_set_jaccard=round(float(np.mean(jac)), 3),
        features_used_in_all_runs=int((cnt >= 1.0).sum()),
        features_used_in_80pct=int((cnt >= 0.8).sum()),
    )])
    summary.to_csv(A / "rule_stability.csv", index=False)
    print("\n=== RULE-BASE STABILITY (bootstrap refits of the CART tree) ===")
    print(summary.to_string(index=False))
    print("\nTop features by selection frequency across bootstraps:")
    print(top.round(2).to_string())
    print(f"\n[ARTIFACT] {A/'rule_stability.csv'}")


if __name__ == "__main__":
    main()
