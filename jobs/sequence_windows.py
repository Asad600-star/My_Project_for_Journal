"""Input windows of the deep baselines (kept free of PyTorch so that the unit tests
can check them without the deep-learning stack)."""
from __future__ import annotations

import numpy as np


def make_sequences(X_all, positions, y, seq_len):
    """Input windows X_all[p - seq_len + 1 .. p] (the decision day p included) for the
    rows at the given positions of the full feature matrix. The window of the first
    validation or test day reaches back into the preceding block, which is past
    information, so every evaluation row gets a forecast and the deep baselines are
    scored on exactly the same rows as the other models. Training rows without a
    full window are skipped."""
    positions = np.asarray(positions)
    keep = positions >= seq_len - 1
    pos = positions[keep]
    Xs = np.zeros((len(pos), seq_len, X_all.shape[1]), np.float32)
    for j, p in enumerate(pos):
        Xs[j] = X_all[p - seq_len + 1:p + 1]
    return Xs, np.asarray(y, dtype=np.float32)[keep]
