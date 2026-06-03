"""Metrics & Evaluation From Scratch — end-to-end demo.

Run once you've solved (enough of) the steps:

    python projects.py metrics-and-evaluation-from-scratch --scaffold

Imports your assembled solution.py and runs the whole thing end-to-end. Keep it
small enough to finish in well under a minute on CPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from solution import *  # noqa: E402,F401,F403


def main():
    rng = np.random.default_rng(0)

    # A toy binary classifier: scores correlate with the true label + noise.
    n = 400
    y = rng.integers(0, 2, n)
    scores = np.clip(0.5 * y + 0.25 * rng.standard_normal(n) + 0.25, 0, 1)
    preds = (scores >= 0.5).astype(int)

    X = np.zeros((n, 1))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.2, seed=1)
    print(f"[split] train={len(Xtr)} test={len(Xte)};  3-fold sizes "
          f"{[len(te) for _, te in k_fold_indices(n, 3)]}")

    cm = confusion_matrix(y, preds, 2)
    p, r, f1 = precision_recall_f1(y, preds)
    print(f"[classification] acc={accuracy_score(y, preds):.3f}  "
          f"P={p:.3f} R={r:.3f} F1={f1:.3f}  macroF1={macro_f1(cm):.3f}")
    print(f"[ranking] ROC AUC = {roc_auc(scores, y):.3f}")

    # A toy regressor.
    yt = rng.standard_normal(n)
    yp = yt + 0.1 * rng.standard_normal(n)
    print(f"[regression] MAE={mean_absolute_error(yt, yp):.3f}  "
          f"RMSE={rmse(yt, yp):.3f}  R^2={r2_score(yt, yp):.3f}")


if __name__ == "__main__":
    main()
