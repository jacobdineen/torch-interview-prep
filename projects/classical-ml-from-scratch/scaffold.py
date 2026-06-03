"""Classical ML From Scratch — end-to-end demo.

Run once you've solved (enough of) the steps:

    python projects.py classical-ml-from-scratch --scaffold

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

    # --- Linear regression: closed form vs gradient descent ---
    n, d = 200, 3
    Xf = rng.standard_normal((n, d))
    X = np.hstack([np.ones((n, 1)), Xf])             # bias column
    w_true = np.array([0.5, 2.0, -1.0, 0.7])
    y = X @ w_true + 0.1 * rng.standard_normal(n)
    w = linreg_normal_equation(X, y)        # w[0] is the bias, w[1:] the feature weights
    print(f"[linreg] normal-equation R^2 = {r2_score(predict_linear(Xf, w[1:], w[0]), y):.4f}")
    wg = np.zeros(d + 1)
    for _ in range(2000):
        wg -= 0.1 * linreg_gradient(X, y, wg)
    print(f"[linreg] gradient-descent loss {mse_loss(X @ wg, y):.4f} (vs {mse_loss(X @ w, y):.4f})")

    # --- Logistic regression on two Gaussian blobs ---
    m = 300
    Xpos = rng.standard_normal((m, 2)) + np.array([2.0, 2.0])
    Xneg = rng.standard_normal((m, 2)) + np.array([-2.0, -2.0])
    Xl = np.vstack([Xpos, Xneg]); yl = np.concatenate([np.ones(m), np.zeros(m)])
    w, b = np.zeros(2), 0.0
    for _ in range(500):
        p = logreg_predict_proba(Xl, w, b)
        w -= 0.1 * logreg_gradient(Xl, yl, p)
        b -= 0.1 * float(np.mean(p - yl))
    preds = (logreg_predict_proba(Xl, w, b) >= 0.5).astype(float)
    print(f"[logreg] accuracy = {accuracy(preds, yl):.3f}, BCE = {bce_loss(logreg_predict_proba(Xl, w, b), yl):.4f}")

    # --- K-means on three blobs ---
    centers = np.array([[0., 0.], [6., 6.], [0., 8.]])
    Xk = np.vstack([c + rng.standard_normal((80, 2)) for c in centers])
    C = Xk[rng.choice(len(Xk), 3, replace=False)]
    for _ in range(10):
        labels = assign_clusters(Xk, C)
        C = update_centroids(Xk, labels, 3)
    print(f"[kmeans] final inertia = {inertia(Xk, assign_clusters(Xk, C), C):.1f}")

    # --- PCA on correlated data ---
    A = rng.standard_normal((4, 4))
    Xp = rng.standard_normal((300, 4)) @ A
    Xc = center_data(Xp)
    comps, ev = pca_fit(Xc, 2)
    ratio = ev / np.trace(covariance_matrix(Xc))
    print(f"[pca] top-2 explained variance ratio = {ratio.round(3)}  (proj shape {pca_transform(Xc, comps).shape})")


if __name__ == "__main__":
    main()
