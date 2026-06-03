"""Hidden reference implementations for classical-ml-from-scratch.

One top-level def per step, named exactly as in spec.STEPS. Pure NumPy — no
autograd, no framework. Steps may call earlier reference functions directly.
"""
import numpy as np


# ===== Part 1: Linear Regression =====

def predict_linear(X, w, b):
    """Linear model prediction: X @ w + b."""
    return X @ w + b


def mse_loss(y_pred, y_true):
    """Mean squared error between predictions and targets (a float)."""
    return float(np.mean((y_pred - y_true) ** 2))


def linreg_normal_equation(X, y):
    """Least-squares weights solving X w ~= y (X already includes any bias column)."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def linreg_gradient(X, y, w):
    """Gradient of the MSE (predictions X @ w) w.r.t. w: (2/n) X^T (X w - y)."""
    n = X.shape[0]
    return (2.0 / n) * (X.T @ (X @ w - y))


def r2_score(y_pred, y_true):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot (a float)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


# ===== Part 2: Logistic Regression =====

def sigmoid(z):
    """Numerically stable logistic sigmoid, elementwise."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_loss(p, y):
    """Mean binary cross-entropy; p are predicted probabilities in (0, 1)."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def logreg_predict_proba(X, w, b):
    """Predicted P(y=1) = sigmoid(X w + b)."""
    return sigmoid(X @ w + b)


def logreg_gradient(X, y, p):
    """Gradient of mean BCE w.r.t. w given predicted probs p: (1/n) X^T (p - y)."""
    n = X.shape[0]
    return (X.T @ (p - y)) / n


def accuracy(y_pred, y_true):
    """Fraction of exactly-correct predictions (a float)."""
    return float(np.mean(np.asarray(y_pred) == np.asarray(y_true)))


# ===== Part 3: K-Means Clustering =====

def pairwise_sq_dists(X, C):
    """(n, k) squared Euclidean distances between rows of X (n,d) and C (k,d)."""
    return (np.sum(X ** 2, axis=1)[:, None]
            + np.sum(C ** 2, axis=1)[None, :]
            - 2.0 * X @ C.T)


def assign_clusters(X, C):
    """Index of the nearest centroid for each row of X."""
    return np.argmin(pairwise_sq_dists(X, C), axis=1)


def update_centroids(X, labels, k):
    """New centroids = mean of the points assigned to each of k clusters
    (an empty cluster keeps a zero row)."""
    d = X.shape[1]
    C = np.zeros((k, d))
    for j in range(k):
        pts = X[labels == j]
        if len(pts):
            C[j] = pts.mean(axis=0)
    return C


def inertia(X, labels, C):
    """Sum of squared distances from each point to its assigned centroid (a float)."""
    diff = X - C[labels]
    return float(np.sum(diff ** 2))


# ===== Part 4: PCA =====

def center_data(X):
    """Subtract the per-column mean from X."""
    return X - X.mean(axis=0)


def covariance_matrix(Xc):
    """Sample covariance of already-centered data: Xc^T Xc / (n - 1)."""
    n = Xc.shape[0]
    return (Xc.T @ Xc) / (n - 1)


def pca_fit(Xc, k):
    """Top-k principal components of centered data Xc.
    Returns (components, explained_variance): components is (k, d) with one unit
    eigenvector per row, explained_variance is (k,), both sorted by descending
    eigenvalue."""
    cov = covariance_matrix(Xc)
    vals, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return vecs[:, :k].T, vals[:k]


def pca_transform(Xc, components):
    """Project centered data onto the components: Xc @ components.T -> (n, k)."""
    return Xc @ components.T
