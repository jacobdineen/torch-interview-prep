"""Hidden tests for classical-ml-from-scratch. One test_<id>_<name>(ns) per step;
assert on ns["<name>"]. Pure NumPy, deterministic fixtures, fast."""
import numpy as np


# ===== Part 1: Linear Regression =====

def test_0001_predict_linear(ns):
    X = np.array([[1., 2.], [3., 4.]]); w = np.array([1., 1.]); b = 0.5
    assert np.allclose(ns["predict_linear"](X, w, b), [3.5, 7.5])


def test_0002_mse_loss(ns):
    assert abs(ns["mse_loss"](np.array([1., 2., 3.]), np.array([1., 2., 5.])) - 4.0 / 3.0) < 1e-9


def test_0003_linreg_normal_equation(ns):
    X = np.array([[1., 1.], [1., 2.], [1., 3.]]); y = np.array([1., 2., 3.])
    assert np.allclose(ns["linreg_normal_equation"](X, y), [0., 1.], atol=1e-8)


def test_0004_linreg_gradient(ns):
    X = np.array([[1., 1.], [1., 2.], [1., 3.]]); y = np.array([1., 2., 3.])
    w = np.array([0.2, 0.3])
    g = ns["linreg_gradient"](X, y, w)

    def L(ww):
        r = X @ ww - y
        return np.mean(r ** 2)
    num = np.zeros(2); eps = 1e-6
    for i in range(2):
        wp = w.copy(); wp[i] += eps; wm = w.copy(); wm[i] -= eps
        num[i] = (L(wp) - L(wm)) / (2 * eps)
    assert np.allclose(g, num, atol=1e-5)


def test_0005_r2_score(ns):
    assert abs(ns["r2_score"](np.array([1., 2., 3.]), np.array([1., 2., 3.])) - 1.0) < 1e-9
    assert abs(ns["r2_score"](np.array([2., 2., 2.]), np.array([1., 2., 3.])) - 0.0) < 1e-9


# ===== Part 2: Logistic Regression =====

def test_0006_sigmoid(ns):
    s = ns["sigmoid"]
    assert abs(float(s(np.array([0.0]))[0]) - 0.5) < 1e-12
    big = s(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(big)) and big[0] < 1e-9 and big[1] > 1 - 1e-9
    z = np.array([-2., -0.5, 0., 0.5, 2.])
    assert np.allclose(s(z), 1.0 / (1.0 + np.exp(-z)), atol=1e-9)


def test_0007_bce_loss(ns):
    assert abs(ns["bce_loss"](np.array([0.5, 0.5]), np.array([1., 0.])) - (-np.log(0.5))) < 1e-9
    assert ns["bce_loss"](np.array([0.9]), np.array([1.])) < ns["bce_loss"](np.array([0.1]), np.array([1.]))


def test_0008_logreg_predict_proba(ns):
    X = np.array([[1., 2.], [3., 4.]])
    assert np.allclose(ns["logreg_predict_proba"](X, np.array([0., 0.]), 0.0), [0.5, 0.5])
    w = np.array([0.5, -0.5]); b = 0.1
    assert np.allclose(ns["logreg_predict_proba"](X, w, b),
                       1.0 / (1.0 + np.exp(-(X @ w + b))), atol=1e-9)


def test_0009_logreg_gradient(ns):
    X = np.array([[1., 1.], [1., 2.]]); y = np.array([0., 1.])
    assert np.allclose(ns["logreg_gradient"](X, y, np.array([0.5, 0.5])), [0.0, -0.25], atol=1e-9)
    # gradient check against mean BCE of sigmoid(X w)
    w0 = np.array([0.3, -0.2])

    def L(ww):
        p = 1.0 / (1.0 + np.exp(-(X @ ww)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    p0 = 1.0 / (1.0 + np.exp(-(X @ w0)))
    g = ns["logreg_gradient"](X, y, p0)
    num = np.zeros(2); eps = 1e-6
    for i in range(2):
        wp = w0.copy(); wp[i] += eps; wm = w0.copy(); wm[i] -= eps
        num[i] = (L(wp) - L(wm)) / (2 * eps)
    assert np.allclose(g, num, atol=1e-5)


def test_0010_accuracy(ns):
    assert abs(ns["accuracy"](np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1])) - 0.75) < 1e-12


# ===== Part 3: K-Means Clustering =====

def test_0011_pairwise_sq_dists(ns):
    X = np.array([[0., 0.], [3., 4.]]); C = np.array([[0., 0.]])
    assert np.allclose(ns["pairwise_sq_dists"](X, C), [[0.], [25.]], atol=1e-9)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3)); C = rng.standard_normal((2, 3))
    bf = np.array([[np.sum((x - c) ** 2) for c in C] for x in X])
    assert np.allclose(ns["pairwise_sq_dists"](X, C), bf, atol=1e-9)


def test_0012_assign_clusters(ns):
    X = np.array([[0., 0.], [10., 10.], [0.1, 0.]]); C = np.array([[0., 0.], [10., 10.]])
    assert np.array_equal(ns["assign_clusters"](X, C), [0, 1, 0])


def test_0013_update_centroids(ns):
    X = np.array([[0., 0.], [2., 2.], [10., 10.]])
    C = ns["update_centroids"](X, np.array([0, 0, 1]), 2)
    assert np.allclose(C, [[1., 1.], [10., 10.]])
    C2 = ns["update_centroids"](X, np.array([0, 0, 0]), 2)
    assert np.allclose(C2[1], [0., 0.])  # empty cluster -> zero row


def test_0014_inertia(ns):
    X = np.array([[0., 0.], [3., 4.]]); C = np.array([[0., 0.]])
    assert abs(ns["inertia"](X, np.array([0, 0]), C) - 25.0) < 1e-9


# ===== Part 4: PCA =====

def test_0015_center_data(ns):
    X = np.array([[1., 2.], [3., 4.]])
    Xc = ns["center_data"](X)
    assert np.allclose(Xc, [[-1., -1.], [1., 1.]]) and np.allclose(Xc.mean(axis=0), 0.0)


def test_0016_covariance_matrix(ns):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 3)); Xc = X - X.mean(axis=0)
    assert np.allclose(ns["covariance_matrix"](Xc), np.cov(Xc, rowvar=False), atol=1e-9)


def test_0017_pca_fit(ns):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 4)); Xc = X - X.mean(axis=0)
    comps, ev = ns["pca_fit"](Xc, 2)
    assert comps.shape == (2, 4) and ev.shape == (2,)
    assert np.allclose(comps @ comps.T, np.eye(2), atol=1e-8)   # orthonormal rows
    assert ev[0] >= ev[1]                                       # descending
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    assert np.allclose(cov @ comps.T, comps.T * ev, atol=1e-7)  # eigen relation (sign-robust)
    top = np.sort(np.linalg.eigvalsh(cov))[::-1][:2]
    assert np.allclose(ev, top, atol=1e-7)


def test_0018_pca_transform(ns):
    Xc = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    assert np.allclose(ns["pca_transform"](Xc, np.array([[1., 0.], [0., 1.]])), Xc)
    assert np.allclose(ns["pca_transform"](Xc, np.array([[0., 1.]])), [[0.], [1.], [0.]])
