"""Hidden reference implementations for metrics-and-evaluation-from-scratch.

One top-level def per step, named exactly as in spec.STEPS. Pure NumPy — the
model-evaluation toolkit you're often asked to write by hand in interviews.
"""
import numpy as np


# ===== Part 1: Splitting & Folds =====

def train_test_split(X, y, test_frac=0.25, seed=0):
    """Shuffle (seeded) and split into (X_train, X_test, y_train, y_test); the
    test set is round(n * test_frac) rows."""
    X, y = np.asarray(X), np.asarray(y)
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_frac))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def k_fold_indices(n, k):
    """Return k (train_idx, test_idx) pairs of contiguous folds over range(n).
    The test folds partition all n indices; train is the complement."""
    folds = np.array_split(np.arange(n), k)
    out = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i]) if k > 1 else np.array([], int)
        out.append((tr, te))
    return out


# ===== Part 2: Classification Metrics =====

def confusion_matrix(y_true, y_pred, num_classes):
    """(num_classes, num_classes) integer matrix; rows are true, cols predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def accuracy_score(y_true, y_pred):
    """Fraction of exactly-correct predictions (a float)."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_recall_f1(y_true, y_pred):
    """Binary precision, recall, F1 for the positive class (label 1), as a tuple.
    Each is 0.0 when its denominator is zero."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def macro_f1(cm):
    """Unweighted mean of per-class F1 from a confusion matrix (rows=true)."""
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


# ===== Part 3: Ranking & Regression Metrics =====

def roc_auc(scores, y_true):
    """Binary ROC AUC via the rank (Mann-Whitney) formula, tie-safe. y in {0,1};
    returns 0.5 if a class is absent."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y_true)
    n_pos = int(np.sum(y == 1)); n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # 1-based average rank
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mean_absolute_error(y_true, y_pred):
    """Mean absolute error (a float)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    """Root mean squared error (a float)."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score(y_true, y_pred):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot (a float)."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)
