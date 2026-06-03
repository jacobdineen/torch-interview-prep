"""Hidden tests for metrics-and-evaluation-from-scratch. One test_<id>_<name>(ns)
per step; assert on ns["<name>"]. Pure NumPy, deterministic, fast."""
import numpy as np


# ===== Part 1: Splitting & Folds =====

def test_0001_train_test_split(ns):
    X = np.arange(20).reshape(10, 2); y = np.arange(10)   # row i = [2i, 2i+1], label i
    Xtr, Xte, ytr, yte = ns["train_test_split"](X, y, 0.3, seed=42)
    assert len(Xte) == 3 and len(Xtr) == 7
    rows = np.vstack([Xtr, Xte]); labels = np.concatenate([ytr, yte])
    assert sorted(labels.tolist()) == list(range(10))     # no rows lost/duplicated
    assert np.array_equal(rows[:, 0] // 2, labels)        # X/y stay aligned
    # deterministic
    Xtr2, _, _, _ = ns["train_test_split"](X, y, 0.3, seed=42)
    assert np.array_equal(Xtr, Xtr2)


def test_0002_k_fold_indices(ns):
    folds = ns["k_fold_indices"](10, 3)
    assert len(folds) == 3
    all_te = np.concatenate([te for _, te in folds])
    assert sorted(all_te.tolist()) == list(range(10))     # test folds partition all
    for tr, te in folds:
        s_tr, s_te = set(tr.tolist()), set(te.tolist())
        assert s_tr | s_te == set(range(10)) and not (s_tr & s_te)


# ===== Part 2: Classification Metrics =====

def test_0003_confusion_matrix(ns):
    yt = [0, 0, 1, 1, 2, 2]; yp = [0, 1, 1, 1, 2, 0]
    cm = ns["confusion_matrix"](yt, yp, 3)
    assert np.array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])


def test_0004_accuracy_score(ns):
    assert abs(ns["accuracy_score"]([1, 0, 1, 1], [1, 0, 0, 1]) - 0.75) < 1e-12


def test_0005_precision_recall_f1(ns):
    p, r, f1 = ns["precision_recall_f1"]([1, 1, 0, 0, 1], [1, 0, 0, 0, 1])
    assert abs(p - 1.0) < 1e-9 and abs(r - 2.0 / 3.0) < 1e-9 and abs(f1 - 0.8) < 1e-9
    # zero-division is safe
    assert ns["precision_recall_f1"]([0, 0], [0, 0]) == (0.0, 0.0, 0.0)


def test_0006_macro_f1(ns):
    cm = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    # per-class F1: 0.5, 0.8, 2/3  ->  mean
    assert abs(ns["macro_f1"](cm) - (0.5 + 0.8 + 2.0 / 3.0) / 3.0) < 1e-9


# ===== Part 3: Ranking & Regression Metrics =====

def test_0007_roc_auc(ns):
    assert abs(ns["roc_auc"]([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9
    # perfectly separable -> 1.0; reversed -> 0.0
    assert abs(ns["roc_auc"]([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) - 1.0) < 1e-9
    assert abs(ns["roc_auc"]([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) - 0.0) < 1e-9
    # ties average to 0.5
    assert abs(ns["roc_auc"]([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 0.5) < 1e-9


def test_0008_mean_absolute_error(ns):
    assert abs(ns["mean_absolute_error"]([1., 2., 3.], [1., 2., 5.]) - 2.0 / 3.0) < 1e-9


def test_0009_rmse(ns):
    assert abs(ns["rmse"]([1., 2., 3.], [1., 2., 5.]) - np.sqrt(4.0 / 3.0)) < 1e-9


def test_0010_r2_score(ns):
    assert abs(ns["r2_score"]([1., 2., 3.], [1., 2., 3.]) - 1.0) < 1e-9
    assert abs(ns["r2_score"]([1., 2., 3.], [2., 2., 2.]) - 0.0) < 1e-9
