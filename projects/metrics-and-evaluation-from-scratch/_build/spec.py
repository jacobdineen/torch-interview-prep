"""Build spec for the metrics-and-evaluation-from-scratch project.

PARTS: (title, description) in order.
STEPS: (name, part_index) in order. The step id is its 1-based position,
zero-padded to 4 digits (0001, 0002, ...).
"""

TITLE = 'Metrics & Evaluation From Scratch'

PARTS = [
    ("Splitting & Folds",
     "Seeded train/test splitting and contiguous k-fold cross-validation indices."),
    ("Classification Metrics",
     "Confusion matrix, accuracy, binary precision/recall/F1, and macro-F1."),
    ("Ranking & Regression Metrics",
     "Tie-safe ROC AUC, plus MAE, RMSE, and R^2."),
]

# (function_or_class_name, part_index) in solve order.
STEPS = [
    # Part 1 — Splitting & Folds
    ("train_test_split", 0), ("k_fold_indices", 0),
    # Part 2 — Classification Metrics
    ("confusion_matrix", 1), ("accuracy_score", 1),
    ("precision_recall_f1", 1), ("macro_f1", 1),
    # Part 3 — Ranking & Regression Metrics
    ("roc_auc", 2), ("mean_absolute_error", 2), ("rmse", 2), ("r2_score", 2),
]


def step_id(i):
    return f"{i:04d}"
