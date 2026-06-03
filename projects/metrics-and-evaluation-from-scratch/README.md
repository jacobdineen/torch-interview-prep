# Metrics & Evaluation From Scratch

The model-evaluation toolkit you're often asked to implement by hand in
interviews, in pure NumPy: train/test splitting, k-fold cross-validation, the
confusion matrix, precision/recall/F1 and macro-F1, a tie-safe ROC AUC, and the
regression metrics (MAE, RMSE, R²). No scikit-learn.

**10 steps across 3 parts.** Each step is one function in `steps/NNNN_<fn>.py`,
graded in isolation against a hidden reference with exact, deterministic checks.

## Parts

| # | Part | Steps | Focus |
|---|------|-------|-------|
| 1 | Splitting & Folds | 2 | seeded train/test split, contiguous k-fold indices |
| 2 | Classification Metrics | 4 | confusion matrix, accuracy, precision/recall/F1, macro-F1 |
| 3 | Ranking & Regression Metrics | 4 | ROC AUC (rank formula, tie-safe), MAE, RMSE, R² |

## Working through it

```bash
uv run python projects.py metrics-and-evaluation-from-scratch              # parts + steps
uv run python projects.py metrics-and-evaluation-from-scratch --next       # next unsolved step
uv run python projects.py metrics-and-evaluation-from-scratch 0007 --explain
```

In Neovim, the `<leader>p` maps work on these step files too (or use the web app).

## End-to-end scaffold

```bash
uv run python projects.py metrics-and-evaluation-from-scratch --scaffold
```

Splits a toy dataset, then scores a toy classifier (accuracy, P/R/F1, macro-F1,
ROC AUC) and a toy regressor (MAE/RMSE/R²) entirely with your solved functions.
