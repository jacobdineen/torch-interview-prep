# Classical ML From Scratch

The pre-deep-learning toolkit, in pure NumPy: linear and logistic regression,
k-means clustering, and PCA — built from the math up, no scikit-learn. A
deliberately *different* project from the neural ones (tiny-gpt, alphazero, …):
closed-form solutions, convex losses, and unsupervised methods.

**18 steps across 4 parts.** Each step is one function in `steps/NNNN_<fn>.py`,
graded in isolation against a hidden reference (gradients are finite-difference
checked).

## Parts

| # | Part | Steps | Focus |
|---|------|-------|-------|
| 1 | Linear Regression | 5 | predict, MSE, the normal equation, the gradient, R² |
| 2 | Logistic Regression | 5 | stable sigmoid, BCE, predicted probabilities, the gradient, accuracy |
| 3 | K-Means Clustering | 4 | pairwise distances, assignment, centroid updates, inertia |
| 4 | PCA | 4 | centering, covariance, top components (eigendecomposition), projection |

## Working through it

```bash
uv run python projects.py classical-ml-from-scratch              # parts + steps, [x]/[ ] solved
uv run python projects.py classical-ml-from-scratch --next       # jump to the next unsolved step
uv run python projects.py classical-ml-from-scratch 0001         # check one step
uv run python projects.py classical-ml-from-scratch 0001 --explain
```

In Neovim, the `<leader>p` maps work on these step files too (or use the web app).

## End-to-end scaffold

```bash
uv run python projects.py classical-ml-from-scratch --scaffold
```

Fits linear regression two ways (closed form vs gradient descent, R² ≈ 0.998),
trains logistic regression on two Gaussian blobs to ~100% accuracy, runs k-means
to convergence on three blobs, and does PCA on correlated data — all from your
solved steps.
