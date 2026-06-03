"""Build spec for the classical-ml-from-scratch project.

PARTS: (title, description) in order.
STEPS: (name, part_index) in order. The step id is its 1-based position,
zero-padded to 4 digits (0001, 0002, ...).
"""

TITLE = 'Classical ML From Scratch'

PARTS = [
    ("Linear Regression",
     "Predict, score, and fit a linear model in closed form and by gradient descent."),
    ("Logistic Regression",
     "A numerically stable sigmoid, BCE loss, predicted probabilities, and the gradient."),
    ("K-Means Clustering",
     "Pairwise distances, cluster assignment, centroid updates, and inertia."),
    ("PCA",
     "Center data, build the covariance, find the top components, and project."),
]

# (function_or_class_name, part_index) in solve order.
STEPS = [
    # Part 1 — Linear Regression
    ("predict_linear", 0), ("mse_loss", 0), ("linreg_normal_equation", 0),
    ("linreg_gradient", 0), ("r2_score", 0),
    # Part 2 — Logistic Regression
    ("sigmoid", 1), ("bce_loss", 1), ("logreg_predict_proba", 1),
    ("logreg_gradient", 1), ("accuracy", 1),
    # Part 3 — K-Means Clustering
    ("pairwise_sq_dists", 2), ("assign_clusters", 2), ("update_centroids", 2),
    ("inertia", 2),
    # Part 4 — PCA
    ("center_data", 3), ("covariance_matrix", 3), ("pca_fit", 3), ("pca_transform", 3),
]


def step_id(i):
    return f"{i:04d}"
