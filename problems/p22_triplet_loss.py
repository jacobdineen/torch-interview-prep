"""
Problem 58: Triplet Margin Loss

For (anchor, positive, negative) triplets in an embedding space:

    L = max( 0, d(a, p) - d(a, n) + margin )

where d is L2 distance (not squared).

  - triplet_margin_loss(anchor, positive, negative, margin=1.0, reduction="mean")
        Shapes: each is (N, D) FloatTensor.
        Returns a 0-D Tensor (or (N,) Tensor if reduction="none").

  - hardest_triplet_loss(embeddings, labels, margin=1.0)
        embeddings: (N, D)
        labels:     (N,) long
        For each anchor i, find the hardest positive (same label, max distance) and
        the hardest negative (different label, min distance), then compute
        triplet_margin_loss with margin. Average over anchors.
        Assume each class has at least 2 samples in the batch.
"""

import torch


def triplet_margin_loss(anchor, positive, negative, margin=1.0, reduction="mean"):
    raise NotImplementedError


def hardest_triplet_loss(embeddings, labels, margin=1.0):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
