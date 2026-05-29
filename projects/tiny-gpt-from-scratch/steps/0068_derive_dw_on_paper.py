"""
Step 0068: derive_dw_on_paper

Part 4 — Single-Layer Neural Bigram
Gradient w.r.t. W. Since logits = W[x], dW = onehot(x).T @ dlogits.
"""
import numpy as np  # noqa: F401


def derive_dw_on_paper(x, dlogits, vocab_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
