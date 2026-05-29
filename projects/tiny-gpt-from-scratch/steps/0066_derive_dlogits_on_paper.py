"""
Step 0066: derive_dlogits_on_paper

Part 4 — Single-Layer Neural Bigram
Gradient of softmax+cross-entropy w.r.t. the logits: (probs - onehot)/N.
"""
import numpy as np  # noqa: F401


def derive_dlogits_on_paper(probs, y):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
