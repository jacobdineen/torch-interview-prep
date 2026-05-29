"""
Step 0069: compute_dw_scatter_add

Part 4 — Single-Layer Neural Bigram
Same dW, computed by scattering each row's dlogits into row W[x[b]].
"""
import numpy as np  # noqa: F401


def compute_dw_scatter_add(x, dlogits, vocab_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
