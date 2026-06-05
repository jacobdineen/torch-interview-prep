"""
Step 0037: adam_update_m

Part 4 — Fused Loss and Optimizers
Adam first-moment (mean) EMA update: beta1*m + (1-beta1)*grad.
"""
import numpy as np  # noqa: F401


def adam_update_m(m, grad, beta1):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
