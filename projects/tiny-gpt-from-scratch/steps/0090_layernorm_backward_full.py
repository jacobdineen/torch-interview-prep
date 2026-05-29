"""
Step 0090: layernorm_backward_full

Part 5 — Layer Primitives and Backprop
Full input gradient dx of LayerNorm(x) * gamma, over the feature axis.
"""
import numpy as np  # noqa: F401


def layernorm_backward_full(dout, x, gamma, eps):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
