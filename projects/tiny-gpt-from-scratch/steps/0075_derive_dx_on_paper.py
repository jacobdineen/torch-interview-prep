"""
Step 0075: derive_dx_on_paper

Part 5 — Layer Primitives and Backprop
For y = x @ W, the input gradient is dL/dx = dout @ W.T.
"""
import numpy as np  # noqa: F401


def derive_dx_on_paper(dout, w):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
