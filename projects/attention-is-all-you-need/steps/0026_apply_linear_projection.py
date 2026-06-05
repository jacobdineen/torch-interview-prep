"""
Step 0026: apply_linear_projection

Part 4 — Multi-Head Attention
Apply an affine linear map using an (in, out)-shaped weight and bias.
"""
import torch  # noqa: F401


def apply_linear_projection(x, W, b):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
