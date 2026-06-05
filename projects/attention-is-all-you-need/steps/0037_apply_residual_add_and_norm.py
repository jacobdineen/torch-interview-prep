"""
Step 0037: apply_residual_add_and_norm

Part 5 — Feed-Forward, LayerNorm, and Dropout
Add the sublayer output to its input and layer-normalize the sum (post-norm residual).
"""
import torch  # noqa: F401


def apply_residual_add_and_norm(x, sublayer_out, ln_params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
