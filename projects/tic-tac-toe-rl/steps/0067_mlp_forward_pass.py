"""
Step 0067: mlp_forward_pass

Part 5 — Deep Q-Network Agent
Forward pass of the MLP: relu(x@W1+b1)@W2+b2. x is (batch, input_dim).
"""
import numpy as np  # noqa: F401


def mlp_forward_pass(params, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
