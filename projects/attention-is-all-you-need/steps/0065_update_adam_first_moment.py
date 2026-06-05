"""
Step 0065: update_adam_first_moment

Part 9 — Adam Optimizer From Scratch
Return the exponential moving average of the gradient (Adam first moment update).
"""
import torch  # noqa: F401


def update_adam_first_moment(m, grad, beta1):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
