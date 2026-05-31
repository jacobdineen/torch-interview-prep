"""
Step 0024: clip_grad_norm

Part 3 — SFT Training Loop
Scale a list of gradient tensors so their global L2 norm is at most max_norm.
"""
import torch  # noqa: F401


def clip_grad_norm(grads, max_norm):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
