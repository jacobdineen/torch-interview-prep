"""
Step 0069: zero_all_parameter_gradients

Part 9 — Adam Optimizer From Scratch
Clear the stored gradient of every parameter so the next backward pass starts fresh.
"""
import torch  # noqa: F401


def zero_all_parameter_gradients(param_list):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
