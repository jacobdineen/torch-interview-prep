"""
Step 0025: accumulate_gradients

Part 3 — SFT Training Loop
Average a list of per-microbatch gradient lists into one gradient list.
"""
import torch  # noqa: F401


def accumulate_gradients(microbatch_grads):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
