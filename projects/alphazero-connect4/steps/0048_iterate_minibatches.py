"""
Step 0048: iterate_minibatches

Part 6 — Losses and Training Loop
Split ``data`` into minibatches (optionally shuffled). Returns a list of batches.
"""
import torch  # noqa: F401


def iterate_minibatches(data, batch_size, shuffle=False, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
