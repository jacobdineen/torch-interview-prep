"""
Step 0050: training_epoch

Part 6 — Losses and Training Loop
One pass over ``data`` in minibatches. Returns the mean loss.
"""
import torch  # noqa: F401


def training_epoch(net, data, optimizer, batch_size, weight_decay, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
