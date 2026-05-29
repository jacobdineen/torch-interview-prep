"""
Step 0155: logging_and_validation_loss

Part 8 — Adam, Training Loop, and Generation
Mean validation cross-entropy over a few random batches (no parameter update).
"""
import numpy as np  # noqa: F401


def logging_and_validation_loss(params, val_ids, block_size, batch_size, n_eval_batches):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
