"""
Step 0154: wire_full_training_loop

Part 8 — Adam, Training Loop, and Generation
One full training step: forward, cross-entropy loss, backward, and an Adam
update over the whole parameter tree. ``opt_state`` is None on the first call.
Returns (params, opt_state, loss).
"""
import numpy as np  # noqa: F401


def wire_full_training_loop(params, opt_state, x, y, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
