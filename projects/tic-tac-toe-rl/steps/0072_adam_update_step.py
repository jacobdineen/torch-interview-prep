"""
Step 0072: adam_update_step

Part 5 — Deep Q-Network Agent
One Adam step over the parameter dict. ``state`` is None on the first call.
Returns (new_params, state).
"""
import numpy as np  # noqa: F401


def adam_update_step(params, grads, state, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
