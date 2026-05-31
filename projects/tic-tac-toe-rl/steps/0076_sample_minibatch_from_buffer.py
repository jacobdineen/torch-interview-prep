"""
Step 0076: sample_minibatch_from_buffer

Part 5 — Deep Q-Network Agent
Sample ``batch_size`` transitions uniformly (with replacement).
"""
import numpy as np  # noqa: F401


def sample_minibatch_from_buffer(buffer, batch_size, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
