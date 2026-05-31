"""
Step 0077: build_target_network_copy

Part 5 — Deep Q-Network Agent
A detached copy of the parameters for the target network.
"""
import numpy as np  # noqa: F401


def build_target_network_copy(params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
