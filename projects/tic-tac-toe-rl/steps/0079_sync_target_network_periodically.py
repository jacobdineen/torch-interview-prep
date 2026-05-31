"""
Step 0079: sync_target_network_periodically

Part 5 — Deep Q-Network Agent
Every ``sync_every`` steps, copy the online params into the target net.
"""
import numpy as np  # noqa: F401


def sync_target_network_periodically(params, target_params, step, sync_every):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
