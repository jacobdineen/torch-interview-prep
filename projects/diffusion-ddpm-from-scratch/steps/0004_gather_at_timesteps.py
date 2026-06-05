"""
Step 0004: gather_at_timesteps

Part 1 — Noise Schedule
Index a 1-D schedule buffer at the given timesteps and reshape to (B,1) for broadcasting.
"""
import torch  # noqa: F401


def gather_at_timesteps(buf, t):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
