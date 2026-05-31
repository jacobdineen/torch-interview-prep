"""
Step 0056: perspective_reward_sign

Part 4 — Self-Play, Evaluation & Persistence
Sign to convert an X-perspective reward to ``player``'s perspective (+1/-1).
"""
import numpy as np  # noqa: F401


def perspective_reward_sign(player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
