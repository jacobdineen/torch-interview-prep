"""
Step 0041: assign_value_targets

Part 5 — Self-Play Data Generation
Turn (state, policy, player) records into (state, policy, value) where value is
+1 if that player won, -1 if they lost, 0 for a draw.
"""
import torch  # noqa: F401


def assign_value_targets(records, winner):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
