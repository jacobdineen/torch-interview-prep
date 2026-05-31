"""
Step 0074: append_transition_to_buffer

Part 5 — Deep Q-Network Agent
Append a (state, action, reward, next_state, done) transition; return buffer.
"""
import numpy as np  # noqa: F401


def append_transition_to_buffer(buffer, transition):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
