"""
Step 0047: encode_batch_states

Part 6 — Losses and Training Loop
Stack a list of encoded (2,6,7) states into a (B,2,6,7) float tensor.
"""
import torch  # noqa: F401


def encode_batch_states(states):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
