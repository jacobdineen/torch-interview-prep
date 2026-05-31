"""
Step 0016: board_to_torch_tensor

Part 2 — Board Encoding and Policy-Value Network
Turn an encoded board into a batched float tensor (1, C, 6, 7).
"""
import torch  # noqa: F401


def board_to_torch_tensor(encoded):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
