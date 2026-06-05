"""
Step 0005: pad_id_sequence

Part 1 — Tokenization and Batching
Right-pad with pad_id or truncate the id tensor to exactly `length`, returning a LongTensor.
"""
import torch  # noqa: F401


def pad_id_sequence(ids, length, pad_id=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
