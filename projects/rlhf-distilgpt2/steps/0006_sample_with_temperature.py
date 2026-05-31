"""
Step 0006: sample_with_temperature

Part 1 — Model Setup and Decoding Strategies
Sample a token id from softmax(logits / temperature).
"""
import torch  # noqa: F401


def sample_with_temperature(logits, temperature, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
