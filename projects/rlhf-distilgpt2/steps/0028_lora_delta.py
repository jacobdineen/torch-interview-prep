"""
Step 0028: lora_delta

Part 4 — LoRA Adapters
Low-rank update applied to x: (alpha/r) * (x @ A^T) @ B^T.
A is (r, in), B is (out, r); result is (..., out).
"""
import torch  # noqa: F401


def lora_delta(x, lora_a, lora_b, alpha, r):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
