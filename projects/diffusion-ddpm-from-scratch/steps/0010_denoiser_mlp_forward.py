"""
Step 0010: denoiser_mlp_forward

Part 3 — Time Embedding & Denoiser
Run the MLP forward with ReLU after every layer except the linear last.
"""
import torch  # noqa: F401


def denoiser_mlp_forward(params, h):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
