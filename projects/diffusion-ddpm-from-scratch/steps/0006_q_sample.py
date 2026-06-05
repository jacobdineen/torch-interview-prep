"""
Step 0006: q_sample

Part 2 — Forward Diffusion Process
Apply the closed-form forward noising to produce x_t from x0 and noise at timesteps t.
"""
import torch  # noqa: F401


def q_sample(x0, t, noise, alpha_bars):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
