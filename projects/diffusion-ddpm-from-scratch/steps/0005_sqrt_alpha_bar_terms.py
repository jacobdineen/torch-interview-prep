"""
Step 0005: sqrt_alpha_bar_terms

Part 2 — Forward Diffusion Process
Return sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t) gathered at timesteps t.
"""
import torch  # noqa: F401


def sqrt_alpha_bar_terms(alpha_bars, t):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
