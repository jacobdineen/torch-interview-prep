"""
Step 0007: sample_timesteps_and_noise

Part 2 — Forward Diffusion Process
Draw random integer timesteps in [0, T) and standard-normal noise shaped like x0.
"""
import torch  # noqa: F401


def sample_timesteps_and_noise(x0, T, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
