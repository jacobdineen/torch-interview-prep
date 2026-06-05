"""
Step 0068: apply_adam_step_to_all_parameters

Part 9 — Adam Optimizer From Scratch
Advance the step, update moments and apply the in-place Adam parameter update for every parameter with a gradient.
"""
import torch  # noqa: F401


def apply_adam_step_to_all_parameters(param_list, opt_state, lr, beta1=0.9, beta2=0.98, eps=1e-09):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
