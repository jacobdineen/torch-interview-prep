"""
Step 0072: run_training_loop_for_steps

Part 10 — Training Step and Loop
Train on a fixed batch for n_steps using a Noam-scheduled learning rate per step and return the list of per-step loss values.
"""
import torch  # noqa: F401


def run_training_loop_for_steps(params, src_ids, tgt_ids, n_steps, d_model, warmup, smoothing):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
