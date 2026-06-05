"""
Step 0071: run_training_step_with_backprop

Part 10 — Training Step and Loop
Run one optimization step: zero grads, compute the batch loss, backpropagate, apply an Adam update, and return the new optimizer state with the float loss value.
"""
import torch  # noqa: F401


def run_training_step_with_backprop(params, param_list, opt_state, src_ids, tgt_ids, lr, smoothing):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
