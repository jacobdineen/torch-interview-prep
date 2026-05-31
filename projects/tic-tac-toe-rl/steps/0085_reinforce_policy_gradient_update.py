"""
Step 0085: reinforce_policy_gradient_update

Part 6 — Policy Gradients & Extensions
Per-step gradient of the REINFORCE loss (-sum_t G_t log pi(a_t)) w.r.t. the
logits: G_t * (softmax(logits_t) - onehot(a_t)). Shape (T, num_actions).
"""
import numpy as np  # noqa: F401


def reinforce_policy_gradient_update(logits_list, actions, returns):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
