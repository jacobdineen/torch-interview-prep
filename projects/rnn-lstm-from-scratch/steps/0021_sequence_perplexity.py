"""
Step 0021: sequence_perplexity

Part 5 — Sampling & Evaluation
Return the perplexity (exp of mean cross-entropy) of the model on the given data.
"""
import torch  # noqa: F401


def sequence_perplexity(params, X_onehot, targets, forward_fn):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
