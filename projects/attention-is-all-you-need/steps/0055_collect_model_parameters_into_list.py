"""
Step 0055: collect_model_parameters_into_list

Part 7 — Parameter Initialization
Gather every trainable leaf tensor in the model (embedding, output bias, and all encoder/decoder layer weights, biases, gammas, betas) into a single deterministically ordered list.
"""
import torch  # noqa: F401


def collect_model_parameters_into_list(params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
