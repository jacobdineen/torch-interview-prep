"""
Step 0054: init_embedding_and_projection_parameters

Part 7 — Parameter Initialization
Create the token embedding matrix (vocab, d_model) and the output projection bias (vocab,) as leaf tensors that require gradients.
"""
import torch  # noqa: F401


def init_embedding_and_projection_parameters(vocab_size, d_model):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
