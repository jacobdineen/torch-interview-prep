"""
Step 0021: predict_classes

Part 7 — Training Loop & Inference
Return the predicted integer class for each row as the argmax of the MLP logits.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def predict_classes(params, X):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
