"""
Step 0153: adam_parameter_update

Part 8 — Adam, Training Loop, and Generation
Adam step: param - lr * mhat / (sqrt(vhat) + eps).
"""
import numpy as np  # noqa: F401


def adam_parameter_update(param, m_hat, v_hat, lr, eps):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
