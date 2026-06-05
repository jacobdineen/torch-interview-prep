"""
Step 0009: online_softmax_step

Part 2 — Online Softmax
One fused flash step folding a key/value block into the running (max, denominator, unnormalized output).

Return (m_new, l_new, O_new) in that order, where m_new=update_running_max(m_prev, stable_rowmax(S_block)), p=block_exp_scores(S_block, m_new), l_new=update_normalizer(l_prev, m_prev, m_new, p), and O_new = correction(m_prev, m_new)[:,None]*O_prev + p @ V_block. The old output accumulator O_prev is rescaled by the SAME correction factor as l before adding the new block.

Conventions: single-head, no batch. Q (N,d), K (M,d), V (M,d), output O (N,d); scores S (N,M);
per-row stats m, l, L, D are shape (N,). `scale` is the scalar the caller passes (1/sqrt(d)).
The flash forward tiles over key/value blocks and keeps an UNNORMALIZED running output that is
divided by the running denominator l only at the end.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py flash-attention-from-scratch` (or the outline drawer) to see all signatures."""
import numpy as np  # noqa: F401


def online_softmax_step(m_prev, l_prev, O_prev, S_block, V_block):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
