"""
Step 0018: flash_attention_backward

Part 5 — Flash Backward (Recomputation)
Gradients dQ, dK, dV, recomputing the softmax from the saved log-sum-exp instead of storing it.

Return (dQ, dK, dV). Recompute S=attention_scores(Q,K,scale); P=exp(S - L[:,None]) (P is the normalized softmax, since L is the full log-sum-exp); D=attention_backward_D(dO,O); then dV=P.T@dO, dP=dO@V.T, dS=P*(dP - D[:,None]), dQ=scale*(dS@K), dK=scale*(dS.T@Q). Shapes: dQ (N,d), dK (M,d), dV (M,d).

Conventions: single-head, no batch. Q (N,d), K (M,d), V (M,d), output O (N,d); scores S (N,M);
per-row stats m, l, L, D are shape (N,). `scale` is the scalar the caller passes (1/sqrt(d)).
The flash forward tiles over key/value blocks and keeps an UNNORMALIZED running output that is
divided by the running denominator l only at the end.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py flash-attention-from-scratch` (or the outline drawer) to see all signatures."""
import numpy as np  # noqa: F401


def flash_attention_backward(Q, K, V, O, dO, L, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
