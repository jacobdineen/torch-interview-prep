"""Build Flash Attention from Scratch — end-to-end demo.

    python projects.py flash-attention-from-scratch --scaffold

Shows the whole point of FlashAttention: tiled, online-softmax attention that
matches standard attention to machine precision while never materializing the
full N x M score matrix — plus a causal variant and a backward pass whose
gradients match finite differences. Pure NumPy.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution import *  # noqa: F401,F403


def main():
    rng = np.random.default_rng(0)
    N, M, d = 64, 64, 16
    scale = 1.0 / np.sqrt(d)
    Q = rng.standard_normal((N, d))
    K = rng.standard_normal((M, d))
    V = rng.standard_normal((M, d))

    # (1) Tiled flash forward matches full attention exactly, block-by-block.
    O_ref = attention_reference(Q, K, V, scale)
    O_flash, L = flash_attention_forward(Q, K, V, scale, block_size=16)
    fwd_err = np.max(np.abs(O_ref - O_flash))
    print("flash forward vs reference  max|diff| = %.2e  (block_size=16, never builds the %dx%d matrix)"
          % (fwd_err, N, M))

    # (2) Numerical stability: huge scores would overflow a naive exp; online softmax is fine.
    big = Q * 50.0
    O_big, _ = flash_attention_forward(big, K, V, scale, block_size=16)
    print("large-magnitude scores: output finite =", bool(np.all(np.isfinite(O_big))))

    # (3) Causal flash matches the masked reference.
    O_causal_ref = masked_attention_reference(Q, K, V, scale)
    O_causal = flash_attention_causal(Q, K, V, scale, block_size=16)
    causal_err = np.max(np.abs(O_causal_ref - O_causal))
    print("causal flash vs masked reference  max|diff| = %.2e" % causal_err)

    # (4) Backward pass: recompute P from the saved L, check grads vs finite differences.
    G = rng.standard_normal((N, d))
    dQ, dK, dV = flash_attention_backward(Q, K, V, O_ref, G, L, scale)

    def loss(Qx, Kx, Vx):
        return np.sum(attention_reference(Qx, Kx, Vx, scale) * G)

    eps = 1e-6
    num_dQ = np.zeros_like(Q)
    for i in range(N):
        for j in range(d):
            Qp = Q.copy(); Qp[i, j] += eps
            Qm = Q.copy(); Qm[i, j] -= eps
            num_dQ[i, j] = (loss(Qp, K, V) - loss(Qm, K, V)) / (2 * eps)
    bwd_err = np.max(np.abs(num_dQ - dQ))
    print("backward dQ vs finite differences  max|diff| = %.2e" % bwd_err)

    ok = fwd_err < 1e-10 and causal_err < 1e-10 and bwd_err < 1e-5 and np.all(np.isfinite(O_big))
    print("ok" if ok else "warning")


if __name__ == "__main__":
    main()
