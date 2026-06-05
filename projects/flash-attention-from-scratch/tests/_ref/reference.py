"""Hidden reference implementations for flash-attention-from-scratch. Steps call earlier references
(shared namespace). One def per step with a one-line docstring."""
import numpy as np

import numpy as np


def attention_scores(Q, K, scale):
    """Compute the scaled dot-product attention score matrix S (N, M)."""
    return scale * (Q @ K.T)


def stable_rowmax(S):
    """Return the per-row maximum of S over the last (keys) axis."""
    return np.max(S, axis=-1)


def softmax_rows(S):
    """Compute a numerically stable row-wise softmax of S."""
    shifted = S - stable_rowmax(S)[:, None]
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def attention_reference(Q, K, V, scale):
    """Compute ground-truth attention: row-softmax of scaled scores applied to V."""
    return softmax_rows(attention_scores(Q, K, scale)) @ V

def correction(m_old, m_new):
    """Rescaling factor exp(m_old - m_new) for old stats when the running max grows (m_old=-inf -> 0)."""
    return np.exp(m_old - m_new)


def update_running_max(m_prev, m_block):
    """Elementwise running maximum of the previous max and the new block's row max."""
    return np.maximum(m_prev, m_block)


def block_exp_scores(S_block, m_new):
    """Unnormalized block probabilities: exp of the block scores shifted by the new running max."""
    return np.exp(S_block - m_new[:, None])


def update_normalizer(l_prev, m_prev, m_new, p_block):
    """Updated softmax denominator: rescaled old denominator plus the new block's exp-score row sums."""
    return correction(m_prev, m_new) * l_prev + p_block.sum(axis=-1)


def online_softmax_step(m_prev, l_prev, O_prev, S_block, V_block):
    """One fused flash step folding a key/value block into the running (max, denominator, unnormalized output)."""
    m_block = stable_rowmax(S_block)
    m_new = update_running_max(m_prev, m_block)
    p = block_exp_scores(S_block, m_new)
    alpha = correction(m_prev, m_new)
    l_new = update_normalizer(l_prev, m_prev, m_new, p)
    O_new = alpha[:, None] * O_prev + p @ V_block
    return m_new, l_new, O_new

def make_blocks(n, block_size):
    """Contiguous (start, end) index ranges that tile 0..n; the last may be shorter."""
    return [(s, min(s + block_size, n)) for s in range(0, n, block_size)]


def finalize(O_acc, l):
    """Normalize the unnormalized output accumulator by the running denominator."""
    return O_acc / l[:, None]


def logsumexp(m, l):
    """Per-row log-sum-exp recovered from the running max and denominator."""
    return m + np.log(l)


def flash_attention_forward(Q, K, V, scale, block_size):
    """Exact attention by tiling over key/value blocks; return output and saved log-sum-exp."""
    N = Q.shape[0]
    M = K.shape[0]
    d = V.shape[1]
    m = np.full(N, -np.inf)
    l = np.zeros(N)
    O = np.zeros((N, d))
    for ks, ke in make_blocks(M, block_size):
        S_block = attention_scores(Q, K[ks:ke], scale)
        m, l, O = online_softmax_step(m, l, O, S_block, V[ks:ke])
    O = finalize(O, l)
    return O, logsumexp(m, l)

def causal_block_mask(S_block, q_start, k_start):
    """Return a copy of S_block with entries whose absolute key index exceeds the absolute query index set to -inf."""
    S = S_block.copy().astype(np.float64)
    br, bc = S.shape
    rows = q_start + np.arange(br)[:, None]
    cols = k_start + np.arange(bc)[None, :]
    S[cols > rows] = -np.inf
    return S


def masked_attention_reference(Q, K, V, scale):
    """Compute full causal self-attention by masking future keys, softmaxing each row, and weighting V."""
    S = attention_scores(Q, K, scale)
    S = causal_block_mask(S, q_start=0, k_start=0)
    P = softmax_rows(S)
    return P @ V


def flash_attention_causal(Q, K, V, scale, block_size):
    """Compute causal self-attention by tiling over key blocks, masking future keys per block before each online-softmax step."""
    N = Q.shape[0]
    M = K.shape[0]
    d = Q.shape[1]
    m = np.full(N, -np.inf, dtype=np.float64)
    l = np.zeros(N, dtype=np.float64)
    O = np.zeros((N, d), dtype=np.float64)
    for ks, ke in make_blocks(M, block_size):
        S_block = attention_scores(Q, K[ks:ke], scale)
        S_block = causal_block_mask(S_block, q_start=0, k_start=ks)
        m, l, O = online_softmax_step(m, l, O, S_block, V[ks:ke])
    return finalize(O, l)

def attention_backward_D(dO, O):
    """Per-row dot product of the upstream gradient and the attention output."""
    return np.sum(dO * O, axis=-1)


def flash_attention_backward(Q, K, V, O, dO, L, scale):
    """Gradients dQ, dK, dV, recomputing the softmax from the saved log-sum-exp instead of storing it."""
    S = attention_scores(Q, K, scale)
    P = np.exp(S - L[:, None])
    D = attention_backward_D(dO, O)
    dV = P.T @ dO
    dP = dO @ V.T
    dS = P * (dP - D[:, None])
    dQ = scale * (dS @ K)
    dK = scale * (dS.T @ Q)
    return dQ, dK, dV
