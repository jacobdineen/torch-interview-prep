"""Hidden tests for flash-attention-from-scratch. One test_<id>_<name>(ns) per step; deterministic, independent oracles."""
import numpy as np

import numpy as np


def test_0001_attention_scores(ns):
    f = ns["attention_scores"]
    rng = np.random.RandomState(0)
    N, M, d = 6, 6, 4
    Q = rng.randn(N, d)
    K = rng.randn(M, d)
    scale = 1.0 / np.sqrt(d)
    S = f(Q, K, scale)
    assert S.shape == (N, M)
    # Independent oracle: manual scaled dot products entry by entry.
    expected = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            expected[i, j] = scale * float(np.dot(Q[i], K[j]))
    assert np.allclose(S, expected, atol=1e-12)


def test_0002_stable_rowmax(ns):
    f = ns["stable_rowmax"]
    rng = np.random.RandomState(1)
    N, M = 6, 6
    S = rng.randn(N, M)
    m = f(S)
    assert m.shape == (N,)
    # Independent oracle: Python max per row.
    for i in range(N):
        assert m[i] == max(S[i, j] for j in range(M))


def test_0003_softmax_rows(ns):
    f = ns["softmax_rows"]
    rng = np.random.RandomState(2)
    N, M = 6, 6
    S = rng.randn(N, M)
    P = f(S)
    assert P.shape == (N, M)
    # Rows sum to 1.
    assert np.allclose(P.sum(axis=-1), np.ones(N), atol=1e-12)
    assert np.all(P >= 0)
    # Independent oracle: manual stable softmax.
    expected = np.empty((N, M))
    for i in range(N):
        row = S[i]
        mx = max(row)
        ex = np.array([np.exp(v - mx) for v in row])
        expected[i] = ex / ex.sum()
    assert np.allclose(P, expected, atol=1e-12)
    # Large-value stability: must not overflow.
    big = np.array([[1000.0, 1000.0, 1001.0]])
    Pb = f(big)
    assert np.all(np.isfinite(Pb))
    assert np.isclose(Pb.sum(), 1.0)


def test_0004_attention_reference(ns):
    f = ns["attention_reference"]
    rng = np.random.RandomState(3)
    N, M, d = 6, 6, 4
    Q = rng.randn(N, d)
    K = rng.randn(M, d)
    V = rng.randn(M, d)
    scale = 1.0 / np.sqrt(d)
    O = f(Q, K, V, scale)
    assert O.shape == (N, d)
    # Independent oracle: fully manual scaled-dot-product attention.
    S = scale * (Q @ K.T)
    expected = np.empty((N, d))
    for i in range(N):
        row = S[i]
        mx = np.max(row)
        ex = np.exp(row - mx)
        w = ex / ex.sum()
        expected[i] = w @ V
    assert np.allclose(O, expected, atol=1e-10)
    # Convexity invariant: each output row lies within the bounds of V columns.
    for k in range(d):
        assert O[:, k].max() <= V[:, k].max() + 1e-9
        assert O[:, k].min() >= V[:, k].min() - 1e-9

def test_0005_correction(ns):
    correction = ns["correction"]
    m_old = np.array([1.0, -2.0, 0.5, -np.inf])
    m_new = np.array([3.0, -1.0, 0.5, 2.0])
    out = correction(m_old, m_new)
    expected = np.exp(m_old - m_new)
    assert out.shape == (4,)
    assert np.allclose(out, expected, atol=1e-12)
    # m_old = -inf must give exactly 0
    assert out[3] == 0.0
    # when max unchanged the factor is 1
    assert np.isclose(out[2], 1.0)


def test_0006_update_running_max(ns):
    update_running_max = ns["update_running_max"]
    m_prev = np.array([-np.inf, 2.0, -1.0, 5.0])
    m_block = np.array([1.0, 1.0, 3.0, 4.0])
    out = update_running_max(m_prev, m_block)
    expected = np.array([1.0, 2.0, 3.0, 5.0])
    assert out.shape == (4,)
    assert np.allclose(out, expected, atol=1e-12)


def test_0007_block_exp_scores(ns):
    block_exp_scores = ns["block_exp_scores"]
    np.random.seed(0)
    S_block = np.random.randn(6, 2)
    m_new = np.max(S_block, axis=-1)
    out = block_exp_scores(S_block, m_new)
    expected = np.exp(S_block - m_new[:, None])
    assert out.shape == (6, 2)
    assert np.allclose(out, expected, atol=1e-12)
    # shifting by the row max means each row's largest entry is exp(0)=1, all <= 1
    assert np.all(out <= 1.0 + 1e-12)
    assert np.allclose(np.max(out, axis=-1), 1.0)


def test_0008_update_normalizer(ns):
    update_normalizer = ns["update_normalizer"]
    l_prev = np.array([2.0, 0.0, 1.5])
    m_prev = np.array([1.0, -np.inf, 0.0])
    m_new = np.array([2.0, 0.0, 0.0])
    p_block = np.array([[0.3, 0.4], [0.5, 0.5], [1.0, 2.0]])
    out = update_normalizer(l_prev, m_prev, m_new, p_block)
    alpha = np.exp(m_prev - m_new)
    expected = alpha * l_prev + p_block.sum(axis=-1)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-12)
    # row 1 had m_prev=-inf so old l contributes nothing
    assert np.isclose(out[1], 1.0)


def test_0009_online_softmax_step(ns):
    # A single full-block online step must reproduce a plain stable softmax @ V.
    online_softmax_step = ns["online_softmax_step"]
    np.random.seed(0)
    N, M, d = 6, 6, 4
    S = np.random.randn(N, M)
    V = np.random.randn(M, d)

    m_prev = np.full(N, -np.inf)
    l_prev = np.zeros(N)
    O_prev = np.zeros((N, d))
    m_new, l_new, O_new = online_softmax_step(m_prev, l_prev, O_prev, S, V)

    # independent stable softmax oracle
    mm = np.max(S, axis=-1, keepdims=True)
    e = np.exp(S - mm)
    denom = e.sum(axis=-1, keepdims=True)
    P = e / denom
    O_expected = P @ V

    assert m_new.shape == (N,)
    assert l_new.shape == (N,)
    assert O_new.shape == (N, d)
    # running max equals the true row max
    assert np.allclose(m_new, mm[:, 0], atol=1e-12)
    # denominator equals sum of exp(S - rowmax)
    assert np.allclose(l_new, denom[:, 0], atol=1e-12)
    # unnormalized output divided by denominator recovers softmax @ V
    assert np.allclose(O_new / l_new[:, None], O_expected, atol=1e-10)

    # determinism
    m2, l2, O2 = online_softmax_step(m_prev, l_prev, O_prev, S, V)
    assert np.allclose(O_new, O2) and np.allclose(l_new, l2) and np.allclose(m_new, m2)

def test_0010_make_blocks(ns):
    f = ns["make_blocks"]
    assert f(6, 2) == [(0, 2), (2, 4), (4, 6)]
    assert f(6, 3) == [(0, 3), (3, 6)]
    assert f(6, 6) == [(0, 6)]
    assert f(7, 3) == [(0, 3), (3, 6), (6, 7)]
    assert f(0, 2) == []
    for n, bs in [(6, 2), (7, 3), (5, 4), (10, 1)]:
        blks = f(n, bs)
        assert blks[0][0] == 0 and blks[-1][1] == n
        covered = []
        for s, e in blks:
            assert 0 <= s < e <= n
            assert e - s <= bs
            covered.extend(range(s, e))
        assert covered == list(range(n))


def test_0011_finalize(ns):
    f = ns["finalize"]
    rng = np.random.RandomState(0)
    O_acc = rng.randn(6, 4)
    l = rng.rand(6) + 0.5
    out = f(O_acc, l)
    assert out.shape == (6, 4)
    for i in range(6):
        for j in range(4):
            assert abs(out[i, j] - O_acc[i, j] / l[i]) < 1e-12


def test_0012_logsumexp(ns):
    f = ns["logsumexp"]
    np.random.seed(0)
    S = np.random.randn(6, 6)
    mx = np.max(S, axis=-1)
    l = np.exp(S - mx[:, None]).sum(axis=-1)
    L = f(mx, l)
    assert L.shape == (6,)
    direct = np.log(np.exp(S).sum(axis=-1))
    assert np.allclose(L, direct, atol=1e-10)


def test_0013_flash_attention_forward(ns):
    flash = ns["flash_attention_forward"]
    np.random.seed(0)
    N, M, d = 6, 6, 4
    Q = np.random.randn(N, d)
    K = np.random.randn(M, d)
    V = np.random.randn(M, d)
    scale = 1.0 / np.sqrt(d)
    # Independent oracle: direct stable softmax @ V, and a direct logsumexp of the scores.
    S = scale * (Q @ K.T)
    P = np.exp(S - np.max(S, axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)
    O_ref = P @ V
    L_ref = np.log(np.exp(S).sum(axis=-1))
    for bs in (2, 3, 6):
        O, L = flash(Q, K, V, scale, bs)
        assert O.shape == (N, d)
        assert L.shape == (N,)
        assert np.allclose(O, O_ref, atol=1e-10)
        assert np.allclose(L, L_ref, atol=1e-10)

def test_0014_causal_block_mask(ns):
    causal_block_mask = ns["causal_block_mask"]
    np.random.seed(0)
    # Sub-block with offsets: 3 query rows starting at q_start, 4 key cols at k_start.
    q_start, k_start = 2, 1
    S = np.random.randn(3, 4).astype(np.float64)
    out = causal_block_mask(S, q_start, k_start)
    # original must be untouched (copy semantics)
    assert out.shape == (3, 4)
    # manual expected
    expected = S.copy()
    for r in range(3):
        for c in range(4):
            if (k_start + c) > (q_start + r):
                expected[r, c] = -np.inf
    assert np.array_equal(np.isneginf(out), np.isneginf(expected))
    finite = ~np.isneginf(expected)
    assert np.allclose(out[finite], expected[finite], atol=1e-12)
    # input not mutated
    assert not np.any(np.isneginf(S))


def test_0015_masked_attention_reference(ns):
    masked_attention_reference = ns["masked_attention_reference"]
    np.random.seed(0)
    N = M = 6
    d = 4
    scale = 1.0 / np.sqrt(d)
    Q = np.random.randn(N, d)
    K = np.random.randn(M, d)
    V = np.random.randn(M, d)
    out = masked_attention_reference(Q, K, V, scale)
    # independent manual causal softmax
    S = scale * (Q @ K.T)
    for i in range(N):
        for j in range(M):
            if j > i:
                S[i, j] = -np.inf
    Smax = np.max(S, axis=-1, keepdims=True)
    P = np.exp(S - Smax)
    P = P / P.sum(axis=-1, keepdims=True)
    expected = P @ V
    assert out.shape == (N, d)
    assert np.allclose(out, expected, atol=1e-12)
    # row 0 attends only to key 0 -> output equals V[0]
    assert np.allclose(out[0], V[0], atol=1e-12)


def test_0016_flash_attention_causal(ns):
    flash_attention_causal = ns["flash_attention_causal"]
    masked_attention_reference = ns["masked_attention_reference"]
    np.random.seed(0)
    N = M = 6
    d = 4
    scale = 1.0 / np.sqrt(d)
    Q = np.random.randn(N, d)
    K = np.random.randn(M, d)
    V = np.random.randn(M, d)
    expected = masked_attention_reference(Q, K, V, scale)
    for bs in (2, 3, 6):
        out = flash_attention_causal(Q, K, V, scale, bs)
        assert out.shape == (N, d)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, expected, atol=1e-10)

def test_0017_attention_backward_D(ns):
    np.random.seed(0)
    dO = np.random.randn(6, 4)
    O = np.random.randn(6, 4)
    D = ns["attention_backward_D"](dO, O)
    expected = np.array([np.dot(dO[i], O[i]) for i in range(6)])
    assert D.shape == (6,)
    assert np.allclose(D, expected, atol=1e-12)


def test_0018_flash_attention_backward(ns):
    np.random.seed(0)
    N, M, d = 6, 6, 4
    Q = np.random.randn(N, d)
    K = np.random.randn(M, d)
    V = np.random.randn(M, d)
    scale = 1.0 / np.sqrt(d)
    G = np.random.randn(N, d)

    # Independent stable-softmax forward (oracle), not the contract reference.
    def ref_forward(Q, K, V):
        S = scale * (Q @ K.T)
        m = np.max(S, axis=-1, keepdims=True)
        e = np.exp(S - m)
        P = e / e.sum(axis=-1, keepdims=True)
        return P @ V

    O = ref_forward(Q, K, V)
    # Saved L = per-row log-sum-exp of the scores, recomputed directly here.
    S = scale * (Q @ K.T)
    mrow = np.max(S, axis=-1)
    L = mrow + np.log(np.exp(S - mrow[:, None]).sum(axis=-1))

    # loss = sum(O * G)  =>  dO = G
    dO = G
    dQ, dK, dV = ns["flash_attention_backward"](Q, K, V, O, dO, L, scale)
    assert dQ.shape == (N, d)
    assert dK.shape == (M, d)
    assert dV.shape == (M, d)
    assert np.all(np.isfinite(dQ)) and np.all(np.isfinite(dK)) and np.all(np.isfinite(dV))

    eps = 1e-5

    def loss(Q, K, V):
        return np.sum(ref_forward(Q, K, V) * G)

    def fd(X, f):
        g = np.zeros_like(X)
        for idx in np.ndindex(X.shape):
            Xp = X.copy(); Xp[idx] += eps
            Xm = X.copy(); Xm[idx] -= eps
            g[idx] = (f(Xp) - f(Xm)) / (2 * eps)
        return g

    dQ_fd = fd(Q, lambda x: loss(x, K, V))
    dK_fd = fd(K, lambda x: loss(Q, x, V))
    dV_fd = fd(V, lambda x: loss(Q, K, x))
    assert np.allclose(dQ, dQ_fd, atol=1e-5)
    assert np.allclose(dK, dK_fd, atol=1e-5)
    assert np.allclose(dV, dV_fd, atol=1e-5)

    # Determinism.
    dQ2, dK2, dV2 = ns["flash_attention_backward"](Q, K, V, O, dO, L, scale)
    assert np.array_equal(dQ, dQ2) and np.array_equal(dK, dK2) and np.array_equal(dV, dV2)
