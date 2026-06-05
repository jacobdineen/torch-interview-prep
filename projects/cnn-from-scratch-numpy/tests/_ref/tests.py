"""Hidden tests for cnn-from-scratch-numpy. One test_<id>_<name>(ns) per step;
deterministic, independent NumPy oracles incl. finite-difference gradient checks."""
import numpy as np

import numpy as np


def test_0001_argmax_rows(ns):
    f = ns["argmax_rows"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3))
    out = f(x)
    assert out.shape == (4,)
    # independent oracle: per-row scan
    for i in range(4):
        best = 0
        for j in range(1, 3):
            if x[i, j] > x[i, best]:
                best = j
        assert out[i] == best


def test_0002_row_max(ns):
    f = ns["row_max"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3))
    out = f(x)
    assert out.shape == (4, 1)
    for i in range(4):
        assert np.allclose(out[i, 0], max(x[i, :]), atol=1e-6)


def test_0003_row_sum(ns):
    f = ns["row_sum"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3))
    out = f(x)
    assert out.shape == (4, 1)
    for i in range(4):
        s = 0.0
        for j in range(3):
            s += x[i, j]
        assert np.allclose(out[i, 0], s, atol=1e-6)


def test_0004_exp_shifted(ns):
    f = ns["exp_shifted"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3))
    out = f(x)
    assert out.shape == (4, 3)
    # oracle: shifted by per-row max computed independently
    for i in range(4):
        m = max(x[i, :])
        for j in range(3):
            assert np.allclose(out[i, j], np.exp(x[i, j] - m), atol=1e-6)
    # max element of each row maps to exp(0)=1
    assert np.allclose(np.max(out, axis=1), 1.0, atol=1e-6)
    # invariant to additive constant on a row
    x2 = x.copy()
    x2[0, :] += 100.0
    assert np.allclose(f(x2)[0], out[0], atol=1e-6)


def test_0005_stable_softmax(ns):
    f = ns["stable_softmax"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3))
    out = f(x)
    assert out.shape == (4, 3)
    # rows sum to 1
    assert np.allclose(np.sum(out, axis=1), 1.0, atol=1e-6)
    assert np.all(out > 0)
    # independent oracle on a tiny known vector
    v = np.array([[1.0, 2.0, 3.0]])
    ev = np.exp(v[0])
    expect = ev / ev.sum()
    assert np.allclose(f(v)[0], expect, atol=1e-6)
    # shift-invariance: adding a constant to a row leaves softmax unchanged
    x2 = x.copy()
    x2[1, :] += 50.0
    assert np.allclose(f(x2)[1], out[1], atol=1e-6)


def test_0006_one_hot(ns):
    f = ns["one_hot"]
    labels = np.array([0, 2, 1, 2])
    out = f(labels, 3)
    assert out.shape == (4, 3)
    assert out.dtype == np.float64
    # oracle: manual rows
    expect = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(out, expect, atol=1e-6)
    # exactly one 1 per row
    assert np.all(np.sum(out, axis=1) == 1.0)


def test_0007_gather_true_class_probs(ns):
    f = ns["gather_true_class_probs"]
    probs = np.array([
        [0.1, 0.7, 0.2],
        [0.5, 0.3, 0.2],
        [0.2, 0.2, 0.6],
        [0.8, 0.1, 0.1],
    ])
    labels = np.array([1, 0, 2, 0])
    out = f(probs, labels)
    assert out.shape == (4,)
    expect = np.array([0.7, 0.5, 0.6, 0.8])
    assert np.allclose(out, expect, atol=1e-6)


def test_0008_cross_entropy_loss(ns):
    f = ns["cross_entropy_loss"]
    probs = np.array([
        [0.1, 0.7, 0.2],
        [0.5, 0.3, 0.2],
        [0.2, 0.2, 0.6],
        [0.8, 0.1, 0.1],
    ])
    labels = np.array([1, 0, 2, 0])
    out = f(probs, labels)
    # independent oracle: mean of -log(true prob + 1e-12)
    true_p = np.array([0.7, 0.5, 0.6, 0.8])
    expect = float(np.mean(-np.log(true_p + 1e-12)))
    assert np.allclose(out, expect, atol=1e-6)
    assert isinstance(out, float)
    # perfect prediction -> near-zero loss
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert f(perfect, np.array([0, 1])) < 1e-5


def test_0009_accuracy(ns):
    f = ns["accuracy"]
    logits = np.array([
        [0.1, 0.7, 0.2],   # argmax 1
        [0.5, 0.3, 0.2],   # argmax 0
        [0.2, 0.2, 0.6],   # argmax 2
        [0.8, 0.1, 0.1],   # argmax 0
    ])
    labels = np.array([1, 0, 1, 0])  # 3 of 4 correct
    out = f(logits, labels)
    assert np.allclose(out, 0.75, atol=1e-6)
    assert isinstance(out, float)
    # all correct
    assert np.allclose(f(logits, np.array([1, 0, 2, 0])), 1.0, atol=1e-6)

def test_0010_he_std(ns):
    f = ns["he_std"]
    assert np.allclose(f(8.0), np.sqrt(2.0 / 8.0), atol=1e-6)
    assert np.allclose(f(2), 1.0, atol=1e-6)
    # larger fan_in -> smaller std
    assert f(100) < f(10)


def test_0011_he_init(ns):
    f = ns["he_init"]
    rng = np.random.default_rng(0)
    shape = (200, 200)
    fan_in = 50
    a = f(shape, fan_in, rng)
    assert a.shape == shape
    # empirical std close to he_std
    assert np.allclose(np.std(a), np.sqrt(2.0 / fan_in), atol=2e-2)
    assert abs(np.mean(a)) < 2e-2
    # determinism
    rng2 = np.random.default_rng(0)
    assert np.allclose(a, f(shape, fan_in, rng2))


def test_0012_init_zero_bias(ns):
    f = ns["init_zero_bias"]
    b = f(5)
    assert b.shape == (5,)
    assert np.allclose(b, 0.0)


def test_0013_pad_2d(ns):
    f = ns["pad_2d"]
    np.random.seed(0)
    x = np.random.randn(4, 1, 8, 8)
    p = 2
    y = f(x, p)
    assert y.shape == (4, 1, 12, 12)
    # interior preserved
    assert np.allclose(y[:, :, p:p + 8, p:p + 8], x)
    # borders zero
    assert np.allclose(y[:, :, :p, :], 0.0)
    assert np.allclose(y[:, :, -p:, :], 0.0)
    # pad 0 is identity
    assert np.allclose(f(x, 0), x)


def test_0014_output_spatial_size(ns):
    f = ns["output_spatial_size"]
    assert f(8, 3, 1, 0) == 6
    assert f(8, 3, 1, 1) == 8
    assert f(8, 2, 2, 0) == 4
    # independent formula
    for in_size in [5, 8, 10]:
        for k in [2, 3]:
            for stride in [1, 2]:
                for pad in [0, 1]:
                    expected = (in_size + 2 * pad - k) // stride + 1
                    assert f(in_size, k, stride, pad) == expected


def test_0015_im2col(ns):
    def _naive_im2col(x, kh, kw, stride, pad):
        N, C, H, W = x.shape
        OH = (H + 2 * pad - kh) // stride + 1
        OW = (W + 2 * pad - kw) // stride + 1
        xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
        xp[:, :, pad:pad + H, pad:pad + W] = x
        out = np.zeros((N, C * kh * kw, OH * OW))
        for n in range(N):
            for oh in range(OH):
                for ow in range(OW):
                    col = oh * OW + ow
                    r = 0
                    for c in range(C):
                        for ki in range(kh):
                            for kj in range(kw):
                                out[n, r, col] = xp[n, c, oh * stride + ki, ow * stride + kj]
                                r += 1
        return out

    f = ns["im2col"]
    np.random.seed(0)
    x = np.random.randn(4, 1, 8, 8)
    for stride in [1, 2]:
        for pad in [0, 1]:
            kh = kw = 3
            got = f(x, kh, kw, stride, pad)
            N, C, H, W = x.shape
            OH = (H + 2 * pad - kh) // stride + 1
            OW = (W + 2 * pad - kw) // stride + 1
            assert got.shape == (N, C * kh * kw, OH * OW)
            ref = _naive_im2col(x, kh, kw, stride, pad)
            assert np.allclose(got, ref, atol=1e-6)
    # multi-channel sanity
    xc = np.random.randn(2, 3, 6, 6)
    assert np.allclose(f(xc, 2, 2, 2, 0),
                       _naive_im2col(xc, 2, 2, 2, 0), atol=1e-6)


def test_0016_col2im(ns):
    def _naive_col2im(cols, x_shape, kh, kw, stride, pad):
        N, C, H, W = x_shape
        OH = (H + 2 * pad - kh) // stride + 1
        OW = (W + 2 * pad - kw) // stride + 1
        xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
        for n in range(N):
            for oh in range(OH):
                for ow in range(OW):
                    col = oh * OW + ow
                    r = 0
                    for c in range(C):
                        for ki in range(kh):
                            for kj in range(kw):
                                xp[n, c, oh * stride + ki, ow * stride + kj] += cols[n, r, col]
                                r += 1
        return xp[:, :, pad:pad + H, pad:pad + W]

    im2col = ns["im2col"]
    col2im = ns["col2im"]
    np.random.seed(0)
    x = np.random.randn(4, 1, 8, 8)
    x_shape = x.shape

    for stride in [1, 2]:
        for pad in [0, 1]:
            kh = kw = 3
            cols = im2col(x, kh, kw, stride, pad)
            # matches independent naive col2im
            got = col2im(cols, x_shape, kh, kw, stride, pad)
            ref = _naive_col2im(cols, x_shape, kh, kw, stride, pad)
            assert got.shape == x_shape
            assert np.allclose(got, ref, atol=1e-6)

    # Adjoint property: <im2col(x), c> == <x, col2im(c)> for any c.
    # This verifies col2im is the true transpose of im2col (accumulating overlaps).
    rng = np.random.default_rng(0)
    for stride in [1, 2]:
        for pad in [0, 1]:
            kh = kw = 3
            cols_x = im2col(x, kh, kw, stride, pad)
            c = rng.normal(size=cols_x.shape)
            lhs = np.sum(cols_x * c)
            rhs = np.sum(x * col2im(c, x_shape, kh, kw, stride, pad))
            assert np.allclose(lhs, rhs, atol=1e-6)

import numpy as np


def _naive_conv(x, W, b, stride, pad):
    N, C, H, Wd = x.shape
    out_ch, _, kh, kw = W.shape
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (Wd + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((N, out_ch, OH, OW))
    for n in range(N):
        for o in range(out_ch):
            for oh in range(OH):
                for ow in range(OW):
                    acc = b[o]
                    for c in range(C):
                        for ki in range(kh):
                            for kj in range(kw):
                                acc += xp[n, c, oh * stride + ki, ow * stride + kj] * W[o, c, ki, kj]
                    out[n, o, oh, ow] = acc
    return out


def _fd_grad(f, arr, dout, eps=1e-5):
    g = np.zeros_like(arr)
    flat = arr.reshape(-1)
    gflat = g.reshape(-1)
    for i in range(flat.size):
        save = flat[i]
        flat[i] = save + eps
        o1 = f(arr)
        flat[i] = save - eps
        o2 = f(arr)
        flat[i] = save
        gflat[i] = ((o1 - o2) * dout).sum() / (2 * eps)
    return g


def _setup(rng):
    N, C, H, Wd, out_ch, k = 4, 1, 8, 8, 2, 3
    x = rng.standard_normal((N, C, H, Wd))
    W = rng.standard_normal((out_ch, C, k, k))
    b = rng.standard_normal(out_ch)
    return x, W, b


def test_0017_conv2d_forward(ns):
    conv2d_forward = ns["conv2d_forward"]
    rng = np.random.default_rng(0)
    x, W, b = _setup(rng)
    for stride, pad in [(1, 0), (1, 1), (2, 1)]:
        out, cache = conv2d_forward(x, W, b, stride, pad)
        ref = _naive_conv(x, W, b, stride, pad)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        assert np.allclose(out, ref, atol=1e-6)


def test_0018_conv2d_grad_input(ns):
    conv2d_forward = ns["conv2d_forward"]
    conv2d_grad_input = ns["conv2d_grad_input"]
    rng = np.random.default_rng(0)
    x, W, b = _setup(rng)
    for stride, pad in [(1, 0), (1, 1), (2, 1)]:
        out, _ = conv2d_forward(x, W, b, stride, pad)
        dout = rng.standard_normal(out.shape)
        dx = conv2d_grad_input(dout, x, W, stride, pad)
        assert dx.shape == x.shape
        dx_num = _fd_grad(lambda a: conv2d_forward(a, W, b, stride, pad)[0], x, dout)
        assert np.allclose(dx, dx_num, atol=1e-5)


def test_0019_conv2d_grad_weights(ns):
    conv2d_forward = ns["conv2d_forward"]
    conv2d_grad_weights = ns["conv2d_grad_weights"]
    rng = np.random.default_rng(0)
    x, W, b = _setup(rng)
    for stride, pad in [(1, 0), (1, 1), (2, 1)]:
        out, _ = conv2d_forward(x, W, b, stride, pad)
        dout = rng.standard_normal(out.shape)
        dW = conv2d_grad_weights(dout, x, W, stride, pad)
        assert dW.shape == W.shape
        dW_num = _fd_grad(lambda a: conv2d_forward(x, a, b, stride, pad)[0], W, dout)
        assert np.allclose(dW, dW_num, atol=1e-5)


def test_0020_conv2d_grad_bias(ns):
    conv2d_forward = ns["conv2d_forward"]
    conv2d_grad_bias = ns["conv2d_grad_bias"]
    rng = np.random.default_rng(0)
    x, W, b = _setup(rng)
    for stride, pad in [(1, 0), (2, 1)]:
        out, _ = conv2d_forward(x, W, b, stride, pad)
        dout = rng.standard_normal(out.shape)
        db = conv2d_grad_bias(dout)
        assert db.shape == b.shape
        db_num = np.zeros_like(b)
        for i in range(b.size):
            bp = b.copy(); bp[i] += 1e-5
            bm = b.copy(); bm[i] -= 1e-5
            o1 = conv2d_forward(x, W, bp, stride, pad)[0]
            o2 = conv2d_forward(x, W, bm, stride, pad)[0]
            db_num[i] = ((o1 - o2) * dout).sum() / (2e-5)
        assert np.allclose(db, db_num, atol=1e-5)


def test_0021_conv2d_backward(ns):
    conv2d_forward = ns["conv2d_forward"]
    conv2d_backward = ns["conv2d_backward"]
    conv2d_grad_input = ns["conv2d_grad_input"]
    conv2d_grad_weights = ns["conv2d_grad_weights"]
    conv2d_grad_bias = ns["conv2d_grad_bias"]
    rng = np.random.default_rng(0)
    x, W, b = _setup(rng)
    for stride, pad in [(1, 0), (1, 1), (2, 1)]:
        out, cache = conv2d_forward(x, W, b, stride, pad)
        dout = rng.standard_normal(out.shape)
        dx, dW, db = conv2d_backward(dout, cache)
        assert np.allclose(dx, conv2d_grad_input(dout, x, W, stride, pad), atol=1e-8)
        assert np.allclose(dW, conv2d_grad_weights(dout, x, W, stride, pad), atol=1e-8)
        assert np.allclose(db, conv2d_grad_bias(dout), atol=1e-8)
        dx_num = _fd_grad(lambda a: conv2d_forward(a, W, b, stride, pad)[0], x, dout)
        dW_num = _fd_grad(lambda a: conv2d_forward(x, a, b, stride, pad)[0], W, dout)
        assert np.allclose(dx, dx_num, atol=1e-5)
        assert np.allclose(dW, dW_num, atol=1e-5)

def test_0022_maxpool2d_forward(ns):
    f = ns["maxpool2d_forward"]
    np.random.seed(0)
    x = np.random.randn(4, 1, 8, 8)
    pool, stride = 2, 2
    out, cache = f(x, pool, stride)
    OH = (8 - pool) // stride + 1
    OW = (8 - pool) // stride + 1
    assert out.shape == (4, 1, OH, OW)
    # independent naive max oracle
    expected = np.empty_like(out)
    for n in range(4):
        for c in range(1):
            for oh in range(OH):
                for ow in range(OW):
                    win = x[n, c, oh * stride:oh * stride + pool, ow * stride:ow * stride + pool]
                    expected[n, c, oh, ow] = win.max()
    assert np.allclose(out, expected, atol=1e-6)


def test_0023_scatter_grad_window(ns):
    f = ns["scatter_grad_window"]
    np.random.seed(0)
    window = np.random.randn(3, 3)
    grad = f(2.5, window)
    assert grad.shape == window.shape
    # exactly one nonzero, at the argmax, equal to dout_val
    assert np.count_nonzero(grad) == 1
    idx = np.unravel_index(np.argmax(window), window.shape)
    assert np.isclose(grad[idx], 2.5, atol=1e-6)
    assert np.isclose(grad.sum(), 2.5, atol=1e-6)


def test_0024_maxpool2d_backward(ns):
    fwd = ns["maxpool2d_forward"]
    bwd = ns["maxpool2d_backward"]
    np.random.seed(0)
    x = np.random.randn(4, 1, 8, 8)
    pool, stride = 2, 2
    out, cache = fwd(x, pool, stride)
    np.random.seed(1)
    dout = np.random.randn(*out.shape)
    dx = bwd(dout, cache)
    assert dx.shape == x.shape

    # finite-difference gradient check on scalar loss = sum(dout * pool(x))
    def loss(xx):
        o, _ = fwd(xx, pool, stride)
        return np.sum(o * dout)

    eps = 1e-5
    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        mi = it.multi_index
        xp = x.copy(); xp[mi] += eps
        xm = x.copy(); xm[mi] -= eps
        num[mi] = (loss(xp) - loss(xm)) / (2 * eps)
        it.iternext()
    assert np.allclose(dx, num, atol=1e-5)


def test_0025_relu_forward(ns):
    f = ns["relu_forward"]
    np.random.seed(0)
    x = np.random.randn(4, 5)
    out, cache = f(x)
    assert np.allclose(out, np.where(x > 0, x, 0.0), atol=1e-6)
    assert np.all(out >= 0)


def test_0026_relu_backward(ns):
    fwd = ns["relu_forward"]
    bwd = ns["relu_backward"]
    np.random.seed(0)
    x = np.random.randn(4, 5)
    out, cache = fwd(x)
    np.random.seed(2)
    dout = np.random.randn(4, 5)
    dx = bwd(dout, cache)

    def loss(xx):
        o, _ = fwd(xx)
        return np.sum(o * dout)

    eps = 1e-5
    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        mi = it.multi_index
        # avoid kink at exactly 0 (random normals essentially never hit it)
        xp = x.copy(); xp[mi] += eps
        xm = x.copy(); xm[mi] -= eps
        num[mi] = (loss(xp) - loss(xm)) / (2 * eps)
        it.iternext()
    assert np.allclose(dx, num, atol=1e-5)


def test_0027_flatten_forward(ns):
    f = ns["flatten_forward"]
    np.random.seed(0)
    x = np.random.randn(4, 1, 8, 8)
    out, cache = f(x)
    assert out.shape == (4, 1 * 8 * 8)
    # row-major reshape oracle
    assert np.allclose(out, x.reshape(4, -1), atol=1e-6)
    # values preserved per sample
    assert np.allclose(out[0], x[0].ravel(), atol=1e-6)


def test_0028_flatten_backward(ns):
    fwd = ns["flatten_forward"]
    bwd = ns["flatten_backward"]
    np.random.seed(0)
    x = np.random.randn(4, 1, 8, 8)
    out, cache = fwd(x)
    np.random.seed(3)
    dout = np.random.randn(*out.shape)
    dx = bwd(dout, cache)
    assert dx.shape == x.shape
    # round-trip: flattening dx must give back dout
    assert np.allclose(dx.reshape(4, -1), dout, atol=1e-6)


def test_0029_linear_forward(ns):
    f = ns["linear_forward"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6))
    W = rng.standard_normal((6, 2))
    b = rng.standard_normal((2,))
    out, cache = f(x, W, b)
    assert out.shape == (4, 2)
    # independent nested-loop oracle
    expected = np.empty((4, 2))
    for i in range(4):
        for j in range(2):
            expected[i, j] = np.dot(x[i], W[:, j]) + b[j]
    assert np.allclose(out, expected, atol=1e-6)


def test_0030_linear_grad_input(ns):
    fwd = ns["linear_forward"]
    g = ns["linear_grad_input"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6))
    W = rng.standard_normal((6, 2))
    b = rng.standard_normal((2,))
    dout = rng.standard_normal((4, 2))
    dx = g(dout, W)
    assert dx.shape == x.shape

    def loss(xx):
        o, _ = fwd(xx, W, b)
        return np.sum(o * dout)

    eps = 1e-6
    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        mi = it.multi_index
        xp = x.copy(); xp[mi] += eps
        xm = x.copy(); xm[mi] -= eps
        num[mi] = (loss(xp) - loss(xm)) / (2 * eps)
        it.iternext()
    assert np.allclose(dx, num, atol=1e-5)


def test_0031_linear_grad_weights(ns):
    fwd = ns["linear_forward"]
    g = ns["linear_grad_weights"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6))
    W = rng.standard_normal((6, 2))
    b = rng.standard_normal((2,))
    dout = rng.standard_normal((4, 2))
    dW = g(dout, x)
    assert dW.shape == W.shape

    def loss(WW):
        o, _ = fwd(x, WW, b)
        return np.sum(o * dout)

    eps = 1e-6
    num = np.zeros_like(W)
    it = np.nditer(W, flags=["multi_index"])
    while not it.finished:
        mi = it.multi_index
        Wp = W.copy(); Wp[mi] += eps
        Wm = W.copy(); Wm[mi] -= eps
        num[mi] = (loss(Wp) - loss(Wm)) / (2 * eps)
        it.iternext()
    assert np.allclose(dW, num, atol=1e-5)


def test_0032_linear_grad_bias(ns):
    fwd = ns["linear_forward"]
    g = ns["linear_grad_bias"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6))
    W = rng.standard_normal((6, 2))
    b = rng.standard_normal((2,))
    dout = rng.standard_normal((4, 2))
    db = g(dout)
    assert db.shape == b.shape

    def loss(bb):
        o, _ = fwd(x, W, bb)
        return np.sum(o * dout)

    eps = 1e-6
    num = np.zeros_like(b)
    for i in range(b.shape[0]):
        bp = b.copy(); bp[i] += eps
        bm = b.copy(); bm[i] -= eps
        num[i] = (loss(bp) - loss(bm)) / (2 * eps)
    assert np.allclose(db, num, atol=1e-5)


def test_0033_linear_backward(ns):
    fwd = ns["linear_forward"]
    bwd = ns["linear_backward"]
    gi = ns["linear_grad_input"]
    gw = ns["linear_grad_weights"]
    gb = ns["linear_grad_bias"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6))
    W = rng.standard_normal((6, 2))
    b = rng.standard_normal((2,))
    dout = rng.standard_normal((4, 2))
    out, cache = fwd(x, W, b)
    dx, dW, db = bwd(dout, cache)
    # consistent with the individual grad functions
    assert np.allclose(dx, gi(dout, W), atol=1e-6)
    assert np.allclose(dW, gw(dout, x), atol=1e-6)
    assert np.allclose(db, gb(dout), atol=1e-6)

    # full finite-difference check on all three
    def loss(xx, WW, bb):
        o, _ = fwd(xx, WW, bb)
        return np.sum(o * dout)

    eps = 1e-6
    num_x = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        mi = it.multi_index
        xp = x.copy(); xp[mi] += eps
        xm = x.copy(); xm[mi] -= eps
        num_x[mi] = (loss(xp, W, b) - loss(xm, W, b)) / (2 * eps)
        it.iternext()
    assert np.allclose(dx, num_x, atol=1e-5)

def test_0034_softmax_cross_entropy_forward(ns):
    """softmax_cross_entropy_forward: loss matches independent stable softmax + NLL, cache holds probs/labels."""
    f = ns["softmax_cross_entropy_forward"]
    rng = np.random.default_rng(0)
    N, K = 4, 3
    logits = rng.standard_normal((N, K))
    labels = np.array([0, 2, 1, 2])
    loss, cache = f(logits, labels)
    # Independent oracle: stable softmax then mean negative log-prob of true class.
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    true_p = probs[np.arange(N), labels]
    expected = np.mean(-np.log(true_p + 1e-12))
    assert np.allclose(loss, expected, atol=1e-6)
    assert np.allclose(cache["probs"], probs, atol=1e-6)
    assert np.array_equal(np.asarray(cache["labels"]), labels)


def test_0035_softmax_cross_entropy_backward(ns):
    """softmax_cross_entropy_backward: matches finite-difference gradient of the fused loss w.r.t. logits."""
    fwd = ns["softmax_cross_entropy_forward"]
    bwd = ns["softmax_cross_entropy_backward"]
    rng = np.random.default_rng(0)
    N, K = 4, 3
    logits = rng.standard_normal((N, K))
    labels = np.array([0, 2, 1, 2])
    _, cache = fwd(logits, labels)
    dlogits = bwd(cache)

    # Independent analytic oracle.
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    oh = np.zeros((N, K))
    oh[np.arange(N), labels] = 1.0
    expected = (probs - oh) / N
    assert np.allclose(dlogits, expected, atol=1e-6)

    # Finite-difference gradient check on the scalar loss.
    def loss_of(L):
        zz = L - L.max(axis=1, keepdims=True)
        ee = np.exp(zz)
        pp = ee / ee.sum(axis=1, keepdims=True)
        return np.mean(-np.log(pp[np.arange(N), labels] + 1e-12))

    eps = 1e-6
    num = np.zeros_like(logits)
    for i in range(N):
        for j in range(K):
            lp = logits.copy(); lp[i, j] += eps
            lm = logits.copy(); lm[i, j] -= eps
            num[i, j] = (loss_of(lp) - loss_of(lm)) / (2 * eps)
    assert np.allclose(dlogits, num, atol=1e-5)


def test_0036_sgd_step(ns):
    """sgd_step: returns param - lr*grad and does not mutate the input."""
    f = ns["sgd_step"]
    rng = np.random.default_rng(0)
    param = rng.standard_normal((3, 4))
    grad = rng.standard_normal((3, 4))
    lr = 0.1
    orig = param.copy()
    out = f(param, grad, lr)
    assert np.allclose(out, param - lr * grad, atol=1e-6)
    assert np.allclose(param, orig, atol=1e-12)


def test_0037_adam_update_m(ns):
    """adam_update_m: beta1*m + (1-beta1)*grad."""
    f = ns["adam_update_m"]
    rng = np.random.default_rng(0)
    m = rng.standard_normal((3, 4))
    grad = rng.standard_normal((3, 4))
    beta1 = 0.9
    out = f(m, grad, beta1)
    assert np.allclose(out, beta1 * m + (1 - beta1) * grad, atol=1e-6)


def test_0038_adam_update_v(ns):
    """adam_update_v: beta2*v + (1-beta2)*grad**2."""
    f = ns["adam_update_v"]
    rng = np.random.default_rng(0)
    v = np.abs(rng.standard_normal((3, 4)))
    grad = rng.standard_normal((3, 4))
    beta2 = 0.999
    out = f(v, grad, beta2)
    assert np.allclose(out, beta2 * v + (1 - beta2) * grad ** 2, atol=1e-6)


def test_0039_adam_bias_correct(ns):
    """adam_bias_correct: moment / (1 - beta**t) for a few t."""
    f = ns["adam_bias_correct"]
    rng = np.random.default_rng(0)
    moment = rng.standard_normal((3, 4))
    for beta in (0.9, 0.999):
        for t in (1, 2, 5):
            out = f(moment, beta, t)
            assert np.allclose(out, moment / (1 - beta ** t), atol=1e-6)


def test_0040_adam_param_step(ns):
    """adam_param_step: param - lr*m_hat/(sqrt(v_hat)+eps)."""
    f = ns["adam_param_step"]
    rng = np.random.default_rng(0)
    param = rng.standard_normal((3, 4))
    m_hat = rng.standard_normal((3, 4))
    v_hat = np.abs(rng.standard_normal((3, 4)))
    lr, eps = 1e-3, 1e-8
    out = f(param, m_hat, v_hat, lr, eps)
    assert np.allclose(out, param - lr * m_hat / (np.sqrt(v_hat) + eps), atol=1e-6)


def test_0041_adam_step(ns):
    """adam_step: full update matches a manual Adam reference over several steps."""
    f = ns["adam_step"]
    rng = np.random.default_rng(0)
    shape = (3, 4)
    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    param = rng.standard_normal(shape)
    m = np.zeros(shape)
    v = np.zeros(shape)

    # Manual reference state.
    rparam = param.copy()
    rm = np.zeros(shape)
    rv = np.zeros(shape)

    for t in range(1, 6):
        grad = rng.standard_normal(shape)
        param, m, v = f(param, grad, m, v, t, lr, beta1, beta2, eps)
        # Manual Adam (t already incremented, matching adam_step convention).
        rm = beta1 * rm + (1 - beta1) * grad
        rv = beta2 * rv + (1 - beta2) * grad ** 2
        mhat = rm / (1 - beta1 ** t)
        vhat = rv / (1 - beta2 ** t)
        rparam = rparam - lr * mhat / (np.sqrt(vhat) + eps)

        assert np.allclose(m, rm, atol=1e-8)
        assert np.allclose(v, rv, atol=1e-8)
        assert np.allclose(param, rparam, atol=1e-8)

def _fd_grad_param(loss_of, arr, eps=1e-5, n_probe=10):
    """Central finite-difference gradient of a scalar loss wrt selected entries of arr."""
    flat = arr.ravel()
    out = np.zeros_like(flat)
    idxs = np.unique(np.linspace(0, flat.size - 1, min(n_probe, flat.size)).astype(int))
    for i in idxs:
        old = flat[i]
        flat[i] = old + eps; lp = loss_of()
        flat[i] = old - eps; lm = loss_of()
        flat[i] = old
        out[i] = (lp - lm) / (2 * eps)
    return out.reshape(arr.shape), idxs


def test_0042_init_conv_layer(ns):
    f = ns["init_conv_layer"]
    rng = np.random.default_rng(0)
    layer = f(1, 2, 3, rng)
    assert layer["W"].shape == (2, 1, 3, 3)
    assert layer["b"].shape == (2,)
    assert np.allclose(layer["b"], 0.0)
    # He std oracle: std ~ sqrt(2/fan_in), fan_in = 1*3*3 = 9
    big = f(1, 64, 3, np.random.default_rng(1))["W"]
    assert abs(np.std(big) - np.sqrt(2.0 / 9)) < 0.05


def test_0043_init_linear_layer(ns):
    f = ns["init_linear_layer"]
    rng = np.random.default_rng(0)
    layer = f(7, 5, rng)
    assert layer["W"].shape == (7, 5)
    assert layer["b"].shape == (5,)
    assert np.allclose(layer["b"], 0.0)
    big = f(50, 50, np.random.default_rng(2))["W"]
    assert abs(np.std(big) - np.sqrt(2.0 / 50)) < 0.05


def test_0044_init_lenet(ns):
    f = ns["init_lenet"]
    rng = np.random.default_rng(0)
    N, C, H, K = 4, 1, 8, 3
    params = f(C, K, H, rng)
    # internal consistency: forward must yield logits of shape (N, K)
    x = np.random.default_rng(3).normal(size=(N, C, H, H))
    logits, _ = ns["lenet_forward"](params, x)
    assert logits.shape == (N, K)
    # first conv layer must accept C input channels
    assert params["convs"][0]["W"].shape[1] == C


def test_0045_forward_conv_block(ns):
    f = ns["forward_conv_block"]
    rng = np.random.default_rng(0)
    N, C, H = 4, 1, 8
    x = rng.normal(size=(N, C, H, H))
    layer = ns["init_conv_layer"](C, 2, 3, rng)
    out, cache = f(x, layer, 2, 2)
    # independent oracle: naive conv(same-pad) -> relu -> maxpool, all nested loops
    W, b = layer["W"], layer["b"]
    pad, stride = layer.get("pad", 0), layer.get("stride", 1)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oc, kh, kw = W.shape[0], W.shape[2], W.shape[3]
    OH = (H + 2 * pad - kh) // stride + 1
    OW = OH
    conv = np.zeros((N, oc, OH, OW))
    for n in range(N):
        for o in range(oc):
            for i in range(OH):
                for j in range(OW):
                    patch = xp[n, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                    conv[n, o, i, j] = np.sum(patch * W[o]) + b[o]
    relu = np.maximum(conv, 0)
    pool, pstride = 2, 2
    POH = (OH - pool) // pstride + 1
    ref = np.zeros((N, oc, POH, POH))
    for n in range(N):
        for o in range(oc):
            for i in range(POH):
                for j in range(POH):
                    win = relu[n, o, i * pstride:i * pstride + pool, j * pstride:j * pstride + pool]
                    ref[n, o, i, j] = np.max(win)
    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=1e-6)


def test_0046_forward_classifier_block(ns):
    f = ns["forward_classifier_block"]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5))
    layer = ns["init_linear_layer"](5, 3, rng)
    # without activation: pure affine oracle
    out, _ = f(x, layer, activation=False)
    assert np.allclose(out, x @ layer["W"] + layer["b"], atol=1e-6)
    # with activation: relu of affine
    out_a, _ = f(x, layer, activation=True)
    assert np.allclose(out_a, np.maximum(x @ layer["W"] + layer["b"], 0), atol=1e-6)


def test_0047_lenet_forward(ns):
    f = ns["lenet_forward"]
    rng = np.random.default_rng(0)
    N, C, H, K = 4, 1, 8, 3
    params = ns["init_lenet"](C, K, H, rng)
    x = rng.normal(size=(N, C, H, H))
    logits, caches = f(params, x)
    assert logits.shape == (N, K)
    assert np.all(np.isfinite(logits))
    # determinism: same inputs -> same logits
    logits2, _ = f(params, x)
    assert np.allclose(logits, logits2)


def test_0048_backward_conv_block(ns):
    f = ns["backward_conv_block"]
    rng = np.random.default_rng(0)
    N, C, H = 4, 1, 8
    x = rng.normal(size=(N, C, H, H))
    layer = ns["init_conv_layer"](C, 2, 3, rng)
    out, cache = ns["forward_conv_block"](x, layer, 2, 2)
    dout = rng.normal(size=out.shape)
    dx, grads = f(dout, cache)

    def scalar_loss():
        o, _ = ns["forward_conv_block"](x, layer, 2, 2)
        return np.sum(o * dout)

    num_dx, idxs = _fd_grad_param(scalar_loss, x, n_probe=12)
    assert np.allclose(num_dx.ravel()[idxs], dx.ravel()[idxs], atol=1e-5)
    num_dW, idxs = _fd_grad_param(scalar_loss, layer["W"], n_probe=12)
    assert np.allclose(num_dW.ravel()[idxs], grads["W"].ravel()[idxs], atol=1e-5)
    num_db, idxs = _fd_grad_param(scalar_loss, layer["b"], n_probe=2)
    assert np.allclose(num_db.ravel()[idxs], grads["b"].ravel()[idxs], atol=1e-5)


def test_0049_backward_classifier_block(ns):
    f = ns["backward_classifier_block"]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5))
    layer = ns["init_linear_layer"](5, 3, rng)
    out, cache = ns["forward_classifier_block"](x, layer, activation=True)
    dout = rng.normal(size=out.shape)
    dx, grads = f(dout, cache)

    def scalar_loss():
        o, _ = ns["forward_classifier_block"](x, layer, activation=True)
        return np.sum(o * dout)

    num_dx, idxs = _fd_grad_param(scalar_loss, x, n_probe=20)
    assert np.allclose(num_dx.ravel()[idxs], dx.ravel()[idxs], atol=1e-5)
    num_dW, idxs = _fd_grad_param(scalar_loss, layer["W"], n_probe=15)
    assert np.allclose(num_dW.ravel()[idxs], grads["W"].ravel()[idxs], atol=1e-5)
    num_db, idxs = _fd_grad_param(scalar_loss, layer["b"], n_probe=3)
    assert np.allclose(num_db.ravel()[idxs], grads["b"].ravel()[idxs], atol=1e-5)


def test_0050_lenet_backward(ns):
    f = ns["lenet_backward"]
    rng = np.random.default_rng(0)
    N, C, H, K = 4, 1, 8, 3
    params = ns["init_lenet"](C, K, H, rng)
    x = rng.normal(size=(N, C, H, H))
    y = rng.integers(0, K, size=N)

    logits, caches = ns["lenet_forward"](params, x)
    loss, ce_cache = ns["softmax_cross_entropy_forward"](logits, y)
    dlogits = ns["softmax_cross_entropy_backward"](ce_cache)
    grads = f(dlogits, caches)

    def full_loss():
        lg, _ = ns["lenet_forward"](params, x)
        l, _ = ns["softmax_cross_entropy_forward"](lg, y)
        return l

    # finite-difference check across conv, fc, and output layers
    checks = [
        (params["convs"][0]["W"], grads["convs"][0]["W"]),
        (params["convs"][1]["W"], grads["convs"][1]["W"]),
        (params["fcs"][0]["W"], grads["fcs"][0]["W"]),
        (params["out"]["W"], grads["out"]["W"]),
        (params["out"]["b"], grads["out"]["b"]),
    ]
    for arr, garr in checks:
        num, idxs = _fd_grad_param(full_loss, arr, n_probe=8)
        assert np.allclose(num.ravel()[idxs], garr.ravel()[idxs], atol=1e-5)


def test_0051_lenet_predict(ns):
    f = ns["lenet_predict"]
    rng = np.random.default_rng(0)
    N, C, H, K = 4, 1, 8, 3
    params = ns["init_lenet"](C, K, H, rng)
    x = rng.normal(size=(N, C, H, H))
    preds = f(params, x)
    assert preds.shape == (N,)
    assert preds.dtype.kind in "iu"
    assert np.all((preds >= 0) & (preds < K))
    # independent recompute: argmax of forward logits
    logits, _ = ns["lenet_forward"](params, x)
    assert np.array_equal(preds, np.argmax(logits, axis=1))

def test_0052_build_synthetic_image_dataset(ns):
    """Shapes, dtypes, label range, class-separability of synthetic dataset."""
    build = ns["build_synthetic_image_dataset"]
    rng = np.random.default_rng(0)
    n_samples, image_size, num_classes = 40, 8, 3
    X, y = build(n_samples, image_size, num_classes, rng)
    assert X.shape == (n_samples, 1, image_size, image_size)
    assert y.shape == (n_samples,)
    assert np.issubdtype(y.dtype, np.integer)
    assert y.min() >= 0 and y.max() < num_classes
    # Determinism: same seed -> same data.
    rng2 = np.random.default_rng(0)
    X2, y2 = build(n_samples, image_size, num_classes, rng2)
    assert np.allclose(X, X2, atol=1e-12)
    assert np.array_equal(y, y2)
    # Class signal: per-class mean images must differ (so a CNN can separate them).
    means = []
    for c in range(num_classes):
        mc = X[y == c]
        if mc.shape[0] > 0:
            means.append(mc.mean(axis=0))
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            assert not np.allclose(means[i], means[j], atol=1e-2)


def test_0053_shuffle_indices(ns):
    """shuffle_indices returns a deterministic valid permutation."""
    shuffle_indices = ns["shuffle_indices"]
    n = 17
    rng = np.random.default_rng(0)
    perm = shuffle_indices(n, rng)
    assert perm.shape == (n,)
    # It is a permutation of 0..n-1.
    assert np.array_equal(np.sort(perm), np.arange(n))
    # Deterministic given seed.
    rng2 = np.random.default_rng(0)
    perm2 = shuffle_indices(n, rng2)
    assert np.array_equal(perm, perm2)
    # Actually shuffles (extremely unlikely to be identity for n=17).
    assert not np.array_equal(perm, np.arange(n))


def test_0054_train_test_split(ns):
    """Split sizes, disjointness, and coverage are correct."""
    split = ns["train_test_split"]
    n = 20
    X = np.arange(n * 1 * 8 * 8, dtype=np.float64).reshape(n, 1, 8, 8)
    y = np.arange(n)
    Xtr, ytr, Xte, yte = split(X, y, 0.25)
    assert Xtr.shape[0] == 15
    assert Xte.shape[0] == 5
    assert ytr.shape[0] == 15 and yte.shape[0] == 5
    # Coverage + disjointness via labels (unique ids).
    union = np.concatenate([ytr, yte])
    assert np.array_equal(np.sort(union), np.arange(n))
    assert len(set(ytr.tolist()) & set(yte.tolist())) == 0
    # X and y stay aligned (label equals index here).
    for i in range(Xtr.shape[0]):
        assert np.allclose(Xtr[i], X[ytr[i]])
    for i in range(Xte.shape[0]):
        assert np.allclose(Xte[i], X[yte[i]])
    # Deterministic.
    Xtr2, ytr2, Xte2, yte2 = split(X, y, 0.25)
    assert np.array_equal(ytr, ytr2) and np.array_equal(yte, yte2)


def test_0055_iterate_minibatches(ns):
    """Minibatches cover all data exactly once with aligned X/y; shuffle is deterministic."""
    iterate = ns["iterate_minibatches"]
    n = 10
    X = np.arange(n * 1 * 8 * 8, dtype=np.float64).reshape(n, 1, 8, 8)
    y = np.arange(n)
    # No shuffle: in-order, full coverage, aligned.
    batches = list(iterate(X, y, 4, np.random.default_rng(0), shuffle=False))
    sizes = [xb.shape[0] for xb, _ in batches]
    assert sizes == [4, 4, 2]
    seen_y = np.concatenate([yb for _, yb in batches])
    assert np.array_equal(seen_y, np.arange(n))
    for xb, yb in batches:
        for i in range(xb.shape[0]):
            assert np.allclose(xb[i], X[yb[i]])
    # Shuffle: covers all once but in different order; deterministic given seed.
    b1 = list(iterate(X, y, 4, np.random.default_rng(0), shuffle=True))
    seen1 = np.concatenate([yb for _, yb in b1])
    assert np.array_equal(np.sort(seen1), np.arange(n))
    for xb, yb in b1:
        for i in range(xb.shape[0]):
            assert np.allclose(xb[i], X[yb[i]])
    b2 = list(iterate(X, y, 4, np.random.default_rng(0), shuffle=True))
    seen2 = np.concatenate([yb for _, yb in b2])
    assert np.array_equal(seen1, seen2)

def _ts_setup(ns, n=24, K=3, size=8, seed=0):
    rng = np.random.default_rng(seed)
    params = ns["init_lenet"](1, K, size, rng)
    X, y = ns["build_synthetic_image_dataset"](n, size, K, rng)
    return params, X, y


def _ts_opt_state(params):
    def z(layer):
        return {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)}
                for k, v in layer.items() if isinstance(v, np.ndarray)}
    state = {}
    for g in ("convs", "fcs", "out"):
        if g not in params:
            continue
        val = params[g]
        state[g] = [z(l) for l in val] if isinstance(val, list) else z(val)
    return state


def test_0056_train_step(ns):
    """train_step reports the pre-step loss, applies a correct Adam update, and preserves structure."""
    np.random.seed(0)
    params, X, y = _ts_setup(ns, n=8)
    opt = _ts_opt_state(params)
    logits0, _ = ns["lenet_forward"](params, X)
    loss0 = float(ns["softmax_cross_entropy_forward"](logits0, y)[0])
    W_before = params["convs"][0]["W"].copy()
    new_params, new_opt, loss = ns["train_step"](params, X, y, 1e-2, opt, 0)
    # reported loss is the pre-update forward loss
    assert np.isclose(float(loss), loss0, atol=1e-5), (float(loss), loss0)
    assert set(new_params.keys()) == set(params.keys())
    for g in ("convs", "fcs"):
        assert len(new_params[g]) == len(params[g])
    # the update actually moved the weights
    assert not np.allclose(new_params["convs"][0]["W"], W_before)
    # rigorous independent manual-Adam oracle for one leaf conv[0]["W"] (t=0 -> uses t=1 internally)
    p0, X0, y0 = _ts_setup(ns, n=8)
    o0 = _ts_opt_state(p0)
    lg, caches = ns["lenet_forward"](p0, X0)
    _, sc = ns["softmax_cross_entropy_forward"](lg, y0)
    grads = ns["lenet_backward"](ns["softmax_cross_entropy_backward"](sc), caches)
    g = grads["convs"][0]["W"]
    W = p0["convs"][0]["W"].copy()
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-2
    m_hat = ((1 - b1) * g) / (1 - b1)
    v_hat = ((1 - b2) * g ** 2) / (1 - b2)
    expected = W - lr * m_hat / (np.sqrt(v_hat) + eps)
    out_params, _, _ = ns["train_step"](p0, X0, y0, lr, o0, 0)
    assert np.allclose(out_params["convs"][0]["W"], expected, atol=1e-6)


def test_0057_train_one_epoch(ns):
    """train_one_epoch covers every minibatch, advances the step counter, and lowers loss epoch over epoch."""
    np.random.seed(0)
    params, X, y = _ts_setup(ns, n=24)
    opt = _ts_opt_state(params)
    bs = 8
    new_params, new_opt, t, ep_loss = ns["train_one_epoch"](
        params, X, y, bs, 1e-2, opt, 0, np.random.default_rng(5))
    n_batches = int(np.ceil(len(X) / bs))
    assert t == n_batches, (t, n_batches)
    assert np.isfinite(ep_loss)
    # some weight actually changed
    moved = any(not np.allclose(new_params[g][i]["W"], params[g][i]["W"])
                for g in params for i in range(len(params[g])))
    assert moved
    # a second epoch has a lower mean loss
    _, _, _, ep_loss2 = ns["train_one_epoch"](
        new_params, X, y, bs, 1e-2, new_opt, t, np.random.default_rng(6))
    assert ep_loss2 < ep_loss, (ep_loss2, ep_loss)


def test_0058_train_loop(ns):
    """train_loop runs the requested epochs from fresh optimizer state and drives the loss down."""
    np.random.seed(0)
    params, X, y = _ts_setup(ns, n=24, seed=1)
    trained, history = ns["train_loop"](
        params, X, y, n_epochs=8, batch_size=8, lr=5e-3, rng=np.random.default_rng(3))
    losses = history["loss"] if isinstance(history, dict) else history
    losses = list(np.ravel(np.asarray(losses, dtype=float)))
    assert len(losses) == 8, len(losses)
    assert losses[-1] < losses[0], losses
    # clear downward trend
    assert np.mean(losses[-3:]) < np.mean(losses[:3]), losses
    # determinism: same seeds reproduce the same history
    params2, X2, y2 = _ts_setup(ns, n=24, seed=1)
    _, history2 = ns["train_loop"](
        params2, X2, y2, n_epochs=8, batch_size=8, lr=5e-3, rng=np.random.default_rng(3))
    losses2 = list(np.ravel(np.asarray(
        (history2["loss"] if isinstance(history2, dict) else history2), dtype=float)))
    assert np.allclose(losses, losses2, atol=1e-12)


def test_0059_evaluate(ns):
    """evaluate returns mean CE loss and accuracy matching an independent manual computation."""
    np.random.seed(0)
    params, X, y = _ts_setup(ns, n=16, seed=2)
    loss, acc = ns["evaluate"](params, X, y)
    logits, _ = ns["lenet_forward"](params, X)
    probs = ns["stable_softmax"](logits)
    man_loss = float(np.mean(-np.log(probs[np.arange(len(y)), y] + 1e-12)))
    man_acc = float(np.mean(np.argmax(logits, axis=1) == y))
    assert np.isclose(loss, man_loss, atol=1e-6), (loss, man_loss)
    assert np.isclose(acc, man_acc, atol=1e-12), (acc, man_acc)
    assert 0.0 <= acc <= 1.0
    # training reduces the evaluated loss on the same data
    trained, _ = ns["train_loop"](params, X, y, 10, 8, 5e-3, np.random.default_rng(7))
    loss_t, _ = ns["evaluate"](trained, X, y)
    assert loss_t < loss, (loss_t, loss)
