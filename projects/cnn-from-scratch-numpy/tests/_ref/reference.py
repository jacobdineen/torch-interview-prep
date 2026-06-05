"""Hidden reference implementations for cnn-from-scratch-numpy.

One top-level def per step; steps call earlier references directly (shared
namespace). A LeNet-style CNN in pure NumPy: im2col conv, pooling, backprop, Adam.
"""
import numpy as np

import numpy as np


def argmax_rows(x):
    """Index of the max per row, shape (N,)."""
    return np.argmax(x, axis=1)


def row_max(x):
    """Max per row with keepdims, shape (N,1)."""
    return np.max(x, axis=1, keepdims=True)


def row_sum(x):
    """Sum per row with keepdims, shape (N,1)."""
    return np.sum(x, axis=1, keepdims=True)


def exp_shifted(x):
    """Numerically stable exp(x - row_max(x)), shape (N,K)."""
    return np.exp(x - row_max(x))


def stable_softmax(x):
    """Row-wise softmax via exp_shifted / row_sum(exp_shifted), shape (N,K)."""
    e = exp_shifted(x)
    return e / row_sum(e)


def one_hot(labels, num_classes):
    """One-hot encode integer labels into float rows, shape (N, num_classes)."""
    labels = np.asarray(labels).astype(int)
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def gather_true_class_probs(probs, labels):
    """Pick probs[i, labels[i]] for each row, shape (N,)."""
    labels = np.asarray(labels).astype(int)
    return probs[np.arange(probs.shape[0]), labels]


def cross_entropy_loss(probs, labels):
    """Mean of -log(true-class prob + 1e-12) over the batch (scalar float)."""
    p = gather_true_class_probs(probs, labels)
    return float(np.mean(-np.log(p + 1e-12)))


def accuracy(logits, labels):
    """Mean fraction of rows whose argmax equals the label (scalar float)."""
    labels = np.asarray(labels).astype(int)
    return float(np.mean(argmax_rows(logits) == labels))

def he_std(fan_in):
    """He initialization standard deviation sqrt(2/fan_in)."""
    return np.sqrt(2.0 / fan_in)


def he_init(shape, fan_in, rng):
    """Sample an array of `shape` from N(0, he_std(fan_in)) using rng."""
    return rng.normal(0.0, he_std(fan_in), size=shape)


def init_zero_bias(n):
    """Return a length-n zero bias vector."""
    return np.zeros(n)


def pad_2d(x, pad):
    """Zero-pad the spatial dims of an NCHW array by `pad` on each side."""
    if pad == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")


def output_spatial_size(in_size, k, stride, pad):
    """Convolution/pool output size along one spatial dim."""
    return (in_size + 2 * pad - k) // stride + 1


def im2col(x, kh, kw, stride, pad):
    """Extract sliding patches into (N, C*kh*kw, OH*OW), row=channel-major (c,ki,kj)."""
    N, C, H, W = x.shape
    OH = output_spatial_size(H, kh, stride, pad)
    OW = output_spatial_size(W, kw, stride, pad)
    xp = pad_2d(x, pad)
    cols = np.empty((N, C * kh * kw, OH * OW), dtype=xp.dtype)
    row = 0
    for c in range(C):
        for ki in range(kh):
            for kj in range(kw):
                # patch[n, oh, ow] = xp[n, c, ki + oh*stride, kj + ow*stride]
                patch = xp[:, c,
                           ki:ki + stride * OH:stride,
                           kj:kj + stride * OW:stride]
                cols[:, row, :] = patch.reshape(N, OH * OW)
                row += 1
    return cols


def col2im(cols, x_shape, kh, kw, stride, pad):
    """Inverse of im2col, accumulating overlapping gradient contributions."""
    N, C, H, W = x_shape
    OH = output_spatial_size(H, kh, stride, pad)
    OW = output_spatial_size(W, kw, stride, pad)
    Hp = H + 2 * pad
    Wp = W + 2 * pad
    xp = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)
    row = 0
    for c in range(C):
        for ki in range(kh):
            for kj in range(kw):
                contrib = cols[:, row, :].reshape(N, OH, OW)
                # accumulate into the strided slice
                for oh in range(OH):
                    hh = ki + oh * stride
                    xp[:, c, hh, kj:kj + stride * OW:stride] += contrib[:, oh, :]
                row += 1
    if pad == 0:
        return xp
    return xp[:, :, pad:pad + H, pad:pad + W]

def conv2d_forward(x, W, b, stride, pad):
    """Forward 2D convolution via im2col returning output and cache."""
    N, C, H, Wd = x.shape
    out_ch = W.shape[0]
    kh, kw = W.shape[2], W.shape[3]
    OH = output_spatial_size(H, kh, stride, pad)
    OW = output_spatial_size(Wd, kw, stride, pad)
    cols = im2col(x, kh, kw, stride, pad)
    Wr = W.reshape(out_ch, -1)
    out = np.einsum('oc,ncl->nol', Wr, cols) + b[None, :, None]
    out = out.reshape(N, out_ch, OH, OW)
    cache = (x, W, b, stride, pad)
    return out, cache


def conv2d_grad_input(dout, x, W, stride, pad):
    """Gradient of conv output w.r.t. input via col2im."""
    N = x.shape[0]
    out_ch = W.shape[0]
    kh, kw = W.shape[2], W.shape[3]
    OH, OW = dout.shape[2], dout.shape[3]
    dout_r = dout.reshape(N, out_ch, OH * OW)
    Wr = W.reshape(out_ch, -1)
    dcols = np.einsum('oc,nol->ncl', Wr, dout_r)
    dx = col2im(dcols, x.shape, kh, kw, stride, pad)
    return dx


def conv2d_grad_weights(dout, x, W, stride, pad):
    """Gradient of conv output w.r.t. weights."""
    N = x.shape[0]
    out_ch = W.shape[0]
    kh, kw = W.shape[2], W.shape[3]
    OH, OW = dout.shape[2], dout.shape[3]
    cols = im2col(x, kh, kw, stride, pad)
    dout_r = dout.reshape(N, out_ch, OH * OW)
    dWr = np.einsum('nol,ncl->oc', dout_r, cols)
    dW = dWr.reshape(W.shape)
    return dW


def conv2d_grad_bias(dout):
    """Gradient of conv output w.r.t. bias (sum over N, OH, OW)."""
    return dout.sum(axis=(0, 2, 3))


def conv2d_backward(dout, cache):
    """Compose conv input/weight/bias gradients from cache."""
    x, W, b, stride, pad = cache
    dx = conv2d_grad_input(dout, x, W, stride, pad)
    dW = conv2d_grad_weights(dout, x, W, stride, pad)
    db = conv2d_grad_bias(dout)
    return dx, dW, db

def maxpool2d_forward(x, pool, stride):
    """Channelwise max over pool x pool windows; returns (out, cache)."""
    N, C, H, W = x.shape
    OH = (H - pool) // stride + 1
    OW = (W - pool) // stride + 1
    out = np.empty((N, C, OH, OW), dtype=x.dtype)
    argmax = np.empty((N, C, OH, OW, 2), dtype=np.int64)
    for oh in range(OH):
        hs = oh * stride
        for ow in range(OW):
            ws = ow * stride
            window = x[:, :, hs:hs + pool, ws:ws + pool]  # (N,C,pool,pool)
            flat = window.reshape(N, C, -1)
            idx = np.argmax(flat, axis=2)  # (N,C)
            out[:, :, oh, ow] = np.max(flat, axis=2)
            argmax[:, :, oh, ow, 0] = idx // pool
            argmax[:, :, oh, ow, 1] = idx % pool
    cache = (x.shape, pool, stride, argmax)
    return out, cache


def scatter_grad_window(dout_val, window):
    """Route gradient to argmax position of a pooling window (1 at max else 0, times dout_val)."""
    grad = np.zeros_like(window, dtype=np.float64)
    idx = np.unravel_index(np.argmax(window), window.shape)
    grad[idx] = dout_val
    return grad


def maxpool2d_backward(dout, cache):
    """Scatter gradients back to the stored max positions; returns dx."""
    x_shape, pool, stride, argmax = cache
    N, C, H, W = x_shape
    dx = np.zeros(x_shape, dtype=np.float64)
    OH, OW = dout.shape[2], dout.shape[3]
    n_idx, c_idx = np.meshgrid(np.arange(N), np.arange(C), indexing="ij")
    for oh in range(OH):
        hs = oh * stride
        for ow in range(OW):
            ws = ow * stride
            di = argmax[:, :, oh, ow, 0]  # (N,C)
            dj = argmax[:, :, oh, ow, 1]  # (N,C)
            np.add.at(dx, (n_idx, c_idx, hs + di, ws + dj), dout[:, :, oh, ow])
    return dx


def relu_forward(x):
    """Elementwise max(x, 0); returns (out, cache)."""
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """dout * (x > 0); returns dx."""
    x = cache
    return dout * (x > 0)


def flatten_forward(x):
    """Reshape (N,C,H,W) -> (N, C*H*W); cache original shape; returns (out, cache)."""
    out = x.reshape(x.shape[0], -1)
    cache = x.shape
    return out, cache


def flatten_backward(dout, cache):
    """Reshape (N, C*H*W) back to original shape; returns dx."""
    return dout.reshape(cache)


def linear_forward(x, W, b):
    """x @ W + b; returns (out, cache)."""
    out = x @ W + b
    cache = (x, W, b)
    return out, cache


def linear_grad_input(dout, W):
    """dout @ W.T; returns dx."""
    return dout @ W.T


def linear_grad_weights(dout, x):
    """x.T @ dout; returns dW."""
    return x.T @ dout


def linear_grad_bias(dout):
    """Sum over rows; returns db."""
    return dout.sum(axis=0)


def linear_backward(dout, cache):
    """Compose linear grads from cache; returns (dx, dW, db)."""
    x, W, b = cache
    dx = linear_grad_input(dout, W)
    dW = linear_grad_weights(dout, x)
    db = linear_grad_bias(dout)
    return dx, dW, db

def softmax_cross_entropy_forward(logits, labels):
    """Fused softmax + cross-entropy loss; returns (loss, cache) with probs and labels."""
    probs = stable_softmax(logits)
    loss = cross_entropy_loss(probs, labels)
    cache = {"probs": probs, "labels": np.asarray(labels)}
    return loss, cache


def softmax_cross_entropy_backward(cache):
    """Gradient of fused softmax cross-entropy w.r.t. logits: (probs - one_hot)/N."""
    probs = cache["probs"]
    labels = cache["labels"]
    N, K = probs.shape
    dlogits = (probs - one_hot(labels, K)) / N
    return dlogits


def sgd_step(param, grad, lr):
    """One vanilla SGD update: param - lr*grad."""
    return param - lr * grad


def adam_update_m(m, grad, beta1):
    """Adam first-moment (mean) EMA update: beta1*m + (1-beta1)*grad."""
    return beta1 * m + (1.0 - beta1) * grad


def adam_update_v(v, grad, beta2):
    """Adam second-moment (uncentered variance) EMA update: beta2*v + (1-beta2)*grad**2."""
    return beta2 * v + (1.0 - beta2) * (grad ** 2)


def adam_bias_correct(moment, beta, t):
    """Bias-correct an Adam moment estimate: moment / (1 - beta**t)."""
    return moment / (1.0 - beta ** t)


def adam_param_step(param, m_hat, v_hat, lr, eps):
    """Apply a bias-corrected Adam step: param - lr*m_hat/(sqrt(v_hat)+eps)."""
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)


def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Full Adam update for one param at step t (t is the current, already-incremented step); returns (new_param, new_m, new_v)."""
    m = adam_update_m(m, grad, beta1)
    v = adam_update_v(v, grad, beta2)
    m_hat = adam_bias_correct(m, beta1, t)
    v_hat = adam_bias_correct(v, beta2, t)
    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)
    return new_param, m, v

def init_conv_layer(in_ch, out_ch, k, rng):
    """Init a conv layer dict with He weights (out_ch,in_ch,k,k), zero bias, stride 1, same-pad."""
    fan_in = in_ch * k * k
    return {"W": he_init((out_ch, in_ch, k, k), fan_in, rng), "b": init_zero_bias(out_ch),
            "stride": 1, "pad": k // 2}

def init_linear_layer(in_dim, out_dim, rng):
    """Init a linear layer dict with He weights (in,out) and zero bias (out,)."""
    return {"W": he_init((in_dim, out_dim), in_dim, rng), "b": init_zero_bias(out_dim)}

def init_lenet(in_ch, num_classes, image_size, rng):
    """Build a small LeNet: 2 conv blocks (3x3 same-pad, pool2) then 2 FC + logits layer."""
    c1_out, c2_out = 6, 16
    k, pad, pool, ps = 3, 1, 2, 2
    conv1 = init_conv_layer(in_ch, c1_out, k, rng)
    conv2 = init_conv_layer(c1_out, c2_out, k, rng)
    # same-padding conv keeps spatial size; pool halves it. Two blocks.
    s = image_size
    for _ in range(2):
        s = (s + 2 * pad - k) + 1     # conv stride 1
        s = (s - pool) // ps + 1      # pool 2x2 stride 2
    flat = c2_out * s * s
    fc1 = init_linear_layer(flat, 120, rng)
    fc2 = init_linear_layer(120, 84, rng)
    out = init_linear_layer(84, num_classes, rng)
    return {"convs": [conv1, conv2], "fcs": [fc1, fc2], "out": out,
            "pool": pool, "pool_stride": ps, "conv_stride": 1, "conv_pad": pad}

def forward_conv_block(x, conv_layer, pool, stride):
    """Conv -> ReLU -> MaxPool; returns (out, cache) holding the three sub-caches."""
    cstride = conv_layer.get("stride", 1)
    cpad = conv_layer.get("pad", 0)
    c_out, c_cache = conv2d_forward(x, conv_layer["W"], conv_layer["b"], cstride, cpad)
    r_out, r_cache = relu_forward(c_out)
    p_out, p_cache = maxpool2d_forward(r_out, pool, stride)
    return p_out, (c_cache, r_cache, p_cache)

def forward_classifier_block(x, linear_layer, activation=True):
    """Linear (optionally followed by ReLU); returns (out, cache)."""
    l_out, l_cache = linear_forward(x, linear_layer["W"], linear_layer["b"])
    if activation:
        a_out, a_cache = relu_forward(l_out)
        return a_out, (l_cache, a_cache)
    return l_out, (l_cache, None)

def lenet_forward(params, x):
    """Run conv blocks -> flatten -> FC blocks (ReLU) -> final linear logits; returns (logits, caches)."""
    pool, ps = params["pool"], params["pool_stride"]
    conv_caches = []
    h = x
    for cl in params["convs"]:
        h, cc = forward_conv_block(h, cl, pool, ps)
        conv_caches.append(cc)
    h, flat_cache = flatten_forward(h)
    fc_caches = []
    for fl in params["fcs"]:
        h, fcc = forward_classifier_block(h, fl, activation=True)
        fc_caches.append(fcc)
    logits, out_cache = forward_classifier_block(h, params["out"], activation=False)
    caches = {"convs": conv_caches, "flat": flat_cache, "fcs": fc_caches, "out": out_cache}
    return logits, caches

def backward_conv_block(dout, cache):
    """Invert MaxPool -> ReLU -> Conv; returns (dx, {'W':dW,'b':db})."""
    c_cache, r_cache, p_cache = cache
    dr = maxpool2d_backward(dout, p_cache)
    dc = relu_backward(dr, r_cache)
    dx, dW, db = conv2d_backward(dc, c_cache)
    return dx, {"W": dW, "b": db}

def backward_classifier_block(dout, cache):
    """Invert (ReLU ->) Linear; returns (dx, {'W':dW,'b':db})."""
    l_cache, a_cache = cache
    if a_cache is not None:
        dout = relu_backward(dout, a_cache)
    dx, dW, db = linear_backward(dout, l_cache)
    return dx, {"W": dW, "b": db}

def lenet_backward(dlogits, caches):
    """Backprop through the whole net; returns grads with the same structure as params."""
    dh, out_grads = backward_classifier_block(dlogits, caches["out"])
    fc_grads = [None] * len(caches["fcs"])
    for i in reversed(range(len(caches["fcs"]))):
        dh, g = backward_classifier_block(dh, caches["fcs"][i])
        fc_grads[i] = g
    dh = flatten_backward(dh, caches["flat"])
    conv_grads = [None] * len(caches["convs"])
    for i in reversed(range(len(caches["convs"]))):
        dh, g = backward_conv_block(dh, caches["convs"][i])
        conv_grads[i] = g
    return {"convs": conv_grads, "fcs": fc_grads, "out": out_grads}

def lenet_predict(params, x):
    """Return predicted class labels (N,) via argmax of lenet_forward logits."""
    logits, _ = lenet_forward(params, x)
    return argmax_rows(logits)

def build_synthetic_image_dataset(n_samples, image_size, num_classes, rng):
    """Build (X (N,1,H,W), y (N,)) of class-dependent images a CNN can learn."""
    H = W = image_size
    y = rng.integers(0, num_classes, size=n_samples).astype(np.int64)
    X = np.zeros((n_samples, 1, H, W), dtype=np.float64)
    # Each class gets a deterministic spatial pattern (a bright band whose
    # position depends on the class), plus a class-dependent additive offset,
    # plus small noise. Patterns are linearly/spatially separable so a CNN learns them.
    for i in range(n_samples):
        c = int(y[i])
        img = rng.normal(0.0, 0.1, size=(H, W))
        # class-specific horizontal band
        row = (c * H) // num_classes
        band_hi = min(row + max(1, H // num_classes), H)
        img[row:band_hi, :] += 1.0 + c
        # class-specific offset
        img += 0.5 * c
        X[i, 0] = img
    return X, y


def shuffle_indices(n, rng):
    """Return a random permutation of range(n) using rng."""
    return rng.permutation(n)


def train_test_split(X, y, test_frac):
    """Deterministically split into (Xtr, ytr, Xte, yte) by fraction, no shuffle here."""
    n = X.shape[0]
    n_test = int(round(n * test_frac))
    n_train = n - n_test
    Xtr = X[:n_train]
    ytr = y[:n_train]
    Xte = X[n_train:]
    yte = y[n_train:]
    return Xtr, ytr, Xte, yte


def iterate_minibatches(X, y, batch_size, rng, shuffle=True):
    """Yield (Xb, yb) minibatches covering all data once, optionally shuffled."""
    n = X.shape[0]
    if shuffle:
        order = shuffle_indices(n, rng)
    else:
        order = np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]

_PARAM_GROUPS = ("convs", "fcs", "out")


def _init_opt_state(params):
    """Build per-parameter Adam moment buffers (m, v) mirroring the params param-groups."""
    def z(layer):
        return {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)}
                for k, v in layer.items() if isinstance(v, np.ndarray)}
    state = {}
    for g in _PARAM_GROUPS:
        if g not in params:
            continue
        val = params[g]
        state[g] = [z(l) for l in val] if isinstance(val, list) else z(val)
    return state


def train_step(params, X, y, lr, opt_state, t):
    """One optimization step: lenet forward, softmax-CE loss/grad, lenet backward, Adam-update every param."""
    t = t + 1
    logits, caches = lenet_forward(params, X)
    loss, sc_cache = softmax_cross_entropy_forward(logits, y)
    dlogits = softmax_cross_entropy_backward(sc_cache)
    grads = lenet_backward(dlogits, caches)

    def upd_layer(layer, glayer, slayer):
        new_layer = dict(layer)  # preserve any config keys (stride, pad, ...)
        for key, val in layer.items():
            if not isinstance(val, np.ndarray) or key not in glayer:
                continue
            st = slayer[key]
            nv, m, v = adam_step(val, glayer[key], st["m"], st["v"], t, lr=lr)
            new_layer[key] = nv
            st["m"], st["v"] = m, v
        return new_layer

    new_params = dict(params)  # preserve config keys (pool, conv_pad, ...)
    for g in _PARAM_GROUPS:
        if g not in params:
            continue
        pval, gval, sval = params[g], grads[g], opt_state[g]
        if isinstance(pval, list):
            new_params[g] = [upd_layer(pval[i], gval[i], sval[i]) for i in range(len(pval))]
        else:
            new_params[g] = upd_layer(pval, gval, sval)
    return new_params, opt_state, loss


def train_one_epoch(params, X, y, batch_size, lr, opt_state, t, rng):
    """Run one epoch of minibatch Adam training; return params, opt_state, step t, and mean batch loss."""
    losses = []
    for Xb, yb in iterate_minibatches(X, y, batch_size, rng, shuffle=True):
        params, opt_state, loss = train_step(params, Xb, yb, lr, opt_state, t)
        t = t + 1
        losses.append(loss)
    epoch_loss = float(np.mean(losses)) if losses else 0.0
    return params, opt_state, t, epoch_loss


def train_loop(params, X, y, n_epochs, batch_size, lr, rng):
    """Initialize optimizer state and train for n_epochs of Adam; return params and a loss history."""
    opt_state = _init_opt_state(params)
    t = 0
    history = {"loss": []}
    for _ in range(n_epochs):
        params, opt_state, t, epoch_loss = train_one_epoch(
            params, X, y, batch_size, lr, opt_state, t, rng)
        history["loss"].append(epoch_loss)
    return params, history


def evaluate(params, X, y):
    """Forward pass on held-out data; return (mean cross-entropy loss, accuracy)."""
    logits, _ = lenet_forward(params, X)
    loss, _ = softmax_cross_entropy_forward(logits, y)
    return float(loss), accuracy(logits, y)
