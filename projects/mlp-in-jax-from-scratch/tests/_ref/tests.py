"""Hidden tests for mlp-in-jax-from-scratch. One test_<id>_<name>(ns) per step;
deterministic, independent JAX oracles. Tiny + fast."""
import jax
import jax.numpy as jnp

def test_0001_make_prng_key(ns):
    make_prng_key = ns["make_prng_key"]
    k0 = make_prng_key(0)
    # Same seed yields identical keys; matches jax.random.PRNGKey directly.
    assert jnp.allclose(k0, jax.random.PRNGKey(0), atol=1e-5)
    assert jnp.allclose(make_prng_key(0), make_prng_key(0), atol=1e-5)
    # Different seeds yield different keys.
    assert not jnp.allclose(make_prng_key(0), make_prng_key(1), atol=1e-5)


def test_0002_split_prng_key(ns):
    split_prng_key = ns["split_prng_key"]
    key = jax.random.PRNGKey(0)
    keys = split_prng_key(key, 2)
    assert keys.shape[0] == 2
    # Matches jax.random.split.
    assert jnp.allclose(keys, jax.random.split(key, 2), atol=1e-5)
    # The two produced keys differ from each other.
    assert not jnp.allclose(keys[0], keys[1], atol=1e-5)
    # Default num=2.
    assert split_prng_key(key).shape[0] == 2
    # num=4 produces 4 distinct keys.
    keys4 = split_prng_key(key, 4)
    assert keys4.shape[0] == 4
    assert jnp.allclose(keys4, jax.random.split(key, 4), atol=1e-5)


def test_0003_sample_normal_matrix(ns):
    sample_normal_matrix = ns["sample_normal_matrix"]
    key = jax.random.PRNGKey(0)
    shape = (16, 4)
    A = sample_normal_matrix(key, shape)
    assert A.shape == shape
    # Matches jax.random.normal for the same key/shape.
    assert jnp.allclose(A, jax.random.normal(key, shape), atol=1e-5)
    # Deterministic given the same key.
    assert jnp.allclose(A, sample_normal_matrix(key, shape), atol=1e-5)
    # Different keys give different draws.
    B = sample_normal_matrix(jax.random.PRNGKey(1), shape)
    assert not jnp.allclose(A, B, atol=1e-5)
    # Standard normal: empirical mean near 0 over a large sample.
    big = sample_normal_matrix(key, (10000,))
    assert abs(float(jnp.mean(big))) < 0.1
    assert abs(float(jnp.std(big)) - 1.0) < 0.1

def test_0004_sample_input_features(ns):
    f = ns["sample_input_features"]
    key = jax.random.PRNGKey(0)
    X = f(key, 16, 4)
    assert X.shape == (16, 4)
    expected = jax.random.normal(key, (16, 4))
    assert jnp.allclose(X, expected, atol=1e-5)
    # different key -> different draw
    X2 = f(jax.random.PRNGKey(1), 16, 4)
    assert not jnp.allclose(X, X2, atol=1e-5)


def test_0005_assign_class_labels(ns):
    f = ns["assign_class_labels"]
    n_classes = 3
    X = jax.random.normal(jax.random.PRNGKey(0), (16, 4))
    labels = f(X, n_classes)
    assert labels.shape == (16,)
    assert jnp.issubdtype(labels.dtype, jnp.integer)
    assert jnp.all(labels >= 0) and jnp.all(labels < n_classes)
    # independent oracle: split 4 cols into 3 contiguous groups -> [0,1], [2], [3]
    groups = jnp.array_split(jnp.arange(4), n_classes)
    sums = jnp.stack([X[:, g].sum(axis=1) for g in groups], axis=1)
    expected = jnp.argmax(sums, axis=1)
    assert jnp.array_equal(labels, expected)
    # explicit hand-built case
    Xm = jnp.array([[5.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 9.0, 0.0],
                    [0.0, 0.0, 0.0, 7.0]])
    lm = f(Xm, 3)
    assert jnp.array_equal(lm, jnp.array([0, 1, 2]))


def test_0006_one_hot_encode_labels(ns):
    f = ns["one_hot_encode_labels"]
    n_classes = 3
    labels = jnp.array([0, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    oh = f(labels, n_classes)
    assert oh.shape == (16, n_classes)
    # rows are valid one-hot
    assert jnp.allclose(oh.sum(axis=1), jnp.ones(16), atol=1e-5)
    assert jnp.array_equal(jnp.argmax(oh, axis=1), labels)
    # independent oracle via manual eye construction
    expected = jnp.eye(n_classes)[labels]
    assert jnp.allclose(oh, expected, atol=1e-5)

def test_0007_init_linear_layer(ns):
    init_linear_layer = ns["init_linear_layer"]
    key = jax.random.PRNGKey(0)
    in_dim, out_dim, scale = 4, 3, 0.1
    W, b = init_linear_layer(key, in_dim, out_dim, scale)
    # shapes
    assert W.shape == (in_dim, out_dim)
    assert b.shape == (out_dim,)
    # bias is zeros
    assert jnp.allclose(b, jnp.zeros((out_dim,)), atol=1e-5)
    # W matches independent recompute of scale * normal
    expected_W = scale * jax.random.normal(key, (in_dim, out_dim))
    assert jnp.allclose(W, expected_W, atol=1e-5)
    # default scale = 0.1
    W2, b2 = init_linear_layer(key, in_dim, out_dim)
    assert jnp.allclose(W2, expected_W, atol=1e-5)
    # determinism: same key -> same draw
    Wd, bd = init_linear_layer(key, in_dim, out_dim, scale)
    assert jnp.allclose(Wd, W, atol=1e-5)
    # different key -> different draw
    Wo, bo = init_linear_layer(jax.random.PRNGKey(1), in_dim, out_dim, scale)
    assert not jnp.allclose(Wo, W, atol=1e-5)


def test_0008_init_mlp_params(ns):
    init_mlp_params = ns["init_mlp_params"]
    init_linear_layer = ns["init_linear_layer"]
    key = jax.random.PRNGKey(0)
    layer_sizes = [4, 8, 3]
    params = init_mlp_params(key, layer_sizes)
    # one tuple per layer transition
    assert len(params) == len(layer_sizes) - 1
    # shapes correct per layer
    for i, (W, b) in enumerate(params):
        assert W.shape == (layer_sizes[i], layer_sizes[i + 1])
        assert b.shape == (layer_sizes[i + 1],)
        assert jnp.allclose(b, jnp.zeros((layer_sizes[i + 1],)), atol=1e-5)
    # independent oracle: replicate key splitting and per-layer init
    num_layers = len(layer_sizes) - 1
    keys = jax.random.split(key, num_layers)
    for i in range(num_layers):
        eW = 0.1 * jax.random.normal(keys[i], (layer_sizes[i], layer_sizes[i + 1]))
        assert jnp.allclose(params[i][0], eW, atol=1e-5)
    # layers use distinct keys -> distinct weight stats (not identical scaled draws)
    # determinism: same key reproduces
    params2 = init_mlp_params(key, layer_sizes)
    for (W, b), (W2, b2) in zip(params, params2):
        assert jnp.allclose(W, W2, atol=1e-5)

def test_0009_linear_forward(ns):
    linear_forward = ns["linear_forward"]
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    N, in_dim, out_dim = 16, 4, 3
    x = jax.random.normal(k1, (N, in_dim))
    W = jax.random.normal(k2, (in_dim, out_dim))
    b = jax.random.normal(k3, (out_dim,))
    got = linear_forward((W, b), x)
    expected = jnp.dot(x, W) + b
    assert got.shape == (N, out_dim)
    assert jnp.allclose(got, expected, atol=1e-5)
    # check a single element manually
    manual_00 = jnp.sum(x[0] * W[:, 0]) + b[0]
    assert jnp.allclose(got[0, 0], manual_00, atol=1e-5)


def test_0010_relu_activation(ns):
    relu_activation = ns["relu_activation"]
    x = jnp.array([[-2.0, -0.5, 0.0, 0.5, 2.0]], dtype=jnp.float32)
    got = relu_activation(x)
    expected = jnp.where(x > 0, x, 0.0)
    assert jnp.allclose(got, expected, atol=1e-5)
    assert jnp.all(got >= 0)
    key = jax.random.PRNGKey(1)
    r = jax.random.normal(key, (16, 8))
    assert jnp.allclose(relu_activation(r), jnp.maximum(r, 0), atol=1e-5)


def test_0011_softmax_probabilities(ns):
    softmax_probabilities = ns["softmax_probabilities"]
    key = jax.random.PRNGKey(2)
    logits = jax.random.normal(key, (16, 3))
    got = softmax_probabilities(logits)
    # rows sum to 1
    assert jnp.allclose(jnp.sum(got, axis=-1), jnp.ones(16), atol=1e-5)
    assert jnp.all(got > 0)
    # matches jax.nn.softmax
    assert jnp.allclose(got, jax.nn.softmax(logits, axis=-1), atol=1e-5)
    # numerical stability: large logits do not produce nan/inf
    big = jnp.array([[1000.0, 1001.0, 999.0]], dtype=jnp.float32)
    out_big = softmax_probabilities(big)
    assert jnp.all(jnp.isfinite(out_big))
    assert jnp.allclose(jnp.sum(out_big, axis=-1), jnp.ones(1), atol=1e-5)
    # shift invariance
    shifted = softmax_probabilities(logits + 7.0)
    assert jnp.allclose(got, shifted, atol=1e-5)


def test_0012_mlp_forward(ns):
    mlp_forward = ns["mlp_forward"]
    key = jax.random.PRNGKey(3)
    N, in_dim, hidden, out_dim = 16, 4, 8, 3
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = jax.random.normal(k1, (N, in_dim))
    W1 = jax.random.normal(k2, (in_dim, hidden))
    b1 = jax.random.normal(k3, (hidden,))
    W2 = jax.random.normal(k4, (hidden, out_dim))
    b2 = jnp.zeros((out_dim,))
    params = [(W1, b1), (W2, b2)]
    got = mlp_forward(params, x)
    # independent recompute: relu on hidden, linear on output (no activation)
    h = jnp.maximum(jnp.dot(x, W1) + b1, 0)
    expected = jnp.dot(h, W2) + b2
    assert got.shape == (N, out_dim)
    assert jnp.allclose(got, expected, atol=1e-5)
    # last layer must be linear, not relu: outputs should contain negatives here
    assert jnp.any(got < 0)

def test_0013_log_softmax_logits(ns):
    f = ns["log_softmax_logits"]
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (16, 3)) * 5.0
    out = f(logits)
    # Independent oracle: jax.nn.log_softmax
    expected = jax.nn.log_softmax(logits, axis=-1)
    assert jnp.allclose(out, expected, atol=1e-5)
    # log-softmax rows: exp sums to 1
    assert jnp.allclose(jnp.sum(jnp.exp(out), axis=-1), jnp.ones(16), atol=1e-5)
    # numerical stability: large logits should not be nan/inf
    big = jnp.array([[1000.0, 1001.0, 999.0]])
    assert jnp.all(jnp.isfinite(f(big)))


def test_0014_cross_entropy_loss(ns):
    f = ns["cross_entropy_loss"]
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    N, C = 16, 3
    logits = jax.random.normal(k1, (N, C)) * 2.0
    labels = jax.random.randint(k2, (N,), 0, C)
    one_hot = jax.nn.one_hot(labels, C)
    out = f(logits, one_hot)
    # Independent oracle: manual cross-entropy via jax.nn.log_softmax + gather
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    picked = log_probs[jnp.arange(N), labels]
    expected = jnp.mean(-picked)
    assert jnp.allclose(out, expected, atol=1e-5)
    assert jnp.ndim(out) == 0
    # Perfect prediction => near-zero loss
    perfect = one_hot * 50.0
    assert f(perfect, one_hot) < 1e-3


def test_0015_classification_accuracy(ns):
    f = ns["classification_accuracy"]
    key = jax.random.PRNGKey(0)
    N, C = 16, 3
    labels = jax.random.randint(key, (N,), 0, C)
    one_hot = jax.nn.one_hot(labels, C)
    # Logits that strongly favor the true label => accuracy 1.0
    perfect_logits = one_hot * 10.0
    assert jnp.allclose(f(perfect_logits, labels), 1.0, atol=1e-5)
    # Logits favoring (label+1) mod C => accuracy 0.0
    wrong = jax.nn.one_hot((labels + 1) % C, C) * 10.0
    assert jnp.allclose(f(wrong, labels), 0.0, atol=1e-5)
    # Independent oracle on arbitrary logits
    logits = jax.random.normal(jax.random.PRNGKey(1), (N, C))
    expected = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    assert jnp.allclose(f(logits, labels), expected, atol=1e-5)

def test_0016_loss_fn_of_params(ns):
    loss_fn_of_params = ns["loss_fn_of_params"]
    mlp_forward = ns["mlp_forward"]
    init_mlp_params = ns["init_mlp_params"]
    sample_input_features = ns["sample_input_features"]
    assign_class_labels = ns["assign_class_labels"]
    one_hot_encode_labels = ns["one_hot_encode_labels"]
    make_prng_key = ns["make_prng_key"]

    in_dim, hidden, out_dim, N = 4, [8], 3, 16
    key = make_prng_key(0)
    k1, k2 = jax.random.split(key, 2)
    X = sample_input_features(k1, N, in_dim)
    labels = assign_class_labels(X, out_dim)
    Y = one_hot_encode_labels(labels, out_dim)
    params = init_mlp_params(k2, [in_dim] + hidden + [out_dim])

    loss = loss_fn_of_params(params, X, Y)

    # Independent oracle: manual log-softmax cross-entropy on the logits.
    logits = mlp_forward(params, X)
    m = jnp.max(logits, axis=-1, keepdims=True)
    logsumexp = jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True)) + m
    log_probs = logits - logsumexp
    expected = jnp.mean(-jnp.sum(Y * log_probs, axis=-1))

    assert jnp.ndim(loss) == 0
    assert jnp.allclose(loss, expected, atol=1e-5)


def test_0017_compute_param_grads(ns):
    compute_param_grads = ns["compute_param_grads"]
    loss_fn_of_params = ns["loss_fn_of_params"]
    init_mlp_params = ns["init_mlp_params"]
    sample_input_features = ns["sample_input_features"]
    assign_class_labels = ns["assign_class_labels"]
    one_hot_encode_labels = ns["one_hot_encode_labels"]
    make_prng_key = ns["make_prng_key"]

    in_dim, hidden, out_dim, N = 4, [8], 3, 16
    key = make_prng_key(0)
    k1, k2 = jax.random.split(key, 2)
    X = sample_input_features(k1, N, in_dim)
    labels = assign_class_labels(X, out_dim)
    Y = one_hot_encode_labels(labels, out_dim)
    params = init_mlp_params(k2, [in_dim] + hidden + [out_dim])

    grads = compute_param_grads(params, X, Y)

    # Structure matches params.
    assert len(grads) == len(params)
    for (gW, gb), (W, b) in zip(grads, params):
        assert gW.shape == W.shape
        assert gb.shape == b.shape

    # Independent finite-difference check on a few coordinates.
    eps = 1e-3
    layer = 0
    for (i, j) in [(0, 0), (1, 2), (3, 1)]:
        plus = [list(t) for t in params]
        minus = [list(t) for t in params]
        plus[layer][0] = params[layer][0].at[i, j].add(eps)
        minus[layer][0] = params[layer][0].at[i, j].add(-eps)
        plus = [tuple(t) for t in plus]
        minus = [tuple(t) for t in minus]
        fd = (loss_fn_of_params(plus, X, Y) - loss_fn_of_params(minus, X, Y)) / (2 * eps)
        assert jnp.allclose(grads[layer][0][i, j], fd, atol=1e-4)

    # Finite difference on a bias coordinate of the last layer.
    last = len(params) - 1
    for j in [0, 2]:
        plus = [list(t) for t in params]
        minus = [list(t) for t in params]
        plus[last][1] = params[last][1].at[j].add(eps)
        minus[last][1] = params[last][1].at[j].add(-eps)
        plus = [tuple(t) for t in plus]
        minus = [tuple(t) for t in minus]
        fd = (loss_fn_of_params(plus, X, Y) - loss_fn_of_params(minus, X, Y)) / (2 * eps)
        assert jnp.allclose(grads[last][1][j], fd, atol=1e-4)


def test_0018_sgd_update_params(ns):
    sgd_update_params = ns["sgd_update_params"]
    compute_param_grads = ns["compute_param_grads"]
    loss_fn_of_params = ns["loss_fn_of_params"]
    init_mlp_params = ns["init_mlp_params"]
    sample_input_features = ns["sample_input_features"]
    assign_class_labels = ns["assign_class_labels"]
    one_hot_encode_labels = ns["one_hot_encode_labels"]
    make_prng_key = ns["make_prng_key"]

    in_dim, hidden, out_dim, N = 4, [8], 3, 16
    key = make_prng_key(0)
    k1, k2 = jax.random.split(key, 2)
    X = sample_input_features(k1, N, in_dim)
    labels = assign_class_labels(X, out_dim)
    Y = one_hot_encode_labels(labels, out_dim)
    params = init_mlp_params(k2, [in_dim] + hidden + [out_dim])
    grads = compute_param_grads(params, X, Y)

    lr = 0.1
    new_params = sgd_update_params(params, grads, lr)

    # Independent oracle: every leaf updated by p - lr*g.
    assert len(new_params) == len(params)
    for (nW, nb), (W, b), (gW, gb) in zip(new_params, params, grads):
        assert jnp.allclose(nW, W - lr * gW, atol=1e-5)
        assert jnp.allclose(nb, b - lr * gb, atol=1e-5)

    # A single SGD step should reduce the loss.
    loss_before = loss_fn_of_params(params, X, Y)
    loss_after = loss_fn_of_params(new_params, X, Y)
    assert float(loss_after) < float(loss_before)

def test_0019_training_step(ns):
    training_step = ns["training_step"]
    mlp_forward = ns["mlp_forward"]
    init_mlp_params = ns["init_mlp_params"]
    sample_input_features = ns["sample_input_features"]
    assign_class_labels = ns["assign_class_labels"]
    one_hot_encode_labels = ns["one_hot_encode_labels"]

    key = jax.random.PRNGKey(0)
    params = init_mlp_params(key, [4, 8, 3])
    X = sample_input_features(jax.random.PRNGKey(1), 16, 4)
    labels = assign_class_labels(X, 3)
    targets = one_hot_encode_labels(labels, 3)
    lr = 0.1

    # Independent oracle for the loss returned (pre-update loss)
    def manual_loss(p):
        logits = mlp_forward(p, X)
        m = jnp.max(logits, axis=-1, keepdims=True)
        logsm = logits - m - jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True))
        return jnp.mean(-(targets * logsm).sum(axis=-1))

    expected_loss = manual_loss(params)

    # Independent grads via finite differences, then manual SGD update
    eps = 1e-3
    new_params, loss = training_step(params, X, targets, lr)
    assert jnp.allclose(loss, expected_loss, atol=1e-5)

    # Check one updated entry via finite-difference grad: W of layer 0, element [0,0]
    W0 = params[0][0]
    base = manual_loss(params)
    Wp = W0.at[0, 0].add(eps)
    pp = [(Wp, params[0][1])] + params[1:]
    Wm = W0.at[0, 0].add(-eps)
    pm = [(Wm, params[0][1])] + params[1:]
    fd_grad = (manual_loss(pp) - manual_loss(pm)) / (2 * eps)
    expected_W00 = W0[0, 0] - lr * fd_grad
    assert jnp.allclose(new_params[0][0][0, 0], expected_W00, atol=1e-4)

    # Structure preserved
    assert len(new_params) == len(params)
    for (Wn, bn), (W, b) in zip(new_params, params):
        assert Wn.shape == W.shape and bn.shape == b.shape


def test_0020_train_mlp(ns):
    train_mlp = ns["train_mlp"]
    mlp_forward = ns["mlp_forward"]
    init_mlp_params = ns["init_mlp_params"]
    sample_input_features = ns["sample_input_features"]
    assign_class_labels = ns["assign_class_labels"]
    one_hot_encode_labels = ns["one_hot_encode_labels"]

    key = jax.random.PRNGKey(0)
    params = init_mlp_params(key, [4, 8, 3])
    X = sample_input_features(jax.random.PRNGKey(1), 16, 4)
    labels = assign_class_labels(X, 3)
    targets = one_hot_encode_labels(labels, 3)

    n_steps = 200
    final_params, loss_history = train_mlp(params, X, targets, 0.2, n_steps)

    assert isinstance(loss_history, list)
    assert len(loss_history) == n_steps
    assert all(isinstance(l, float) for l in loss_history)

    # Loss should trend down: final much lower than initial
    assert loss_history[-1] < loss_history[0]

    # Accuracy on separable synthetic data should rise
    init_acc = jnp.mean(jnp.argmax(mlp_forward(params, X), -1) == labels)
    final_acc = jnp.mean(jnp.argmax(mlp_forward(final_params, X), -1) == labels)
    assert final_acc >= init_acc
    assert final_acc > 0.7


def test_0021_predict_classes(ns):
    predict_classes = ns["predict_classes"]
    mlp_forward = ns["mlp_forward"]
    init_mlp_params = ns["init_mlp_params"]
    sample_input_features = ns["sample_input_features"]

    key = jax.random.PRNGKey(0)
    params = init_mlp_params(key, [4, 8, 3])
    X = sample_input_features(jax.random.PRNGKey(1), 16, 4)

    preds = predict_classes(params, X)
    logits = mlp_forward(params, X)
    expected = jnp.argmax(logits, axis=-1)

    assert preds.shape == (16,)
    assert jnp.array_equal(preds, expected)
    assert jnp.all(preds >= 0) and jnp.all(preds < 3)
