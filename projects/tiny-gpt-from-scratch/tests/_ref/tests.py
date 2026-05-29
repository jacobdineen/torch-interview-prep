"""Hidden step tests for tiny-gpt-from-scratch.

Each ``test_<id>_<name>(ns)`` grades one step. ``ns`` is a namespace holding the
reference implementation of every function with the USER's implementation of the
step's own function(s) swapped in — so a step is tested in isolation.

Assertion messages deliberately contain markers ("shape mismatch", "values
differ", "max abs diff") that the runner's likely-cause heuristic keys on.
"""
import numpy as np
from contextlib import contextmanager


# --------------------------- harness ---------------------------

@contextmanager
def step(label):
    try:
        yield
    except AssertionError as e:
        raise AssertionError(f"step {label!r}: {e}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e


def expect_eq(got, want):
    if got != want:
        raise AssertionError(f"got {got!r} but expected {want!r}")


def expect_true(cond, msg="expected True"):
    if not cond:
        raise AssertionError(msg)


def expect_shape(got, shape):
    got = np.asarray(got)
    if tuple(got.shape) != tuple(shape):
        raise AssertionError(f"shape mismatch: got {tuple(got.shape)} vs want {tuple(shape)}")


def expect_allclose(got, want, atol=1e-6, rtol=1e-5):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    if got.shape != want.shape:
        raise AssertionError(f"shape mismatch: got {got.shape} vs want {want.shape}")
    if np.allclose(got, want, atol=atol, rtol=rtol, equal_nan=True):
        return
    d = np.abs(got - want)
    idx = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.size else ()
    raise AssertionError(
        f"values differ: max abs diff={np.nanmax(d):.3e} at index {idx}; "
        f"got={got[idx]:.6g} want={want[idx]:.6g}")


def grad_check(f, x, dout, analytic_dx, eps=1e-5, atol=1e-4, name="dx"):
    """Finite-difference check: numeric d/dx of (f(x)*dout).sum() vs analytic_dx."""
    x = np.asarray(x, dtype=float)
    analytic_dx = np.asarray(analytic_dx, dtype=float)
    if analytic_dx.shape != x.shape:
        raise AssertionError(f"{name} shape mismatch: got {analytic_dx.shape} vs want {x.shape}")
    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = np.sum(np.asarray(f(x)) * dout)
        x[i] = old - eps
        fm = np.sum(np.asarray(f(x)) * dout)
        x[i] = old
        num[i] = (fp - fm) / (2 * eps)
        it.iternext()
    d = np.abs(num - analytic_dx)
    denom = np.maximum(1.0, np.abs(num) + np.abs(analytic_dx))
    if np.max(d / denom) > atol:
        idx = np.unravel_index(int(np.argmax(d / denom)), d.shape)
        raise AssertionError(
            f"gradient check failed for {name}: max rel diff={np.max(d / denom):.3e} "
            f"at {idx}; analytic={analytic_dx[idx]:.6g} numeric={num[idx]:.6g}")


# --------------------------- Part 1 — Tokenizer ---------------------------

def test_0001_build_vocab(ns):
    build_vocab = ns["build_vocab"]
    with step("sorted unique chars"):
        expect_eq(build_vocab("banana"), ["a", "b", "n"])
    with step("empty text -> empty vocab"):
        expect_eq(build_vocab(""), [])
    with step("whitespace and newline are chars too"):
        expect_eq(build_vocab("b a\n"), ["\n", " ", "a", "b"])


def test_0002_build_stoi(ns):
    build_stoi = ns["build_stoi"]
    with step("char -> index in vocab order"):
        expect_eq(build_stoi(["a", "b", "c"]), {"a": 0, "b": 1, "c": 2})
    with step("single element"):
        expect_eq(build_stoi(["z"]), {"z": 0})


def test_0003_build_itos(ns):
    build_itos = ns["build_itos"]
    with step("index -> char"):
        expect_eq(build_itos(["a", "b", "c"]), {0: "a", 1: "b", 2: "c"})


def test_0004_encode_char(ns):
    encode_char = ns["encode_char"]
    stoi = {"a": 0, "b": 1, "c": 2}
    with step("looks up the id"):
        expect_eq(encode_char("b", stoi), 1)


def test_0005_encode_string(ns):
    encode_string = ns["encode_string"]
    stoi = {"a": 0, "b": 1, "c": 2}
    with step("encodes each char in order"):
        expect_eq(encode_string("cab", stoi), [2, 0, 1])
    with step("empty string -> empty list"):
        expect_eq(encode_string("", stoi), [])


def test_0006_decode_int(ns):
    decode_int = ns["decode_int"]
    itos = {0: "a", 1: "b", 2: "c"}
    with step("looks up the char"):
        expect_eq(decode_int(2, itos), "c")


def test_0007_decode_ids(ns):
    decode_ids = ns["decode_ids"]
    itos = {0: "a", 1: "b", 2: "c"}
    with step("joins chars into a string"):
        expect_eq(decode_ids([2, 0, 1], itos), "cab")
    with step("round-trips with encode_string"):
        expect_eq(decode_ids([0, 1, 2], itos), "abc")


# --------------------- Part 2 — NumPy and Softmax Foundations ---------------------

def test_0008_make_1d_array(ns):
    with step("builds a 1-D array"):
        expect_allclose(ns["make_1d_array"]([1, 2, 3]), [1.0, 2.0, 3.0])


def test_0009_get_array_shape(ns):
    with step("returns the shape tuple"):
        expect_eq(tuple(ns["get_array_shape"](np.zeros((2, 3)))), (2, 3))


def test_0010_get_array_dtype(ns):
    with step("returns the dtype"):
        expect_true(ns["get_array_dtype"](np.zeros(3)) == np.float64, "expected float64")


def test_0011_make_2d_zeros(ns):
    out = ns["make_2d_zeros"](2, 3)
    with step("right shape, all zeros"):
        expect_shape(out, (2, 3))
        expect_allclose(out, np.zeros((2, 3)))


def test_0012_make_2d_random(ns):
    rng = np.random.default_rng(0)
    got = ns["make_2d_random"](2, 3, rng)
    want = np.random.default_rng(0).standard_normal((2, 3))
    with step("draws standard-normal from the rng"):
        expect_shape(got, (2, 3))
        expect_allclose(got, want)


def test_0013_index_element(ns):
    a = np.arange(6).reshape(2, 3)
    with step("indexes [i, j]"):
        expect_eq(ns["index_element"](a, 1, 2), 5)


def test_0014_slice_row(ns):
    a = np.arange(6).reshape(2, 3)
    with step("returns the row"):
        expect_allclose(ns["slice_row"](a, 1), [3, 4, 5])


def test_0015_slice_column(ns):
    a = np.arange(6).reshape(2, 3)
    with step("returns the column"):
        expect_allclose(ns["slice_column"](a, 1), [1, 4])


def test_0016_slice_subblock(ns):
    a = np.arange(12).reshape(3, 4)
    with step("returns the sub-block"):
        expect_allclose(ns["slice_subblock"](a, 0, 2, 1, 3), [[1, 2], [5, 6]])


def test_0017_elementwise_add(ns):
    with step("adds elementwise"):
        expect_allclose(ns["elementwise_add"](np.array([1, 2]), np.array([3, 4])), [4, 6])


def test_0018_elementwise_multiply(ns):
    with step("multiplies elementwise"):
        expect_allclose(ns["elementwise_multiply"](np.array([1, 2]), np.array([3, 4])), [3, 8])


def test_0019_scalar_broadcast_add(ns):
    with step("adds a scalar to all"):
        expect_allclose(ns["scalar_broadcast_add"](np.array([[1, 2], [3, 4]]), 10),
                        [[11, 12], [13, 14]])


def test_0020_vector_matrix_broadcast_add(ns):
    with step("adds a row vector to each row"):
        expect_allclose(ns["vector_matrix_broadcast_add"](np.zeros((2, 3)), np.array([1, 2, 3])),
                        [[1, 2, 3], [1, 2, 3]])


def test_0021_array_exp(ns):
    with step("element-wise exp"):
        expect_allclose(ns["array_exp"](np.array([0.0, 1.0])), [1.0, np.e])


def test_0022_array_log(ns):
    with step("element-wise log"):
        expect_allclose(ns["array_log"](np.array([1.0, np.e])), [0.0, 1.0])


def test_0023_sum_all(ns):
    with step("sums everything"):
        expect_allclose(ns["sum_all"](np.arange(6).reshape(2, 3)), 15.0)


def test_0024_sum_axis0(ns):
    with step("sums over axis 0"):
        expect_allclose(ns["sum_axis0"](np.arange(6).reshape(2, 3)), [3, 5, 7])


def test_0025_sum_axis1(ns):
    with step("sums over axis 1"):
        expect_allclose(ns["sum_axis1"](np.arange(6).reshape(2, 3)), [3, 12])


def test_0026_max_along_axis(ns):
    a = np.array([[1, 9], [7, 3]])
    with step("max over axis 0"):
        expect_allclose(ns["max_along_axis"](a, 0), [7, 9])
    with step("max over axis 1"):
        expect_allclose(ns["max_along_axis"](a, 1), [9, 7])


def test_0027_matmul(ns):
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    with step("matrix product"):
        expect_allclose(ns["matmul"](a, b), a @ b)


def test_0028_transpose_matrix(ns):
    with step("transposes"):
        expect_allclose(ns["transpose_matrix"](np.arange(6).reshape(2, 3)),
                        np.arange(6).reshape(2, 3).T)


def test_0029_sum_keepdims(ns):
    a = np.arange(6).reshape(2, 3)
    out = ns["sum_keepdims"](a, 1)
    with step("keeps the reduced dim"):
        expect_shape(out, (2, 1))
        expect_allclose(out, [[3], [12]])


def test_0030_naive_softmax_1d(ns):
    z = np.array([1.0, 2.0, 3.0])
    out = ns["naive_softmax_1d"](z)
    with step("sums to 1 and matches exp/sum(exp)"):
        expect_allclose(out.sum(), 1.0)
        e = np.exp(z)
        expect_allclose(out, e / e.sum())


def test_0031_softmax_overflow_demo(ns):
    out = np.asarray(ns["softmax_overflow_demo"]())
    with step("naive softmax on huge logits produces nan/inf"):
        expect_true(np.isnan(out).any() or np.isinf(out).any(),
                    "expected nan/inf to demonstrate overflow")


def test_0032_stable_softmax_1d(ns):
    z = np.array([1.0, 2.0, 3.0])
    big = np.array([1000.0, 1001.0, 1002.0])
    with step("matches softmax and is finite on huge logits"):
        e = np.exp(z - z.max())
        expect_allclose(ns["stable_softmax_1d"](z), e / e.sum())
        out = np.asarray(ns["stable_softmax_1d"](big))
        expect_true(np.all(np.isfinite(out)), "must be finite (subtract the max!)")
        expect_allclose(out.sum(), 1.0)


def test_0033_stable_softmax_2d_rowwise(ns):
    z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    out = ns["stable_softmax_2d_rowwise"](z)
    with step("each row sums to 1"):
        expect_allclose(out.sum(axis=1), [1.0, 1.0])
    with step("matches reference rows"):
        ez = np.exp(z - z.max(axis=1, keepdims=True))
        expect_allclose(out, ez / ez.sum(axis=1, keepdims=True))


# ------------------- Part 3 — Data Pipeline and Bigram Baseline -------------------

def test_0034_read_text_file(ns):
    with step("returns the corpus text"):
        expect_eq(ns["read_text_file"]("hello"), "hello")


def test_0035_encode_corpus_to_int_array(ns):
    stoi = {"a": 0, "b": 1, "c": 2}
    out = ns["encode_corpus_to_int_array"]("cab", stoi)
    with step("encodes to an int array"):
        expect_allclose(out, [2, 0, 1])
        expect_true(np.issubdtype(np.asarray(out).dtype, np.integer), "ids must be ints")


def test_0036_pick_split_point(ns):
    with step("90% split of 100"):
        expect_eq(ns["pick_split_point"](100, 0.9), 90)


def test_0037_slice_train_and_val(ns):
    data = np.arange(10)
    tr, va = ns["slice_train_and_val"](data, 7)
    with step("splits at index"):
        expect_allclose(tr, np.arange(7))
        expect_allclose(va, np.arange(7, 10))


def test_0038_pick_block_size(ns):
    with step("passes block size through"):
        expect_eq(ns["pick_block_size"](8), 8)


def test_0039_slice_x_at_offset(ns):
    data = np.arange(10)
    with step("input window"):
        expect_allclose(ns["slice_x_at_offset"](data, 2, 3), [2, 3, 4])


def test_0040_slice_y_at_offset(ns):
    data = np.arange(10)
    with step("target window is shifted by one"):
        expect_allclose(ns["slice_y_at_offset"](data, 2, 3), [3, 4, 5])


def test_0041_sample_random_batch_offsets(ns):
    rng = np.random.default_rng(0)
    got = ns["sample_random_batch_offsets"](100, 8, 4, rng)
    want = np.random.default_rng(0).integers(0, 100 - 8, size=4)
    with step("matches seeded rng draws, all in range"):
        expect_allclose(got, want)
        expect_true(np.all((np.asarray(got) >= 0) & (np.asarray(got) <= 100 - 8)), "offsets out of range")


def test_0042_stack_x_batch(ns):
    data = np.arange(10)
    with step("stacks input windows"):
        expect_allclose(ns["stack_x_batch"](data, [0, 5], 3), [[0, 1, 2], [5, 6, 7]])


def test_0043_stack_y_batch(ns):
    data = np.arange(10)
    with step("stacks shifted target windows"):
        expect_allclose(ns["stack_y_batch"](data, [0, 5], 3), [[1, 2, 3], [6, 7, 8]])


def test_0044_get_batch(ns):
    data = np.arange(50)
    X, Y = ns["get_batch"](data, 4, 3, np.random.default_rng(0))
    with step("shapes are (batch, block)"):
        expect_shape(X, (3, 4))
        expect_shape(Y, (3, 4))
    with step("Y is X shifted by one"):
        expect_allclose(np.asarray(Y), np.asarray(X) + 1)


def test_0045_allocate_count_matrix(ns):
    with step("vocab x vocab zeros"):
        out = ns["allocate_count_matrix"](4)
        expect_shape(out, (4, 4))
        expect_allclose(out, np.zeros((4, 4)))


def test_0046_loop_fill_counts(ns):
    data = np.array([0, 1, 0, 1, 2])
    counts = ns["loop_fill_counts"](data, np.zeros((3, 3)))
    with step("tallies adjacent pairs"):
        want = np.zeros((3, 3))
        want[0, 1] = 2  # (0->1) twice
        want[1, 0] = 1
        want[1, 2] = 1
        expect_allclose(counts, want)


def test_0047_vectorize_counts_add_at(ns):
    data = np.array([0, 1, 0, 1, 2])
    loop = ns["loop_fill_counts"](data, np.zeros((3, 3)))
    vec = ns["vectorize_counts_add_at"](data, np.zeros((3, 3)))
    with step("matches the loop version"):
        expect_allclose(vec, loop)


def test_0048_add_one_smoothing(ns):
    with step("adds one everywhere"):
        expect_allclose(ns["add_one_smoothing"](np.zeros((2, 2))), np.ones((2, 2)))


def test_0049_row_sums_of_counts(ns):
    c = np.array([[1.0, 3.0], [2.0, 2.0]])
    out = ns["row_sums_of_counts"](c)
    with step("keepdims row sums"):
        expect_shape(out, (2, 1))
        expect_allclose(out, [[4.0], [4.0]])


def test_0050_normalize_counts_to_probs(ns):
    c = np.array([[1.0, 3.0], [2.0, 2.0]])
    out = ns["normalize_counts_to_probs"](c)
    with step("each row sums to 1"):
        expect_allclose(out.sum(axis=1), [1.0, 1.0])
        expect_allclose(out, [[0.25, 0.75], [0.5, 0.5]])


def test_0051_sample_next_token(ns):
    probs = np.array([0.0, 0.0, 1.0])  # deterministic
    with step("samples from the distribution"):
        expect_eq(ns["sample_next_token"](probs, np.random.default_rng(0)), 2)


def test_0052_generate_sequence(ns):
    probs = np.array([[0.0, 1.0], [1.0, 0.0]])  # toggles 0<->1 deterministically
    out = ns["generate_sequence"](probs, 0, 3, np.random.default_rng(0))
    with step("starts at start_token, length n+1, follows transitions"):
        expect_eq(list(out), [0, 1, 0, 1])


def test_0053_decode_generated_sequence(ns):
    itos = {0: "a", 1: "b"}
    with step("decodes ids to text"):
        expect_eq(ns["decode_generated_sequence"]([0, 1, 0], itos), "aba")


def test_0054_log_prob_of_pair(ns):
    probs = np.array([[0.25, 0.75], [0.5, 0.5]])
    with step("log of the entry"):
        expect_allclose(ns["log_prob_of_pair"](probs, 0, 1), np.log(0.75))


def test_0055_sum_negative_log_probs(ns):
    probs = np.array([[0.25, 0.75], [0.5, 0.5]])
    data = np.array([0, 1, 0])
    with step("sums NLL over pairs (0->1),(1->0)"):
        expect_allclose(ns["sum_negative_log_probs"](probs, data),
                        -(np.log(0.75) + np.log(0.5)))


def test_0056_average_nll(ns):
    probs = np.array([[0.25, 0.75], [0.5, 0.5]])
    data = np.array([0, 1, 0])
    with step("mean NLL per pair"):
        expect_allclose(ns["average_nll"](probs, data),
                        -(np.log(0.75) + np.log(0.5)) / 2)


# ------------------- Part 4 — Single-Layer Neural Bigram -------------------

def test_0057_initialize_w_random(ns):
    got = ns["initialize_w_random"](4, np.random.default_rng(0))
    want = np.random.default_rng(0).standard_normal((4, 4))
    with step("(V,V) standard-normal from rng"):
        expect_shape(got, (4, 4))
        expect_allclose(got, want)


def test_0058_scale_w_small(ns):
    w = np.ones((2, 2))
    with step("scales the weights"):
        expect_allclose(ns["scale_w_small"](w, 0.01), np.full((2, 2), 0.01))


def test_0059_one_hot_encode_batch(ns):
    out = ns["one_hot_encode_batch"](np.array([0, 2]), 3)
    with step("(B, V) one-hot"):
        expect_allclose(out, [[1, 0, 0], [0, 0, 1]])


def test_0060_forward_logits_onehot(ns):
    w = np.arange(9).reshape(3, 3).astype(float)
    oh = ns["one_hot_encode_batch"](np.array([0, 2]), 3)
    with step("onehot @ W selects rows"):
        expect_allclose(ns["forward_logits_onehot"](oh, w), [w[0], w[2]])


def test_0061_observe_lookup_equivalence(ns):
    w = np.random.default_rng(1).standard_normal((4, 4))
    with step("one-hot matmul equals row lookup"):
        expect_true(ns["observe_lookup_equivalence"](np.array([0, 3, 1]), w) is True
                    or ns["observe_lookup_equivalence"](np.array([0, 3, 1]), w) == True,  # noqa: E712
                    "should confirm equivalence (True)")


def test_0062_forward_logits_lookup(ns):
    w = np.arange(9).reshape(3, 3).astype(float)
    with step("W[x] row lookup"):
        expect_allclose(ns["forward_logits_lookup"](np.array([2, 0]), w), [w[2], w[0]])


def test_0063_logits_to_probs_rowwise(ns):
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = ns["logits_to_probs_rowwise"](logits)
    with step("rows are softmax distributions"):
        expect_allclose(out.sum(axis=1), [1.0, 1.0])
        expect_allclose(out[1], [1 / 3, 1 / 3, 1 / 3])


def test_0064_gather_correct_token_probs(ns):
    probs = np.array([[0.1, 0.9], [0.7, 0.3]])
    with step("gathers p[i, y[i]]"):
        expect_allclose(ns["gather_correct_token_probs"](probs, np.array([1, 0])), [0.9, 0.7])


def test_0065_cross_entropy_loss(ns):
    probs = np.array([[0.1, 0.9], [0.7, 0.3]])
    y = np.array([1, 0])
    with step("mean -log p(correct)"):
        expect_allclose(ns["cross_entropy_loss"](probs, y),
                        -(np.log(0.9) + np.log(0.7)) / 2)


def _ce_of_logits(ns, logits, y):
    return ns["cross_entropy_loss"](ns["logits_to_probs_rowwise"](logits), y)


def test_0066_derive_dlogits_on_paper(ns):
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((4, 5))
    y = rng.integers(0, 5, size=4)
    probs = ns["logits_to_probs_rowwise"](logits)
    analytic = ns["derive_dlogits_on_paper"](probs, y)
    with step("matches numeric dL/dlogits (finite differences)"):
        grad_check(lambda L: _ce_of_logits(ns, L, y), logits, 1.0, analytic, name="dlogits")


def test_0067_compute_dlogits(ns):
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((4, 5))
    y = rng.integers(0, 5, size=4)
    probs = ns["logits_to_probs_rowwise"](logits)
    analytic = ns["compute_dlogits"](probs, y)
    with step("gradient-checks against the loss"):
        grad_check(lambda L: _ce_of_logits(ns, L, y), logits, 1.0, analytic, name="dlogits")


def test_0068_derive_dw_on_paper(ns):
    rng = np.random.default_rng(5)
    x = rng.integers(0, 4, size=6)
    dlogits = rng.standard_normal((6, 4))
    with step("dW = onehot(x).T @ dlogits"):
        expect_allclose(ns["derive_dw_on_paper"](x, dlogits, 4),
                        np.eye(4)[x].T @ dlogits)


def test_0069_compute_dw_scatter_add(ns):
    rng = np.random.default_rng(6)
    x = rng.integers(0, 4, size=6)
    w = rng.standard_normal((4, 4))
    y = rng.integers(0, 4, size=6)

    def loss_of_w(W):
        return _ce_of_logits(ns, ns["forward_logits_lookup"](x, W), y)

    probs = ns["logits_to_probs_rowwise"](ns["forward_logits_lookup"](x, w))
    dlogits = ns["compute_dlogits"](probs, y)
    analytic = ns["compute_dw_scatter_add"](x, dlogits, 4)
    with step("matches scatter form and gradient-checks vs W"):
        expect_allclose(analytic, ns["derive_dw_on_paper"](x, dlogits, 4))
        grad_check(loss_of_w, w, 1.0, analytic, name="dW")


def test_0070_sgd_update_w(ns):
    w = np.ones((2, 2))
    dw = np.full((2, 2), 0.5)
    with step("W - lr*dW"):
        expect_allclose(ns["sgd_update_w"](w, dw, 0.1), np.full((2, 2), 0.95))


def test_0071_run_one_training_step(ns):
    rng = np.random.default_rng(7)
    w = rng.standard_normal((4, 4)) * 0.1
    x = np.array([0, 1, 2])
    y = np.array([1, 2, 3])
    w2, loss = ns["run_one_training_step"](w, x, y, 1.0)
    with step("returns updated W (same shape) and a scalar loss"):
        expect_shape(w2, (4, 4))
        expect_true(np.isscalar(loss) or np.ndim(loss) == 0, "loss should be a scalar")
        expect_true(not np.allclose(w2, w), "W should change after a step")


def test_0072_train_neural_bigram_loop(ns):
    rng = np.random.default_rng(8)
    data = np.array([0, 1, 2, 3] * 50)  # strong bigram structure
    w = rng.standard_normal((4, 4)) * 0.1
    w2, losses = ns["train_neural_bigram_loop"](w, data, 300, 32, 1.0, rng)
    with step("loss decreases over training"):
        expect_true(losses[-1] < losses[0] * 0.7,
                    f"expected loss to drop; start={losses[0]:.3f} end={losses[-1]:.3f}")


def test_0073_sample_from_neural_bigram(ns):
    rng = np.random.default_rng(9)
    w = rng.standard_normal((4, 4))
    out = ns["sample_from_neural_bigram"](w, 0, 5, rng)
    with step("starts at start_token, length n+1, valid ids"):
        expect_eq(out[0], 0)
        expect_eq(len(out), 6)
        expect_true(all(0 <= t < 4 for t in out), "ids must be in range")


# ------------------- Part 5 — Layer Primitives and Backprop -------------------

def test_0074_linear_forward(ns):
    x = np.array([[1.0, 2.0]])
    w = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    with step("y = x @ W"):
        expect_allclose(ns["linear_forward"](x, w), x @ w)


def test_0075_derive_dx_on_paper(ns):
    rng = np.random.default_rng(10)
    x = rng.standard_normal((4, 3))
    w = rng.standard_normal((3, 5))
    dout = rng.standard_normal((4, 5))
    with step("dx = dout @ W.T, gradient-checked"):
        analytic = ns["derive_dx_on_paper"](dout, w)
        grad_check(lambda X: ns["linear_forward"](X, w), x, dout, analytic, name="dx")


def test_0076_derive_linear_dw_on_paper(ns):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 3))
    w = rng.standard_normal((3, 5))
    dout = rng.standard_normal((4, 5))
    with step("dW = x.T @ dout, gradient-checked"):
        analytic = ns["derive_linear_dw_on_paper"](x, dout)
        grad_check(lambda W: ns["linear_forward"](x, W), w, dout, analytic, name="dW")


def test_0077_linear_backward_dx(ns):
    rng = np.random.default_rng(12)
    x = rng.standard_normal((4, 3))
    w = rng.standard_normal((3, 5))
    dout = rng.standard_normal((4, 5))
    with step("gradient check dx"):
        grad_check(lambda X: ns["linear_forward"](X, w), x,
                   dout, ns["linear_backward_dx"](dout, w), name="dx")


def test_0078_linear_backward_dw(ns):
    rng = np.random.default_rng(13)
    x = rng.standard_normal((4, 3))
    w = rng.standard_normal((3, 5))
    dout = rng.standard_normal((4, 5))
    with step("gradient check dW"):
        grad_check(lambda W: ns["linear_forward"](x, W), w,
                   dout, ns["linear_backward_dw"](x, dout), name="dW")


def test_0079_bias_add_forward(ns):
    x = np.zeros((2, 3))
    b = np.array([1.0, 2.0, 3.0])
    with step("broadcasts bias over rows"):
        expect_allclose(ns["bias_add_forward"](x, b), [[1, 2, 3], [1, 2, 3]])


def test_0080_bias_add_backward_db(ns):
    rng = np.random.default_rng(14)
    x = rng.standard_normal((4, 3))
    b = rng.standard_normal((3,))
    dout = rng.standard_normal((4, 3))
    with step("db sums dout over batch, gradient-checked"):
        grad_check(lambda B: ns["bias_add_forward"](x, B), b,
                   dout, ns["bias_add_backward_db"](dout), name="db")


def test_0081_relu_forward(ns):
    with step("max(x,0)"):
        expect_allclose(ns["relu_forward"](np.array([-1.0, 0.0, 2.0])), [0, 0, 2])


def test_0082_relu_backward(ns):
    rng = np.random.default_rng(15)
    x = rng.standard_normal((4, 3))
    x[np.abs(x) < 0.1] = 0.5  # avoid kinks near 0 for the finite-diff check
    dout = rng.standard_normal((4, 3))
    with step("passes grad where x>0, gradient-checked"):
        grad_check(lambda X: ns["relu_forward"](X), x,
                   dout, ns["relu_backward"](dout, x), name="dx")


def test_0083_softmax_cross_entropy_backward(ns):
    rng = np.random.default_rng(16)
    logits = rng.standard_normal((4, 5))
    y = rng.integers(0, 5, size=4)
    probs = ns["stable_softmax_2d_rowwise"](logits)
    analytic = ns["softmax_cross_entropy_backward"](probs, y)

    def loss(L):
        p = ns["stable_softmax_2d_rowwise"](L)
        return -np.mean(np.log(p[np.arange(len(y)), y]))

    with step("dlogits gradient-checked against softmax+CE"):
        grad_check(loss, logits, 1.0, analytic, name="dlogits")


def test_0084_layernorm_forward_mean(ns):
    x = np.array([[1.0, 2.0, 3.0]])
    with step("feature-axis mean, keepdims"):
        out = ns["layernorm_forward_mean"](x)
        expect_shape(out, (1, 1))
        expect_allclose(out, [[2.0]])


def test_0085_layernorm_forward_variance(ns):
    x = np.array([[1.0, 2.0, 3.0]])
    with step("feature-axis (population) variance"):
        expect_allclose(ns["layernorm_forward_variance"](x), [[np.var([1, 2, 3])]])


def test_0086_layernorm_forward_normalize(ns):
    rng = np.random.default_rng(17)
    x = rng.standard_normal((3, 6))
    out = ns["layernorm_forward_normalize"](x, 1e-5)
    with step("normalized rows have ~0 mean, ~1 variance"):
        expect_allclose(out.mean(axis=-1), np.zeros(3), atol=1e-6)
        expect_allclose(out.var(axis=-1), np.ones(3), atol=1e-3)


def test_0087_layernorm_forward_affine(ns):
    xhat = np.ones((2, 3))
    gamma = np.array([2.0, 2.0, 2.0])
    beta = np.array([1.0, 1.0, 1.0])
    with step("gamma*xhat + beta"):
        expect_allclose(ns["layernorm_forward_affine"](xhat, gamma, beta), np.full((2, 3), 3.0))


def test_0088_layernorm_backward_subtract_mean(ns):
    rng = np.random.default_rng(18)
    x = rng.standard_normal((4, 5))
    g = rng.standard_normal((4, 5))
    with step("backward of centering, gradient-checked"):
        analytic = ns["layernorm_backward_subtract_mean"](g)
        grad_check(lambda X: X - X.mean(axis=-1, keepdims=True), x, g, analytic, name="dx")


def test_0089_layernorm_backward_divide_std(ns):
    rng = np.random.default_rng(19)
    c = rng.standard_normal((4, 5))
    std = np.abs(rng.standard_normal((4, 1))) + 0.5
    g = rng.standard_normal((4, 5))
    with step("backward of dividing by std (constant), gradient-checked"):
        analytic = ns["layernorm_backward_divide_std"](g, std)
        grad_check(lambda C: C / std, c, g, analytic, name="dc")


def _ln_forward(ns, x, gamma, beta, eps):
    return ns["layernorm_forward_affine"](ns["layernorm_forward_normalize"](x, eps), gamma, beta)


def test_0090_layernorm_backward_full(ns):
    rng = np.random.default_rng(20)
    x = rng.standard_normal((4, 6))
    gamma = rng.standard_normal((6,))
    beta = rng.standard_normal((6,))
    dout = rng.standard_normal((4, 6))
    eps = 1e-5
    with step("dx gradient-checked against full LayerNorm"):
        analytic = ns["layernorm_backward_full"](dout, x, gamma, eps)
        grad_check(lambda X: _ln_forward(ns, X, gamma, beta, eps), x, dout, analytic, name="dx")


def test_0091_layernorm_backward_implementation(ns):
    rng = np.random.default_rng(21)
    x = rng.standard_normal((4, 6))
    gamma = rng.standard_normal((6,))
    beta = rng.standard_normal((6,))
    dout = rng.standard_normal((4, 6))
    eps = 1e-5
    dx, dgamma, dbeta = ns["layernorm_backward_implementation"](dout, x, gamma, eps)
    with step("dx, dgamma, dbeta all gradient-checked"):
        grad_check(lambda X: _ln_forward(ns, X, gamma, beta, eps), x, dout, dx, name="dx")
        grad_check(lambda G: _ln_forward(ns, x, G, beta, eps), gamma, dout, dgamma, name="dgamma")
        grad_check(lambda B: _ln_forward(ns, x, gamma, B, eps), beta, dout, dbeta, name="dbeta")


# ------------------- Part 6 — Embeddings and Self-Attention -------------------

def test_0092_create_token_embedding(ns):
    np.random.seed(0)
    got = ns["create_token_embedding"](5, 4)
    np.random.seed(0)
    want = np.random.randn(5, 4) * 0.02
    with step("(vocab, d_model) small init"):
        expect_shape(got, (5, 4))
        expect_allclose(got, want)


def test_0093_token_embedding_forward(ns):
    emb = np.arange(12).reshape(4, 3).astype(float)
    x = np.array([[0, 2], [1, 3]])
    with step("(B,T) -> (B,T,d) lookup"):
        out = ns["token_embedding_forward"](emb, x)
        expect_shape(out, (2, 2, 3))
        expect_allclose(out[0, 1], emb[2])


def test_0094_token_embedding_backward(ns):
    rng = np.random.default_rng(30)
    emb = rng.standard_normal((4, 3))
    x = np.array([[0, 2], [2, 1]])
    dout = rng.standard_normal((2, 2, 3))
    analytic = ns["token_embedding_backward"](dout, x, 4, 3)
    with step("scatter-add gradient, gradient-checked"):
        grad_check(lambda E: ns["token_embedding_forward"](E, x), emb, dout, analytic, name="dE")


def test_0095_create_positional_embedding(ns):
    np.random.seed(1)
    got = ns["create_positional_embedding"](6, 4)
    np.random.seed(1)
    want = np.random.randn(6, 4) * 0.02
    with step("(block_size, d_model)"):
        expect_allclose(got, want)


def test_0096_slice_positional_embedding(ns):
    pos = np.arange(20).reshape(5, 4).astype(float)
    with step("first t rows"):
        expect_allclose(ns["slice_positional_embedding"](pos, 3), pos[:3])


def test_0097_add_token_and_positional_embeddings(ns):
    tok = np.ones((2, 3, 4))
    pos = np.full((3, 4), 2.0)
    with step("broadcasts pos over batch"):
        expect_allclose(ns["add_token_and_positional_embeddings"](tok, pos), np.full((2, 3, 4), 3.0))


def test_0098_embedding_sum_backward(ns):
    rng = np.random.default_rng(31)
    tok = rng.standard_normal((2, 3, 4))
    pos = rng.standard_normal((3, 4))
    dout = rng.standard_normal((2, 3, 4))
    dtok, dpos = ns["embedding_sum_backward"](dout)
    with step("dtok and dpos gradient-checked"):
        grad_check(lambda T: ns["add_token_and_positional_embeddings"](T, pos), tok, dout, dtok, name="dtok")
        grad_check(lambda P: ns["add_token_and_positional_embeddings"](tok, P), pos, dout, dpos, name="dpos")


def test_0099_create_qkv_projections(ns):
    np.random.seed(2)
    p = ns["create_qkv_projections"](4)
    with step("Wq/Wk/Wv each (d,d)"):
        for key in ("Wq", "Wk", "Wv"):
            expect_shape(p[key], (4, 4))


def test_0100_compute_query(ns):
    x = np.array([[1.0, 2.0]])
    wq = np.array([[1.0, 0.0], [0.0, 1.0]])
    with step("Q = x @ Wq"):
        expect_allclose(ns["compute_query"](x, wq), x @ wq)


def test_0101_compute_key(ns):
    x = np.array([[1.0, 2.0]])
    wk = np.eye(2)
    with step("K = x @ Wk"):
        expect_allclose(ns["compute_key"](x, wk), x)


def test_0102_compute_value(ns):
    x = np.array([[1.0, 2.0]])
    wv = np.eye(2)
    with step("V = x @ Wv"):
        expect_allclose(ns["compute_value"](x, wv), x)


def test_0103_compute_attention_scores(ns):
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    k = np.array([[1.0, 0.0], [0.0, 1.0]])
    with step("scores = Q @ K.T"):
        expect_allclose(ns["compute_attention_scores"](q, k), q @ k.T)


def test_0104_scale_attention_scores(ns):
    s = np.array([[2.0, 4.0]])
    with step("divide by sqrt(d_head)"):
        expect_allclose(ns["scale_attention_scores"](s, 4), s / 2.0)


def test_0105_build_causal_mask(ns):
    m = np.asarray(ns["build_causal_mask"](3))
    with step("upper triangle (future) is masked"):
        expect_shape(m, (3, 3))
        expect_true(bool(m[0, 1]) and bool(m[0, 2]) and bool(m[1, 2]), "future should be masked")
        expect_true(not m[1, 0] and not m[0, 0], "past/self not masked")


def test_0106_apply_causal_mask(ns):
    scores = np.zeros((3, 3))
    mask = ns["build_causal_mask"](3)
    out = ns["apply_causal_mask"](scores, mask)
    with step("future entries become very negative"):
        expect_true(out[0, 1] < -1e8, "future not masked to -inf-ish")
        expect_allclose(out[1, 0], 0.0)


def test_0107_softmax_attention_weights(ns):
    scores = ns["apply_causal_mask"](np.zeros((3, 3)), ns["build_causal_mask"](3))
    w = ns["softmax_attention_weights"](scores)
    with step("rows sum to 1; causal (no attention to future)"):
        expect_allclose(w.sum(axis=1), np.ones(3))
        expect_true(w[0, 1] < 1e-6 and w[0, 2] < 1e-6, "should not attend to future")


def test_0108_attention_weighted_values(ns):
    w = np.array([[1.0, 0.0], [0.5, 0.5]])
    v = np.array([[2.0, 0.0], [0.0, 4.0]])
    with step("weights @ V"):
        expect_allclose(ns["attention_weighted_values"](w, v), w @ v)


def test_0109_apply_output_projection(ns):
    a = np.array([[1.0, 2.0]])
    wo = np.eye(2) * 3
    with step("attn_out @ Wo"):
        expect_allclose(ns["apply_output_projection"](a, wo), a @ wo)


def test_0110_output_projection_backward(ns):
    rng = np.random.default_rng(32)
    a = rng.standard_normal((3, 4))
    wo = rng.standard_normal((4, 4))
    dout = rng.standard_normal((3, 4))
    da, dwo = ns["output_projection_backward"](dout, a, wo)
    with step("d_attn_out and dWo gradient-checked"):
        grad_check(lambda A: ns["apply_output_projection"](A, wo), a, dout, da, name="d_attn_out")
        grad_check(lambda W: ns["apply_output_projection"](a, W), wo, dout, dwo, name="dWo")


def test_0111_attention_value_backward(ns):
    rng = np.random.default_rng(33)
    w = rng.standard_normal((3, 3))
    v = rng.standard_normal((3, 4))
    dout = rng.standard_normal((3, 4))
    dw, dv = ns["attention_value_backward"](dout, w, v)
    with step("dweights and dV gradient-checked"):
        grad_check(lambda W: ns["attention_weighted_values"](W, v), w, dout, dw, name="dweights")
        grad_check(lambda V: ns["attention_weighted_values"](w, V), v, dout, dv, name="dV")


def test_0112_masked_softmax_backward(ns):
    rng = np.random.default_rng(34)
    scores = rng.standard_normal((3, 3))
    weights = ns["softmax_attention_weights"](scores)
    dweights = rng.standard_normal((3, 3))
    dscores = ns["masked_softmax_backward"](dweights, weights)
    with step("dscores gradient-checked through the softmax"):
        grad_check(lambda S: ns["softmax_attention_weights"](S), scores, dweights, dscores, name="dscores")


def test_0113_scale_scores_backward(ns):
    rng = np.random.default_rng(35)
    scores = rng.standard_normal((3, 3))
    dscaled = rng.standard_normal((3, 3))
    ds = ns["scale_scores_backward"](dscaled, 4)
    with step("gradient-checked"):
        grad_check(lambda S: ns["scale_attention_scores"](S, 4), scores, dscaled, ds, name="dscores")


def test_0114_qk_scores_backward(ns):
    rng = np.random.default_rng(36)
    q = rng.standard_normal((3, 4))
    k = rng.standard_normal((3, 4))
    dscores = rng.standard_normal((3, 3))
    dq, dk = ns["qk_scores_backward"](dscores, q, k)
    with step("dQ and dK gradient-checked"):
        grad_check(lambda Q: ns["compute_attention_scores"](Q, k), q, dscores, dq, name="dQ")
        grad_check(lambda K: ns["compute_attention_scores"](q, K), k, dscores, dk, name="dK")


def test_0115_qkv_projection_backward(ns):
    rng = np.random.default_rng(37)
    x = rng.standard_normal((3, 4))
    wq, wk, wv = (rng.standard_normal((4, 4)) for _ in range(3))
    dq, dk, dv = (rng.standard_normal((3, 4)) for _ in range(3))
    dx, dwq, dwk, dwv = ns["qkv_projection_backward"](dq, dk, dv, x, wq, wk, wv)

    def loss_x(X):
        return np.sum((X @ wq) * dq) + np.sum((X @ wk) * dk) + np.sum((X @ wv) * dv)

    with step("dx and dWq/dWk/dWv gradient-checked"):
        grad_check(loss_x, x, 1.0, dx, name="dx")
        grad_check(lambda W: np.sum((x @ W) * dq), wq, 1.0, dwq, name="dWq")
        grad_check(lambda W: np.sum((x @ W) * dk), wk, 1.0, dwk, name="dWk")
        grad_check(lambda W: np.sum((x @ W) * dv), wv, 1.0, dwv, name="dWv")


def test_0116_choose_attention_head_config(ns):
    with step("d_head = d_model // n_heads"):
        expect_eq(ns["choose_attention_head_config"](8, 2), 4)


def test_0117_create_multihead_qkv_projections(ns):
    np.random.seed(3)
    p = ns["create_multihead_qkv_projections"](8)
    with step("Wq/Wk/Wv each (d,d)"):
        for key in ("Wq", "Wk", "Wv"):
            expect_shape(p[key], (8, 8))


def test_0118_create_multihead_output_projection(ns):
    np.random.seed(4)
    with step("(d,d)"):
        expect_shape(ns["create_multihead_output_projection"](8), (8, 8))


def test_0119_reshape_to_heads(ns):
    x = np.arange(2 * 3 * 8).reshape(2, 3, 8).astype(float)
    out = ns["reshape_to_heads"](x, 2)
    with step("(B,T,d) -> (B,T,H,d_head)"):
        expect_shape(out, (2, 3, 2, 4))
        expect_allclose(out.reshape(2, 3, 8), x)


def test_0120_transpose_heads_to_front(ns):
    x = np.zeros((2, 3, 2, 4))
    with step("(B,T,H,dh) -> (B,H,T,dh)"):
        expect_shape(ns["transpose_heads_to_front"](x), (2, 2, 3, 4))


def test_0121_get_multihead_n_heads(ns):
    with step("heads at axis 1"):
        expect_eq(ns["get_multihead_n_heads"](np.zeros((2, 2, 3, 4))), 2)


def test_0122_get_multihead_sequence_length(ns):
    with step("T at axis 2"):
        expect_eq(ns["get_multihead_sequence_length"](np.zeros((2, 2, 3, 4))), 3)


def test_0123_compute_d_head(ns):
    with step("d_head at axis 3"):
        expect_eq(ns["compute_d_head"](np.zeros((2, 2, 3, 4))), 4)


def test_0124_multihead_masked_softmax_scores(ns):
    rng = np.random.default_rng(38)
    q = rng.standard_normal((2, 2, 3, 4))
    k = rng.standard_normal((2, 2, 3, 4))
    mask = ns["build_causal_mask"](3)
    w = ns["multihead_masked_softmax_scores"](q, k, mask)
    with step("(B,H,T,T), rows sum to 1, causal"):
        expect_shape(w, (2, 2, 3, 3))
        expect_allclose(w.sum(axis=-1), np.ones((2, 2, 3)))
        expect_true(np.all(w[..., 0, 1:] < 1e-6), "must not attend to future")


def test_0125_multihead_weighted_sum(ns):
    w = np.zeros((2, 2, 3, 3))
    w[..., 0] = 1.0  # attend only to position 0
    v = np.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4).astype(float)
    out = ns["multihead_weighted_sum"](w, v)
    with step("(B,H,T,dh); picks value 0 for every query"):
        expect_shape(out, (2, 2, 3, 4))
        expect_allclose(out[0, 0, 1], v[0, 0, 0])


def test_0126_transpose_heads_to_back(ns):
    x = np.zeros((2, 2, 3, 4))
    with step("(B,H,T,dh) -> (B,T,H,dh)"):
        expect_shape(ns["transpose_heads_to_back"](x), (2, 3, 2, 4))


def test_0127_get_multihead_output_sequence_length(ns):
    with step("T at axis 1"):
        expect_eq(ns["get_multihead_output_sequence_length"](np.zeros((2, 3, 2, 4))), 3)


def test_0128_merge_heads_to_d_model(ns):
    x = np.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4).astype(float)
    out = ns["merge_heads_to_d_model"](x)
    with step("(B,T,H,dh) -> (B,T,d_model)"):
        expect_shape(out, (2, 3, 8))
        expect_allclose(out, x.reshape(2, 3, 8))


def test_0129_multihead_output_projection_forward(ns):
    x = np.ones((2, 3, 8))
    wo = np.eye(8) * 2
    with step("(B,T,d) @ Wo"):
        expect_allclose(ns["multihead_output_projection_forward"](x, wo), x * 2)


def test_0130_multihead_reshape_transpose_backward(ns):
    rng = np.random.default_rng(39)
    x = rng.standard_normal((2, 3, 8))
    dheads = rng.standard_normal((2, 2, 3, 4))

    def fwd(X):
        return ns["transpose_heads_to_front"](ns["reshape_to_heads"](X, 2))

    analytic = ns["multihead_reshape_transpose_backward"](dheads, 2)
    with step("inverse of reshape+transpose, gradient-checked"):
        grad_check(fwd, x, dheads, analytic, name="dx")
