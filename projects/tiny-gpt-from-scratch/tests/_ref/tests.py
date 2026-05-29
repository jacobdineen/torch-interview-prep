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
