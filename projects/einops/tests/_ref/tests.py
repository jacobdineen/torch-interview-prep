"""Hidden tests for einops. One test_<id>_<name>(ns) per step; assert on
ns["<name>"]. Each expected value is computed with plain NumPy (transpose,
reshape, matmul, ...) so the test is an independent oracle, not a re-run of the
einops solution. Deterministic and fast."""
import numpy as np


# ===== Part 1: Rearrange =====

def test_0001_transpose_2d(ns):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(ns["transpose_2d"](x), x.T)


def test_0002_flatten_image(ns):
    x = np.array([[1, 2], [3, 4]])
    assert np.array_equal(ns["flatten_image"](x), np.array([1, 2, 3, 4]))


def test_0003_merge_batch_and_time(ns):
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    out = ns["merge_batch_and_time"](x)
    assert out.shape == (6, 4)
    assert np.array_equal(out, x.reshape(6, 4))


def test_0004_split_into_heads(ns):
    x = np.arange(2 * 5 * 8).reshape(2, 5, 8)
    out = ns["split_into_heads"](x, 4)
    assert out.shape == (2, 4, 5, 2)
    assert np.array_equal(out, x.reshape(2, 5, 4, 2).transpose(0, 2, 1, 3))


def test_0005_channels_last_to_first(ns):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 5, 3))
    out = ns["channels_last_to_first"](x)
    assert out.shape == (2, 3, 4, 5)
    assert np.array_equal(out, np.transpose(x, (0, 3, 1, 2)))


# ===== Part 2: Reduce =====

def test_0006_global_average_pool(ns):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 4, 4))
    out = ns["global_average_pool"](x)
    assert out.shape == (2, 3)
    assert np.allclose(out, x.mean(axis=(2, 3)))


def test_0007_max_pool_2x2(ns):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 4, 4))
    out = ns["max_pool_2x2"](x)
    assert out.shape == (2, 3, 2, 2)
    oracle = x.reshape(2, 3, 2, 2, 2, 2).max(axis=(3, 5))
    assert np.allclose(out, oracle)


def test_0008_sequence_mean(ns):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 5, 3))
    out = ns["sequence_mean"](x)
    assert out.shape == (2, 3)
    assert np.allclose(out, x.mean(axis=1))


# ===== Part 3: Repeat =====

def test_0009_gray_to_rgb(ns):
    x = np.array([[1, 2], [3, 4]])
    out = ns["gray_to_rgb"](x)
    assert out.shape == (2, 2, 3)
    assert np.array_equal(out, np.repeat(x[:, :, None], 3, axis=2))


def test_0010_tile_rows(ns):
    x = np.array([1, 2, 3])
    out = ns["tile_rows"](x, 3)
    assert out.shape == (3, 3)
    assert np.array_equal(out, np.tile(x, (3, 1)))


def test_0011_upsample_nearest(ns):
    x = np.array([[1, 2], [3, 4]])
    out = ns["upsample_nearest"](x, 2)
    oracle = np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)
    assert out.shape == (4, 4)
    assert np.array_equal(out, oracle)


# ===== Part 4: Einsum =====

def test_0012_batched_matmul(ns):
    rng = np.random.default_rng(4)
    a = rng.standard_normal((2, 3, 4)); b = rng.standard_normal((2, 4, 5))
    out = ns["batched_matmul"](a, b)
    assert out.shape == (2, 3, 5)
    assert np.allclose(out, a @ b)


def test_0013_attention_scores(ns):
    rng = np.random.default_rng(5)
    q = rng.standard_normal((2, 3, 6)); k = rng.standard_normal((2, 4, 6))
    out = ns["attention_scores"](q, k)
    assert out.shape == (2, 3, 4)
    assert np.allclose(out, q @ np.transpose(k, (0, 2, 1)))


def test_0014_weighted_token_sum(ns):
    rng = np.random.default_rng(6)
    w = rng.standard_normal((2, 3, 4)); v = rng.standard_normal((2, 4, 5))
    out = ns["weighted_token_sum"](w, v)
    assert out.shape == (2, 3, 5)
    assert np.allclose(out, w @ v)


def test_0015_trace_batch(ns):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 4, 4))
    out = ns["trace_batch"](x)
    assert out.shape == (3,)
    assert np.allclose(out, np.trace(x, axis1=1, axis2=2))


# ===== Part 5: Pack & unpack =====

def test_0016_flatten_keep_batch(ns):
    rng = np.random.default_rng(8)
    x = rng.standard_normal((2, 3, 4))
    out = ns["flatten_keep_batch"](x)
    assert out.shape == (2, 12)
    assert np.allclose(out, x.reshape(2, -1))


def test_0017_concat_features(ns):
    rng = np.random.default_rng(9)
    a = rng.standard_normal((4, 5, 2)); b = rng.standard_normal((4, 5, 3))
    out = ns["concat_features"](a, b)
    assert out.shape == (4, 5, 5)
    assert np.allclose(out, np.concatenate([a, b], axis=2))
