"""Hidden reference implementations for einops.

One top-level def per step, named exactly as in spec.STEPS, each with a one-line
docstring (it becomes the stub's prompt). Every solution is a single einops call
-- the whole point is to express the operation declaratively rather than with a
chain of transpose/reshape/sum.
"""
import numpy as np  # noqa: F401
from einops import rearrange, reduce, repeat, einsum, pack, unpack  # noqa: F401


# -- Part 1: Rearrange ------------------------------------------------------

def transpose_2d(x):
    """Transpose a 2-D array: swap its two axes."""
    return rearrange(x, "h w -> w h")


def flatten_image(x):
    """Flatten a 2-D array into a 1-D vector in row-major (C) order."""
    return rearrange(x, "h w -> (h w)")


def merge_batch_and_time(x):
    """Collapse the leading (batch, time) axes of a 3-D array into one, giving (batch*time, features)."""
    return rearrange(x, "b t f -> (b t) f")


def split_into_heads(x, heads):
    """Reshape (batch, seq, dim) into (batch, heads, seq, head_dim) by splitting the last axis into `heads` equal groups."""
    return rearrange(x, "b s (h d) -> b h s d", h=heads)


def channels_last_to_first(x):
    """Convert an image batch from NHWC to NCHW layout (move the channel axis from last to second)."""
    return rearrange(x, "b h w c -> b c h w")


# -- Part 2: Reduce ---------------------------------------------------------

def global_average_pool(x):
    """Average each channel of an NCHW image batch over its spatial dims, giving (N, C)."""
    return reduce(x, "b c h w -> b c", "mean")


def max_pool_2x2(x):
    """2x2 max-pool an NCHW image batch (H and W are even), halving each spatial dim."""
    return reduce(x, "b c (h ph) (w pw) -> b c h w", "max", ph=2, pw=2)


def sequence_mean(x):
    """Mean-pool a (batch, seq, dim) array over the sequence axis, giving (batch, dim)."""
    return reduce(x, "b s d -> b d", "mean")


# -- Part 3: Repeat ---------------------------------------------------------

def gray_to_rgb(x):
    """Turn a 2-D grayscale image (H, W) into an (H, W, 3) RGB image by copying it across 3 channels."""
    return repeat(x, "h w -> h w c", c=3)


def tile_rows(x, n):
    """Stack `n` identical copies of a 1-D vector into an (n, len) array."""
    return repeat(x, "w -> n w", n=n)


def upsample_nearest(x, factor):
    """Nearest-neighbour upsample a 2-D array by an integer `factor` along both axes."""
    return repeat(x, "h w -> (h ph) (w pw)", ph=factor, pw=factor)


# -- Part 4: Einsum ---------------------------------------------------------

def batched_matmul(a, b):
    """Batched matrix multiply: (B, I, K) times (B, K, J) gives (B, I, J)."""
    return einsum(a, b, "b i k, b k j -> b i j")


def attention_scores(q, k):
    """Dot-product attention scores: queries (B, Q, D) and keys (B, K, D) give (B, Q, K)."""
    return einsum(q, k, "b q d, b k d -> b q k")


def weighted_token_sum(weights, values):
    """Weighted sum of value vectors: weights (B, Q, K) and values (B, K, D) give (B, Q, D)."""
    return einsum(weights, values, "b q k, b k d -> b q d")


def trace_batch(x):
    """Trace (sum of the diagonal) of each matrix in a batch (B, N, N), giving (B,)."""
    return einsum(x, "b i i -> b")


# -- Part 5: Pack & unpack --------------------------------------------------

def flatten_keep_batch(x):
    """Flatten every axis except the first into one, giving (B, M) from a batch of any rank >= 1."""
    packed, _ = pack([x], "b *")
    return packed


def concat_features(a, b):
    """Concatenate two feature maps (H, W, C1) and (H, W, C2) along the channel axis, giving (H, W, C1+C2)."""
    packed, _ = pack([a, b], "h w *")
    return packed
