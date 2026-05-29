"""Reference implementations for tiny-gpt-from-scratch (HIDDEN).

Every step's test grades the user's function against these by swapping the
user's implementation in over the reference namespace, so each step is checked
in isolation with correct versions of all other functions available.

Functions are grouped by part. Each carries a one-line docstring used to derive
the editable stub and the --explain text.
"""
import numpy as np


# ============================== Part 1 — Tokenizer ==============================

def build_vocab(text):
    """Return the sorted list of unique characters in ``text``."""
    return sorted(set(text))


def build_stoi(vocab):
    """Map each character to its index: {char: i} for the vocab list."""
    return {ch: i for i, ch in enumerate(vocab)}


def build_itos(vocab):
    """Inverse map index -> character: {i: char} for the vocab list."""
    return {i: ch for i, ch in enumerate(vocab)}


def encode_char(ch, stoi):
    """Encode a single character to its integer id."""
    return stoi[ch]


def encode_string(s, stoi):
    """Encode a string to a list of integer ids."""
    return [stoi[ch] for ch in s]


def decode_int(i, itos):
    """Decode a single integer id back to its character."""
    return itos[i]


def decode_ids(ids, itos):
    """Decode a sequence of integer ids back into a string."""
    return "".join(itos[int(i)] for i in ids)


# =================== Part 2 — NumPy and Softmax Foundations ===================

def make_1d_array(values):
    """Make a 1-D float array from a Python list of numbers."""
    return np.array(values, dtype=float)


def get_array_shape(a):
    """Return the shape tuple of an array."""
    return a.shape


def get_array_dtype(a):
    """Return the dtype of an array."""
    return a.dtype


def make_2d_zeros(rows, cols):
    """Return a (rows, cols) array of zeros."""
    return np.zeros((rows, cols))


def make_2d_random(rows, cols, rng):
    """Return a (rows, cols) array of standard-normal samples drawn from ``rng``."""
    return rng.standard_normal((rows, cols))


def index_element(a, i, j):
    """Return the scalar at row i, column j."""
    return a[i, j]


def slice_row(a, i):
    """Return row i of a 2-D array."""
    return a[i]


def slice_column(a, j):
    """Return column j of a 2-D array (as a 1-D array)."""
    return a[:, j]


def slice_subblock(a, r0, r1, c0, c1):
    """Return the sub-block a[r0:r1, c0:c1]."""
    return a[r0:r1, c0:c1]


def elementwise_add(a, b):
    """Element-wise sum of two same-shaped arrays."""
    return a + b


def elementwise_multiply(a, b):
    """Element-wise (Hadamard) product of two same-shaped arrays."""
    return a * b


def scalar_broadcast_add(a, s):
    """Add a scalar to every element (broadcasting)."""
    return a + s


def vector_matrix_broadcast_add(m, v):
    """Add a length-``cols`` vector to each row of matrix ``m`` (broadcasting)."""
    return m + v


def array_exp(a):
    """Element-wise exponential."""
    return np.exp(a)


def array_log(a):
    """Element-wise natural log."""
    return np.log(a)


def sum_all(a):
    """Sum of every element (a scalar)."""
    return a.sum()


def sum_axis0(a):
    """Sum down columns (over axis 0) -> shape (cols,)."""
    return a.sum(axis=0)


def sum_axis1(a):
    """Sum across rows (over axis 1) -> shape (rows,)."""
    return a.sum(axis=1)


def max_along_axis(a, axis):
    """Maximum over the given axis."""
    return a.max(axis=axis)


def matmul(a, b):
    """Matrix product a @ b."""
    return a @ b


def transpose_matrix(a):
    """Transpose of a 2-D array."""
    return a.T


def sum_keepdims(a, axis):
    """Sum over ``axis`` while keeping that dimension (size 1) for broadcasting."""
    return a.sum(axis=axis, keepdims=True)


def naive_softmax_1d(z):
    """Softmax of a 1-D vector, the textbook (overflow-prone) way: exp / sum(exp)."""
    e = np.exp(z)
    return e / e.sum()


def softmax_overflow_demo():
    """Show why naive softmax overflows: run it on large logits and return the
    (nan/inf-polluted) result so the test can confirm the failure mode."""
    z = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(z)
        return e / e.sum()


def stable_softmax_1d(z):
    """Numerically stable softmax: subtract the max before exponentiating."""
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def stable_softmax_2d_rowwise(z):
    """Row-wise stable softmax of a 2-D array: each row sums to 1."""
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ================= Part 3 — Data Pipeline and Bigram Baseline =================

def read_text_file(path_or_text):
    """Return the corpus text. Here the corpus is passed in directly as a string,
    so this is effectively identity (real datasets would read from ``path``)."""
    return path_or_text


def encode_corpus_to_int_array(text, stoi):
    """Encode the whole corpus into a 1-D int array of token ids."""
    return np.array([stoi[ch] for ch in text], dtype=np.int64)


def pick_split_point(n, frac):
    """Index that splits ``n`` tokens into a ``frac`` train / rest val."""
    return int(n * frac)


def slice_train_and_val(data, split_idx):
    """Split the token array into (train, val) at ``split_idx``."""
    return data[:split_idx], data[split_idx:]


def pick_block_size(n):
    """The context length (block size). Passed through here."""
    return n


def slice_x_at_offset(data, i, block_size):
    """Input window: tokens [i, i+block_size)."""
    return data[i:i + block_size]


def slice_y_at_offset(data, i, block_size):
    """Target window: the inputs shifted by one, tokens [i+1, i+block_size+1)."""
    return data[i + 1:i + block_size + 1]


def sample_random_batch_offsets(n, block_size, batch_size, rng):
    """Sample ``batch_size`` random start offsets so each window fits in ``n``."""
    return rng.integers(0, n - block_size, size=batch_size)


def stack_x_batch(data, offsets, block_size):
    """Stack input windows for each offset into a (batch, block) array."""
    return np.stack([data[i:i + block_size] for i in offsets])


def stack_y_batch(data, offsets, block_size):
    """Stack target windows (shifted by one) into a (batch, block) array."""
    return np.stack([data[i + 1:i + block_size + 1] for i in offsets])


def get_batch(data, block_size, batch_size, rng):
    """Sample a random (X, Y) batch of (batch_size, block_size) token windows."""
    offsets = sample_random_batch_offsets(len(data), block_size, batch_size, rng)
    return stack_x_batch(data, offsets, block_size), stack_y_batch(data, offsets, block_size)


def allocate_count_matrix(vocab_size):
    """A (vocab, vocab) matrix of zeros to tally bigram counts into."""
    return np.zeros((vocab_size, vocab_size))


def loop_fill_counts(data, counts):
    """Tally bigram (current -> next) counts with an explicit Python loop."""
    for i in range(len(data) - 1):
        counts[data[i], data[i + 1]] += 1
    return counts


def vectorize_counts_add_at(data, counts):
    """Same tally, vectorized with np.add.at over all adjacent pairs."""
    np.add.at(counts, (data[:-1], data[1:]), 1)
    return counts


def add_one_smoothing(counts):
    """Laplace smoothing: add 1 to every count so no bigram has zero probability."""
    return counts + 1


def row_sums_of_counts(counts):
    """Per-row totals, kept 2-D (vocab, 1) for broadcasting."""
    return counts.sum(axis=1, keepdims=True)


def normalize_counts_to_probs(counts):
    """Normalize each row of counts into a probability distribution."""
    return counts / counts.sum(axis=1, keepdims=True)


def sample_next_token(probs_row, rng):
    """Sample one token id from a probability row."""
    return int(rng.choice(len(probs_row), p=probs_row))


def generate_sequence(probs, start_token, n, rng):
    """Generate ``n`` tokens from the bigram table, starting at ``start_token``."""
    out = [start_token]
    cur = start_token
    for _ in range(n):
        cur = sample_next_token(probs[cur], rng)
        out.append(cur)
    return out


def decode_generated_sequence(ids, itos):
    """Decode generated ids back to text."""
    return decode_ids(ids, itos)


def log_prob_of_pair(probs, a, b):
    """Log probability the model assigns to the bigram (a -> b)."""
    return np.log(probs[a, b])


def sum_negative_log_probs(probs, data):
    """Total negative log-likelihood of every adjacent pair in ``data``."""
    return -np.sum(np.log(probs[data[:-1], data[1:]]))


def average_nll(probs, data):
    """Mean negative log-likelihood per bigram (the bigram model's loss)."""
    return sum_negative_log_probs(probs, data) / (len(data) - 1)
