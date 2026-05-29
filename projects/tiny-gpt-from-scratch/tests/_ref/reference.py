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


# ================= Part 4 — Single-Layer Neural Bigram =================

def initialize_w_random(vocab_size, rng):
    """Random (vocab, vocab) weight matrix — the learned bigram 'table'."""
    return rng.standard_normal((vocab_size, vocab_size))


def scale_w_small(w, scale):
    """Scale the weights down so initial logits are small."""
    return w * scale


def one_hot_encode_batch(x, vocab_size):
    """One-hot encode a batch of token ids (B,) into (B, vocab_size)."""
    return np.eye(vocab_size)[x]


def forward_logits_onehot(onehot, w):
    """Logits as a matmul of one-hot inputs with the weights: (B,V) @ (V,V)."""
    return onehot @ w


def observe_lookup_equivalence(x, w):
    """One-hot @ W equals a plain row lookup W[x]; return True to confirm."""
    vocab_size = w.shape[0]
    return bool(np.allclose(np.eye(vocab_size)[x] @ w, w[x]))


def forward_logits_lookup(x, w):
    """Logits via row lookup W[x] — same result as one-hot @ W, far cheaper."""
    return w[x]


def logits_to_probs_rowwise(logits):
    """Row-wise softmax turning logits into next-token probabilities."""
    return stable_softmax_2d_rowwise(logits)


def gather_correct_token_probs(probs, y):
    """Pick out the probability assigned to each true next token y. Shape (B,)."""
    return probs[np.arange(len(y)), y]


def cross_entropy_loss(probs, y):
    """Mean negative log-prob of the correct tokens."""
    return -np.mean(np.log(probs[np.arange(len(y)), y]))


def derive_dlogits_on_paper(probs, y):
    """Gradient of softmax+cross-entropy w.r.t. the logits: (probs - onehot)/N."""
    n = len(y)
    onehot = np.eye(probs.shape[1])[y]
    return (probs - onehot) / n


def compute_dlogits(probs, y):
    """Implement dL/dlogits = (probs - onehot(y)) / N."""
    n = len(y)
    onehot = np.eye(probs.shape[1])[y]
    return (probs - onehot) / n


def derive_dw_on_paper(x, dlogits, vocab_size):
    """Gradient w.r.t. W. Since logits = W[x], dW = onehot(x).T @ dlogits."""
    return np.eye(vocab_size)[x].T @ dlogits


def compute_dw_scatter_add(x, dlogits, vocab_size):
    """Same dW, computed by scattering each row's dlogits into row W[x[b]]."""
    dw = np.zeros((vocab_size, dlogits.shape[1]))
    np.add.at(dw, x, dlogits)
    return dw


def sgd_update_w(w, dw, lr):
    """One SGD step: W <- W - lr * dW."""
    return w - lr * dw


def run_one_training_step(w, x, y, lr):
    """Forward, loss, backward, and SGD update for one batch. Returns (W, loss)."""
    logits = forward_logits_lookup(x, w)
    probs = logits_to_probs_rowwise(logits)
    loss = cross_entropy_loss(probs, y)
    dlogits = compute_dlogits(probs, y)
    dw = compute_dw_scatter_add(x, dlogits, w.shape[0])
    w = sgd_update_w(w, dw, lr)
    return w, loss


def train_neural_bigram_loop(w, data, n_steps, batch_size, lr, rng):
    """Train the neural bigram for ``n_steps`` SGD steps. Returns (W, losses)."""
    losses = []
    for _ in range(n_steps):
        idx = rng.integers(0, len(data) - 1, size=batch_size)
        x, y = data[idx], data[idx + 1]
        w, loss = run_one_training_step(w, x, y, lr)
        losses.append(loss)
    return w, losses


def sample_from_neural_bigram(w, start_token, n, rng):
    """Generate ``n`` tokens by softmaxing each row W[cur] and sampling."""
    out = [start_token]
    cur = start_token
    for _ in range(n):
        cur = sample_next_token(stable_softmax_1d(w[cur]), rng)
        out.append(cur)
    return out


# ================= Part 5 — Layer Primitives and Backprop =================

def linear_forward(x, w):
    """Linear layer without bias: y = x @ W."""
    return x @ w


def derive_dx_on_paper(dout, w):
    """For y = x @ W, the input gradient is dL/dx = dout @ W.T."""
    return dout @ w.T


def derive_linear_dw_on_paper(x, dout):
    """For y = x @ W, the weight gradient is dL/dW = x.T @ dout."""
    return x.T @ dout


def linear_backward_dx(dout, w):
    """Input gradient of a linear layer: dout @ W.T."""
    return dout @ w.T


def linear_backward_dw(x, dout):
    """Weight gradient of a linear layer: x.T @ dout."""
    return x.T @ dout


def bias_add_forward(x, b):
    """Add a per-feature bias, broadcasting over the batch: y = x + b."""
    return x + b


def bias_add_backward_db(dout):
    """Bias gradient is dout summed over the batch (axis 0)."""
    return dout.sum(axis=0)


def relu_forward(x):
    """ReLU: max(x, 0)."""
    return np.maximum(x, 0.0)


def relu_backward(dout, x):
    """ReLU gradient: pass dout through only where the input was positive."""
    return dout * (x > 0)


def softmax_cross_entropy_backward(probs, y):
    """dL/dlogits for mean softmax+cross-entropy over a batch: (probs - onehot)/N."""
    n = len(y)
    onehot = np.eye(probs.shape[1])[y]
    return (probs - onehot) / n


def layernorm_forward_mean(x):
    """Per-row mean over the feature axis, kept 2-D for broadcasting."""
    return x.mean(axis=-1, keepdims=True)


def layernorm_forward_variance(x):
    """Per-row (population) variance over the feature axis, kept for broadcasting."""
    return x.var(axis=-1, keepdims=True)


def layernorm_forward_normalize(x, eps):
    """Center and scale to unit variance: (x - mean) / sqrt(var + eps)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def layernorm_forward_affine(xhat, gamma, beta):
    """Learnable scale and shift: gamma * xhat + beta."""
    return gamma * xhat + beta


def layernorm_backward_subtract_mean(g):
    """Backward through centering c = x - mean(x): dx = g - mean(g) over features."""
    return g - g.mean(axis=-1, keepdims=True)


def layernorm_backward_divide_std(g, std):
    """Backward through dividing by a (constant) std: dc = g / std."""
    return g / std


def layernorm_backward_full(dout, x, gamma, eps):
    """Full input gradient dx of LayerNorm(x) * gamma, over the feature axis."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x - mu) / std
    dxhat = dout * gamma
    return (dxhat - dxhat.mean(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)) / std


def layernorm_backward_implementation(dout, x, gamma, eps):
    """Complete LayerNorm backward: returns (dx, dgamma, dbeta)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x - mu) / std
    axes = tuple(range(dout.ndim - 1))  # sum over all but the feature axis
    dgamma = (dout * xhat).sum(axis=axes)
    dbeta = dout.sum(axis=axes)
    dxhat = dout * gamma
    dx = (dxhat - dxhat.mean(axis=-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)) / std
    return dx, dgamma, dbeta


# ================= Part 6 — Embeddings and Self-Attention =================

def create_token_embedding(vocab_size, d_model):
    """Token embedding table (vocab, d_model), small random init."""
    return np.random.randn(vocab_size, d_model) * 0.02


def token_embedding_forward(tok_emb, x):
    """Look up embeddings for token ids x (B,T) -> (B, T, d_model)."""
    return tok_emb[x]


def token_embedding_backward(dout, x, vocab_size, d_model):
    """Gradient w.r.t. the table: scatter-add dout into the rows that were used."""
    d_emb = np.zeros((vocab_size, d_model))
    np.add.at(d_emb, x, dout)
    return d_emb


def create_positional_embedding(block_size, d_model):
    """Positional embedding table (block_size, d_model), small random init."""
    return np.random.randn(block_size, d_model) * 0.02


def slice_positional_embedding(pos_emb, t):
    """Take the first ``t`` positions: (t, d_model)."""
    return pos_emb[:t]


def add_token_and_positional_embeddings(tok, pos):
    """Sum token (B,T,d) and positional (T,d) embeddings (pos broadcasts over batch)."""
    return tok + pos


def embedding_sum_backward(dout):
    """Backward of tok+pos: (dtok, dpos). dtok = dout; dpos sums over the batch."""
    return dout, dout.sum(axis=0)


def create_qkv_projections(d_model):
    """Single-head Q/K/V projection matrices, each (d_model, d_model)."""
    return {"Wq": np.random.randn(d_model, d_model) * 0.02,
            "Wk": np.random.randn(d_model, d_model) * 0.02,
            "Wv": np.random.randn(d_model, d_model) * 0.02}


def compute_query(x, wq):
    """Query projection: Q = x @ Wq."""
    return x @ wq


def compute_key(x, wk):
    """Key projection: K = x @ Wk."""
    return x @ wk


def compute_value(x, wv):
    """Value projection: V = x @ Wv."""
    return x @ wv


def compute_attention_scores(q, k):
    """Raw attention scores Q @ K.T -> (T, T)."""
    return q @ k.T


def scale_attention_scores(scores, d_head):
    """Scale scores by 1/sqrt(d_head) to stabilize the softmax."""
    return scores / np.sqrt(d_head)


def build_causal_mask(t):
    """Boolean (T,T) mask, True where a position must be hidden (strictly future)."""
    return np.triu(np.ones((t, t), dtype=bool), k=1)


def apply_causal_mask(scores, mask):
    """Set masked (future) score entries to a large negative number."""
    out = scores.copy()
    out[mask] = -1e9
    return out


def softmax_attention_weights(scores):
    """Row-wise softmax over the key axis -> attention weights."""
    return stable_softmax_2d_rowwise(scores)


def attention_weighted_values(weights, v):
    """Weighted sum of values: weights @ V."""
    return weights @ v


def apply_output_projection(attn_out, wo):
    """Project the attention output back to d_model: attn_out @ Wo."""
    return attn_out @ wo


def output_projection_backward(dout, attn_out, wo):
    """Backward of attn_out @ Wo: (d_attn_out, dWo)."""
    return dout @ wo.T, attn_out.T @ dout


def attention_value_backward(dattn_out, weights, v):
    """Backward of attn_out = weights @ V: (dweights, dV)."""
    return dattn_out @ v.T, weights.T @ dattn_out


def masked_softmax_backward(dweights, weights):
    """Backward through the row-wise softmax: dscores from dweights and weights."""
    return weights * (dweights - (dweights * weights).sum(axis=-1, keepdims=True))


def scale_scores_backward(dscaled, d_head):
    """Backward of dividing scores by sqrt(d_head)."""
    return dscaled / np.sqrt(d_head)


def qk_scores_backward(dscores, q, k):
    """Backward of scores = Q @ K.T: (dQ, dK)."""
    return dscores @ k, dscores.T @ q


def qkv_projection_backward(dq, dk, dv, x, wq, wk, wv):
    """Backward of the Q/K/V projections: (dx, dWq, dWk, dWv)."""
    dx = dq @ wq.T + dk @ wk.T + dv @ wv.T
    return dx, x.T @ dq, x.T @ dk, x.T @ dv


def choose_attention_head_config(d_model, n_heads):
    """Per-head dimension d_head = d_model // n_heads (must divide evenly)."""
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    return d_model // n_heads


def create_multihead_qkv_projections(d_model):
    """Combined Q/K/V projections for all heads, each (d_model, d_model)."""
    return {"Wq": np.random.randn(d_model, d_model) * 0.02,
            "Wk": np.random.randn(d_model, d_model) * 0.02,
            "Wv": np.random.randn(d_model, d_model) * 0.02}


def create_multihead_output_projection(d_model):
    """Output projection Wo, (d_model, d_model)."""
    return np.random.randn(d_model, d_model) * 0.02


def reshape_to_heads(x, n_heads):
    """Split the feature axis into heads: (B,T,d_model) -> (B,T,n_heads,d_head)."""
    b, t, d = x.shape
    return x.reshape(b, t, n_heads, d // n_heads)


def transpose_heads_to_front(x):
    """(B,T,H,d_head) -> (B,H,T,d_head) so each head is an independent (T,d_head) block."""
    return x.transpose(0, 2, 1, 3)


def get_multihead_n_heads(x):
    """Number of heads from a (B,H,T,d_head) tensor."""
    return x.shape[1]


def get_multihead_sequence_length(x):
    """Sequence length T from a (B,H,T,d_head) tensor."""
    return x.shape[2]


def compute_d_head(x):
    """Per-head dimension d_head from a (B,H,T,d_head) tensor."""
    return x.shape[3]


def multihead_masked_softmax_scores(q, k, mask):
    """Per-head causal attention weights: softmax(QK^T/sqrt(d_head) + mask)."""
    d_head = q.shape[-1]
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_head)
    scores = np.where(mask, -1e9, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=-1, keepdims=True)


def multihead_weighted_sum(weights, v):
    """Per-head weighted sum of values: weights @ V -> (B,H,T,d_head)."""
    return weights @ v


def transpose_heads_to_back(x):
    """(B,H,T,d_head) -> (B,T,H,d_head), ready to merge heads."""
    return x.transpose(0, 2, 1, 3)


def get_multihead_output_sequence_length(x):
    """Sequence length T from a (B,T,H,d_head) tensor."""
    return x.shape[1]


def merge_heads_to_d_model(x):
    """Concatenate heads back into d_model: (B,T,H,d_head) -> (B,T,d_model)."""
    b, t, h, dh = x.shape
    return x.reshape(b, t, h * dh)


def multihead_output_projection_forward(x, wo):
    """Final output projection: (B,T,d_model) @ Wo."""
    return x @ wo


def multihead_reshape_transpose_backward(dheads, n_heads):
    """Inverse of reshape_to_heads + transpose_heads_to_front: take a per-head
    gradient (B,H,T,d_head) back to (B,T,d_model)."""
    b, h, t, dh = dheads.shape
    return dheads.transpose(0, 2, 1, 3).reshape(b, t, h * dh)
