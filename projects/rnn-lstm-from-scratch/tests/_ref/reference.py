"""Hidden reference implementations for rnn-lstm-from-scratch. Steps call earlier references
(shared namespace). One def per step with a one-line docstring."""
import math

import torch
import torch.nn.functional as F  # noqa: F401

import torch
import torch.nn.functional as F
import math


def build_char_vocab(text):
    """Build sorted character-to-id and id-to-character mappings from text."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode_text(text, stoi):
    """Map each character in text to its integer id, returning a LongTensor (L,)."""
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def make_training_pairs(ids, seq_len):
    """Split ids into non-overlapping windows X and next-char targets Y, dropping the remainder."""
    n = (ids.shape[0] - 1) // seq_len
    usable = n * seq_len
    X = ids[:usable].reshape(n, seq_len)
    Y = ids[1:usable + 1].reshape(n, seq_len)
    return X, Y


def one_hot(ids, vocab_size):
    """One-hot encode an integer id tensor along a new last axis of size vocab_size.

    Return a float32 tensor (NOT long); the hot index along the new last axis is 1.0.
    """
    return F.one_hot(ids, num_classes=vocab_size).to(torch.float32)

def init_rnn_params(vocab_size, hidden, seed=0):
    """Create the trainable parameter dict for a vanilla RNN.

    Return a dict with keys "Wxh" (vocab_size,hidden), "Whh" (hidden,hidden), "bh" (hidden,), "Why" (hidden,vocab_size), "by" (vocab_size,); weights small random (~0.1*randn) seeded by seed, biases zero, all float32 with requires_grad=True.
    """
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return (0.1 * torch.randn(*shape, generator=g)).requires_grad_(True)
    return {
        "Wxh": rnd(vocab_size, hidden),
        "Whh": rnd(hidden, hidden),
        "bh": torch.zeros(hidden, requires_grad=True),
        "Why": rnd(hidden, vocab_size),
        "by": torch.zeros(vocab_size, requires_grad=True),
    }


def rnn_cell_step(x_t, h_prev, params):
    """Advance the RNN hidden state by one time step.

    Return h_t = tanh(x_t @ params["Wxh"] + h_prev @ params["Whh"] + params["bh"]); shape (B, hidden).
    """
    return torch.tanh(x_t @ params["Wxh"] + h_prev @ params["Whh"] + params["bh"])


def rnn_forward(X_onehot, h0, params):
    """Unroll the RNN cell over the sequence axis starting from h0.

    If h0 is None use a zeros(B,hidden) initial state. Apply rnn_cell_step at each t; return (H_seq, h_last) where H_seq stacks all hidden states along dim=1 with shape (B,S,hidden) and h_last is the final (B,hidden) state.
    """
    B, S, V = X_onehot.shape
    if h0 is None:
        h0 = torch.zeros(B, params["Whh"].shape[0])
    h = h0
    outs = []
    for t in range(S):
        h = rnn_cell_step(X_onehot[:, t, :], h, params)
        outs.append(h)
    H_seq = torch.stack(outs, dim=1)
    return H_seq, h


def output_logits(H_seq, params):
    """Project hidden states to vocabulary logits.

    Return H_seq @ params["Why"] + params["by"], shape (B,S,vocab_size).
    """
    return H_seq @ params["Why"] + params["by"]


def rnn_logits(params, X_onehot):
    """Run the RNN from a zero initial state and produce logits."""
    H_seq, _ = rnn_forward(X_onehot, None, params)
    return output_logits(H_seq, params)

def init_lstm_params(vocab_size, hidden, seed=0):
    """Initialize LSTM weight/bias tensors (4H gate block ordered [i,f,o,g]) as small random requires_grad params.

    Return a dict with keys "Wx" (vocab_size,4*hidden), "Wh" (hidden,4*hidden), "b" (4*hidden,), "Why" (hidden,vocab_size), "by" (vocab_size,); the 4*hidden axis is the concatenated [i,f,o,g] gate block; weights small random (~0.01*randn) seeded by seed, biases zero, all float32 with requires_grad=True.
    """
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.01).requires_grad_(True)

    return {
        "Wx": rand(vocab_size, 4 * hidden),
        "Wh": rand(hidden, 4 * hidden),
        "b": torch.zeros(4 * hidden, requires_grad=True),
        "Why": rand(hidden, vocab_size),
        "by": torch.zeros(vocab_size, requires_grad=True),
    }


def lstm_gates(x_t, h_prev, params):
    """Compute the LSTM input, forget, output, and candidate gates for one time step.

    Compute z = x_t @ params["Wx"] + h_prev @ params["Wh"] + params["b"], split z into four (B,hidden) blocks in order [i,f,o,g], and return (sigmoid(i), sigmoid(f), sigmoid(o), tanh(g)).
    """
    z = x_t @ params["Wx"] + h_prev @ params["Wh"] + params["b"]
    i, f, o, g = z.chunk(4, dim=-1)
    return torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)


def lstm_cell_step(x_t, h_prev, c_prev, params):
    """Advance the LSTM cell and hidden state by one time step using the gated update.

    Using gates (i,f,o,g) from lstm_gates, compute c_t = f*c_prev + i*g and h_t = o*tanh(c_t); return (h_t, c_t), each (B,hidden).
    """
    i, f, o, g = lstm_gates(x_t, h_prev, params)
    c_t = f * c_prev + i * g
    h_t = o * torch.tanh(c_t)
    return h_t, c_t


def lstm_forward(X_onehot, h0, c0, params):
    """Unroll the LSTM cell over the sequence axis, returning all hidden states and the final h and c.

    If h0 or c0 is None use a zeros(B,hidden) initial state for it. Return (H_seq, h_last, c_last) where H_seq stacks all hidden states along dim=1 with shape (B,S,hidden) and h_last,c_last are the final (B,hidden) states.
    """
    B, S, _ = X_onehot.shape
    H = params["Wh"].shape[0]
    h = torch.zeros(B, H) if h0 is None else h0
    c = torch.zeros(B, H) if c0 is None else c0
    outs = []
    for t in range(S):
        h, c = lstm_cell_step(X_onehot[:, t, :], h, c, params)
        outs.append(h)
    H_seq = torch.stack(outs, dim=1)
    return H_seq, h, c


def lstm_logits(params, X_onehot):
    """Run the LSTM from zero initial states and project the hidden sequence to vocabulary logits."""
    H_seq, _, _ = lstm_forward(X_onehot, None, None, params)
    return output_logits(H_seq, params)

def sequence_cross_entropy(logits, targets):
    """Mean cross-entropy of (B,S,V) logits against (B,S) integer targets."""
    B, S, V = logits.shape
    flat_logits = logits.reshape(B * S, V)
    flat_targets = targets.reshape(B * S)
    return F.cross_entropy(flat_logits, flat_targets)


def compute_sequence_loss(params, X_onehot, targets, forward_fn):
    """Run forward_fn to get logits, then return their sequence cross-entropy."""
    logits = forward_fn(params, X_onehot)
    return sequence_cross_entropy(logits, targets)


def train_step(params, X_onehot, targets, lr, forward_fn):
    """One SGD step: zero grads, backprop the sequence loss, update params in place, return the float loss."""
    for p in params.values():
        if p.grad is not None:
            p.grad = None
    loss = compute_sequence_loss(params, X_onehot, targets, forward_fn)
    loss.backward()
    with torch.no_grad():
        for p in params.values():
            p -= lr * p.grad
    return float(loss.detach())


def train_model(params, X_onehot, targets, lr, n_steps, forward_fn):
    """Run n_steps of train_step and return the list of per-step losses."""
    losses = []
    for _ in range(n_steps):
        losses.append(train_step(params, X_onehot, targets, lr, forward_fn))
    return losses

def sample_next_char(logits, temperature=1.0, generator=None):
    """Sample one character id from the temperature-scaled softmax of a (V,) logit vector."""
    probs = torch.softmax(logits / temperature, dim=-1)
    idx = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(idx.item())


def generate_text(params, stoi, itos, prompt, length, forward_fn, temperature=1.0, generator=None):
    """Autoregressively generate `length` new chars after `prompt` by sampling the model's last-step logits."""
    vocab_size = len(stoi)
    ids = [stoi[ch] for ch in prompt]
    out_chars = []
    for _ in range(length):
        seq = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, S)
        X = one_hot(seq, vocab_size).float()                    # (1, S, V)
        logits = forward_fn(params, X)                          # (1, S, V)
        last = logits[0, -1]                                    # (V,)
        next_id = sample_next_char(last, temperature=temperature, generator=generator)
        ids.append(next_id)
        out_chars.append(itos[next_id])
    return prompt + "".join(out_chars)


def sequence_perplexity(params, X_onehot, targets, forward_fn):
    """Return the perplexity (exp of mean cross-entropy) of the model on the given data."""
    with torch.no_grad():
        logits = forward_fn(params, X_onehot)
        ce = sequence_cross_entropy(logits, targets)
    return float(torch.exp(ce))
