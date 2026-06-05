"""Hidden tests for rnn-lstm-from-scratch. One test_<id>_<name>(ns) per step; deterministic, independent oracles."""
import math

import torch
import torch.nn.functional as F  # noqa: F401

import torch
import torch.nn.functional as F
import math


def test_0001_build_char_vocab(ns):
    build_char_vocab = ns["build_char_vocab"]
    text = "banana"
    stoi, itos = build_char_vocab(text)
    expected_chars = sorted(set(text))  # ['a', 'b', 'n']
    assert len(stoi) == len(expected_chars)
    assert len(itos) == len(expected_chars)
    # ids are 0..V-1 assigned in sorted order
    for i, ch in enumerate(expected_chars):
        assert stoi[ch] == i
        assert itos[i] == ch
    # round-trip consistency
    for ch, i in stoi.items():
        assert itos[i] == ch
    # sorted invariant: ids strictly increase with sorted chars
    ordered = [itos[i] for i in range(len(itos))]
    assert ordered == sorted(ordered)


def test_0002_encode_text(ns):
    build_char_vocab = ns["build_char_vocab"]
    encode_text = ns["encode_text"]
    text = "abcabc"
    stoi, itos = build_char_vocab(text)
    enc = encode_text(text, stoi)
    assert isinstance(enc, torch.Tensor)
    assert enc.dtype == torch.long
    assert enc.shape == (len(text),)
    # independent recompute
    expected = [stoi[c] for c in text]
    assert enc.tolist() == expected
    # decodes back to original text
    decoded = "".join(itos[int(i)] for i in enc)
    assert decoded == text


def test_0003_make_training_pairs(ns):
    make_training_pairs = ns["make_training_pairs"]
    seq_len = 4
    # length 11 ids -> with 1 shift, (11-1)//4 = 2 windows, remainder dropped
    ids = torch.arange(11, dtype=torch.long)
    X, Y = make_training_pairs(ids, seq_len)
    n = (11 - 1) // seq_len
    assert X.shape == (n, seq_len)
    assert Y.shape == (n, seq_len)
    assert X.dtype == torch.long and Y.dtype == torch.long
    # Y is X shifted left by one (next-char targets)
    for r in range(n):
        for c in range(seq_len):
            flat_idx = r * seq_len + c
            assert int(X[r, c]) == int(ids[flat_idx])
            assert int(Y[r, c]) == int(ids[flat_idx + 1])
    # exact match to a manual reshape
    usable = n * seq_len
    assert torch.equal(X, ids[:usable].reshape(n, seq_len))
    assert torch.equal(Y, ids[1:usable + 1].reshape(n, seq_len))


def test_0004_one_hot(ns):
    one_hot = ns["one_hot"]
    V = 5
    ids = torch.tensor([[0, 2, 4, 1]], dtype=torch.long)  # (1,4)
    oh = one_hot(ids, V)
    assert oh.shape == (1, 4, V)
    assert oh.dtype == torch.float32
    # exactly one 1 per row along last axis, rest zeros
    assert torch.allclose(oh.sum(dim=-1), torch.ones(1, 4))
    # the hot index matches the id (independent recompute)
    expected = torch.zeros(1, 4, V)
    for b in range(1):
        for s in range(4):
            expected[b, s, int(ids[b, s])] = 1.0
    assert torch.equal(oh, expected)
    # works on a flat 1-D tensor too
    flat = torch.tensor([3, 0], dtype=torch.long)
    ohf = one_hot(flat, V)
    assert ohf.shape == (2, V)
    assert int(ohf[0].argmax()) == 3 and int(ohf[1].argmax()) == 0

def test_0005_init_rnn_params(ns):
    V, H = 5, 8
    p = ns["init_rnn_params"](V, H, seed=0)
    assert set(p.keys()) == {"Wxh", "Whh", "bh", "Why", "by"}
    assert p["Wxh"].shape == (V, H)
    assert p["Whh"].shape == (H, H)
    assert p["bh"].shape == (H,)
    assert p["Why"].shape == (H, V)
    assert p["by"].shape == (V,)
    for k, t in p.items():
        assert t.requires_grad, k
        assert t.dtype == torch.float32, k
        assert torch.isfinite(t).all(), k
    # determinism by seed
    p2 = ns["init_rnn_params"](V, H, seed=0)
    for k in p:
        assert torch.equal(p[k].detach(), p2[k].detach()), k
    # different seed -> different weights
    p3 = ns["init_rnn_params"](V, H, seed=1)
    assert not torch.equal(p["Wxh"].detach(), p3["Wxh"].detach())


def test_0006_rnn_cell_step(ns):
    V, H, B = 5, 8, 2
    p = ns["init_rnn_params"](V, H, seed=0)
    x_t = torch.zeros(B, V)
    x_t[0, 2] = 1.0
    x_t[1, 4] = 1.0
    torch.manual_seed(0)
    h_prev = torch.randn(B, H)
    h_t = ns["rnn_cell_step"](x_t, h_prev, p)
    expected = torch.tanh(
        x_t @ p["Wxh"].detach() + h_prev @ p["Whh"].detach() + p["bh"].detach()
    )
    assert h_t.shape == (B, H)
    assert torch.allclose(h_t, expected, atol=1e-6)
    # tanh range invariant
    assert (h_t.abs() <= 1.0 + 1e-6).all()


def test_0007_rnn_forward(ns):
    V, H, B, S = 5, 8, 2, 4
    p = ns["init_rnn_params"](V, H, seed=0)
    torch.manual_seed(0)
    ids = torch.randint(0, V, (B, S))
    X = torch.zeros(B, S, V)
    for b in range(B):
        for s in range(S):
            X[b, s, ids[b, s]] = 1.0
    H_seq, h_last = ns["rnn_forward"](X, None, p)
    assert H_seq.shape == (B, S, H)
    assert h_last.shape == (B, H)
    # manual unroll oracle from zero state
    h = torch.zeros(B, H)
    cols = []
    for s in range(S):
        h = torch.tanh(
            X[:, s, :] @ p["Wxh"].detach() + h @ p["Whh"].detach() + p["bh"].detach()
        )
        cols.append(h)
    exp = torch.stack(cols, dim=1)
    assert torch.allclose(H_seq, exp, atol=1e-6)
    assert torch.allclose(h_last, cols[-1], atol=1e-6)
    # explicit nonzero h0 is respected
    h0 = torch.randn(B, H)
    _, hl2 = ns["rnn_forward"](X, h0, p)
    h = h0
    for s in range(S):
        h = torch.tanh(
            X[:, s, :] @ p["Wxh"].detach() + h @ p["Whh"].detach() + p["bh"].detach()
        )
    assert torch.allclose(hl2, h, atol=1e-6)


def test_0008_output_logits(ns):
    V, H, B, S = 5, 8, 2, 4
    p = ns["init_rnn_params"](V, H, seed=0)
    torch.manual_seed(0)
    H_seq = torch.randn(B, S, H)
    logits = ns["output_logits"](H_seq, p)
    assert logits.shape == (B, S, V)
    exp = H_seq @ p["Why"].detach() + p["by"].detach()
    assert torch.allclose(logits, exp, atol=1e-6)


def test_0009_rnn_logits(ns):
    V, H, B, S = 5, 8, 2, 4
    p = ns["init_rnn_params"](V, H, seed=0)
    torch.manual_seed(0)
    ids = torch.randint(0, V, (B, S))
    X = torch.zeros(B, S, V)
    for b in range(B):
        for s in range(S):
            X[b, s, ids[b, s]] = 1.0
    logits = ns["rnn_logits"](p, X)
    assert logits.shape == (B, S, V)
    # equivalent to zero-state forward then output_logits
    H_seq, _ = ns["rnn_forward"](X, None, p)
    exp = ns["output_logits"](H_seq, p)
    assert torch.allclose(logits, exp, atol=1e-6)
    # gradients flow through the uniform (params, X) signature
    loss = logits.sum()
    loss.backward()
    for k in ("Wxh", "Whh", "bh", "Why", "by"):
        assert p[k].grad is not None, k
        assert torch.isfinite(p[k].grad).all(), k

def test_0010_init_lstm_params(ns):
    init_lstm_params = ns["init_lstm_params"]
    V, H = 5, 8
    p = init_lstm_params(V, H, seed=0)
    expected = {"Wx": (V, 4 * H), "Wh": (H, 4 * H), "b": (4 * H,), "Why": (H, V), "by": (V,)}
    assert set(p.keys()) == set(expected.keys())
    for k, shape in expected.items():
        assert tuple(p[k].shape) == shape, (k, tuple(p[k].shape), shape)
        assert p[k].dtype == torch.float32
        assert p[k].requires_grad, k
    # biases start at zero
    assert torch.count_nonzero(p["b"]) == 0
    assert torch.count_nonzero(p["by"]) == 0
    # small random weights
    assert p["Wx"].detach().abs().max() < 0.2
    # determinism by seed
    p2 = init_lstm_params(V, H, seed=0)
    assert torch.allclose(p["Wx"].detach(), p2["Wx"].detach())
    p3 = init_lstm_params(V, H, seed=1)
    assert not torch.allclose(p["Wx"].detach(), p3["Wx"].detach())


def test_0011_lstm_gates(ns):
    init_lstm_params = ns["init_lstm_params"]
    lstm_gates = ns["lstm_gates"]
    torch.manual_seed(0)
    V, H, B = 5, 8, 2
    p = init_lstm_params(V, H, seed=0)
    x_t = torch.zeros(B, V)
    x_t[torch.arange(B), torch.randint(0, V, (B,))] = 1.0
    h_prev = torch.randn(B, H)
    i, f, o, g = lstm_gates(x_t, h_prev, p)
    for t in (i, f, o, g):
        assert tuple(t.shape) == (B, H)
    # ranges
    for gate in (i, f, o):
        assert (gate > 0).all() and (gate < 1).all()
    assert (g > -1).all() and (g < 1).all()
    # independent oracle: z then chunk in [i,f,o,g] order
    z = x_t @ p["Wx"] + h_prev @ p["Wh"] + p["b"]
    zi, zf, zo, zg = z[:, :H], z[:, H:2 * H], z[:, 2 * H:3 * H], z[:, 3 * H:]
    assert torch.allclose(i, torch.sigmoid(zi), atol=1e-6)
    assert torch.allclose(f, torch.sigmoid(zf), atol=1e-6)
    assert torch.allclose(o, torch.sigmoid(zo), atol=1e-6)
    assert torch.allclose(g, torch.tanh(zg), atol=1e-6)


def test_0012_lstm_cell_step(ns):
    init_lstm_params = ns["init_lstm_params"]
    lstm_gates = ns["lstm_gates"]
    lstm_cell_step = ns["lstm_cell_step"]
    torch.manual_seed(0)
    V, H, B = 5, 8, 2
    p = init_lstm_params(V, H, seed=0)
    x_t = torch.zeros(B, V)
    x_t[torch.arange(B), torch.tensor([1, 3])] = 1.0
    h_prev = torch.randn(B, H)
    c_prev = torch.randn(B, H)
    h_t, c_t = lstm_cell_step(x_t, h_prev, c_prev, p)
    assert tuple(h_t.shape) == (B, H)
    assert tuple(c_t.shape) == (B, H)
    # oracle using the gates directly
    i, f, o, g = lstm_gates(x_t, h_prev, p)
    c_exp = f * c_prev + i * g
    h_exp = o * torch.tanh(c_exp)
    assert torch.allclose(c_t, c_exp, atol=1e-6)
    assert torch.allclose(h_t, h_exp, atol=1e-6)


def test_0013_lstm_forward(ns):
    init_lstm_params = ns["init_lstm_params"]
    lstm_cell_step = ns["lstm_cell_step"]
    lstm_forward = ns["lstm_forward"]
    torch.manual_seed(0)
    V, H, B, S = 5, 8, 2, 4
    p = init_lstm_params(V, H, seed=0)
    ids = torch.randint(0, V, (B, S))
    X = torch.zeros(B, S, V)
    for b in range(B):
        for s in range(S):
            X[b, s, ids[b, s]] = 1.0
    H_seq, h_last, c_last = lstm_forward(X, None, None, p)
    assert tuple(H_seq.shape) == (B, S, H)
    assert tuple(h_last.shape) == (B, H)
    assert tuple(c_last.shape) == (B, H)
    # independent oracle: manual unroll from zeros
    h = torch.zeros(B, H)
    c = torch.zeros(B, H)
    outs = []
    for t in range(S):
        h, c = lstm_cell_step(X[:, t, :], h, c, p)
        outs.append(h)
    H_ref = torch.stack(outs, dim=1)
    assert torch.allclose(H_seq, H_ref, atol=1e-6)
    assert torch.allclose(h_last, h, atol=1e-6)
    assert torch.allclose(c_last, c, atol=1e-6)
    # final hidden equals last slice of sequence
    assert torch.allclose(H_seq[:, -1, :], h_last, atol=1e-6)


def test_0014_lstm_logits(ns):
    init_lstm_params = ns["init_lstm_params"]
    lstm_forward = ns["lstm_forward"]
    lstm_logits = ns["lstm_logits"]
    torch.manual_seed(0)
    V, H, B, S = 5, 8, 2, 4
    p = init_lstm_params(V, H, seed=0)
    ids = torch.randint(0, V, (B, S))
    X = torch.zeros(B, S, V)
    for b in range(B):
        for s in range(S):
            X[b, s, ids[b, s]] = 1.0
    logits = lstm_logits(p, X)
    assert tuple(logits.shape) == (B, S, V)
    # independent oracle: forward from zeros then linear projection by hand
    H_seq, _, _ = lstm_forward(X, None, None, p)
    expected = H_seq @ p["Why"] + p["by"]
    assert torch.allclose(logits, expected, atol=1e-6)

def test_0015_sequence_cross_entropy(ns):
    import torch
    import torch.nn.functional as F
    import math
    torch.manual_seed(0)
    sequence_cross_entropy = ns["sequence_cross_entropy"]
    B, S, V = 2, 4, 5
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    got = sequence_cross_entropy(logits, targets)
    # Independent oracle: manual softmax + nll averaged over all B*S positions.
    total = 0.0
    for b in range(B):
        for s in range(S):
            row = logits[b, s]
            logZ = torch.logsumexp(row, dim=0)
            nll = (logZ - row[targets[b, s]]).item()
            total += nll
    expected = total / (B * S)
    assert got.shape == torch.Size([])
    assert math.isfinite(got.item())
    assert abs(got.item() - expected) < 1e-5
    # Also matches F.cross_entropy on the flattened tensors.
    ref = F.cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
    assert abs(got.item() - ref.item()) < 1e-6


def test_0016_compute_sequence_loss(ns):
    import torch
    import torch.nn.functional as F
    import math
    torch.manual_seed(0)
    compute_sequence_loss = ns["compute_sequence_loss"]
    sequence_cross_entropy = ns["sequence_cross_entropy"]
    B, S, V = 2, 4, 5
    targets = torch.randint(0, V, (B, S))
    fixed_logits = torch.randn(B, S, V)

    def forward_fn(params, X_onehot):
        return fixed_logits

    params = {"dummy": torch.zeros(1)}
    X = torch.zeros(B, S, V)
    got = compute_sequence_loss(params, X, targets, forward_fn)
    # Oracle: it must equal cross-entropy of the logits forward_fn returns.
    expected = sequence_cross_entropy(fixed_logits, targets)
    assert abs(got.item() - expected.item()) < 1e-6
    assert abs(got.item() - F.cross_entropy(
        fixed_logits.reshape(B * S, V), targets.reshape(B * S)).item()) < 1e-6
    assert math.isfinite(got.item())


def test_0017_train_step(ns):
    import torch
    import math
    torch.manual_seed(0)
    train_step = ns["train_step"]
    compute_sequence_loss = ns["compute_sequence_loss"]
    B, S, V, H = 2, 4, 5, 8
    # Self-contained linear model: logits = (X @ W) so grads/SGD are checkable by hand.
    W = torch.randn(V, V, requires_grad=True)
    params = {"W": W}
    targets = torch.randint(0, V, (B, S))
    X = torch.zeros(B, S, V)
    ids = torch.randint(0, V, (B, S))
    for b in range(B):
        for s in range(S):
            X[b, s, ids[b, s]] = 1.0

    def forward_fn(params, X_onehot):
        return X_onehot @ params["W"]

    lr = 0.5
    W_before = W.detach().clone()
    # Independent oracle: recompute loss + grad on a clone, predict the SGD update.
    W_clone = W_before.clone().requires_grad_(True)
    loss_oracle = compute_sequence_loss({"W": W_clone}, X, targets, forward_fn)
    loss_oracle.backward()
    expected_W_after = (W_before - lr * W_clone.grad).clone()
    expected_loss = loss_oracle.item()

    loss = train_step(params, X, targets, lr, forward_fn)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert abs(loss - expected_loss) < 1e-5
    assert torch.allclose(params["W"].detach(), expected_W_after, atol=1e-6)
    # Param actually moved.
    assert not torch.allclose(params["W"].detach(), W_before)


def test_0018_train_model(ns):
    import torch
    import math
    torch.manual_seed(0)
    train_model = ns["train_model"]
    B, S, V = 2, 4, 5
    W = torch.randn(V, V, requires_grad=True)
    params = {"W": W}
    targets = torch.randint(0, V, (B, S))
    X = torch.zeros(B, S, V)
    ids = torch.randint(0, V, (B, S))
    for b in range(B):
        for s in range(S):
            X[b, s, ids[b, s]] = 1.0

    def forward_fn(params, X_onehot):
        return X_onehot @ params["W"]

    n_steps = 30
    losses = train_model(params, X, targets, 0.5, n_steps, forward_fn)
    assert isinstance(losses, list)
    assert len(losses) == n_steps
    assert all(math.isfinite(l) for l in losses)
    # Loss trends down: last is meaningfully below the first.
    assert losses[-1] < losses[0]
    assert losses[-1] < losses[0] - 1e-3

def test_0019_sample_next_char(ns):
    sample_next_char = ns["sample_next_char"]
    # Determinism: same generator state -> same draw.
    logits = torch.tensor([0.1, 2.0, -1.0, 0.5, 0.3])
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = sample_next_char(logits, temperature=1.0, generator=g1)
    b = sample_next_char(logits, temperature=1.0, generator=g2)
    assert a == b
    assert isinstance(a, int)
    assert 0 <= a < logits.numel()

    # Empirical frequencies must match softmax probs (oracle via Monte Carlo).
    probs = torch.softmax(logits, dim=-1)
    g = torch.Generator().manual_seed(0)
    counts = torch.zeros(5)
    N = 20000
    for _ in range(N):
        counts[sample_next_char(logits, temperature=1.0, generator=g)] += 1.0
    emp = counts / N
    assert torch.allclose(emp, probs, atol=0.03), (emp, probs)

    # Very low temperature concentrates on the argmax.
    g = torch.Generator().manual_seed(7)
    hard = [sample_next_char(logits, temperature=0.01, generator=g) for _ in range(50)]
    assert set(hard) == {int(torch.argmax(logits))}


def test_0020_generate_text(ns):
    generate_text = ns["generate_text"]
    rnn_logits = ns["rnn_logits"]
    lstm_logits = ns["lstm_logits"]
    init_rnn_params = ns["init_rnn_params"]
    init_lstm_params = ns["init_lstm_params"]
    build_char_vocab = ns["build_char_vocab"]

    torch.manual_seed(0)
    text = "abcde" * 4
    stoi, itos = build_char_vocab(text)
    V = len(stoi)
    H = 8
    prompt = "ab"
    length = 6

    for init, fwd in ((init_rnn_params, rnn_logits), (init_lstm_params, lstm_logits)):
        params = init(V, H, seed=0)

        # Shape / type: prompt preserved, exactly `length` new chars appended.
        g = torch.Generator().manual_seed(42)
        out = generate_text(params, stoi, itos, prompt, length, fwd,
                            temperature=1.0, generator=g)
        assert isinstance(out, str)
        assert out.startswith(prompt)
        assert len(out) == len(prompt) + length
        # Every char must be a valid vocab char.
        for ch in out:
            assert ch in stoi

        # Determinism with a fixed generator.
        g1 = torch.Generator().manual_seed(99)
        g2 = torch.Generator().manual_seed(99)
        o1 = generate_text(params, stoi, itos, prompt, length, fwd,
                          temperature=1.0, generator=g1)
        o2 = generate_text(params, stoi, itos, prompt, length, fwd,
                          temperature=1.0, generator=g2)
        assert o1 == o2

        # Different seeds generally differ (sanity, not strict).
        g3 = torch.Generator().manual_seed(1)
        g4 = torch.Generator().manual_seed(2)
        a = generate_text(params, stoi, itos, prompt, 20, fwd, temperature=1.0, generator=g3)
        b = generate_text(params, stoi, itos, prompt, 20, fwd, temperature=1.0, generator=g4)
        assert a != b


def test_0021_sequence_perplexity(ns):
    sequence_perplexity = ns["sequence_perplexity"]
    rnn_logits = ns["rnn_logits"]
    lstm_logits = ns["lstm_logits"]
    init_rnn_params = ns["init_rnn_params"]
    init_lstm_params = ns["init_lstm_params"]
    one_hot = ns["one_hot"]

    torch.manual_seed(0)
    V, H, B, S = 5, 8, 2, 4
    ids = torch.randint(0, V, (B, S))
    X = one_hot(ids, V).float()
    targets = torch.randint(0, V, (B, S))

    for init, fwd in ((init_rnn_params, rnn_logits), (init_lstm_params, lstm_logits)):
        params = init(V, H, seed=0)
        ppl = sequence_perplexity(params, X, targets, fwd)
        assert isinstance(ppl, float)
        assert math.isfinite(ppl)
        assert ppl > 0.0

        # Independent oracle: ppl == exp(F.cross_entropy(flattened logits, flattened targets)).
        with torch.no_grad():
            logits = fwd(params, X)
            ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
            expected = float(torch.exp(ce))
        assert abs(ppl - expected) < 1e-4, (ppl, expected)

        # Perplexity for V classes lies in (1, V] for finite logits on random data here,
        # and >= 1 always.
        assert ppl >= 1.0 - 1e-6
