"""Hidden tests for attention-is-all-you-need. One test_<id>_<name>(ns) per step;
each is a deterministic, independent torch oracle. Tiny + fast."""
import math  # noqa: F401

import torch
import torch.nn.functional as F  # noqa: F401

def test_0001_build_token_to_id_vocab(ns):
    f = ns["build_token_to_id_vocab"]
    sentences = ["the cat sat", "the dog ran"]
    vocab = f(sentences)
    assert vocab["<pad>"] == 0 and vocab["<bos>"] == 1
    assert vocab["<eos>"] == 2 and vocab["<unk>"] == 3
    expected_words = sorted({"the", "cat", "sat", "dog", "ran"})
    for i, w in enumerate(expected_words, start=4):
        assert vocab[w] == i, (w, vocab[w], i)
    assert len(vocab) == 4 + len(expected_words)
    # ids contiguous 0..N-1, unique
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_0002_build_id_to_token_vocab(ns):
    f = ns["build_id_to_token_vocab"]
    token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "cat": 4, "dog": 5}
    inv = f(token_to_id)
    assert inv == {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>", 4: "cat", 5: "dog"}
    for tok, i in token_to_id.items():
        assert inv[i] == tok


def test_0003_encode_sentence_to_ids(ns):
    f = ns["encode_sentence_to_ids"]
    token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "the": 4, "cat": 5}
    out = f("the cat zzz the", token_to_id)
    expected = torch.tensor([4, 5, 3, 4], dtype=torch.long)
    assert out.dtype == torch.long
    assert out.shape == (4,)
    assert torch.equal(out, expected)


def test_0004_decode_ids_to_tokens(ns):
    f = ns["decode_ids_to_tokens"]
    id_to_token = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>", 4: "the", 5: "cat"}
    ids = torch.tensor([4, 5, 3, 0], dtype=torch.long)
    out = f(ids, id_to_token)
    assert out == ["the", "cat", "<unk>", "<pad>"]


def test_0005_pad_id_sequence(ns):
    f = ns["pad_id_sequence"]
    ids = torch.tensor([4, 5, 6], dtype=torch.long)
    # pad case
    out_pad = f(ids, 5, pad_id=0)
    assert out_pad.dtype == torch.long and out_pad.shape == (5,)
    assert torch.equal(out_pad, torch.tensor([4, 5, 6, 0, 0], dtype=torch.long))
    # truncate case
    out_trunc = f(ids, 2, pad_id=0)
    assert torch.equal(out_trunc, torch.tensor([4, 5], dtype=torch.long))
    # exact length
    out_exact = f(ids, 3, pad_id=0)
    assert torch.equal(out_exact, ids)
    # custom pad id
    out_cp = f(ids, 4, pad_id=9)
    assert torch.equal(out_cp, torch.tensor([4, 5, 6, 9], dtype=torch.long))


def test_0006_stack_padded_sequences_to_batch(ns):
    f = ns["stack_padded_sequences_to_batch"]
    a = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    b = torch.tensor([8, 9], dtype=torch.long)
    out = f([a, b], pad_id=0)
    assert out.dtype == torch.long
    assert out.shape == (2, 4)
    expected = torch.tensor([[4, 5, 6, 7], [8, 9, 0, 0]], dtype=torch.long)
    assert torch.equal(out, expected)

def test_0007_scale_embeddings_by_sqrt_d_model(ns):
    torch.manual_seed(0)
    d_model = 8
    emb = torch.randn(2, 4, d_model)
    expected = emb * math.sqrt(d_model)
    out = ns["scale_embeddings_by_sqrt_d_model"](emb, d_model)
    assert out.shape == emb.shape
    assert torch.allclose(out, expected, atol=1e-6)


def test_0008_compute_positional_div_term(ns):
    d_model = 8
    out = ns["compute_positional_div_term"](d_model)
    assert out.shape == (d_model // 2,)
    k = torch.arange(0, d_model, 2).float()
    expected = torch.exp(-k * (math.log(10000.0) / d_model))
    assert torch.allclose(out, expected, atol=1e-6)
    # spot-check first entry is 1.0 (k=0)
    assert torch.allclose(out[0], torch.tensor(1.0), atol=1e-6)


def test_0009_build_position_index_column(ns):
    max_len = 5
    out = ns["build_position_index_column"](max_len)
    assert out.shape == (max_len, 1)
    assert out.dtype == torch.float32
    expected = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_0010_fill_even_indices_with_sin(ns):
    max_len, d_model = 5, 8
    position = torch.arange(max_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    out = ns["fill_even_indices_with_sin"](pe.clone(), position, div_term)
    assert out.shape == (max_len, d_model)
    expected_even = torch.sin(position * div_term)
    assert torch.allclose(out[:, 0::2], expected_even, atol=1e-6)
    # odd columns untouched (still zero)
    assert torch.allclose(out[:, 1::2], torch.zeros(max_len, d_model // 2), atol=1e-6)


def test_0011_fill_odd_indices_with_cos(ns):
    max_len, d_model = 5, 8
    position = torch.arange(max_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    out = ns["fill_odd_indices_with_cos"](pe.clone(), position, div_term)
    assert out.shape == (max_len, d_model)
    expected_odd = torch.cos(position * div_term)
    assert torch.allclose(out[:, 1::2], expected_odd, atol=1e-6)
    # even columns untouched (still zero)
    assert torch.allclose(out[:, 0::2], torch.zeros(max_len, d_model // 2), atol=1e-6)


def test_0012_build_sinusoidal_positional_encoding(ns):
    max_len, d_model = 5, 8
    out = ns["build_sinusoidal_positional_encoding"](max_len, d_model)
    assert out.shape == (max_len, d_model)
    position = torch.arange(max_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    expected = torch.zeros(max_len, d_model)
    expected[:, 0::2] = torch.sin(position * div_term)
    expected[:, 1::2] = torch.cos(position * div_term)
    assert torch.allclose(out, expected, atol=1e-6)
    # position 0: sin->0, cos->1
    assert torch.allclose(out[0, 0::2], torch.zeros(d_model // 2), atol=1e-6)
    assert torch.allclose(out[0, 1::2], torch.ones(d_model // 2), atol=1e-6)


def test_0013_add_positional_encoding_to_embeddings(ns):
    torch.manual_seed(0)
    max_len, d_model, B, S = 5, 8, 2, 4
    emb = torch.randn(B, S, d_model)
    pe = torch.randn(max_len, d_model)
    out = ns["add_positional_encoding_to_embeddings"](emb, pe)
    assert out.shape == (B, S, d_model)
    expected = emb + pe[:S].unsqueeze(0)
    assert torch.allclose(out, expected, atol=1e-6)

def test_0014_build_padding_mask(ns):
    f = ns["build_padding_mask"]
    ids = torch.tensor([[1, 5, 0, 0], [3, 0, 7, 2]])
    out = f(ids, pad_id=0)
    assert out.shape == (2, 1, 1, 4)
    assert out.dtype == torch.bool
    expected = (ids != 0)[:, None, None, :]
    assert torch.equal(out, expected)


def test_0015_build_causal_mask(ns):
    f = ns["build_causal_mask"]
    out = f(4)
    assert out.shape == (1, 1, 4, 4)
    assert out.dtype == torch.bool
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
    assert torch.equal(out, expected)
    assert bool(out[0, 0, 0, 1]) is False
    assert bool(out[0, 0, 1, 0]) is True


def test_0016_combine_padding_and_causal_masks(ns):
    f = ns["combine_padding_and_causal_masks"]
    pad = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)[:, None, None, :]
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
    out = f(pad, causal)
    expected = pad & causal
    assert torch.equal(out, expected)
    assert out.shape == (1, 1, 4, 4)


def test_0017_compute_raw_attention_scores(ns):
    f = ns["compute_raw_attention_scores"]
    torch.manual_seed(0)
    Q = torch.randn(2, 2, 4, 4)
    K = torch.randn(2, 2, 4, 4)
    out = f(Q, K)
    expected = torch.matmul(Q, K.transpose(-2, -1))
    assert out.shape == (2, 2, 4, 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_0018_scale_attention_scores(ns):
    f = ns["scale_attention_scores"]
    torch.manual_seed(0)
    scores = torch.randn(2, 2, 4, 4)
    out = f(scores, 8)
    expected = scores / math.sqrt(8)
    assert torch.allclose(out, expected, atol=1e-6)


def test_0019_mask_attention_scores_with_neg_inf(ns):
    f = ns["mask_attention_scores_with_neg_inf"]
    torch.manual_seed(0)
    scores = torch.randn(1, 1, 4, 4)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
    out = f(scores, mask)
    expected = scores.masked_fill(~mask, -1e9)
    assert torch.allclose(out, expected, atol=1e-5)
    assert float(out[0, 0, 0, 1]) == -1e9
    assert float(out[0, 0, 1, 0]) == float(scores[0, 0, 1, 0])


def test_0020_softmax_attention_weights(ns):
    f = ns["softmax_attention_weights"]
    torch.manual_seed(0)
    scores = torch.randn(2, 2, 4, 4)
    out = f(scores)
    expected = F.softmax(scores, dim=-1)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2, 2, 4), atol=1e-5)


def test_0021_apply_attention_weights_to_values(ns):
    f = ns["apply_attention_weights_to_values"]
    torch.manual_seed(0)
    weights = F.softmax(torch.randn(2, 2, 4, 4), dim=-1)
    V = torch.randn(2, 2, 4, 4)
    out = f(weights, V)
    expected = torch.matmul(weights, V)
    assert out.shape == (2, 2, 4, 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_0022_scaled_dot_product_attention(ns):
    f = ns["scaled_dot_product_attention"]
    torch.manual_seed(0)
    Q = torch.randn(2, 2, 4, 4)
    K = torch.randn(2, 2, 4, 4)
    V = torch.randn(2, 2, 4, 4)

    # unmasked oracle
    out = f(Q, K, V)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    expected = torch.matmul(F.softmax(scores, dim=-1), V)
    assert out.shape == (2, 2, 4, 4)
    assert torch.allclose(out, expected, atol=1e-5)

    # masked oracle (causal)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
    out_m = f(Q, K, V, mask)
    scores_m = scores.masked_fill(~mask, -1e9)
    expected_m = torch.matmul(F.softmax(scores_m, dim=-1), V)
    assert torch.allclose(out_m, expected_m, atol=1e-5)

def test_0023_split_last_dim_into_heads(ns):
    torch.manual_seed(0)
    B, S, D, H = 2, 4, 8, 2
    x = torch.randn(B, S, D)
    out = ns["split_last_dim_into_heads"](x, H)
    assert out.shape == (B, S, H, D // H)
    # element at head h, feature k should equal original feature h*Dk+k
    Dk = D // H
    for h in range(H):
        for k in range(Dk):
            assert torch.allclose(out[:, :, h, k], x[:, :, h * Dk + k])


def test_0024_transpose_heads_before_sequence(ns):
    torch.manual_seed(0)
    B, S, H, Dk = 2, 4, 2, 4
    x = torch.randn(B, S, H, Dk)
    out = ns["transpose_heads_before_sequence"](x)
    assert out.shape == (B, H, S, Dk)
    assert torch.allclose(out, x.permute(0, 2, 1, 3))


def test_0025_merge_heads_back_to_model_dim(ns):
    torch.manual_seed(0)
    B, H, S, Dk = 2, 2, 4, 4
    x = torch.randn(B, H, S, Dk)
    out = ns["merge_heads_back_to_model_dim"](x)
    assert out.shape == (B, S, H * Dk)
    expected = x.transpose(1, 2).reshape(B, S, H * Dk)
    assert torch.allclose(out, expected)
    # round-trip: split+transpose then merge returns original (B,S,D)
    D = H * Dk
    y = torch.randn(B, S, D)
    yh = y.reshape(B, S, H, Dk).transpose(1, 2)
    assert torch.allclose(ns["merge_heads_back_to_model_dim"](yh), y)


def test_0026_apply_linear_projection(ns):
    torch.manual_seed(0)
    B, S, Din, Dout = 2, 4, 8, 6
    x = torch.randn(B, S, Din)
    W = torch.randn(Din, Dout)
    b = torch.randn(Dout)
    out = ns["apply_linear_projection"](x, W, b)
    assert out.shape == (B, S, Dout)
    assert torch.allclose(out, torch.matmul(x, W) + b, atol=1e-5)


def test_0027_project_to_query_key_value(ns):
    torch.manual_seed(0)
    B, S, D = 2, 4, 8
    x_q = torch.randn(B, S, D)
    x_kv = torch.randn(B, S, D)
    p = {k: torch.randn(D, D) if k.startswith("W") else torch.randn(D)
         for k in ["Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"]}
    Q, K, V = ns["project_to_query_key_value"](x_q, x_kv, p)
    assert Q.shape == (B, S, D) and K.shape == (B, S, D) and V.shape == (B, S, D)
    assert torch.allclose(Q, x_q @ p["Wq"] + p["bq"], atol=1e-5)
    assert torch.allclose(K, x_kv @ p["Wk"] + p["bk"], atol=1e-5)
    assert torch.allclose(V, x_kv @ p["Wv"] + p["bv"], atol=1e-5)


def test_0028_split_qkv_into_heads(ns):
    torch.manual_seed(0)
    B, S, D, H = 2, 4, 8, 2
    Dk = D // H
    Q = torch.randn(B, S, D)
    K = torch.randn(B, S, D)
    V = torch.randn(B, S, D)
    Qh, Kh, Vh = ns["split_qkv_into_heads"](Q, K, V, H)
    for orig, h in [(Q, Qh), (K, Kh), (V, Vh)]:
        assert h.shape == (B, H, S, Dk)
        expected = orig.reshape(B, S, H, Dk).transpose(1, 2)
        assert torch.allclose(h, expected)


def test_0029_multi_head_scaled_dot_product_attention(ns):
    torch.manual_seed(0)
    B, S, D, H = 2, 4, 8, 2
    Dk = D // H
    Q = torch.randn(B, S, D)
    K = torch.randn(B, S, D)
    V = torch.randn(B, S, D)
    out = ns["multi_head_scaled_dot_product_attention"](Q, K, V, None, H)
    assert out.shape == (B, S, D)
    # independent oracle via F.scaled_dot_product_attention per head
    Qh = Q.reshape(B, S, H, Dk).transpose(1, 2)
    Kh = K.reshape(B, S, H, Dk).transpose(1, 2)
    Vh = V.reshape(B, S, H, Dk).transpose(1, 2)
    expected_h = F.scaled_dot_product_attention(Qh, Kh, Vh)
    expected = expected_h.transpose(1, 2).reshape(B, S, D)
    assert torch.allclose(out, expected, atol=1e-5)


def test_0030_merge_heads_and_project_output(ns):
    torch.manual_seed(0)
    B, S, D = 2, 4, 8
    x = torch.randn(B, S, D)
    p = {"Wo": torch.randn(D, D), "bo": torch.randn(D)}
    out = ns["merge_heads_and_project_output"](x, p)
    assert out.shape == (B, S, D)
    assert torch.allclose(out, x @ p["Wo"] + p["bo"], atol=1e-5)


def test_0031_assemble_multi_head_attention_forward(ns):
    torch.manual_seed(0)
    B, S, D, H = 2, 4, 8, 2
    Dk = D // H
    x_q = torch.randn(B, S, D)
    x_kv = torch.randn(B, S, D)
    p = {k: torch.randn(D, D) if k.startswith("W") else torch.randn(D)
         for k in ["Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"]}
    out = ns["assemble_multi_head_attention_forward"](x_q, x_kv, p, None, H)
    assert out.shape == (B, S, D)
    # full independent oracle
    Q = x_q @ p["Wq"] + p["bq"]
    K = x_kv @ p["Wk"] + p["bk"]
    V = x_kv @ p["Wv"] + p["bv"]
    Qh = Q.reshape(B, S, H, Dk).transpose(1, 2)
    Kh = K.reshape(B, S, H, Dk).transpose(1, 2)
    Vh = V.reshape(B, S, H, Dk).transpose(1, 2)
    att = F.scaled_dot_product_attention(Qh, Kh, Vh)
    merged = att.transpose(1, 2).reshape(B, S, D)
    expected = merged @ p["Wo"] + p["bo"]
    assert torch.allclose(out, expected, atol=1e-5)

def test_0032_apply_ffn_first_linear_and_relu(ns):
    torch.manual_seed(0)
    f = ns["apply_ffn_first_linear_and_relu"]
    B, S, D, Dff = 2, 4, 8, 16
    x = torch.randn(B, S, D)
    ffn = {"W1": torch.randn(D, Dff), "b1": torch.randn(Dff),
           "W2": torch.randn(Dff, D), "b2": torch.randn(D)}
    out = f(x, ffn)
    expected = torch.relu(x @ ffn["W1"] + ffn["b1"])
    assert out.shape == (B, S, Dff)
    assert torch.allclose(out, expected, atol=1e-5)
    assert (out >= 0).all()


def test_0033_apply_ffn_second_linear(ns):
    torch.manual_seed(0)
    f = ns["apply_ffn_second_linear"]
    B, S, D, Dff = 2, 4, 8, 16
    h = torch.randn(B, S, Dff)
    ffn = {"W1": torch.randn(D, Dff), "b1": torch.randn(Dff),
           "W2": torch.randn(Dff, D), "b2": torch.randn(D)}
    out = f(h, ffn)
    expected = h @ ffn["W2"] + ffn["b2"]
    assert out.shape == (B, S, D)
    assert torch.allclose(out, expected, atol=1e-5)


def test_0034_position_wise_feed_forward_network(ns):
    torch.manual_seed(0)
    f = ns["position_wise_feed_forward_network"]
    B, S, D, Dff = 2, 4, 8, 16
    x = torch.randn(B, S, D)
    ffn = {"W1": torch.randn(D, Dff), "b1": torch.randn(Dff),
           "W2": torch.randn(Dff, D), "b2": torch.randn(D)}
    out = f(x, ffn)
    expected = torch.relu(x @ ffn["W1"] + ffn["b1"]) @ ffn["W2"] + ffn["b2"]
    assert out.shape == (B, S, D)
    assert torch.allclose(out, expected, atol=1e-5)


def test_0035_compute_layer_norm_mean_and_variance(ns):
    torch.manual_seed(0)
    f = ns["compute_layer_norm_mean_and_variance"]
    B, S, D = 2, 4, 8
    x = torch.randn(B, S, D)
    mean, var = f(x)
    assert mean.shape == (B, S, 1)
    assert var.shape == (B, S, 1)
    assert torch.allclose(mean, x.mean(dim=-1, keepdim=True), atol=1e-5)
    assert torch.allclose(var, x.var(dim=-1, unbiased=False, keepdim=True), atol=1e-5)


def test_0036_normalize_and_scale_with_gamma_beta(ns):
    torch.manual_seed(0)
    f = ns["normalize_and_scale_with_gamma_beta"]
    B, S, D = 2, 4, 8
    x = torch.randn(B, S, D)
    gamma = torch.randn(D)
    beta = torch.randn(D)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    eps = 1e-5
    out = f(x, mean, var, gamma, beta, eps)
    expected = gamma * (x - mean) / torch.sqrt(var + eps) + beta
    assert out.shape == (B, S, D)
    assert torch.allclose(out, expected, atol=1e-5)
    # Reference check against F.layer_norm with unit gamma / zero beta
    ln = torch.nn.functional.layer_norm(x, (D,), weight=gamma, bias=beta, eps=eps)
    assert torch.allclose(out, ln, atol=1e-4)


def test_0037_apply_residual_add_and_norm(ns):
    torch.manual_seed(0)
    f = ns["apply_residual_add_and_norm"]
    B, S, D = 2, 4, 8
    x = torch.randn(B, S, D)
    sub = torch.randn(B, S, D)
    ln = {"gamma": torch.randn(D), "beta": torch.randn(D)}
    out = f(x, sub, ln)
    expected = torch.nn.functional.layer_norm(x + sub, (D,), weight=ln["gamma"], bias=ln["beta"], eps=1e-5)
    assert out.shape == (B, S, D)
    assert torch.allclose(out, expected, atol=1e-4)


def test_0038_apply_dropout_with_keep_mask(ns):
    torch.manual_seed(0)
    f = ns["apply_dropout_with_keep_mask"]
    B, S, D = 2, 4, 8
    x = torch.randn(B, S, D)
    p = 0.5
    keep_mask = (torch.rand(B, S, D) > p).float()
    out = f(x, keep_mask, p)
    expected = x * keep_mask / (1 - p)
    assert out.shape == (B, S, D)
    assert torch.allclose(out, expected, atol=1e-5)
    # Dropped positions are exactly zero; kept positions are scaled by 1/(1-p)
    assert (out[keep_mask == 0] == 0).all()
    assert torch.allclose(out[keep_mask == 1], x[keep_mask == 1] / (1 - p), atol=1e-5)

def _mk_attn_params(D):
    return {
        "Wq": torch.randn(D, D), "bq": torch.randn(D),
        "Wk": torch.randn(D, D), "bk": torch.randn(D),
        "Wv": torch.randn(D, D), "bv": torch.randn(D),
        "Wo": torch.randn(D, D), "bo": torch.randn(D),
    }


def _mk_ffn_params(D, Dff):
    return {
        "W1": torch.randn(D, Dff), "b1": torch.randn(Dff),
        "W2": torch.randn(Dff, D), "b2": torch.randn(D),
    }


def _mk_ln_params(D):
    return {"gamma": torch.randn(D), "beta": torch.randn(D)}


def _ln(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def _mha(x_q, x_kv, p, mask, n_heads):
    B, Sq, D = x_q.shape
    Sk = x_kv.size(1)
    Dk = D // n_heads
    Q = x_q @ p["Wq"] + p["bq"]
    K = x_kv @ p["Wk"] + p["bk"]
    V = x_kv @ p["Wv"] + p["bv"]
    Q = Q.view(B, Sq, n_heads, Dk).transpose(1, 2)
    K = K.view(B, Sk, n_heads, Dk).transpose(1, 2)
    V = V.view(B, Sk, n_heads, Dk).transpose(1, 2)
    scores = (Q @ K.transpose(-2, -1)) / (Dk ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    w = torch.softmax(scores, dim=-1)
    out = w @ V
    out = out.transpose(1, 2).contiguous().view(B, Sq, D)
    return out @ p["Wo"] + p["bo"]


def _ffn(x, p):
    return torch.relu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def test_0039_encoder_layer_self_attention_sublayer(ns):
    torch.manual_seed(0)
    D, H, B, S = 8, 2, 2, 4
    x = torch.randn(B, S, D)
    layer = {"self_attn": _mk_attn_params(D), "ffn": _mk_ffn_params(D, 16),
             "ln1": _mk_ln_params(D), "ln2": _mk_ln_params(D)}
    mask = torch.ones(B, 1, 1, S, dtype=torch.bool)
    out = ns["encoder_layer_self_attention_sublayer"](x, layer, mask, H)
    attn = _mha(x, x, layer["self_attn"], mask, H)
    exp = _ln(x + attn, layer["ln1"]["gamma"], layer["ln1"]["beta"])
    assert out.shape == (B, S, D)
    assert torch.allclose(out, exp, atol=1e-5)


def test_0040_encoder_layer_feed_forward_sublayer(ns):
    torch.manual_seed(0)
    D, B, S = 8, 2, 4
    x = torch.randn(B, S, D)
    layer = {"ffn": _mk_ffn_params(D, 16), "ln2": _mk_ln_params(D)}
    out = ns["encoder_layer_feed_forward_sublayer"](x, layer)
    exp = _ln(x + _ffn(x, layer["ffn"]), layer["ln2"]["gamma"], layer["ln2"]["beta"])
    assert out.shape == (B, S, D)
    assert torch.allclose(out, exp, atol=1e-5)


def test_0041_assemble_encoder_layer(ns):
    torch.manual_seed(0)
    D, H, B, S = 8, 2, 2, 4
    x = torch.randn(B, S, D)
    layer = {"self_attn": _mk_attn_params(D), "ffn": _mk_ffn_params(D, 16),
             "ln1": _mk_ln_params(D), "ln2": _mk_ln_params(D)}
    mask = torch.ones(B, 1, 1, S, dtype=torch.bool)
    out = ns["assemble_encoder_layer"](x, layer, mask, H)
    h = _ln(x + _mha(x, x, layer["self_attn"], mask, H), layer["ln1"]["gamma"], layer["ln1"]["beta"])
    exp = _ln(h + _ffn(h, layer["ffn"]), layer["ln2"]["gamma"], layer["ln2"]["beta"])
    assert out.shape == (B, S, D)
    assert torch.allclose(out, exp, atol=1e-5)


def test_0042_stack_encoder_layers(ns):
    torch.manual_seed(0)
    D, H, B, S = 8, 2, 2, 4
    x = torch.randn(B, S, D)
    layers = [{"self_attn": _mk_attn_params(D), "ffn": _mk_ffn_params(D, 16),
               "ln1": _mk_ln_params(D), "ln2": _mk_ln_params(D)} for _ in range(2)]
    mask = torch.ones(B, 1, 1, S, dtype=torch.bool)
    out = ns["stack_encoder_layers"](x, layers, mask, H)
    h = x
    for layer in layers:
        a = _ln(h + _mha(h, h, layer["self_attn"], mask, H), layer["ln1"]["gamma"], layer["ln1"]["beta"])
        h = _ln(a + _ffn(a, layer["ffn"]), layer["ln2"]["gamma"], layer["ln2"]["beta"])
    assert out.shape == (B, S, D)
    assert torch.allclose(out, h, atol=1e-5)


def test_0043_decoder_layer_masked_self_attention_sublayer(ns):
    torch.manual_seed(0)
    D, H, B, S = 8, 2, 2, 4
    x = torch.randn(B, S, D)
    layer = {"self_attn": _mk_attn_params(D), "ln1": _mk_ln_params(D)}
    mask = torch.tril(torch.ones(S, S, dtype=torch.bool)).view(1, 1, S, S)
    out = ns["decoder_layer_masked_self_attention_sublayer"](x, layer, mask, H)
    exp = _ln(x + _mha(x, x, layer["self_attn"], mask, H), layer["ln1"]["gamma"], layer["ln1"]["beta"])
    assert out.shape == (B, S, D)
    assert torch.allclose(out, exp, atol=1e-5)


def test_0044_decoder_layer_cross_attention_sublayer(ns):
    torch.manual_seed(0)
    D, H, B, S, Sm = 8, 2, 2, 4, 5
    x = torch.randn(B, S, D)
    memory = torch.randn(B, Sm, D)
    layer = {"cross_attn": _mk_attn_params(D), "ln2": _mk_ln_params(D)}
    mask = torch.ones(B, 1, 1, Sm, dtype=torch.bool)
    out = ns["decoder_layer_cross_attention_sublayer"](x, memory, layer, mask, H)
    exp = _ln(x + _mha(x, memory, layer["cross_attn"], mask, H), layer["ln2"]["gamma"], layer["ln2"]["beta"])
    assert out.shape == (B, S, D)
    assert torch.allclose(out, exp, atol=1e-5)


def test_0045_decoder_layer_feed_forward_sublayer(ns):
    torch.manual_seed(0)
    D, B, S = 8, 2, 4
    x = torch.randn(B, S, D)
    layer = {"ffn": _mk_ffn_params(D, 16), "ln3": _mk_ln_params(D)}
    out = ns["decoder_layer_feed_forward_sublayer"](x, layer)
    exp = _ln(x + _ffn(x, layer["ffn"]), layer["ln3"]["gamma"], layer["ln3"]["beta"])
    assert out.shape == (B, S, D)
    assert torch.allclose(out, exp, atol=1e-5)


def test_0046_assemble_decoder_layer(ns):
    torch.manual_seed(0)
    D, H, B, S, Sm = 8, 2, 2, 4, 5
    x = torch.randn(B, S, D)
    memory = torch.randn(B, Sm, D)
    layer = {"self_attn": _mk_attn_params(D), "cross_attn": _mk_attn_params(D),
             "ffn": _mk_ffn_params(D, 16), "ln1": _mk_ln_params(D),
             "ln2": _mk_ln_params(D), "ln3": _mk_ln_params(D)}
    tgt_mask = torch.tril(torch.ones(S, S, dtype=torch.bool)).view(1, 1, S, S)
    src_mask = torch.ones(B, 1, 1, Sm, dtype=torch.bool)
    out = ns["assemble_decoder_layer"](x, memory, layer, tgt_mask, src_mask, H)
    a = _ln(x + _mha(x, x, layer["self_attn"], tgt_mask, H), layer["ln1"]["gamma"], layer["ln1"]["beta"])
    c = _ln(a + _mha(a, memory, layer["cross_attn"], src_mask, H), layer["ln2"]["gamma"], layer["ln2"]["beta"])
    exp = _ln(c + _ffn(c, layer["ffn"]), layer["ln3"]["gamma"], layer["ln3"]["beta"])
    assert out.shape == (B, S, D)
    assert torch.allclose(out, exp, atol=1e-5)


def test_0047_stack_decoder_layers(ns):
    torch.manual_seed(0)
    D, H, B, S, Sm = 8, 2, 2, 4, 5
    x = torch.randn(B, S, D)
    memory = torch.randn(B, Sm, D)
    layers = [{"self_attn": _mk_attn_params(D), "cross_attn": _mk_attn_params(D),
               "ffn": _mk_ffn_params(D, 16), "ln1": _mk_ln_params(D),
               "ln2": _mk_ln_params(D), "ln3": _mk_ln_params(D)} for _ in range(2)]
    tgt_mask = torch.tril(torch.ones(S, S, dtype=torch.bool)).view(1, 1, S, S)
    src_mask = torch.ones(B, 1, 1, Sm, dtype=torch.bool)
    out = ns["stack_decoder_layers"](x, memory, layers, tgt_mask, src_mask, H)
    h = x
    for layer in layers:
        a = _ln(h + _mha(h, h, layer["self_attn"], tgt_mask, H), layer["ln1"]["gamma"], layer["ln1"]["beta"])
        c = _ln(a + _mha(a, memory, layer["cross_attn"], src_mask, H), layer["ln2"]["gamma"], layer["ln2"]["beta"])
        h = _ln(c + _ffn(c, layer["ffn"]), layer["ln3"]["gamma"], layer["ln3"]["beta"])
    assert out.shape == (B, S, D)
    assert torch.allclose(out, h, atol=1e-5)


def test_0048_apply_final_output_projection(ns):
    torch.manual_seed(0)
    D, vocab, B, S = 8, 12, 2, 4
    h = torch.randn(B, S, D)
    out_weight = torch.randn(vocab, D)
    out_bias = torch.randn(vocab)
    out = ns["apply_final_output_projection"](h, out_weight, out_bias)
    exp = h @ out_weight.t() + out_bias
    assert out.shape == (B, S, vocab)
    assert torch.allclose(out, exp, atol=1e-5)


def test_0049_tie_output_projection_to_token_embeddings(ns):
    torch.manual_seed(0)
    embed = torch.randn(12, 8)
    out = ns["tie_output_projection_to_token_embeddings"](embed)
    assert out.shape == embed.shape
    assert torch.allclose(out, embed)
    assert out is embed


def test_0050_apply_log_softmax_over_vocab(ns):
    torch.manual_seed(0)
    vocab, B, S = 12, 2, 4
    logits = torch.randn(B, S, vocab)
    out = ns["apply_log_softmax_over_vocab"](logits)
    exp = F.log_softmax(logits, dim=-1)
    assert out.shape == logits.shape
    assert torch.allclose(out, exp, atol=1e-5)
    probs = out.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)


def test_0051_run_transformer_forward(ns):
    torch.manual_seed(0)
    D, H, vocab, Dff, B, Ssrc, Stgt, n_layers = 8, 2, 12, 16, 2, 4, 4, 2
    enc = [{"self_attn": _mk_attn_params(D), "ffn": _mk_ffn_params(D, Dff),
            "ln1": _mk_ln_params(D), "ln2": _mk_ln_params(D)} for _ in range(n_layers)]
    dec = [{"self_attn": _mk_attn_params(D), "cross_attn": _mk_attn_params(D),
            "ffn": _mk_ffn_params(D, Dff), "ln1": _mk_ln_params(D),
            "ln2": _mk_ln_params(D), "ln3": _mk_ln_params(D)} for _ in range(n_layers)]
    params = {"d_model": D, "n_heads": H, "vocab_size": vocab, "d_ff": Dff, "n_layers": n_layers,
              "embed": torch.randn(vocab, D), "out_bias": torch.randn(vocab),
              "encoder": enc, "decoder": dec}
    src_ids = torch.randint(1, vocab, (B, Ssrc))
    tgt_ids = torch.randint(1, vocab, (B, Stgt))
    out = ns["run_transformer_forward"](params, src_ids, tgt_ids)
    assert out.shape == (B, Stgt, vocab)
    probs = out.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)
    assert torch.isfinite(out).all()
    assert (out <= 1e-4).all()

def test_0052_init_encoder_layer_parameters(ns):
    f = ns["init_encoder_layer_parameters"]
    D, Dff, H = 8, 16, 2
    torch.manual_seed(0)
    layer = f(D, Dff, H)
    assert set(layer.keys()) == {"self_attn", "ffn", "ln1", "ln2"}
    a = layer["self_attn"]
    assert set(a.keys()) == {"Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"}
    for k in ["Wq", "Wk", "Wv", "Wo"]:
        assert a[k].shape == (D, D)
        assert a[k].requires_grad and a[k].is_leaf
    for k in ["bq", "bk", "bv", "bo"]:
        assert a[k].shape == (D,)
        assert a[k].requires_grad and a[k].is_leaf
    ffn = layer["ffn"]
    assert ffn["W1"].shape == (D, Dff) and ffn["b1"].shape == (Dff,)
    assert ffn["W2"].shape == (Dff, D) and ffn["b2"].shape == (D,)
    for k in ["W1", "b1", "W2", "b2"]:
        assert ffn[k].requires_grad and ffn[k].is_leaf
    for lnk in ["ln1", "ln2"]:
        ln = layer[lnk]
        assert set(ln.keys()) == {"gamma", "beta"}
        assert ln["gamma"].shape == (D,) and ln["beta"].shape == (D,)
        assert ln["gamma"].requires_grad and ln["beta"].requires_grad


def test_0053_init_decoder_layer_parameters(ns):
    f = ns["init_decoder_layer_parameters"]
    D, Dff, H = 8, 16, 2
    torch.manual_seed(0)
    layer = f(D, Dff, H)
    assert set(layer.keys()) == {"self_attn", "cross_attn", "ffn", "ln1", "ln2", "ln3"}
    for ak in ["self_attn", "cross_attn"]:
        a = layer[ak]
        assert set(a.keys()) == {"Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"}
        for k in ["Wq", "Wk", "Wv", "Wo"]:
            assert a[k].shape == (D, D) and a[k].requires_grad and a[k].is_leaf
        for k in ["bq", "bk", "bv", "bo"]:
            assert a[k].shape == (D,) and a[k].requires_grad
    ffn = layer["ffn"]
    assert ffn["W1"].shape == (D, Dff) and ffn["W2"].shape == (Dff, D)
    for lnk in ["ln1", "ln2", "ln3"]:
        ln = layer[lnk]
        assert ln["gamma"].shape == (D,) and ln["beta"].shape == (D,)
        assert ln["gamma"].requires_grad and ln["beta"].requires_grad
    # self_attn and cross_attn must be distinct tensor objects
    assert layer["self_attn"]["Wq"] is not layer["cross_attn"]["Wq"]


def test_0054_init_embedding_and_projection_parameters(ns):
    f = ns["init_embedding_and_projection_parameters"]
    vocab, D = 12, 8
    torch.manual_seed(0)
    embed, out_bias = f(vocab, D)
    assert embed.shape == (vocab, D)
    assert out_bias.shape == (vocab,)
    assert embed.requires_grad and embed.is_leaf
    assert out_bias.requires_grad and out_bias.is_leaf
    # embed should be small random, out_bias should be zeros
    assert torch.allclose(out_bias.detach(), torch.zeros(vocab))
    assert embed.detach().abs().max() < 1.0


def test_0055_collect_model_parameters_into_list(ns):
    f = ns["collect_model_parameters_into_list"]
    init_enc = ns["init_encoder_layer_parameters"]
    init_dec = ns["init_decoder_layer_parameters"]
    init_emb = ns["init_embedding_and_projection_parameters"]
    D, Dff, H, vocab, n_layers = 8, 16, 2, 12, 2
    torch.manual_seed(0)
    embed, out_bias = init_emb(vocab, D)
    params = {
        "d_model": D, "n_heads": H, "vocab_size": vocab, "d_ff": Dff, "n_layers": n_layers,
        "embed": embed, "out_bias": out_bias,
        "encoder": [init_enc(D, Dff, H) for _ in range(n_layers)],
        "decoder": [init_dec(D, Dff, H) for _ in range(n_layers)],
    }
    lst = f(params)
    # all entries are leaf tensors requiring grad
    for t in lst:
        assert isinstance(t, torch.Tensor)
        assert t.requires_grad and t.is_leaf
    # no duplicates (every leaf appears exactly once)
    ids = [id(t) for t in lst]
    assert len(ids) == len(set(ids))
    # expected count: embed + out_bias + per enc layer (8 attn + 4 ffn + 2*2 ln = 16)
    # per dec layer (16 attn + 4 ffn + 3*2 ln = 26)
    expected = 2 + n_layers * 16 + n_layers * 26
    assert len(lst) == expected
    # first two are embed, out_bias in order
    assert lst[0] is params["embed"]
    assert lst[1] is params["out_bias"]
    # deterministic: contains every leaf in params
    leaves = set()
    leaves.add(id(params["embed"]))
    leaves.add(id(params["out_bias"]))
    for grp in (params["encoder"], params["decoder"]):
        for layer in grp:
            for sub in layer.values():
                for v in sub.values():
                    leaves.add(id(v))
    assert set(ids) == leaves

def test_0056_shift_targets_right_with_start_token(ns):
    f = ns["shift_targets_right_with_start_token"]
    torch.manual_seed(0)
    tgt = torch.randint(4, 12, (2, 4))
    out = f(tgt, bos_id=1)
    assert out.shape == (2, 4)
    assert out.dtype == tgt.dtype
    assert torch.equal(out[:, 0], torch.full((2,), 1, dtype=tgt.dtype))
    assert torch.equal(out[:, 1:], tgt[:, :-1])


def test_0057_compute_noam_learning_rate(ns):
    f = ns["compute_noam_learning_rate"]
    d_model, warmup = 8, 5
    for step in [1, 3, 5, 7, 20]:
        s = max(step, 1)
        expected = d_model ** -0.5 * min(s ** -0.5, s * warmup ** -1.5)
        got = f(step, d_model, warmup)
        assert abs(got - expected) < 1e-9
    # warmup region increases, decay region decreases
    assert f(1, d_model, warmup) < f(warmup, d_model, warmup)
    assert f(warmup, d_model, warmup) > f(20, d_model, warmup)


def test_0058_build_uniform_smoothing_distribution(ns):
    f = ns["build_uniform_smoothing_distribution"]
    vocab, smoothing = 12, 0.1
    out = f(vocab, smoothing, pad_id=0)
    assert out.shape == (vocab,)
    expected_fill = smoothing / (vocab - 2)
    assert torch.allclose(out, torch.full((vocab,), expected_fill), atol=1e-7)


def test_0059_set_confidence_on_gold_tokens(ns):
    f = ns["set_confidence_on_gold_tokens"]
    vocab = 12
    smoothing, confidence = 0.1, 0.9
    fill = smoothing / (vocab - 2)
    dist = torch.full((3, vocab), fill)
    gold = torch.tensor([4, 7, 2])
    out = f(dist, gold, confidence)
    expected = torch.full((3, vocab), fill)
    for i, g in enumerate(gold.tolist()):
        expected[i, g] = confidence
    assert torch.allclose(out, expected, atol=1e-7)
    # original unchanged
    assert torch.allclose(dist, torch.full((3, vocab), fill), atol=1e-7)


def test_0060_zero_pad_column_and_pad_token_rows(ns):
    f = ns["zero_pad_column_and_pad_token_rows"]
    vocab = 12
    dist = torch.rand(3, vocab) + 0.1
    gold = torch.tensor([4, 0, 7])
    out = f(dist, gold, pad_id=0)
    expected = dist.clone()
    expected[:, 0] = 0.0
    expected[1] = 0.0  # gold==pad row
    assert torch.allclose(out, expected, atol=1e-7)
    assert torch.all(out[:, 0] == 0.0)
    assert torch.all(out[1] == 0.0)


def test_0061_compute_label_smoothed_kl_loss(ns):
    f = ns["compute_label_smoothed_kl_loss"]
    torch.manual_seed(0)
    N, vocab = 6, 12
    logits = torch.randn(N, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)
    target = torch.rand(N, vocab)
    loss = f(log_probs, target)
    expected = -(target * log_probs).sum()
    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.isfinite(loss)
    assert loss.dim() == 0


def test_0062_average_loss_over_non_pad_tokens(ns):
    f = ns["average_loss_over_non_pad_tokens"]
    total = torch.tensor(12.0)
    n = torch.tensor(4.0)
    out = f(total, n)
    assert torch.allclose(out, torch.tensor(3.0), atol=1e-7)
    out2 = f(torch.tensor(7.5), 3)
    assert torch.allclose(out2, torch.tensor(2.5), atol=1e-7)


def test_0063_compute_token_accuracy_ignoring_pad(ns):
    f = ns["compute_token_accuracy_ignoring_pad"]
    vocab = 12
    # build logits whose argmax we control
    logits = torch.full((2, 3, vocab), -10.0)
    # set predicted argmax per position
    preds = torch.tensor([[4, 5, 6], [7, 0, 9]])
    for b in range(2):
        for s in range(3):
            logits[b, s, preds[b, s]] = 10.0
    gold = torch.tensor([[4, 0, 6], [7, 8, 1]])  # pad=0 at [0,1]
    out = f(logits, gold, pad_id=0)
    non_pad = (gold != 0)
    correct = ((preds == gold) & non_pad).sum().float()
    total = non_pad.sum().float()
    expected = float(correct / total)
    assert abs(out - expected) < 1e-7
    assert 0.0 <= out <= 1.0

def test_0064_initialize_adam_optimizer_state(ns):
    f = ns["initialize_adam_optimizer_state"]
    torch.manual_seed(0)
    plist = [torch.randn(3, 4), torch.randn(5)]
    st = f(plist)
    assert st["t"] == 0
    assert len(st["m"]) == 2 and len(st["v"]) == 2
    for p, m, v in zip(plist, st["m"], st["v"]):
        assert m.shape == p.shape and v.shape == p.shape
        assert torch.allclose(m, torch.zeros_like(p))
        assert torch.allclose(v, torch.zeros_like(p))


def test_0065_update_adam_first_moment(ns):
    f = ns["update_adam_first_moment"]
    torch.manual_seed(0)
    m = torch.randn(4, 3)
    grad = torch.randn(4, 3)
    beta1 = 0.9
    expected = beta1 * m + (1 - beta1) * grad
    assert torch.allclose(f(m, grad, beta1), expected, atol=1e-6)


def test_0066_update_adam_second_moment(ns):
    f = ns["update_adam_second_moment"]
    torch.manual_seed(0)
    v = torch.randn(4, 3).abs()
    grad = torch.randn(4, 3)
    beta2 = 0.98
    expected = beta2 * v + (1 - beta2) * grad.pow(2)
    assert torch.allclose(f(v, grad, beta2), expected, atol=1e-6)


def test_0067_apply_adam_bias_correction(ns):
    f = ns["apply_adam_bias_correction"]
    torch.manual_seed(0)
    moment = torch.randn(5)
    beta = 0.9
    t = 3
    expected = moment / (1 - beta ** t)
    assert torch.allclose(f(moment, beta, t), expected, atol=1e-6)


def test_0068_apply_adam_step_to_all_parameters(ns):
    f = ns["apply_adam_step_to_all_parameters"]
    torch.manual_seed(0)
    p = torch.randn(3, 4, requires_grad=True)
    g = torch.randn(3, 4)
    p.grad = g.clone()
    p_before = p.detach().clone()

    lr, beta1, beta2, eps = 0.01, 0.9, 0.98, 1e-9
    opt_state = {"m": [torch.zeros_like(p)], "v": [torch.zeros_like(p)], "t": 0}

    # independent oracle for one step (t=1)
    m = beta1 * torch.zeros_like(p) + (1 - beta1) * g
    v = beta2 * torch.zeros_like(p) + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** 1)
    v_hat = v / (1 - beta2 ** 1)
    expected = p_before - lr * m_hat / (torch.sqrt(v_hat) + eps)

    out_state = f([p], opt_state, lr, beta1, beta2, eps)
    assert out_state["t"] == 1
    assert torch.allclose(p.detach(), expected, atol=1e-6)
    assert torch.allclose(out_state["m"][0], m, atol=1e-6)
    assert torch.allclose(out_state["v"][0], v, atol=1e-6)
    # param with no grad must be untouched
    p2 = torch.randn(2, 2, requires_grad=True)
    p2_before = p2.detach().clone()
    st2 = {"m": [torch.zeros_like(p2)], "v": [torch.zeros_like(p2)], "t": 0}
    f([p2], st2, lr)
    assert torch.allclose(p2.detach(), p2_before)


def test_0069_zero_all_parameter_gradients(ns):
    f = ns["zero_all_parameter_gradients"]
    torch.manual_seed(0)
    a = torch.randn(3, requires_grad=True)
    b = torch.randn(2, 2, requires_grad=True)
    a.grad = torch.randn(3)
    b.grad = torch.randn(2, 2)
    ret = f([a, b])
    assert ret is None
    assert a.grad is None and b.grad is None

def test_0070_compute_batch_training_loss(ns):
    torch.manual_seed(0)
    d_model, n_heads, vocab, d_ff, n_layers = 8, 2, 12, 16, 2
    pad_id, bos_id = 0, 1
    smoothing = 0.1
    encoder = [ns["init_encoder_layer_parameters"](d_model, d_ff, n_heads) for _ in range(n_layers)]
    decoder = [ns["init_decoder_layer_parameters"](d_model, d_ff, n_heads) for _ in range(n_layers)]
    embed, out_bias = ns["init_embedding_and_projection_parameters"](vocab, d_model)
    params = {"d_model": d_model, "n_heads": n_heads, "vocab_size": vocab,
              "d_ff": d_ff, "n_layers": n_layers, "embed": embed, "out_bias": out_bias,
              "encoder": encoder, "decoder": decoder}
    src_ids = torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0]], dtype=torch.long)
    tgt_ids = torch.tensor([[4, 5, 2, 0], [9, 2, 0, 0]], dtype=torch.long)

    loss, n_tokens = ns["compute_batch_training_loss"](params, src_ids, tgt_ids, smoothing, pad_id, bos_id)

    # n_tokens = number of non-pad gold tokens
    expected_n = int((tgt_ids != pad_id).sum())
    assert int(n_tokens) == expected_n
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    # Independent oracle for the loss value.
    decoder_input = ns["shift_targets_right_with_start_token"](tgt_ids, bos_id)
    log_probs = ns["run_transformer_forward"](params, src_ids, decoder_input)
    V = log_probs.size(-1)
    lp = log_probs.reshape(-1, V)
    gold = tgt_ids.reshape(-1)
    conf = 1.0 - smoothing
    fill = smoothing / (V - 2)
    dist = torch.full((gold.size(0), V), fill, dtype=lp.dtype)
    dist[torch.arange(gold.size(0)), gold] = conf
    dist[:, pad_id] = 0.0
    dist[gold == pad_id] = 0.0
    total = -(dist * lp).sum()
    expected_loss = total / (gold != pad_id).sum()
    assert torch.allclose(loss, expected_loss, atol=1e-5)


def test_0071_run_training_step_with_backprop(ns):
    torch.manual_seed(0)
    d_model, n_heads, vocab, d_ff, n_layers = 8, 2, 12, 16, 2
    smoothing = 0.1
    encoder = [ns["init_encoder_layer_parameters"](d_model, d_ff, n_heads) for _ in range(n_layers)]
    decoder = [ns["init_decoder_layer_parameters"](d_model, d_ff, n_heads) for _ in range(n_layers)]
    embed, out_bias = ns["init_embedding_and_projection_parameters"](vocab, d_model)
    params = {"d_model": d_model, "n_heads": n_heads, "vocab_size": vocab,
              "d_ff": d_ff, "n_layers": n_layers, "embed": embed, "out_bias": out_bias,
              "encoder": encoder, "decoder": decoder}
    src_ids = torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0]], dtype=torch.long)
    tgt_ids = torch.tensor([[4, 5, 2, 0], [9, 2, 0, 0]], dtype=torch.long)

    param_list = ns["collect_model_parameters_into_list"](params)
    opt_state = ns["initialize_adam_optimizer_state"](param_list)

    # Snapshot a parameter to confirm an update happened.
    p0_before = param_list[0].detach().clone()

    new_state, loss_value = ns["run_training_step_with_backprop"](
        params, param_list, opt_state, src_ids, tgt_ids, lr=0.01, smoothing=smoothing)

    assert isinstance(loss_value, float)
    import math as _m
    assert _m.isfinite(loss_value)
    assert loss_value > 0.0
    # Adam time-step advanced.
    assert new_state["t"] == 1
    assert len(new_state["m"]) == len(param_list)
    assert len(new_state["v"]) == len(param_list)
    # Parameters actually moved.
    assert not torch.allclose(param_list[0].detach(), p0_before)


def test_0072_run_training_loop_for_steps(ns):
    torch.manual_seed(0)
    d_model, n_heads, vocab, d_ff, n_layers = 8, 2, 12, 16, 2
    smoothing = 0.1
    encoder = [ns["init_encoder_layer_parameters"](d_model, d_ff, n_heads) for _ in range(n_layers)]
    decoder = [ns["init_decoder_layer_parameters"](d_model, d_ff, n_heads) for _ in range(n_layers)]
    embed, out_bias = ns["init_embedding_and_projection_parameters"](vocab, d_model)
    params = {"d_model": d_model, "n_heads": n_heads, "vocab_size": vocab,
              "d_ff": d_ff, "n_layers": n_layers, "embed": embed, "out_bias": out_bias,
              "encoder": encoder, "decoder": decoder}
    src_ids = torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0]], dtype=torch.long)
    tgt_ids = torch.tensor([[4, 5, 2, 0], [9, 2, 0, 0]], dtype=torch.long)

    n_steps = 40
    history = ns["run_training_loop_for_steps"](
        params, src_ids, tgt_ids, n_steps, d_model, warmup=8, smoothing=smoothing)

    assert isinstance(history, list)
    assert len(history) == n_steps
    import math as _m
    assert all(_m.isfinite(x) for x in history)
    # Loss should trend downward on a fixed tiny batch.
    assert history[-1] < history[0]
    assert history[-1] < 0.9 * history[0]

def test_0073_pick_next_token_by_argmax(ns):
    torch.manual_seed(0)
    f = ns["pick_next_token_by_argmax"]
    B, S, vocab = 2, 4, 12
    logits = torch.randn(B, S, vocab)
    out = f(logits)
    assert out.shape == (B, S)
    assert out.dtype == torch.long
    expected = logits.max(dim=-1).indices
    assert torch.equal(out, expected)
    # also a flat 1D logits case
    flat = torch.randn(vocab)
    assert int(f(flat)) == int(flat.argmax())


def test_0074_compute_length_penalty(ns):
    f = ns["compute_length_penalty"]
    for length, alpha in [(1, 0.0), (5, 0.6), (10, 1.0), (3, 0.7)]:
        out = float(f(length, alpha))
        expected = ((5.0 + length) / 6.0) ** alpha
        assert abs(out - expected) < 1e-9
    # alpha=0 -> always 1.0
    assert abs(float(f(7, 0.0)) - 1.0) < 1e-12
    # length=1 -> always 1.0 regardless of alpha
    assert abs(float(f(1, 0.6)) - 1.0) < 1e-12


def test_0075_compute_candidate_scores(ns):
    torch.manual_seed(0)
    f = ns["compute_candidate_scores"]
    beam, vocab = 3, 12
    beam_logprob_sums = torch.randn(beam)
    next_log_probs = torch.randn(beam, vocab)
    out = f(beam_logprob_sums, next_log_probs)
    assert out.shape == (beam, vocab)
    expected = torch.empty(beam, vocab)
    for b in range(beam):
        for v in range(vocab):
            expected[b, v] = beam_logprob_sums[b] + next_log_probs[b, v]
    assert torch.allclose(out, expected, atol=1e-6)


def test_0076_select_top_k_candidates(ns):
    torch.manual_seed(0)
    f = ns["select_top_k_candidates"]
    beam, vocab, k = 3, 12, 4
    scores = torch.randn(beam, vocab)
    values, flat_indices = f(scores, k)
    assert values.shape == (k,)
    assert flat_indices.shape == (k,)
    flat = scores.reshape(-1)
    # recovered values match flat lookup at returned indices
    assert torch.allclose(values, flat[flat_indices], atol=1e-6)
    # they are indeed the k largest, and sorted descending
    sorted_flat = torch.sort(flat, descending=True).values
    assert torch.allclose(values, sorted_flat[:k], atol=1e-6)
    assert torch.all(values[:-1] >= values[1:])


def test_0077_append_tokens_to_beam_sequences(ns):
    f = ns["append_tokens_to_beam_sequences"]
    sequences = torch.tensor([[1, 5, 7],
                              [1, 4, 9],
                              [1, 2, 3]], dtype=torch.long)
    beam_indices = torch.tensor([2, 0, 0], dtype=torch.long)
    token_indices = torch.tensor([8, 6, 11], dtype=torch.long)
    out = f(sequences, beam_indices, token_indices)
    assert out.shape == (3, 4)
    expected = torch.tensor([[1, 2, 3, 8],
                             [1, 5, 7, 6],
                             [1, 5, 7, 11]], dtype=torch.long)
    assert torch.equal(out, expected)


def test_0078_mark_finished_beams(ns):
    f = ns["mark_finished_beams"]
    eos_id = 2
    sequences = torch.tensor([[1, 5, 2],
                              [1, 4, 9],
                              [1, 2, 7],
                              [1, 8, 2]], dtype=torch.long)
    out = f(sequences, eos_id=eos_id)
    assert out.dtype == torch.bool
    assert out.shape == (4,)
    expected = sequences[:, -1] == eos_id
    assert torch.equal(out, expected)
    assert out.tolist() == [True, False, False, True]


def test_0079_select_best_finished_beam(ns):
    f = ns["select_best_finished_beam"]
    sequences = torch.tensor([[1, 5, 2, 0],
                              [1, 4, 9, 2],
                              [1, 2, 7, 2]], dtype=torch.long)
    scores = torch.tensor([-3.0, -5.0, -4.0])
    lengths = torch.tensor([3, 4, 4])
    alpha = 0.7
    out = f(sequences, scores, lengths, alpha)
    # independent oracle
    pen = ((5.0 + lengths.float()) / 6.0) ** alpha
    norm = scores / pen
    best = int(torch.argmax(norm))
    assert torch.equal(out, sequences[best])
    # sanity: output is one of the input rows, 1D of seq length
    assert out.shape == (sequences.shape[1],)
