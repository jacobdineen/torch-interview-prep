"""Hidden reference implementations for attention-is-all-you-need.

One top-level def per step, named exactly as in spec.STEPS, each with a one-line
docstring (it becomes the stub's prompt). Steps call earlier reference functions
directly -- they share this namespace. From-scratch encoder-decoder Transformer.
"""
import math

import torch
import torch.nn.functional as F  # noqa: F401

def build_token_to_id_vocab(sentences):
    """Build a token-to-id dict: the four special tokens first, then sorted unique whitespace tokens starting at id 4."""
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    tokens = set()
    for sentence in sentences:
        tokens.update(sentence.split())
    for i, tok in enumerate(sorted(tokens), start=4):
        vocab[tok] = i
    return vocab


def build_id_to_token_vocab(token_to_id):
    """Return the inverse mapping from id to token."""
    return {i: tok for tok, i in token_to_id.items()}


def encode_sentence_to_ids(sentence, token_to_id):
    """Map a sentence's whitespace tokens to their ids, sending unknown tokens to the <unk> id 3, as a LongTensor."""
    unk = token_to_id["<unk>"]
    ids = [token_to_id.get(tok, unk) for tok in sentence.split()]
    return torch.tensor(ids, dtype=torch.long)


def decode_ids_to_tokens(ids, id_to_token):
    """Map each id in the sequence back to its token string, returning a list."""
    return [id_to_token[int(i)] for i in ids]


def pad_id_sequence(ids, length, pad_id=0):
    """Right-pad with pad_id or truncate the id tensor to exactly `length`, returning a LongTensor."""
    ids = ids.to(torch.long)
    if ids.size(0) >= length:
        return ids[:length]
    pad = torch.full((length - ids.size(0),), pad_id, dtype=torch.long)
    return torch.cat([ids, pad], dim=0)


def stack_padded_sequences_to_batch(list_of_id_lists, pad_id=0):
    """Pad every id sequence to the batch's max length and stack into a (B, Smax) LongTensor."""
    seqs = [s if torch.is_tensor(s) else torch.tensor(s, dtype=torch.long) for s in list_of_id_lists]
    smax = max(s.size(0) for s in seqs)
    rows = [pad_id_sequence(s, smax, pad_id) for s in seqs]
    return torch.stack(rows, dim=0)

def scale_embeddings_by_sqrt_d_model(emb, d_model):
    """Scale token embeddings up by the square root of the model dimension."""
    return emb * math.sqrt(d_model)


def compute_positional_div_term(d_model):
    """Compute the geometric frequency divisors for the sinusoidal positional encoding."""
    return torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))


def build_position_index_column(max_len):
    """Build a column vector of position indices as floats from 0 to max_len-1."""
    return torch.arange(max_len).float().unsqueeze(1)


def fill_even_indices_with_sin(pe, position, div_term):
    """Write sine values into the even feature columns of the positional encoding table."""
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe


def fill_odd_indices_with_cos(pe, position, div_term):
    """Write cosine values into the odd feature columns of the positional encoding table."""
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the full fixed sinusoidal positional encoding table of shape (max_len, d_model)."""
    pe = torch.zeros(max_len, d_model)
    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)
    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)
    return pe


def add_positional_encoding_to_embeddings(emb, pe):
    """Add the positional encoding to the embeddings, broadcasting over the batch dimension."""
    return emb + pe[:emb.size(-2)]

def build_padding_mask(ids, pad_id=0):
    """Build a boolean mask marking real (non-pad) token positions for attention."""
    mask = ids != pad_id
    return mask[:, None, None, :]


def build_causal_mask(seq_len):
    """Build a lower-triangular boolean mask allowing each position to attend only to earlier or equal positions."""
    m = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return m[None, None, :, :]


def combine_padding_and_causal_masks(pad_mask, causal_mask):
    """Combine padding and causal masks so a position is kept only if both allow it."""
    return pad_mask & causal_mask


def compute_raw_attention_scores(Q, K):
    """Compute unnormalized attention scores as the dot products between queries and keys."""
    return Q @ K.transpose(-2, -1)


def scale_attention_scores(scores, d_k):
    """Scale attention scores down by the square root of the key dimension."""
    return scores / math.sqrt(d_k)


def mask_attention_scores_with_neg_inf(scores, mask):
    """Replace blocked attention scores with a large negative value before softmax."""
    return scores.masked_fill(~mask, -1e9)


def softmax_attention_weights(scores):
    """Turn attention scores into a probability distribution over keys."""
    return torch.softmax(scores, dim=-1)


def apply_attention_weights_to_values(weights, V):
    """Form the attention output as the weighted sum of value vectors."""
    return weights @ V


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention output from queries, keys, values, and an optional keep-mask."""
    scores = compute_raw_attention_scores(Q, K)
    scores = scale_attention_scores(scores, Q.size(-1))
    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)
    weights = softmax_attention_weights(scores)
    return apply_attention_weights_to_values(weights, V)

def split_last_dim_into_heads(x, n_heads):
    """Reshape the trailing model dimension into separate head and per-head feature dimensions."""
    B, S, D = x.shape
    Dk = D // n_heads
    return x.reshape(B, S, n_heads, Dk)


def transpose_heads_before_sequence(x):
    """Move the head dimension ahead of the sequence dimension."""
    return x.transpose(1, 2)


def merge_heads_back_to_model_dim(x):
    """Recombine the per-head features back into a single model dimension."""
    B, H, S, Dk = x.shape
    return x.transpose(1, 2).reshape(B, S, H * Dk)


def apply_linear_projection(x, W, b):
    """Apply an affine linear map using an (in, out)-shaped weight and bias."""
    return x @ W + b


def project_to_query_key_value(x_q, x_kv, attn_params):
    """Linearly project inputs into query, key, and value representations."""
    Q = apply_linear_projection(x_q, attn_params["Wq"], attn_params["bq"])
    K = apply_linear_projection(x_kv, attn_params["Wk"], attn_params["bk"])
    V = apply_linear_projection(x_kv, attn_params["Wv"], attn_params["bv"])
    return Q, K, V


def split_qkv_into_heads(Q, K, V, n_heads):
    """Split each of query, key, and value into multiple attention heads."""
    Q = transpose_heads_before_sequence(split_last_dim_into_heads(Q, n_heads))
    K = transpose_heads_before_sequence(split_last_dim_into_heads(K, n_heads))
    V = transpose_heads_before_sequence(split_last_dim_into_heads(V, n_heads))
    return Q, K, V


def multi_head_scaled_dot_product_attention(Q, K, V, mask, n_heads):
    """Run scaled dot-product attention independently per head and recombine the results."""
    Qh, Kh, Vh = split_qkv_into_heads(Q, K, V, n_heads)
    out = scaled_dot_product_attention(Qh, Kh, Vh, mask)
    return merge_heads_back_to_model_dim(out)


def merge_heads_and_project_output(x, attn_params):
    """Apply the output linear projection to the merged attention result."""
    return apply_linear_projection(x, attn_params["Wo"], attn_params["bo"])


def assemble_multi_head_attention_forward(x_q, x_kv, attn_params, mask, n_heads):
    """Run the full multi-head attention block from inputs to projected output."""
    Q, K, V = project_to_query_key_value(x_q, x_kv, attn_params)
    attended = multi_head_scaled_dot_product_attention(Q, K, V, mask, n_heads)
    return merge_heads_and_project_output(attended, attn_params)

def apply_ffn_first_linear_and_relu(x, ffn_params):
    """Project the input up to the feed-forward inner dimension and apply a ReLU nonlinearity."""
    return torch.relu(apply_linear_projection(x, ffn_params["W1"], ffn_params["b1"]))


def apply_ffn_second_linear(h, ffn_params):
    """Project the inner feed-forward activations back down to the model dimension."""
    return apply_linear_projection(h, ffn_params["W2"], ffn_params["b2"])


def position_wise_feed_forward_network(x, ffn_params):
    """Run the two-layer position-wise feed-forward network with a ReLU in between."""
    return apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x, ffn_params), ffn_params)


def compute_layer_norm_mean_and_variance(x):
    """Compute the per-position mean and biased variance over the last (feature) dimension, keeping dims."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return mean, var


def normalize_and_scale_with_gamma_beta(x, mean, var, gamma, beta, eps=1e-5):
    """Standardize the input with the given mean and variance, then apply the learned scale and shift."""
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def apply_residual_add_and_norm(x, sublayer_out, ln_params):
    """Add the sublayer output to its input and layer-normalize the sum (post-norm residual)."""
    summed = x + sublayer_out
    mean, var = compute_layer_norm_mean_and_variance(summed)
    return normalize_and_scale_with_gamma_beta(summed, mean, var, ln_params["gamma"], ln_params["beta"])


def apply_dropout_with_keep_mask(x, keep_mask, p):
    """Apply inverted dropout by zeroing masked elements and rescaling the survivors by 1/(1-p)."""
    return x * keep_mask / (1 - p)

def encoder_layer_self_attention_sublayer(x, layer, src_mask, n_heads):
    """Run the encoder self-attention sublayer with a residual add-and-norm around it."""
    attn_out = assemble_multi_head_attention_forward(x, x, layer["self_attn"], src_mask, n_heads)
    return apply_residual_add_and_norm(x, attn_out, layer["ln1"])


def encoder_layer_feed_forward_sublayer(x, layer):
    """Run the encoder position-wise feed-forward sublayer with a residual add-and-norm around it."""
    ffn_out = position_wise_feed_forward_network(x, layer["ffn"])
    return apply_residual_add_and_norm(x, ffn_out, layer["ln2"])


def assemble_encoder_layer(x, layer, src_mask, n_heads):
    """Apply one full encoder layer: self-attention sublayer then feed-forward sublayer."""
    x = encoder_layer_self_attention_sublayer(x, layer, src_mask, n_heads)
    x = encoder_layer_feed_forward_sublayer(x, layer)
    return x


def stack_encoder_layers(x, encoder_layers, src_mask, n_heads):
    """Pass the input through every encoder layer in sequence and return the final hidden states."""
    for layer in encoder_layers:
        x = assemble_encoder_layer(x, layer, src_mask, n_heads)
    return x


def decoder_layer_masked_self_attention_sublayer(x, layer, tgt_mask, n_heads):
    """Run the decoder masked self-attention sublayer with a residual add-and-norm around it."""
    attn_out = assemble_multi_head_attention_forward(x, x, layer["self_attn"], tgt_mask, n_heads)
    return apply_residual_add_and_norm(x, attn_out, layer["ln1"])


def decoder_layer_cross_attention_sublayer(x, memory, layer, src_mask, n_heads):
    """Run the decoder cross-attention sublayer attending to encoder memory with a residual add-and-norm."""
    attn_out = assemble_multi_head_attention_forward(x, memory, layer["cross_attn"], src_mask, n_heads)
    return apply_residual_add_and_norm(x, attn_out, layer["ln2"])


def decoder_layer_feed_forward_sublayer(x, layer):
    """Run the decoder position-wise feed-forward sublayer with a residual add-and-norm around it."""
    ffn_out = position_wise_feed_forward_network(x, layer["ffn"])
    return apply_residual_add_and_norm(x, ffn_out, layer["ln3"])


def assemble_decoder_layer(x, memory, layer, tgt_mask, src_mask, n_heads):
    """Apply one full decoder layer: masked self-attention, cross-attention, then feed-forward sublayers."""
    x = decoder_layer_masked_self_attention_sublayer(x, layer, tgt_mask, n_heads)
    x = decoder_layer_cross_attention_sublayer(x, memory, layer, src_mask, n_heads)
    x = decoder_layer_feed_forward_sublayer(x, layer)
    return x


def stack_decoder_layers(x, memory, decoder_layers, tgt_mask, src_mask, n_heads):
    """Pass the input through every decoder layer in sequence and return the final hidden states."""
    for layer in decoder_layers:
        x = assemble_decoder_layer(x, memory, layer, tgt_mask, src_mask, n_heads)
    return x


def apply_final_output_projection(h, out_weight, out_bias):
    """Project decoder hidden states to vocabulary logits using the (vocab,D) output weight and bias."""
    return h @ out_weight.transpose(-2, -1) + out_bias


def tie_output_projection_to_token_embeddings(embed):
    """Return the token embedding matrix to be reused as the tied output projection weight."""
    return embed


def apply_log_softmax_over_vocab(logits):
    """Convert vocabulary logits into log-probabilities over the vocabulary dimension."""
    return F.log_softmax(logits, dim=-1)


def run_transformer_forward(params, src_ids, tgt_ids):
    """Run the full encoder-decoder Transformer and return log-probabilities over the vocabulary."""
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    embed = params["embed"]

    src_pad_mask = build_padding_mask(src_ids)
    tgt_pad_mask = build_padding_mask(tgt_ids)
    causal_mask = build_causal_mask(tgt_ids.size(-1))
    tgt_mask = combine_padding_and_causal_masks(tgt_pad_mask, causal_mask)

    pe = build_sinusoidal_positional_encoding(max(src_ids.size(-1), tgt_ids.size(-1)), d_model)

    src_emb = scale_embeddings_by_sqrt_d_model(embed[src_ids], d_model)
    src_emb = add_positional_encoding_to_embeddings(src_emb, pe)
    tgt_emb = scale_embeddings_by_sqrt_d_model(embed[tgt_ids], d_model)
    tgt_emb = add_positional_encoding_to_embeddings(tgt_emb, pe)

    memory = stack_encoder_layers(src_emb, params["encoder"], src_pad_mask, n_heads)
    dec_out = stack_decoder_layers(tgt_emb, memory, params["decoder"], tgt_mask, src_pad_mask, n_heads)

    out_weight = tie_output_projection_to_token_embeddings(embed)
    logits = apply_final_output_projection(dec_out, out_weight, params["out_bias"])
    return apply_log_softmax_over_vocab(logits)

def init_encoder_layer_parameters(d_model, d_ff, n_heads):
    """Create one encoder layer's parameter dict (self-attention, FFN, two layer norms) as small random leaf tensors that require gradients."""
    def _w(shape):
        return (torch.randn(*shape) * 0.02).requires_grad_(True)
    def _b(shape):
        return torch.zeros(*shape, requires_grad=True)
    attn = {
        "Wq": _w((d_model, d_model)), "bq": _b((d_model,)),
        "Wk": _w((d_model, d_model)), "bk": _b((d_model,)),
        "Wv": _w((d_model, d_model)), "bv": _b((d_model,)),
        "Wo": _w((d_model, d_model)), "bo": _b((d_model,)),
    }
    ffn = {
        "W1": _w((d_model, d_ff)), "b1": _b((d_ff,)),
        "W2": _w((d_ff, d_model)), "b2": _b((d_model,)),
    }
    ln1 = {"gamma": torch.ones(d_model, requires_grad=True), "beta": torch.zeros(d_model, requires_grad=True)}
    ln2 = {"gamma": torch.ones(d_model, requires_grad=True), "beta": torch.zeros(d_model, requires_grad=True)}
    return {"self_attn": attn, "ffn": ffn, "ln1": ln1, "ln2": ln2}


def init_decoder_layer_parameters(d_model, d_ff, n_heads):
    """Create one decoder layer's parameter dict (self-attention, cross-attention, FFN, three layer norms) as small random leaf tensors that require gradients."""
    def _w(shape):
        return (torch.randn(*shape) * 0.02).requires_grad_(True)
    def _b(shape):
        return torch.zeros(*shape, requires_grad=True)
    def _attn():
        return {
            "Wq": _w((d_model, d_model)), "bq": _b((d_model,)),
            "Wk": _w((d_model, d_model)), "bk": _b((d_model,)),
            "Wv": _w((d_model, d_model)), "bv": _b((d_model,)),
            "Wo": _w((d_model, d_model)), "bo": _b((d_model,)),
        }
    self_attn = _attn()
    cross_attn = _attn()
    ffn = {
        "W1": _w((d_model, d_ff)), "b1": _b((d_ff,)),
        "W2": _w((d_ff, d_model)), "b2": _b((d_model,)),
    }
    ln1 = {"gamma": torch.ones(d_model, requires_grad=True), "beta": torch.zeros(d_model, requires_grad=True)}
    ln2 = {"gamma": torch.ones(d_model, requires_grad=True), "beta": torch.zeros(d_model, requires_grad=True)}
    ln3 = {"gamma": torch.ones(d_model, requires_grad=True), "beta": torch.zeros(d_model, requires_grad=True)}
    return {"self_attn": self_attn, "cross_attn": cross_attn, "ffn": ffn, "ln1": ln1, "ln2": ln2, "ln3": ln3}


def init_embedding_and_projection_parameters(vocab_size, d_model):
    """Create the token embedding matrix (vocab, d_model) and the output projection bias (vocab,) as leaf tensors that require gradients."""
    embed = (torch.randn(vocab_size, d_model) * 0.02).requires_grad_(True)
    out_bias = torch.zeros(vocab_size, requires_grad=True)
    return embed, out_bias


def collect_model_parameters_into_list(params):
    """Gather every trainable leaf tensor in the model (embedding, output bias, and all encoder/decoder layer weights, biases, gammas, betas) into a single deterministically ordered list."""
    out = [params["embed"], params["out_bias"]]
    attn_keys = ["Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"]
    ffn_keys = ["W1", "b1", "W2", "b2"]
    ln_keys = ["gamma", "beta"]

    def _attn(a):
        return [a[k] for k in attn_keys]

    def _ffn(f):
        return [f[k] for k in ffn_keys]

    def _ln(l):
        return [l[k] for k in ln_keys]

    for layer in params["encoder"]:
        out += _attn(layer["self_attn"])
        out += _ffn(layer["ffn"])
        out += _ln(layer["ln1"])
        out += _ln(layer["ln2"])
    for layer in params["decoder"]:
        out += _attn(layer["self_attn"])
        out += _attn(layer["cross_attn"])
        out += _ffn(layer["ffn"])
        out += _ln(layer["ln1"])
        out += _ln(layer["ln2"])
        out += _ln(layer["ln3"])
    return out

def shift_targets_right_with_start_token(tgt_ids, bos_id=1):
    """Build the decoder input by prepending a start token column and dropping the final target position."""
    B = tgt_ids.size(0)
    start_col = torch.full((B, 1), bos_id, dtype=tgt_ids.dtype, device=tgt_ids.device)
    return torch.cat([start_col, tgt_ids[:, :-1]], dim=1)


def compute_noam_learning_rate(step, d_model, warmup_steps):
    """Compute the Noam schedule learning rate for the given step, warming up linearly then decaying."""
    step = max(step, 1)
    return float(d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5))


def build_uniform_smoothing_distribution(vocab_size, smoothing, pad_id=0):
    """Create the (vocab,) target distribution filled with the off-target smoothing mass spread over non-target non-pad classes."""
    fill_value = smoothing / (vocab_size - 2)
    return torch.full((vocab_size,), fill_value)


def set_confidence_on_gold_tokens(dist, gold, confidence):
    """Place the confidence probability mass onto each row's gold-token column of the smoothing distribution."""
    out = dist.clone()
    out.scatter_(1, gold.unsqueeze(1), confidence)
    return out


def zero_pad_column_and_pad_token_rows(dist, gold, pad_id=0):
    """Zero out the pad-token column for every row and zero entire rows whose gold token is the pad token."""
    out = dist.clone()
    out[:, pad_id] = 0.0
    pad_rows = (gold == pad_id)
    out[pad_rows] = 0.0
    return out


def compute_label_smoothed_kl_loss(log_probs, smoothed_target):
    """Compute the total label-smoothed cross-entropy as the negative sum of target-weighted log probabilities."""
    return -(smoothed_target * log_probs).sum()


def average_loss_over_non_pad_tokens(total_loss, n_tokens):
    """Normalize the summed loss by the number of non-pad target tokens."""
    return total_loss / n_tokens


def compute_token_accuracy_ignoring_pad(logits_or_logprobs, gold, pad_id=0):
    """Compute the fraction of non-pad positions where the predicted argmax token matches the gold token."""
    preds = logits_or_logprobs.argmax(dim=-1)
    non_pad = (gold != pad_id)
    correct = ((preds == gold) & non_pad).sum().float()
    total = non_pad.sum().float()
    return float(correct / total)

def initialize_adam_optimizer_state(param_list):
    """Create Adam state with zeroed first/second moment buffers for each parameter and a zero step counter."""
    return {
        "m": [torch.zeros_like(p) for p in param_list],
        "v": [torch.zeros_like(p) for p in param_list],
        "t": 0,
    }


def update_adam_first_moment(m, grad, beta1):
    """Return the exponential moving average of the gradient (Adam first moment update)."""
    return beta1 * m + (1 - beta1) * grad


def update_adam_second_moment(v, grad, beta2):
    """Return the exponential moving average of the squared gradient (Adam second moment update)."""
    return beta2 * v + (1 - beta2) * grad * grad


def apply_adam_bias_correction(moment, beta, t):
    """Return the bias-corrected moment estimate by dividing out the accumulated decay at step t."""
    return moment / (1 - beta ** t)


def apply_adam_step_to_all_parameters(param_list, opt_state, lr, beta1=0.9, beta2=0.98, eps=1e-9):
    """Advance the step, update moments and apply the in-place Adam parameter update for every parameter with a gradient."""
    opt_state["t"] += 1
    t = opt_state["t"]
    with torch.no_grad():
        for i, p in enumerate(param_list):
            if p.grad is None:
                continue
            grad = p.grad
            opt_state["m"][i] = update_adam_first_moment(opt_state["m"][i], grad, beta1)
            opt_state["v"][i] = update_adam_second_moment(opt_state["v"][i], grad, beta2)
            m_hat = apply_adam_bias_correction(opt_state["m"][i], beta1, t)
            v_hat = apply_adam_bias_correction(opt_state["v"][i], beta2, t)
            p -= lr * m_hat / (torch.sqrt(v_hat) + eps)
    return opt_state


def zero_all_parameter_gradients(param_list):
    """Clear the stored gradient of every parameter so the next backward pass starts fresh."""
    for p in param_list:
        p.grad = None

def compute_batch_training_loss(params, src_ids, tgt_ids, smoothing, pad_id=0, bos_id=1):
    """Forward the model on shifted decoder inputs and return the label-smoothed KL loss averaged over non-pad target tokens plus that token count."""
    decoder_input = shift_targets_right_with_start_token(tgt_ids, bos_id)
    log_probs = run_transformer_forward(params, src_ids, decoder_input)
    vocab_size = log_probs.size(-1)
    log_probs_flat = log_probs.reshape(-1, vocab_size)
    gold = tgt_ids.reshape(-1)
    fill = build_uniform_smoothing_distribution(vocab_size, smoothing, pad_id)  # (vocab,) template
    dist = fill.to(dtype=log_probs_flat.dtype, device=log_probs_flat.device).unsqueeze(0).repeat(gold.size(0), 1)
    dist = set_confidence_on_gold_tokens(dist, gold, 1.0 - smoothing)
    dist = zero_pad_column_and_pad_token_rows(dist, gold, pad_id)
    total_loss = compute_label_smoothed_kl_loss(log_probs_flat, dist)
    n_tokens = (gold != pad_id).sum()
    loss = average_loss_over_non_pad_tokens(total_loss, n_tokens)
    return loss, n_tokens


def run_training_step_with_backprop(params, param_list, opt_state, src_ids, tgt_ids, lr, smoothing):
    """Run one optimization step: zero grads, compute the batch loss, backpropagate, apply an Adam update, and return the new optimizer state with the float loss value."""
    zero_all_parameter_gradients(param_list)
    loss, _ = compute_batch_training_loss(params, src_ids, tgt_ids, smoothing)
    loss.backward()
    opt_state = apply_adam_step_to_all_parameters(param_list, opt_state, lr)
    return opt_state, loss.item()


def run_training_loop_for_steps(params, src_ids, tgt_ids, n_steps, d_model, warmup, smoothing):
    """Train on a fixed batch for n_steps using a Noam-scheduled learning rate per step and return the list of per-step loss values."""
    param_list = collect_model_parameters_into_list(params)
    opt_state = initialize_adam_optimizer_state(param_list)
    history = []
    for step in range(1, n_steps + 1):
        lr = compute_noam_learning_rate(step, d_model, warmup)
        opt_state, loss_value = run_training_step_with_backprop(
            params, param_list, opt_state, src_ids, tgt_ids, lr, smoothing)
        history.append(loss_value)
    return history

def pick_next_token_by_argmax(logits):
    """Greedily choose each position's next token as the highest-scoring vocab entry."""
    return torch.argmax(logits, dim=-1)


def compute_length_penalty(length, alpha):
    """Return the GNMT length-normalization factor for a hypothesis of the given length."""
    return ((5.0 + length) / (5.0 + 1.0)) ** alpha


def compute_candidate_scores(beam_logprob_sums, next_log_probs):
    """Combine each beam's running log-prob with every possible next token's log-prob."""
    return beam_logprob_sums[:, None] + next_log_probs


def select_top_k_candidates(scores, k):
    """Pick the k best (beam, token) candidates across the flattened score grid."""
    values, flat_indices = torch.topk(scores.reshape(-1), k)
    return values, flat_indices


def append_tokens_to_beam_sequences(sequences, beam_indices, token_indices):
    """Extend each selected beam's sequence by its chosen next token."""
    gathered = sequences[beam_indices]
    return torch.cat([gathered, token_indices.unsqueeze(-1)], dim=-1)


def mark_finished_beams(sequences, eos_id=2):
    """Flag beams whose most recent token is the end-of-sequence symbol."""
    return sequences[:, -1] == eos_id


def select_best_finished_beam(sequences, scores, lengths, alpha):
    """Return the sequence whose length-normalized score is highest."""
    penalties = compute_length_penalty(lengths.to(scores.dtype), alpha)
    normalized = scores / penalties
    best = torch.argmax(normalized)
    return sequences[best]
