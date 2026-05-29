"""Build spec for the tiny-gpt-from-scratch project.

PARTS: (title, description) in order.
STEPS: (name, part_index) in order. The step id is its 1-based position,
zero-padded to 4 digits (0001..0166). Points are a flat 5 per step.
"""

PARTS = [
    ("Tokenizer",
     "Build a tiny character-level tokenizer with vocab, stoi/itos, and encode/decode helpers."),
    ("NumPy and Softmax Foundations",
     "Get fluent with NumPy arrays, indexing, broadcasting, reductions, and numerically stable softmax."),
    ("Data Pipeline and Bigram Baseline",
     "Load the corpus, build batched (X, Y) sequences, and train a counting-based bigram model."),
    ("Single-Layer Neural Bigram",
     "Replace the count table with a learned weight matrix; derive cross-entropy, gradients, SGD."),
    ("Layer Primitives and Backprop",
     "Forward and backward for linear, bias, ReLU, softmax+CE, and LayerNorm building blocks."),
    ("Embeddings and Self-Attention",
     "Token/positional embeddings, then masked single- and multi-head self-attention with backward."),
    ("FFN, Blocks, and Full Model",
     "Feed-forward, residuals, pre-LN Transformer blocks, and the complete GPT forward/backward."),
    ("Adam, Training Loop, and Generation",
     "Adam, the full training/validation loop, and sampling with temperature and top-k decoding."),
]

# (name, part_index) -- order defines the step id.
STEPS = [
    # Part 1 — Tokenizer (1-7)
    ("build_vocab", 0), ("build_stoi", 0), ("build_itos", 0), ("encode_char", 0),
    ("encode_string", 0), ("decode_int", 0), ("decode_ids", 0),
    # Part 2 — NumPy and Softmax Foundations (8-33)
    ("make_1d_array", 1), ("get_array_shape", 1), ("get_array_dtype", 1),
    ("make_2d_zeros", 1), ("make_2d_random", 1), ("index_element", 1), ("slice_row", 1),
    ("slice_column", 1), ("slice_subblock", 1), ("elementwise_add", 1),
    ("elementwise_multiply", 1), ("scalar_broadcast_add", 1), ("vector_matrix_broadcast_add", 1),
    ("array_exp", 1), ("array_log", 1), ("sum_all", 1), ("sum_axis0", 1), ("sum_axis1", 1),
    ("max_along_axis", 1), ("matmul", 1), ("transpose_matrix", 1), ("sum_keepdims", 1),
    ("naive_softmax_1d", 1), ("softmax_overflow_demo", 1), ("stable_softmax_1d", 1),
    ("stable_softmax_2d_rowwise", 1),
    # Part 3 — Data Pipeline and Bigram Baseline (34-56)
    ("read_text_file", 2), ("encode_corpus_to_int_array", 2), ("pick_split_point", 2),
    ("slice_train_and_val", 2), ("pick_block_size", 2), ("slice_x_at_offset", 2),
    ("slice_y_at_offset", 2), ("sample_random_batch_offsets", 2), ("stack_x_batch", 2),
    ("stack_y_batch", 2), ("get_batch", 2), ("allocate_count_matrix", 2),
    ("loop_fill_counts", 2), ("vectorize_counts_add_at", 2), ("add_one_smoothing", 2),
    ("row_sums_of_counts", 2), ("normalize_counts_to_probs", 2), ("sample_next_token", 2),
    ("generate_sequence", 2), ("decode_generated_sequence", 2), ("log_prob_of_pair", 2),
    ("sum_negative_log_probs", 2), ("average_nll", 2),
    # Part 4 — Single-Layer Neural Bigram (57-73)
    ("initialize_w_random", 3), ("scale_w_small", 3), ("one_hot_encode_batch", 3),
    ("forward_logits_onehot", 3), ("observe_lookup_equivalence", 3), ("forward_logits_lookup", 3),
    ("logits_to_probs_rowwise", 3), ("gather_correct_token_probs", 3), ("cross_entropy_loss", 3),
    ("derive_dlogits_on_paper", 3), ("compute_dlogits", 3), ("derive_dw_on_paper", 3),
    ("compute_dw_scatter_add", 3), ("sgd_update_w", 3), ("run_one_training_step", 3),
    ("train_neural_bigram_loop", 3), ("sample_from_neural_bigram", 3),
    # Part 5 — Layer Primitives and Backprop (74-91)
    ("linear_forward", 4), ("derive_dx_on_paper", 4), ("derive_linear_dw_on_paper", 4),
    ("linear_backward_dx", 4), ("linear_backward_dw", 4), ("bias_add_forward", 4),
    ("bias_add_backward_db", 4), ("relu_forward", 4), ("relu_backward", 4),
    ("softmax_cross_entropy_backward", 4), ("layernorm_forward_mean", 4),
    ("layernorm_forward_variance", 4), ("layernorm_forward_normalize", 4),
    ("layernorm_forward_affine", 4), ("layernorm_backward_subtract_mean", 4),
    ("layernorm_backward_divide_std", 4), ("layernorm_backward_full", 4),
    ("layernorm_backward_implementation", 4),
    # Part 6 — Embeddings and Self-Attention (92-130)
    ("create_token_embedding", 5), ("token_embedding_forward", 5), ("token_embedding_backward", 5),
    ("create_positional_embedding", 5), ("slice_positional_embedding", 5),
    ("add_token_and_positional_embeddings", 5), ("embedding_sum_backward", 5),
    ("create_qkv_projections", 5), ("compute_query", 5), ("compute_key", 5), ("compute_value", 5),
    ("compute_attention_scores", 5), ("scale_attention_scores", 5), ("build_causal_mask", 5),
    ("apply_causal_mask", 5), ("softmax_attention_weights", 5), ("attention_weighted_values", 5),
    ("apply_output_projection", 5), ("output_projection_backward", 5),
    ("attention_value_backward", 5), ("masked_softmax_backward", 5), ("scale_scores_backward", 5),
    ("qk_scores_backward", 5), ("qkv_projection_backward", 5), ("choose_attention_head_config", 5),
    ("create_multihead_qkv_projections", 5), ("create_multihead_output_projection", 5),
    ("reshape_to_heads", 5), ("transpose_heads_to_front", 5), ("get_multihead_n_heads", 5),
    ("get_multihead_sequence_length", 5), ("compute_d_head", 5),
    ("multihead_masked_softmax_scores", 5), ("multihead_weighted_sum", 5),
    ("transpose_heads_to_back", 5), ("get_multihead_output_sequence_length", 5),
    ("merge_heads_to_d_model", 5), ("multihead_output_projection_forward", 5),
    ("multihead_reshape_transpose_backward", 5),
    # Part 7 — FFN, Blocks, and Full Model (131-146)
    ("ffn_linear_one_forward", 6), ("ffn_activation_forward", 6), ("ffn_linear_two_forward", 6),
    ("ffn_backward", 6), ("residual_forward", 6), ("residual_backward", 6),
    ("pre_layernorm_sublayer_forward", 6), ("transformer_block_forward", 6),
    ("transformer_block_backward", 6), ("stack_transformer_blocks", 6),
    ("forward_through_all_blocks", 6), ("backward_through_all_blocks", 6),
    ("final_layernorm_forward", 6), ("lm_head_linear_forward", 6), ("full_model_forward", 6),
    ("full_model_backward", 6),
    # Part 8 — Adam, Training Loop, and Generation (147-166)
    ("initialize_adam_moments", 7), ("initialize_adam_step_counter", 7), ("adam_increment_step", 7),
    ("adam_update_first_moment", 7), ("adam_update_second_moment", 7), ("adam_bias_correction", 7),
    ("adam_parameter_update", 7), ("wire_full_training_loop", 7), ("logging_and_validation_loss", 7),
    ("encode_prompt", 7), ("crop_context_to_block_size", 7), ("forward_to_get_logits", 7),
    ("take_last_position_logits", 7), ("apply_temperature", 7), ("top_k_filter", 7),
    ("softmax_to_probs", 7), ("sample_one_token", 7), ("append_token_to_sequence", 7),
    ("generation_loop_for_n_steps", 7), ("decode_final_sequence", 7),
]

assert len(STEPS) == 166, len(STEPS)


def step_id(i):
    """1-based index -> 4-digit id string."""
    return f"{i:04d}"
