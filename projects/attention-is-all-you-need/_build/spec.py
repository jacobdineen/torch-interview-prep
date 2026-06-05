"""Build spec for the attention-is-all-you-need project.

PARTS: (title, description) in order.
STEPS: (name, part_index) in order. The step id is its 1-based position,
zero-padded to 4 digits (0001, 0002, ...).
"""

TITLE = 'Attention Is All You Need: Build the Transformer From Scratch'

PARTS = [
    ("Tokenization and Batching",
     "Build the vocabulary, encode and decode token ids, and pack padded sequences into batched tensors."),
    ("Embeddings and Positional Encoding",
     "Scale embeddings and construct the sinusoidal positional encoding matrix added to input embeddings."),
    ("Masks and Scaled Dot-Product Attention",
     "Build padding and causal masks and assemble scaled dot-product attention step by step."),
    ("Multi-Head Attention",
     "Split, permute, and merge heads, project Q/K/V, and assemble the full multi-head attention module."),
    ("Feed-Forward, LayerNorm, and Dropout",
     "Implement the position-wise feed-forward network, layer normalization, residual add-and-norm, and dropout."),
    ("Encoder, Decoder, and Full Model",
     "Stack encoder and decoder layers, tie output projections to embeddings, and run the complete forward pass."),
    ("Parameter Initialization",
     "Allocate the raw weight tensors (with requires_grad) for encoder/decoder layers and embeddings."),
    ("Training Objective and Schedule",
     "Implement teacher forcing, Noam warmup, label-smoothed KL loss, and token-level accuracy."),
    ("Adam Optimizer From Scratch",
     "Build Adam step by step: moment buffers, EMA updates, bias correction, the update rule, and gradient zeroing."),
    ("Training Step and Loop",
     "Run a forward pass, compute the label-smoothed loss, backpropagate, and step Adam across many iterations."),
    ("Decoding and Beam Search",
     "Generate sequences with greedy argmax and a length-penalized beam search over candidate hypotheses."),
]

# (function_or_class_name, part_index) in solve order.
STEPS = [
    # Part 1 - Tokenization and Batching
    ("build_token_to_id_vocab", 0), ("build_id_to_token_vocab", 0),
    ("encode_sentence_to_ids", 0), ("decode_ids_to_tokens", 0),
    ("pad_id_sequence", 0), ("stack_padded_sequences_to_batch", 0),
    # Part 2 - Embeddings and Positional Encoding
    ("scale_embeddings_by_sqrt_d_model", 1), ("compute_positional_div_term", 1),
    ("build_position_index_column", 1), ("fill_even_indices_with_sin", 1),
    ("fill_odd_indices_with_cos", 1), ("build_sinusoidal_positional_encoding", 1),
    ("add_positional_encoding_to_embeddings", 1),
    # Part 3 - Masks and Scaled Dot-Product Attention
    ("build_padding_mask", 2), ("build_causal_mask", 2),
    ("combine_padding_and_causal_masks", 2), ("compute_raw_attention_scores", 2),
    ("scale_attention_scores", 2), ("mask_attention_scores_with_neg_inf", 2),
    ("softmax_attention_weights", 2), ("apply_attention_weights_to_values", 2),
    ("scaled_dot_product_attention", 2),
    # Part 4 - Multi-Head Attention
    ("split_last_dim_into_heads", 3), ("transpose_heads_before_sequence", 3),
    ("merge_heads_back_to_model_dim", 3), ("apply_linear_projection", 3),
    ("project_to_query_key_value", 3), ("split_qkv_into_heads", 3),
    ("multi_head_scaled_dot_product_attention", 3), ("merge_heads_and_project_output", 3),
    ("assemble_multi_head_attention_forward", 3),
    # Part 5 - Feed-Forward, LayerNorm, and Dropout
    ("apply_ffn_first_linear_and_relu", 4), ("apply_ffn_second_linear", 4),
    ("position_wise_feed_forward_network", 4), ("compute_layer_norm_mean_and_variance", 4),
    ("normalize_and_scale_with_gamma_beta", 4), ("apply_residual_add_and_norm", 4),
    ("apply_dropout_with_keep_mask", 4),
    # Part 6 - Encoder, Decoder, and Full Model
    ("encoder_layer_self_attention_sublayer", 5), ("encoder_layer_feed_forward_sublayer", 5),
    ("assemble_encoder_layer", 5), ("stack_encoder_layers", 5),
    ("decoder_layer_masked_self_attention_sublayer", 5), ("decoder_layer_cross_attention_sublayer", 5),
    ("decoder_layer_feed_forward_sublayer", 5), ("assemble_decoder_layer", 5),
    ("stack_decoder_layers", 5), ("apply_final_output_projection", 5),
    ("tie_output_projection_to_token_embeddings", 5), ("apply_log_softmax_over_vocab", 5),
    ("run_transformer_forward", 5),
    # Part 7 - Parameter Initialization
    ("init_encoder_layer_parameters", 6), ("init_decoder_layer_parameters", 6),
    ("init_embedding_and_projection_parameters", 6), ("collect_model_parameters_into_list", 6),
    # Part 8 - Training Objective and Schedule
    ("shift_targets_right_with_start_token", 7), ("compute_noam_learning_rate", 7),
    ("build_uniform_smoothing_distribution", 7), ("set_confidence_on_gold_tokens", 7),
    ("zero_pad_column_and_pad_token_rows", 7), ("compute_label_smoothed_kl_loss", 7),
    ("average_loss_over_non_pad_tokens", 7), ("compute_token_accuracy_ignoring_pad", 7),
    # Part 9 - Adam Optimizer From Scratch
    ("initialize_adam_optimizer_state", 8), ("update_adam_first_moment", 8),
    ("update_adam_second_moment", 8), ("apply_adam_bias_correction", 8),
    ("apply_adam_step_to_all_parameters", 8), ("zero_all_parameter_gradients", 8),
    # Part 10 - Training Step and Loop
    ("compute_batch_training_loss", 9), ("run_training_step_with_backprop", 9),
    ("run_training_loop_for_steps", 9),
    # Part 11 - Decoding and Beam Search
    ("pick_next_token_by_argmax", 10), ("compute_length_penalty", 10),
    ("compute_candidate_scores", 10), ("select_top_k_candidates", 10),
    ("append_tokens_to_beam_sequences", 10), ("mark_finished_beams", 10),
    ("select_best_finished_beam", 10),
]


def step_id(i):
    return f"{i:04d}"
