"""Build spec for the rlhf-distilgpt2 project.

PARTS: (title, description). STEPS: (name, part_index) in order; step id is the
1-based position zero-padded to 4 digits. 5 points each.
"""

PARTS = [
    ("Model Setup and Decoding Strategies",
     "Load distilgpt2, sanity-check generation, and implement greedy/temperature/top-k/top-p decoding."),
    ("SFT Data Pipeline",
     "A synthetic instruction dataset plus formatting, tokenization, label masking, padding, batching, splits."),
    ("SFT Training Loop",
     "Shifted next-token loss, AdamW, warmup, grad clipping, accumulation, train/eval steps."),
    ("LoRA Adapters",
     "Low-rank adapters: the LoRA delta, forward, init, freezing, counting, and merge-back."),
    ("Reward Modeling",
     "A synthetic preference dataset and a reward head with pairwise / BCE losses and accuracy."),
    ("PPO-Based RLHF",
     "PPO ingredients: log-probs, KL penalty, returns, GAE, clipped surrogate, value loss, entropy, ppo_loss."),
    ("Preference Optimization Alternatives",
     "Reference-based and reference-free preference losses: DPO, IPO, KTO, ORPO, SimPO."),
    ("Evaluation and Chat Interface",
     "Held-out generation, reward scoring, win rate, token streaming, stop-token handling, and chat."),
]

STEPS = [
    # Part 1 — Model Setup and Decoding (1-8)
    ("load_distilgpt2_tokenizer", 0), ("load_distilgpt2_model", 0), ("set_pad_token_to_eos", 0),
    ("generate_and_decode", 0), ("greedy_decode", 0), ("sample_with_temperature", 0),
    ("top_k_filter", 0), ("top_p_filter", 0),
    # Part 2 — SFT Data Pipeline (9-19)
    ("build_synthetic_instruction_dataset", 1), ("format_example", 1), ("apply_template", 1),
    ("tokenize_example", 1), ("build_labels", 1), ("mask_prompt_labels", 1), ("pad_batch", 1),
    ("make_attention_mask", 1), ("collate_lm_batch", 1), ("iterate_minibatches", 1),
    ("train_val_split", 1),
    # Part 3 — SFT Training Loop (20-27)
    ("shift_logits_and_labels", 2), ("cross_entropy_loss", 2), ("adamw_update", 2),
    ("linear_warmup_schedule", 2), ("clip_grad_norm", 2), ("accumulate_gradients", 2),
    ("sft_train_step", 2), ("evaluate_loss", 2),
    # Part 4 — LoRA Adapters (28-33)
    ("lora_delta", 3), ("lora_linear_forward", 3), ("init_lora_weights", 3),
    ("freeze_base_params", 3), ("count_trainable_params", 3), ("merge_lora", 3),
    # Part 5 — Reward Modeling (34-40)
    ("build_synthetic_preference_dataset", 4), ("format_preference", 4), ("reward_head_forward", 4),
    ("pairwise_reward_loss", 4), ("reward_bce_loss", 4), ("pairwise_accuracy", 4),
    ("reward_train_step", 4),
    # Part 6 — PPO-Based RLHF (41-51)
    ("sequence_logprob", 5), ("per_token_kl", 5), ("compute_returns", 5), ("gae_advantages", 5),
    ("policy_ratio", 5), ("clipped_surrogate", 5), ("value_function_loss", 5), ("entropy_bonus", 5),
    ("ppo_loss", 5), ("kl_penalized_reward", 5), ("batch_sequence_logprob", 5),
    # Part 7 — Preference Optimization Alternatives (52-58)
    ("dpo_logratios", 6), ("dpo_ref_logratios", 6), ("dpo_loss", 6), ("ipo_loss", 6),
    ("kto_loss", 6), ("orpo_loss", 6), ("simpo_loss", 6),
    # Part 8 — Evaluation and Chat Interface (59-65)
    ("build_eval_prompt_set", 7), ("generate_completions", 7), ("score_with_reward", 7),
    ("win_rate", 7), ("stream_tokens", 7), ("apply_stop_tokens", 7), ("chat", 7),
]

assert len(STEPS) == 65, len(STEPS)


def step_id(i):
    return f"{i:04d}"
