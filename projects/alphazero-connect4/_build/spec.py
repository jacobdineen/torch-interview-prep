"""Build spec for the alphazero-connect4 project."""

PARTS = [
    ("Connect-4 Game Engine",
     "Board representation, move mechanics, terminal detection, and the env step function."),
    ("Board Encoding and Policy-Value Network",
     "Encode boards as tensors and a conv backbone with policy and value heads."),
    ("Action Masking and Policy Sampling",
     "Mask illegal moves and turn logits into sampled or greedy column actions."),
    ("PUCT Monte Carlo Tree Search",
     "Nodes, PUCT selection, network-guided expansion, backup, and full MCTS rollouts."),
    ("Self-Play Data Generation",
     "MCTS self-play recording (state, policy, outcome) training tuples."),
    ("Losses and Training Loop",
     "Policy, value, and L2 losses and minibatched training over the self-play buffer."),
    ("Iterated Training Loop",
     "Alternate self-play generation and network training across iterations."),
    ("Agents and Evaluation",
     "Baseline agents, head-to-head matches, and win rate against a random policy."),
]

STEPS = [
    # Part 1 — Connect-4 Game Engine (1-14)
    ("make_empty_board", 0), ("column_top_row", 0), ("column_full", 0), ("valid_moves", 0),
    ("drop_piece", 0), ("four_in_a_row_horizontal", 0), ("four_in_a_row_vertical", 0),
    ("four_in_a_row_diagonal_down_right", 0), ("four_in_a_row_diagonal_up_right", 0),
    ("check_winner", 0), ("board_is_full", 0), ("is_terminal", 0), ("other_player", 0),
    ("step_env", 0),
    # Part 2 — Board Encoding and Policy-Value Network (15-21)
    ("encode_board", 1), ("board_to_torch_tensor", 1), ("init_conv_backbone", 1),
    ("init_policy_head", 1), ("init_value_head", 1), ("build_policy_value_net", 1),
    ("policy_value_forward", 1),
    # Part 3 — Action Masking and Policy Sampling (22-26)
    ("action_mask", 2), ("masked_policy_logits", 2), ("masked_log_softmax", 2),
    ("sample_action_from_policy", 2), ("greedy_action_from_policy", 2),
    # Part 4 — PUCT Monte Carlo Tree Search (27-38)
    ("make_mcts_node", 3), ("node_q_value", 3), ("ucb_score", 3), ("select_best_child", 3),
    ("select_leaf", 3), ("evaluate_with_network", 3), ("expand_node", 3), ("backup_value", 3),
    ("run_one_simulation", 3), ("run_mcts", 3), ("visit_count_policy", 3), ("mcts_choose_action", 3),
    # Part 5 — Self-Play Data Generation (39-42)
    ("record_self_play_step", 4), ("play_self_play_game", 4), ("assign_value_targets", 4),
    ("generate_self_play_batch", 4),
    # Part 6 — Losses and Training Loop (43-50)
    ("value_loss_mse", 5), ("policy_loss_cross_entropy", 5), ("l2_regularization_loss", 5),
    ("combined_loss", 5), ("encode_batch_states", 5), ("iterate_minibatches", 5),
    ("training_step", 5), ("training_epoch", 5),
    # Part 7 — Iterated Training Loop (51-52)
    ("self_play_iteration", 6), ("train_loop", 6),
    # Part 8 — Agents and Evaluation (53-57)
    ("random_policy_action", 7), ("greedy_agent_action", 7), ("play_one_match", 7),
    ("match_win_rate", 7), ("evaluate_against_random", 7),
]

assert len(STEPS) == 57, len(STEPS)


def step_id(i):
    return f"{i:04d}"
