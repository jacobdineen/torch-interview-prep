"""Build spec for the tic-tac-toe-rl project.

PARTS: (title, description). STEPS: (name, part_index) in order; step id is the
1-based position zero-padded to 4 digits. 5 points each.
"""

PARTS = [
    ("Board Representation & Game Engine",
     "Encode the 3x3 board, move legality, win/draw detection, turn tracking, and a Game class."),
    ("Random and Minimax Baselines",
     "A random agent, head-to-head rollouts, and full minimax with alpha-beta pruning."),
    ("Tabular Q-Learning Foundations",
     "State hashing with symmetry, hyperparameters, epsilon-greedy, rewards, and the Q-update loop."),
    ("Self-Play, Evaluation & Persistence",
     "Self-play with perspective flipping, evaluation vs random/minimax, and Q-table save/load."),
    ("Deep Q-Network Agent",
     "Board encodings, an MLP with backprop, replay buffer, target network, and DQN training."),
    ("Policy Gradients & Extensions",
     "SARSA, REINFORCE, value-vs-policy comparison, and 8-fold symmetry data augmentation."),
]

STEPS = [
    # Part 1 — Board Representation & Game Engine (1-18)
    ("create_empty_board", 0), ("encode_player", 0), ("print_board", 0),
    ("is_cell_empty", 0), ("place_move", 0), ("get_legal_moves", 0),
    ("check_row_win", 0), ("check_column_win", 0), ("check_main_diagonal_win", 0),
    ("check_anti_diagonal_win", 0), ("is_winner", 0), ("is_draw", 0),
    ("get_game_status", 0), ("get_current_player", 0), ("switch_player", 0),
    ("play_hardcoded_game", 0), ("play_interactive_game", 0), ("TicTacToeGame", 0),
    # Part 2 — Random and Minimax Baselines (19-29)
    ("random_move_agent", 1), ("play_random_vs_random_game", 1),
    ("play_random_vs_random_matches", 1), ("compute_outcome_rates", 1),
    ("minimax_terminal_score", 1), ("minimax_recursive", 1), ("minimax_max_min_step", 1),
    ("minimax_best_move", 1), ("minimax_alpha_beta", 1),
    ("play_minimax_vs_random_matches", 1), ("play_minimax_vs_minimax_matches", 1),
    # Part 3 — Tabular Q-Learning Foundations (30-53)
    ("encode_board_state_key", 2), ("canonical_board_key", 2), ("initialize_q_table", 2),
    ("get_q_value", 2), ("set_q_value", 2), ("choose_learning_rate_alpha", 2),
    ("choose_discount_factor_gamma", 2), ("choose_initial_epsilon", 2),
    ("epsilon_decay_schedule", 2), ("epsilon_greedy_explore_move", 2),
    ("epsilon_greedy_select_action", 2), ("greedy_argmax_over_legal_actions", 2),
    ("random_tie_break_argmax", 2), ("tic_tac_toe_reward", 2),
    ("q_learning_nonterminal_target", 2), ("q_learning_terminal_target", 2),
    ("q_learning_update", 2), ("episode_reset_game", 2), ("episode_agent_pick_action", 2),
    ("episode_apply_action", 2), ("episode_apply_q_update", 2), ("episode_check_terminate", 2),
    ("train_q_learning_agent", 2), ("compute_batched_outcome_stats", 2),
    # Part 4 — Self-Play, Evaluation & Persistence (54-62)
    ("self_play_episode", 3), ("flip_board_perspective", 3), ("perspective_reward_sign", 3),
    ("train_q_agent_self_play", 3), ("evaluate_q_agent_vs_random", 3),
    ("evaluate_q_agent_vs_minimax", 3), ("inspect_q_values_for_state", 3),
    ("serialize_q_table_to_dict", 3), ("deserialize_q_table_from_dict", 3),
    # Part 5 — Deep Q-Network Agent (63-81)
    ("encode_board_flat_length_nine", 4), ("encode_board_one_hot_length_eighteen", 4),
    ("build_mlp_architecture", 4), ("initialize_mlp_parameters", 4), ("mlp_forward_pass", 4),
    ("mask_illegal_actions_neg_inf", 4), ("argmax_action_from_q_values", 4),
    ("mse_loss_on_chosen_action", 4), ("mlp_backward_pass", 4), ("adam_update_step", 4),
    ("create_replay_buffer", 4), ("append_transition_to_buffer", 4),
    ("cap_buffer_size_drop_oldest", 4), ("sample_minibatch_from_buffer", 4),
    ("build_target_network_copy", 4), ("compute_target_q_with_target_network", 4),
    ("sync_target_network_periodically", 4), ("train_dqn_agent", 4),
    ("compare_dqn_tabular_random_minimax", 4),
    # Part 6 — Policy Gradients & Extensions (82-87)
    ("sarsa_on_policy_update", 5), ("reinforce_log_prob_of_action", 5),
    ("reinforce_collect_episode_returns", 5), ("reinforce_policy_gradient_update", 5),
    ("compare_value_vs_policy_learners", 5), ("symmetry_augmented_training", 5),
]

assert len(STEPS) == 87, len(STEPS)


def step_id(i):
    return f"{i:04d}"
