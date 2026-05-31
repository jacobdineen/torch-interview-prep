"""Tic-Tac-Toe RL — end-to-end demo.

Run once you've solved (enough of) the steps:

    python projects.py tic-tac-toe-rl --scaffold

Imports your assembled solution.py and walks the whole arc: game engine ->
random baseline -> optimal minimax -> tabular Q-learning -> DQN -> value-vs-policy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from solution import *  # noqa: E402,F401,F403


def _rates(stats):
    return {k: round(v, 3) for k, v in stats.items()}


def main():
    rng = np.random.default_rng(0)

    print("=== 1) game engine ===")
    board = play_hardcoded_game([0, 3, 1, 4, 2])  # X takes the top row
    print(print_board(board))
    print("status:", get_game_status(board))

    print("\n=== 2) random vs random baseline (200 games) ===")
    print(_rates(compute_outcome_rates(play_random_vs_random_matches(200, rng))))

    print("\n=== 3) minimax baselines ===")
    print("minimax(X) vs random(O), 20 games (X never loses):",
          _rates(compute_outcome_rates(play_minimax_vs_random_matches(20, rng))))
    print("minimax vs minimax, 5 games (all draws):",
          _rates(compute_outcome_rates(play_minimax_vs_minimax_matches(5))))

    print("\n=== 4) tabular Q-learning ===")
    q_table, _ = train_q_learning_agent(4000, 0.2, 0.99, 0.2, rng)
    print("Q vs random: ", _rates(evaluate_q_agent_vs_random(q_table, 200, rng)))
    print("Q vs minimax:", _rates(evaluate_q_agent_vs_minimax(q_table, 20)))

    print("\n=== 5) deep Q-network ===")
    dqn = train_dqn_agent(2500, rng)
    for name, st in compare_dqn_tabular_random_minimax(dqn, q_table, 100, rng).items():
        print(f"  {name:<8} vs random:", _rates(st))

    print("\n=== 6) value-based vs policy-based ===")
    vp = compare_value_vs_policy_learners(2500, rng)
    print("value  (Q-learning) vs random:", _rates(vp["value"]))
    print("policy (REINFORCE)  vs random:", _rates(vp["policy"]))


if __name__ == "__main__":
    main()
