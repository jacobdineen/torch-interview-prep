"""AlphaZero on Connect-4 — end-to-end demo.

Run once you've solved (enough of) the steps:

    python projects.py alphazero-connect4 --scaffold

Builds the policy-value net, shows MCTS finding a tactical win, then runs a few
self-play + training iterations and evaluates against a random opponent. Kept
tiny so it finishes in under a minute or so on CPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from solution import *  # noqa: E402,F401,F403


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = build_policy_value_net(2, 32, 7)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    print("built policy-value net")

    print("\n=== 1) MCTS tactics ===")
    b = make_empty_board()
    b[5, 0:3] = 1  # player 1 threatens a win at column 3
    root = make_mcts_node(b, 1, 0.0)
    run_mcts(root, net, 150, 1.5)
    print(f"  with three in a row, MCTS plays column {mcts_choose_action(root, 0)} (winning move is 3)")

    print("\n=== 2) baseline: untrained net vs random ===")
    print(f"  win rate: {evaluate_against_random(net, 20, rng):.2f}")

    print("\n=== 3) AlphaZero training (self-play + train) ===")
    net, losses = train_loop(net, opt, n_iterations=3, n_games=4, n_simulations=20,
                             c_puct=1.5, temperature=1.0, batch_size=16, weight_decay=1e-4, rng=rng)
    print("  per-iteration loss:", [round(x, 3) for x in losses])

    print("\n=== 4) trained net vs random ===")
    print(f"  win rate: {evaluate_against_random(net, 20, rng):.2f}")

    print("\n=== 5) a self-play game ===")
    records, winner = play_self_play_game(net, 20, 1.5, 1.0, rng)
    print(f"  game length {len(records)} moves, winner {winner}")


if __name__ == "__main__":
    main()
