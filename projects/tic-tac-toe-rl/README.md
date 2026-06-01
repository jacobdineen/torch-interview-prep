# Reinforcement Learning for Tic-Tac-Toe: From Minimax to DQN

A reinforcement-learning lab that climbs from a hand-coded game engine to optimal minimax, tabular Q-learning, and finally a Deep Q-Network — plus policy-gradient and symmetry extensions. Pure NumPy, gradients checked by hand.

**87 steps across 6 parts.** Each step is one function (or class) in `steps/NNNN_<fn>.py`, graded in isolation against a hidden reference.

## Parts

| # | Part | Steps | Focus |
|---|------|-------|-------|
| 1 | Board Representation & Game Engine | 18 | Encode the 3x3 board, move legality, win/draw detection, turn tracking, and a Game class. |
| 2 | Random and Minimax Baselines | 11 | A random agent, head-to-head rollouts, and full minimax with alpha-beta pruning. |
| 3 | Tabular Q-Learning Foundations | 24 | State hashing with symmetry, hyperparameters, epsilon-greedy, rewards, and the Q-update loop. |
| 4 | Self-Play, Evaluation & Persistence | 9 | Self-play with perspective flipping, evaluation vs random/minimax, and Q-table save/load. |
| 5 | Deep Q-Network Agent | 19 | Board encodings, an MLP with backprop, replay buffer, target network, and DQN training. |
| 6 | Policy Gradients & Extensions | 6 | SARSA, REINFORCE, value-vs-policy comparison, and 8-fold symmetry data augmentation. |

## Working through it

Run from the repo root. The CLI tracks progress and unlocks the next step as you pass each one.

```bash
uv run python projects.py tic-tac-toe-rl              # parts + steps, [x]/[ ] solved
uv run python projects.py tic-tac-toe-rl --next       # jump to the next unsolved step
uv run python projects.py tic-tac-toe-rl <id>         # check one step, e.g. 0001
uv run python projects.py tic-tac-toe-rl <id> --explain   # hint / explanation
```

In Neovim, open a step file and use the `<leader>p` practice maps (run / next / hint / explain / solution) — they route to this project automatically.

## End-to-end scaffold

Once enough steps pass, assemble everything and run the full demo:

```bash
uv run python projects.py tic-tac-toe-rl --scaffold
```

The tabular Q-agent learns optimal play (only draws against minimax); the DQN agent beats the random baseline.
