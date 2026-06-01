# AlphaZero on Connect-4 from Scratch

An AlphaZero agent for Connect-4 from scratch: the game engine, a conv policy-value network, PUCT Monte Carlo Tree Search, self-play data generation, the AlphaZero loss, the iterated self-play/train loop, and head-to-head evaluation. PyTorch.

**57 steps across 8 parts.** Each step is one function (or class) in `steps/NNNN_<fn>.py`, graded in isolation against a hidden reference.

## Parts

| # | Part | Steps | Focus |
|---|------|-------|-------|
| 1 | Connect-4 Game Engine | 14 | Board representation, move mechanics, terminal detection, and the env step function. |
| 2 | Board Encoding and Policy-Value Network | 7 | Encode boards as tensors and a conv backbone with policy and value heads. |
| 3 | Action Masking and Policy Sampling | 5 | Mask illegal moves and turn logits into sampled or greedy column actions. |
| 4 | PUCT Monte Carlo Tree Search | 12 | Nodes, PUCT selection, network-guided expansion, backup, and full MCTS rollouts. |
| 5 | Self-Play Data Generation | 4 | MCTS self-play recording (state, policy, outcome) training tuples. |
| 6 | Losses and Training Loop | 8 | Policy, value, and L2 losses and minibatched training over the self-play buffer. |
| 7 | Iterated Training Loop | 2 | Alternate self-play generation and network training across iterations. |
| 8 | Agents and Evaluation | 5 | Baseline agents, head-to-head matches, and win rate against a random policy. |

## Working through it

Run from the repo root. The CLI tracks progress and unlocks the next step as you pass each one.

```bash
uv run python projects.py alphazero-connect4              # parts + steps, [x]/[ ] solved
uv run python projects.py alphazero-connect4 --next       # jump to the next unsolved step
uv run python projects.py alphazero-connect4 <id>         # check one step, e.g. 0001
uv run python projects.py alphazero-connect4 <id> --explain   # hint / explanation
```

In Neovim, open a step file and use the `<leader>p` practice maps (run / next / hint / explain / solution) — they route to this project automatically.

## End-to-end scaffold

Once enough steps pass, assemble everything and run the full demo:

```bash
uv run python projects.py alphazero-connect4 --scaffold
```

MCTS finds tactical wins; a few self-play/train iterations lift the win rate against random from ~0.70 to ~0.95.
