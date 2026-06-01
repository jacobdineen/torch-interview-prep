# RLHF from Scratch on DistilGPT2

The full RLHF pipeline on the real distilgpt2: decoding strategies, supervised fine-tuning, LoRA adapters, reward modeling, PPO, and the modern preference-optimization family (DPO, IPO, KTO, ORPO, SimPO). PyTorch + HuggingFace.

**65 steps across 8 parts.** Each step is one function (or class) in `steps/NNNN_<fn>.py`, graded in isolation against a hidden reference.

## Parts

| # | Part | Steps | Focus |
|---|------|-------|-------|
| 1 | Model Setup and Decoding Strategies | 8 | Load distilgpt2, sanity-check generation, and implement greedy/temperature/top-k/top-p decoding. |
| 2 | SFT Data Pipeline | 11 | A synthetic instruction dataset plus formatting, tokenization, label masking, padding, batching, splits. |
| 3 | SFT Training Loop | 8 | Shifted next-token loss, AdamW, warmup, grad clipping, accumulation, train/eval steps. |
| 4 | LoRA Adapters | 6 | Low-rank adapters: the LoRA delta, forward, init, freezing, counting, and merge-back. |
| 5 | Reward Modeling | 7 | A synthetic preference dataset and a reward head with pairwise / BCE losses and accuracy. |
| 6 | PPO-Based RLHF | 11 | PPO ingredients: log-probs, KL penalty, returns, GAE, clipped surrogate, value loss, entropy, ppo_loss. |
| 7 | Preference Optimization Alternatives | 7 | Reference-based and reference-free preference losses: DPO, IPO, KTO, ORPO, SimPO. |
| 8 | Evaluation and Chat Interface | 7 | Held-out generation, reward scoring, win rate, token streaming, stop-token handling, and chat. |

## Working through it

Run from the repo root. The CLI tracks progress and unlocks the next step as you pass each one.

```bash
uv run python projects.py rlhf-distilgpt2              # parts + steps, [x]/[ ] solved
uv run python projects.py rlhf-distilgpt2 --next       # jump to the next unsolved step
uv run python projects.py rlhf-distilgpt2 <id>         # check one step, e.g. 0001
uv run python projects.py rlhf-distilgpt2 <id> --explain   # hint / explanation
```

In Neovim, open a step file and use the `<leader>p` practice maps (run / next / hint / explain / solution) — they route to this project automatically.

## End-to-end scaffold

Once enough steps pass, assemble everything and run the full demo:

```bash
uv run python projects.py rlhf-distilgpt2 --scaffold
```

Walks the whole pipeline on real distilgpt2: SFT loss ~6.5 -> ~0.7, DPO loss ~0.69 -> ~0.0, then reward-scored generation and chat.
