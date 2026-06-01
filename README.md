# torch-interview-prep

A self-paced PyTorch interview prep curriculum: **313 single-task problems** organized into 15 tiers that build from `torch.zeros` up to a working MiniGPT. Each problem is a stub file you fill in; running the script gives PASS/FAIL feedback with tensor-aware diffs. Hints, reference solutions, and benchmarks are wired in per-problem.

## Setup

Requires `uv` and Python 3.12.

```bash
uv sync
```

That installs torch + numpy into `./.venv/`. No other dependencies.

## The basic loop

```bash
# 0. See where you are and what's next (problems + projects):
uv run python prep.py

# 1. Open a problem stub:
$EDITOR problems/p03a_flatten_batch.py

# 2. Run it (or just `check.py` with no args to jump to the next unsolved):
uv run python check.py 03a
```

The runner prints `PASS Problem 03a (flatten_batch)` plus a concept blurb and a per-tier progress bar; or `FAIL ...` with the failing assertion's source line and a tensor-aware diff of what was expected vs. what your code produced.

`prep.py` is the single "where am I / what's next" dashboard across both the problems and the projects.

## CLI reference

### `check.py` — the main entry point

```
uv run python check.py <id> [flags]
```

`<id>` is either `NNa` (one task, e.g. `03a`) or `NN` (all tasks of a parent, runs them in order, e.g. `03` runs 03a, 03b, 03c, 03d).

`<id>` is optional when using `--next`.

| Flag | What it does |
|---|---|
| (no flag) | Run the test for that problem. Records PASS/FAIL to `.progress.json`. On failure, prints a tensor diff plus a one-line **likely cause** (shape → wrong reduction dim, dtype → missing cast, constant scale factor → missing normalization). |
| `--next` | Print the next unsolved problem (its id, then its path on the last line). Used by the editor to jump to where you left off. |
| `--note "TEXT"` | Save a free-text note for this problem; it resurfaces under `--explain`. Stored in `.notes.json`. |
| `--hint` | Show the next graduated hint for this problem (1 of 3-4). Persists state in `.hint_state.json`. |
| `--reset-hints` | Reset the hint counter for one problem so `--hint` starts over. |
| `--explain` | Show what the test verifies + the function signatures you need + the concept blurb + any notes you saved. |
| `--solution` | Show the reference implementation. Locked until either: (a) you've passed the problem once, OR (b) you pass `--i-give-up`. |
| `--i-give-up` | Unlock `--solution` for that one problem without earning it. State persists in `.solution_unlock.json`. |
| `--time` | Benchmark your implementation vs the reference. Available for 16 compute-sensitive problems. |
| `--redo` | Re-lock a solved problem (clears its pass record, hint counter, and solution unlock) so you can drill it again. The stub is untouched; use `reset.py` to restore the pristine stub. |
| `--help` / `-h` | Print this help summary. |

Running `check.py` with no id and no flags jumps straight to the next unsolved problem (same as `--next`).

Setting `PREP_JSON=1` makes a run print a single machine-readable JSON line instead of the human report (`status`, `problem`, `fail_file`, `fail_line`, `error_type`, `message`, `hint`, `diff`). This is what the Neovim integration parses; normal runs are unchanged.

Examples:

```bash
uv run python check.py 04a                       # run one task
uv run python check.py 04                        # run all of parent 04
uv run python check.py --next                    # where did I leave off?
uv run python check.py 09c --hint                # graduated hint
uv run python check.py 65a --explain             # what is flash-attention-tiled asking for?
uv run python check.py 05a --note "reduce over batch+spatial, keep channel"
uv run python check.py 22a --solution --i-give-up   # unlock + show reference
uv run python check.py 65a --time                # benchmark flash attention vs reference
```

### `run_all.py` — batch runner + dashboard

```
uv run python run_all.py            # run every problem (all 313)
uv run python run_all.py 4          # only parent 4's children (04a, 04b, 04c, 04d)
uv run python run_all.py 30-50      # parents 30 through 50 (inclusive)
uv run python run_all.py 4 7 12     # specific parents
uv run python run_all.py --status   # dashboard from .progress.json — no tests re-run
uv run python run_all.py --help     # this summary
```

The `--status` dashboard looks like:

```
  [##----------------]   4/40  Tensor Fundamentals           next: 02a
  [------------------]   0/25  Autograd & Linear             next: 11a
  ...
  Overall: 4/313 solved
```

### `reset.py` — restore stubs

```
uv run python reset.py 04a           # reset one task (prompts y/N)
uv run python reset.py 04            # reset whole parent (all p04*)
uv run python reset.py 04 --yes      # skip the confirmation
uv run python reset.py 04 11 23a     # multiple at once
uv run python reset.py --all --yes   # nuclear: reset everything (--yes required)
uv run python reset.py --status      # list which problems differ from their stubs
uv run python reset.py --help        # this summary
```

Resetting also clears the matching entry in `.progress.json` so the dashboard reflects that you're starting over.

### `rebuild.py` — recompile tests

```
uv run python rebuild.py
```

Reads `tests/_src/*.py`, AST-rewrites assertions for nicer failure output, then writes `tests/_compiled/*.pyc`. Only needed if you edit a test source by hand.

## Curriculum

15 tiers. Each parent problem `NN` has 2-6 single-task children (`NNa`, `NNb`, ...).

| Tier | Parents | Topic |
|---|---|---|
| 1 | 01-10 | Tensor Fundamentals |
| 2 | 11-16 | Autograd & Linear |
| 3 | 17-22 | Loss Functions |
| 4 | 23-27 | Normalization |
| 5 | 28-29 | Embeddings & Init |
| 6 | 30-34 | Optimizers (SGD, Adam, AdamW, schedules, grad clipping) |
| 7 | 35-43 | Training Infrastructure (Dataset/Loader, accum, AMP, hooks, checkpoints, EMA, distributed) |
| 8 | 44-47 | Convolutions |
| 9 | 48-51 | Sequences (RNN, LSTM, packed BiLSTM) |
| 10 | 52-58 | Attention Primitives (SDPA, MHA, RoPE, ALiBi, GQA) |
| 11 | 59-64 | Transformer Architecture (encoder, SwiGLU, GPT block, KV cache, sliding window, tied embeddings) |
| 12 | 65 | Efficient Attention (tiled flash attention) |
| 13 | 66-70 | Generation (top-k/top-p, repetition penalty, beam, KV-cached greedy, speculative decoding) |
| 14 | 71-74 | Auxiliary Losses (InfoNCE, KD, VAE reparam, Gumbel-softmax) |
| 15 | 75-77 | LLM Specifics & Capstone (LoRA, gradient checkpointing, MiniGPT end-to-end) |

Concept blurbs (one per parent) live in `concepts.py`. Graduated hints (3-4 per parent, 280 total) live in `hints.py`. Reference solutions (one full impl per parent, child-specific subsets extracted at lookup time) live in `solutions.py`. Benchmarks for compute-sensitive problems live in the `BENCHMARKS` dict at the bottom of `solutions.py`.

**Adding a problem.** Scaffold the stub, pristine snapshot, and test skeleton in
one command, then fill in the test and reference:

```bash
python new_problem.py 78a clip_to_unit --sig "(x, lo, hi)" --desc "clamp x into [lo, hi]"
```

See [docs/adding-a-problem.md](docs/adding-a-problem.md) for the full walkthrough.

## Projects (multi-step builds)

Alongside the standalone problems, **projects** are long ordered sequences of small
steps that accumulate into one working artifact. Four are included:

- **tiny-gpt-from-scratch** (NumPy) — 166 steps / 8 parts: a character-level GPT (tokenizer,
  NumPy/softmax foundations, data pipeline + bigram baseline, single-layer neural bigram,
  layer primitives with backprop, embeddings + single/multi-head self-attention,
  FFN/blocks/full model forward **and** backward, Adam + training loop + generation).
- **tic-tac-toe-rl** (NumPy) — 87 steps / 6 parts: an RL lab from minimax to DQN (game engine,
  random + minimax baselines, tabular Q-learning, self-play + evaluation, a from-scratch
  DQN with replay/target-net, and policy gradients — SARSA, REINFORCE, symmetry augmentation).
- **rlhf-distilgpt2** (PyTorch + HuggingFace) — 65 steps / 8 parts: the full RLHF pipeline on
  distilgpt2 (decoding, SFT, LoRA, reward modeling, PPO, and preference optimization — DPO,
  IPO, KTO, ORPO, SimPO — plus evaluation and a chat interface). Needs `transformers` and
  downloads distilgpt2 (~350MB) on first run; uses CUDA if available.
- **alphazero-connect4** (PyTorch) — 57 steps / 8 parts: an AlphaZero agent for Connect-4
  (board engine, conv policy-value net, action masking, PUCT MCTS, self-play data generation,
  the AlphaZero loss + training loop, the iterated self-play/train loop, and evaluation).

Each step is one function (or class) in `projects/<name>/steps/NNNN_<fn>.py`. You solve it
like a problem (run the file, or `<leader>pp` in nvim) — it's graded by a hidden test that
swaps your function over a reference implementation, so each step is checked in isolation.
Every `*_backward*` / policy-gradient step is validated by **finite-difference gradient
checking**. As you pass steps, `solution.py` is re-assembled from your solved code; once
enough is done, the end-to-end `scaffold.py` runs the whole thing (trains the model/agents).

```bash
uv run python projects.py                       # list projects
uv run python projects.py tiny-gpt-from-scratch  # parts + steps, [x]/[ ] solved
uv run python projects.py tiny-gpt-from-scratch --status   # per-part bars
uv run python projects.py tiny-gpt-from-scratch --next     # next unsolved step (id + path)
uv run python projects.py tiny-gpt-from-scratch 44         # run step 0044
uv run python projects.py tiny-gpt-from-scratch 44 --explain    # signature + what it does
uv run python projects.py tiny-gpt-from-scratch 44 --solution --i-give-up  # reference (gated)
uv run python projects.py tiny-gpt-from-scratch --scaffold # end-to-end demo (once solved)
```

(Swap in `tic-tac-toe-rl` for the other project. With a single project the name is
optional; with several, `python projects.py` lists them.) The
nvim `<leader>p` keymaps work on step files too. Build/regenerate a project from its spec
with `python projects/<name>/_build/gen.py`; `_build/verify.py` checks every reference
passes its own test.

Each project has its own README with the full part breakdown and expected results:
[tiny-gpt-from-scratch](projects/tiny-gpt-from-scratch/README.md) ·
[tic-tac-toe-rl](projects/tic-tac-toe-rl/README.md) ·
[rlhf-distilgpt2](projects/rlhf-distilgpt2/README.md) ·
[alphazero-connect4](projects/alphazero-connect4/README.md). To add your own,
scaffold it with `python new_project.py <name> --title "..."` and see
[docs/adding-a-project.md](docs/adding-a-project.md).

## Continuous integration

`python verify_all.py` is the correctness gate. It runs, in order:
`test_framework.py` (the grader + assembler themselves, against a throwaway
project), `verify_problems.py` (every problem's reference solution against its
compiled test, using a temp dir so your working files are never touched), and
each project's `_build/verify.py`. It exits non-zero if anything fails.

GitHub Actions runs it on every push and pull request (`.github/workflows/ci.yml`,
CPU-only torch, HuggingFace model cached), so a change to shared tooling, a
reference, or a test can't silently break another problem or project. Run it
locally before pushing:

```bash
uv run python verify_all.py
```

## File layout

```
mle_prep/
├── pyproject.toml             # uv project, torch + numpy + transformers deps
├── prep.py                    # unified dashboard: problems + projects + what's next
├── check.py                   # main problem CLI
├── run_all.py                 # batch runner + --status dashboard
├── reset.py                   # restore from .stubs/ snapshot
├── rebuild.py                 # recompile tests/_src/ -> tests/_compiled/
├── runner.py                  # dispatcher each problem stub's __main__ calls
├── debug_tools.py             # backs check.py's --hint/--explain/--solution/--time
├── assert_rewriter.py         # AST-rewrites assertions for rich failure output
├── curriculum.py              # tier definitions, parent→tier lookup, has_test
├── concepts.py                # concept blurbs (printed on PASS)
├── hints.py                   # graduated hints
├── solutions.py               # parent reference impls + per-child extractor + BENCHMARKS
├── new_problem.py             # scaffold a new problem (stub + .stubs + test skeleton)
├── new_project.py             # scaffold a new project (_build/steps/tests tree)
├── regen_all.py               # re-run every project's _build/gen.py
├── projects.py                # multi-step project CLI (mirrors check.py)
├── project_runner.py          # grades one project step + assembles solution.py
├── verify_all.py              # CI gate: framework + problems + every project
├── verify_problems.py         # checks all problem reference solutions pass
├── test_framework.py          # tests the grader + assembler themselves
├── problems/                  # 313 stub files you edit
│   └── pNN<letter>_<slug>.py
├── projects/                  # multi-step builds (see docs/adding-a-project.md)
│   └── <name>/{_build,steps,tests,scaffold.py,project.json,README.md}
├── tests/
│   ├── _compiled/             # opaque .pyc the runner loads
│   └── _src/                  # readable source (don't peek before solving)
├── web/                       # browser app: serve-app.sh, serve-nvim.sh, app.py, static/
├── docs/                      # adding-a-problem.md, adding-a-project.md, web-nvim.md
├── .github/workflows/ci.yml   # runs verify_all.py on every push / PR
└── .stubs/                    # pristine snapshot used by reset.py
```

Local state (gitignored, per-machine):

- `.progress.json` — pass/fail history (drives the dashboard + solution gating)
- `.hint_state.json` — how many hints you've revealed per problem
- `.solution_unlock.json` — manual `--i-give-up` unlocks

## Editor setup

For nvim with a Python LSP, point pyright/pylsp at the local venv. Drop a `pyrightconfig.json`:

```json
{ "venvPath": ".", "venv": ".venv" }
```

Then `:LspRestart` from a Python buffer.

### Neovim integration

An additive block in `init.lua` wires the practice loop into the editor (prefix `<leader>p`).
It shells out to `check.py`/`run_all.py`, parses `PREP_JSON=1` output, and turns a failure into a
diagnostic on the exact failing line (also pushed to the quickfix list, viewable in Trouble).
The repo root and venv python are auto-detected by walking up to `check.py`.

| Keymap | Command | What it does |
|---|---|---|
| `<leader>pp` | `:PracticeRun` | Run the current problem. PASS → notify; FAIL → diagnostic on the failing line + a float with the diff and likely cause. |
| `<leader>pn` | `:PracticeNext` | Open the next unsolved problem. |
| `<leader>pf` | `:PracticePick` | Telescope picker over all problems **and project steps**, marked `[x]` solved / `[ ]` unsolved. |
| `<leader>ph` | `:PracticeHint` | Next graduated hint (float). |
| `<leader>pe` | `:PracticeExplain` | What the test checks + your notes (float). |
| `<leader>ps` | `:PracticeSolution` | Reference solution if unlocked (float). |
| `<leader>pg` | `:PracticeGiveUp` | Unlock + show the solution. |
| `<leader>pt` | `:PracticeTime` | Benchmark vs the reference (float). |
| `<leader>po` | `:PracticeNote` | Jot a note for this problem (resurfaces under explain). |
| `<leader>pd` | `:PracticeStatus` | Unified dashboard (problems + projects + next) via `prep.py`; in a project step, that project's status. |
| `<leader>pw` | `:PracticeAutorun` | Toggle run-on-save for problems **and project steps** (off by default). |

The same maps work on project steps (`projects/*/steps/NNNN_*.py`), routing to `projects.py`. On PASS the float shows the concept blurb + tier progress; on FAIL it shows the diff detail. After a failed run, `[d` / `]d` jump between diagnostics and `<leader>xx` opens them in Trouble.

### Browser web app

Practice in the browser with a problem-description UI on the left and **your real nvim** embedded on the right (Run/Submit + a results panel) — self-host, single user:

```bash
sudo apt-get install -y ttyd && sudo systemctl disable --now ttyd   # one-time
./web/serve-app.sh                                                   # http://127.0.0.1:8000
```

The editor is your actual `nvim` (streamed via ttyd + xterm.js), so every `<leader>p` keymap works identically; picking a problem does an `nvim --remote` buffer switch, and Run does a remote `:wa` + `check.py`. For just the editor with no surrounding UI, use `./web/serve-nvim.sh`. Both bind loopback; reach a remote box by tunnelling the ports. See [docs/web-nvim.md](docs/web-nvim.md).

## Workflow tips

- Order doesn't strictly matter, but tiers build on each other — work them in numeric order.
- When stuck: `--explain` to clarify what's being asked, then `--hint` (escalates), then `--solution --i-give-up`.
- When passing: read the concept blurb that prints on PASS — it's where the cross-references between problems live.
- When something's slow: `--time` will tell you if your implementation is algorithmically off vs the reference (the 10x kind, not the 1.2x kind).
- Want to redo a problem? `reset.py <id>` and it's a clean stub again.
