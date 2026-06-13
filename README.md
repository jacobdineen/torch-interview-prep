# torch-interview-prep

A self-paced PyTorch interview prep curriculum: **313 single-task problems** organized into 15 tiers that build from `torch.zeros` up to a working MiniGPT. Each problem is a stub file you fill in; running the script gives PASS/FAIL feedback with tensor-aware diffs. Hints, reference solutions, and benchmarks are wired in per-problem.

## Setup

Requires `uv` and Python 3.12.

```bash
uv sync
```

That installs torch + numpy (plus transformers, einops, jax, and ninja — a few
projects need them) into `./.venv/`.

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

## Repository layout

The repo root holds the **daily commands** plus the two graders; occasional
tools live in `tools/`, and everything they build on lives in `lib/`.

```
prep.py            single dashboard: where am I + what's next  (start here)
check.py           work a standalone problem  (run / hint / explain / solution / note / redo / time)
projects.py        work a multi-step project  (status / run / explain / solution)
verify_all.py      full correctness gate (CI): every reference solution + project passes its tests
runner.py          grader for standalone problems   (imported by each problem file)
project_runner.py  grader for project steps         (imported by each step file)

tools/             run-occasionally commands  (run as `python tools/<name>.py`)
  run_all.py         run / summarize every problem      reset.py            restore a problem's stub
  new_problem.py     scaffold a problem                 new_project.py      scaffold a project
  gen_numpy_stubs.py regen_all.py rebuild.py            build / maintenance
  verify_problems.py verify_numpy.py test_framework.py  verification (verify_problems runs under verify_all)

lib/               internal modules (not run directly)
  store.py           SQLite store — the single source of truth for progress, notes,
                     hints, and unlocks; mirrors each table to its legacy JSON file
  curriculum.py      tier/order metadata        frameworks.py   torch vs numpy detection
  concepts.py hints.py examples.py solutions.py solutions_numpy.py   per-problem content
  debug_tools.py     hint / explain / solution   np_bridge.py    numpy-variant grading
  assert_rewriter.py test authoring helper

problems/          313 standalone problem stubs (PyTorch)     problems_numpy/  numpy variants
projects/          multi-step projects — each has steps/, tests/, _build/, scaffold.py
tests/             hidden graders: _compiled/*.pyc (run) + _src/*.py (readable source)
web/               self-hosted web app (app.py + static/) wrapping your real nvim
.stubs/            pristine stub snapshots that reset.py restores from
docs/              guides (adding a problem/project, numpy variants, the web app)
```

`runner.py`/`project_runner.py` stay at the root because every problem/step file
imports them by name (`from runner import run_test_for`); `check.py`/`projects.py`
stay because the nvim integration locates the repo via `check.py` and runs both.

## Daily flow

```bash
uv run python prep.py                 # 1. see status + the next problem
uv run python check.py                #    jump straight to the next unsolved (or: check.py <id>)
# ...edit the stub, then re-run check.py <id> until it PASSES...
uv run python projects.py <name>      # 2. or work a from-scratch project
./web/serve-app.sh                    # 3. or do it all in a browser-hosted nvim
```

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

### `tools/run_all.py` — batch runner + dashboard

```
uv run python tools/run_all.py            # run every problem (all 313)
uv run python tools/run_all.py 4          # only parent 4's children (04a, 04b, 04c, 04d)
uv run python tools/run_all.py 30-50      # parents 30 through 50 (inclusive)
uv run python tools/run_all.py 4 7 12     # specific parents
uv run python tools/run_all.py --status   # dashboard from recorded progress — no tests re-run
uv run python tools/run_all.py --help     # this summary
```

The `--status` dashboard looks like:

```
  [##----------------]   4/40  Tensor Fundamentals           next: 02a
  [------------------]   0/25  Autograd & Linear             next: 11a
  ...
  Overall: 4/313 solved
```

### `tools/reset.py` — restore stubs

```
uv run python tools/reset.py 04a           # reset one task (prompts y/N)
uv run python tools/reset.py 04            # reset whole parent (all p04*)
uv run python tools/reset.py 04 --yes      # skip the confirmation
uv run python tools/reset.py 04 11 23a     # multiple at once
uv run python tools/reset.py --all --yes   # nuclear: reset everything (--yes required)
uv run python tools/reset.py --status      # list which problems differ from their stubs
uv run python tools/reset.py --help        # this summary
```

Resetting also clears that problem's progress in the store so the dashboard reflects
that you're starting over. The web app has the same thing built in: **⟲ Reset
progress** in the sidebar resets status (and optionally the code) at item / project /
everything scope, and any discarded code is first tarred into `.reset_backups/`.

### `tools/rebuild.py` — recompile tests

```
uv run python tools/rebuild.py
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

Concept blurbs (one per parent) live in `lib/concepts.py`. Graduated hints (3-4 per parent, 280 total) live in `lib/hints.py`. Reference solutions (one full impl per parent, child-specific subsets extracted at lookup time) live in `lib/solutions.py`. Benchmarks for compute-sensitive problems live in the `BENCHMARKS` dict at the bottom of `lib/solutions.py`.

**Adding a problem.** Scaffold the stub, pristine snapshot, and test skeleton in
one command, then fill in the test and reference:

```bash
python tools/new_problem.py 78a clip_to_unit --sig "(x, lo, hi)" --desc "clamp x into [lo, hi]"
```

See [docs/adding-a-problem.md](docs/adding-a-problem.md) for the full walkthrough.

## Projects (multi-step builds)

Alongside the standalone problems, **projects** are long ordered sequences of small
steps that accumulate into one working artifact. Seventeen are included (698 steps);
the six biggest:

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
- **classical-ml-from-scratch** (NumPy) — 18 steps / 4 parts: the pre-deep-learning toolkit
  (linear regression closed-form + gradient, logistic regression, k-means clustering, and PCA),
  built from the math up with gradients finite-difference checked.
- **metrics-and-evaluation-from-scratch** (NumPy) — 10 steps / 3 parts: the model-evaluation
  toolkit (train/test split, k-fold, confusion matrix, precision/recall/F1 + macro-F1, tie-safe
  ROC AUC, and MAE/RMSE/R²).

Plus eleven more: **attention-is-all-you-need** (79 steps — the original Transformer,
encoder-decoder + beam search), **cnn-from-scratch-numpy** (59), **flash-attention-cuda**
(26 — real CUDA kernels JIT-compiled and graded on your GPU), **mlp-in-jax-from-scratch**
(21), **rnn-lstm-from-scratch** (21), **diffusion-ddpm-from-scratch** (21),
**mini-vllm-inference-engine** (20 — paged KV cache + continuous batching),
**micrograd-autograd-from-scratch** (18), **flash-attention-from-scratch** (18 — the
tiled algorithm in pure torch), **einops** (17), and **bpe-tokenizer-from-scratch** (15).

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
[alphazero-connect4](projects/alphazero-connect4/README.md) ·
[classical-ml-from-scratch](projects/classical-ml-from-scratch/README.md) ·
[metrics-and-evaluation-from-scratch](projects/metrics-and-evaluation-from-scratch/README.md). To add your own,
scaffold it with `python tools/new_project.py <name> --title "..."` and see
[docs/adding-a-project.md](docs/adding-a-project.md).

## Continuous integration

`python verify_all.py` is the correctness gate. It runs, in order:
`tools/test_framework.py` (the grader + assembler themselves, against a throwaway
project), `tools/test_store.py` (the SQLite progress store — recording, per-scope
resets, concurrency — against an isolated temp DB), `tools/verify_problems.py`
(every problem's reference solution against its compiled test, using a temp dir so
your working files are never touched), `tools/verify_numpy.py`, and each project's
`_build/verify.py`. CI also runs `web/smoke_test.py --static` (frontend/backend
contract, JS syntax, CSRF wiring). It exits non-zero if anything fails.

GitHub Actions runs it on every push and pull request (`.github/workflows/ci.yml`,
CPU-only torch, HuggingFace model cached), so a change to shared tooling, a
reference, or a test can't silently break another problem or project. Run it
locally before pushing:

```bash
uv run python verify_all.py
```

## Local state (gitignored, per-machine)

All mutable practice state lives in one SQLite database, `.mle_store.db` —
progress, notes, hint counters, solution unlocks, and the full attempt history
(which powers the home-screen heatmap and survives resets). Every write also
re-exports the matching legacy JSON file (`.progress.json`, `.notes.json`,
`.hint_state.json`, `.solution_unlock.json`, per-project
`.project_progress.json`) so anything that still reads JSON keeps working;
the DB is the source of truth. Code resets archive whatever they discard into
`.reset_backups/solutions-<timestamp>.tar.gz`.

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
| `<leader>pp` | `:PracticeRun` | Run the current problem. PASS → notify; FAIL → notify with the likely cause + a diagnostic on the failing line (no popups — the web results panel mirrors every run). |
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

The same maps work on project steps (`projects/*/steps/NNNN_*.py`), routing to `projects.py`. If the web app is open, every nvim-initiated run also lands in its results panel (concept, progress, diff detail) and the page follows your navigation. After a failed run, `[d` / `]d` jump between diagnostics and `<leader>xx` opens them in Trouble.

### Browser web app

Practice in the browser with a problem-description UI on the left and **your real nvim** embedded on the right (Run/Submit + a results panel) — self-host, single user:

```bash
sudo apt-get install -y ttyd && sudo systemctl disable --now ttyd   # one-time
./web/serve-app.sh                                                   # http://127.0.0.1:8000
```

The editor is your actual `nvim` (streamed via ttyd + xterm.js), so every `<leader>p`
keymap works identically; picking a problem does an `nvim --remote` buffer switch, and
Run saves just the graded buffer (a synchronous `:update`) then runs `check.py`. Runs
started from nvim (`<leader>pp`) drive the same UI. Also built in:

- a **home screen** with per-track cards, day-streak stats, and a GitHub-style
  **activity heatmap** built from the attempt history (it survives resets);
- a collapsible **browse sidebar** (search, source/framework/status filters,
  Previous / Next), a project outline drawer (`o`), and a command palette (`Ctrl-K`);
- **⟲ Reset progress** — clear solved status (and optionally restore starting code,
  with the discarded solutions tarred into `.reset_backups/` first) per item,
  per project, or globally;
- a **stale-tab banner**: if the server restarts on newer code under an open tab,
  the tab notices on its next sync poll and tells you to reload instead of
  half-working.

For just the editor with no surrounding UI, use `./web/serve-nvim.sh`. Both bind
loopback; reach a remote box by tunnelling the ports. See [docs/web-nvim.md](docs/web-nvim.md).

## Workflow tips

- Order doesn't strictly matter, but tiers build on each other — work them in numeric order.
- When stuck: `--explain` to clarify what's being asked, then `--hint` (escalates), then `--solution --i-give-up`.
- When passing: read the concept blurb that prints on PASS — it's where the cross-references between problems live.
- When something's slow: `--time` will tell you if your implementation is algorithmically off vs the reference (the 10x kind, not the 1.2x kind).
- Want to redo a problem? `tools/reset.py <id>` (or the web app's ⟲ Reset) and it's a clean stub again.
