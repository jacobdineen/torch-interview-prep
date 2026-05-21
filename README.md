# torch-interview-prep

77 PyTorch problems in a curriculum that builds from tensor basics to a working MiniGPT. Each problem is a stub you fill in; running the script gives PASS/FAIL feedback with tensor-aware diffs, a concept blurb on PASS, and per-tier progress.

## Setup

Requires `uv` and Python 3.12.

```bash
uv sync
```

That installs torch + numpy into `./.venv/`. No other dependencies.

## Working a problem

```bash
# Open the problem you want to solve:
$EDITOR problems/p03_reshape_view_permute.py

# Run it — PASS/FAIL with feedback:
uv run python check.py 03                                 # shortcut by number
uv run python problems/p03_reshape_view_permute.py        # direct
```

On PASS you get a brief concept blurb and a per-tier progress bar. On FAIL you get the failing step's label, the offending line in your code, and a tensor-aware diff if the test compared two tensors.

## Other commands

```bash
uv run python run_all.py                # run every problem
uv run python run_all.py 30-50          # run a range
uv run python run_all.py --status       # progress dashboard from cache (no re-runs)

uv run python reset.py --status         # see which problems you've edited
uv run python reset.py 04 --yes         # reset one problem to its stub
uv run python reset.py --all --yes      # reset everything (asks for explicit --yes)
```

## Layout

```
problems/           77 stubs you edit. Each one has a __main__ that calls runner.py.
tests/
  _compiled/        .pyc bytecode the runner loads. Opaque on purpose.
  _src/             readable test source. Don't peek before solving.
concepts.py         per-problem blurb shown on PASS.
curriculum.py       tier definitions used by runner.py and run_all.py.
.stubs/             pristine baseline used by reset.py.
runner.py           the dispatcher each stub's __main__ calls.
check.py            shortcut: `python check.py NN`.
run_all.py          batch runner + status dashboard.
reset.py            restore a problem (or all) to its stub.
rebuild.py          recompile tests/_src/ → tests/_compiled/ if you edit a test.
```

## Curriculum

15 tiers, ~5 problems each. Run `uv run python run_all.py --status` to see them.

1. Tensor Fundamentals (01–10)
2. Autograd & Linear (11–16)
3. Loss Functions (17–22)
4. Normalization (23–27)
5. Embeddings & Init (28–29)
6. Optimizers (30–34)
7. Training Infrastructure (35–43)
8. Convolutions (44–47)
9. Sequences (48–51)
10. Attention Primitives (52–58)
11. Transformer Architecture (59–64)
12. Efficient Attention (65 — tiled flash attention)
13. Generation (66–70)
14. Auxiliary Losses (71–74)
15. LLM Specifics & Capstone (75–77 — LoRA, grad checkpointing, MiniGPT end-to-end)
