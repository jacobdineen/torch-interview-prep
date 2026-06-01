# Adding a project

A *project* is a long ordered sequence of small steps (one function or class
each) that accumulate into a working program. The four existing ones
(`tiny-gpt-from-scratch`, `tic-tac-toe-rl`, `rlhf-distilgpt2`,
`alphazero-connect4`) all follow the same recipe. This is that recipe.

## The fluid way: `new_project.py`

```bash
python new_project.py my-cool-project --title "My Cool Project"   # add --torch for torch stubs
```

This lays down the whole tree — `_build/{spec,gen,verify}.py`,
`tests/_ref/{reference,tests}.py`, `scaffold.py`, and `README.md` — with a
working, class-aware `gen.py`/`verify.py` (it derives the project name/title from
the directory + `spec.TITLE`, so there's no hardcoded name to forget). It refuses
to overwrite an existing project. Then you only write the content:

1. the reference implementations in `tests/_ref/reference.py`,
2. a test per step in `tests/_ref/tests.py`,
3. the parts + ordered steps in `_build/spec.py`,
4. `python projects/<name>/_build/gen.py` then `_build/verify.py`.

The manual recipe below explains what those files are and how they connect.

## Layout

Everything for a project lives under `projects/<name>/`:

```
projects/<name>/
  _build/
    spec.py      # YOU write: the parts + ordered step list
    gen.py       # generates stubs + project.json + compiles the hidden refs
    verify.py    # checks every reference passes its own test (the CI gate)
  tests/
    _ref/
      reference.py   # YOU write: the hidden correct implementation of every step
      tests.py       # YOU write: test_<id>_<name>(ns) for every step
    _compiled/       # reference.pyc + tests.pyc: committed so a fresh clone works; auto-rebuilt if stale
  steps/             # generated: NNNN_<name>.py stubs the learner edits
  project.json       # generated: the manifest (parts, steps, docs)
  scaffold.py        # YOU write: the end-to-end demo that imports solution.py
  README.md          # the project overview (generatable from project.json)
  solution.py            # generated as the learner solves steps (gitignored)
  .project_progress.json # per-learner progress (gitignored)
```

The shared tooling — `project_runner.py` (the grader), `projects.py` (the CLI) —
is already generic; you don't touch it to add a project.

## The two-and-a-half files you actually write

### 1. `tests/_ref/reference.py` — the answer key

One top-level `def` (or `class`) per step, named exactly as the step. Each is the
correct implementation, with a one-line docstring (it becomes the stub's prompt).
Steps may call earlier functions directly — the grader injects the whole
reference namespace into each step's globals, so dependencies resolve.

### 2. `tests/_ref/tests.py` — one test per step

For each step define `test_<id>_<name>(ns)`, where `ns` is the namespace holding
the learner's function(s) (shadowing the reference). Assert on `ns["<name>"]`.
For numeric/backward steps, do a finite-difference gradient check or
`np.allclose` value check. Keep each test fast — they run in CI for every push.

### 3. `_build/spec.py` — the ordering

```python
PARTS = [
    ("Part Title", "One-line description of the part."),
    ...
]

# (function_name, part_index) in solve order. The step id is its 1-based
# position, zero-padded to 4 digits (0001, 0002, ...).
STEPS = [
    ("first_fn", 0), ("second_fn", 0),
    ("later_fn", 1),
    ...
]
```

The simplest way to start a new project is to copy an existing `_build/` dir and
edit `spec.py` (the `gen.py` and `verify.py` are near-identical across projects —
just update the project `name`/`title` in `gen.py`'s manifest).

## Generate, then verify

From `projects/<name>/`:

```bash
python _build/gen.py        # writes step stubs + project.json, compiles the refs
python _build/verify.py     # every reference must PASS its own test
```

`gen.py` is **idempotent and safe**: it only (re)writes a stub that doesn't exist
yet or is still pristine (untouched / still `raise NotImplementedError`). It never
clobbers real work. Re-run it whenever you add steps to `spec.py` or edit a
reference docstring/signature.

`verify.py` is the gate that your reference math and your tests agree. If a
reference is correct but its test is wrong (or vice versa), it fails here — fix it
before committing. This same script runs in CI (see below).

## The scaffold

`scaffold.py` imports `from solution import *` and runs the whole thing
end-to-end (train the model, play the games, etc.). It's what
`projects.py <name> --scaffold` runs. `solution.py` is assembled automatically
from solved steps, so the scaffold gets richer as the learner progresses. Keep it
tiny enough to finish in well under a minute on CPU — CI doesn't run it, but you
will, often.

## Wire it into CI

Nothing to do — `verify_all.py` globs `projects/*/_build/verify.py`, so a new
project is picked up automatically. Confirm locally before pushing:

```bash
uv run python verify_all.py
```

This runs the framework tests (`test_framework.py`), all problem references, and
every project's `verify.py`. Green here means green in GitHub Actions.

## Checklist

- [ ] `reference.py`: one def/class per step, with docstrings
- [ ] `tests.py`: a `test_<id>_<name>` for every step, fast, with grad/value checks
- [ ] `spec.py`: parts + ordered steps; `gen.py` manifest has the right name/title
- [ ] `python _build/gen.py` then `python _build/verify.py` → all PASS
- [ ] `scaffold.py` runs end-to-end and reports a sensible final metric
- [ ] `README.md` overview (mirror an existing one)
- [ ] `uv run python verify_all.py` green; add the project to the top-level README
