# NumPy variants of problems

Problems are a PyTorch track, but most tensor problems can also be solved in
NumPy. Rather than maintain a second reference + test per problem, a NumPy
solution is graded against the **existing torch test** through a bridge.

## How it works

`np_bridge.py` wraps your NumPy function so the torch test can drive it: torch
tensors in are converted to NumPy arrays, your NumPy result is converted back to
a torch tensor (float64 → torch's float32), and `torch.dtype` args are mapped to
NumPy dtypes. So the same hidden test grades both variants.

A problem **supports NumPy** exactly when a correct NumPy solution passes that
bridge — i.e. it's a pure tensor→tensor map. Torch-specific problems (anything
touching `.backward()`, `requires_grad`, `nn.Module`, or `.device`) can't be
bridged and stay **torch-only**; that's expected, not a gap.

## Solving in NumPy

```bash
uv run python check.py 02a --numpy        # grade the NumPy variant of 02a
uv run python check.py 02 --numpy         # all of parent 02 in NumPy
```

You edit `problems_numpy/pNN<letter>_<slug>.py` (or run the file directly). If a
problem has no NumPy variant, `--numpy` says so (it's torch-only).

## Adding NumPy support to more problems

1. Add the parent's NumPy reference to `NUMPY_PARENTS` in `solutions_numpy.py`
   and list its child ids in `NUMPY_SUPPORTED`.
2. `python tools/gen_numpy_stubs.py` — writes the `problems_numpy/` stubs.
3. `python tools/verify_numpy.py` — every claimed NumPy variant must pass its torch
   test through the bridge (also runs in `verify_all.py` / CI).

## Coverage

Every problem has at least torch support, and **149 of 313 problems** have a
verified NumPy variant — every problem whose function is a pure tensor→tensor
map. The remaining **164 are torch-only by nature**: nn.Module problems (linear,
norm layers, dropout, embedding, MHA/GQA, transformer blocks, LoRA, MiniGPT),
optimizers and gradient clipping, autograd (`.backward()`/`autograd.grad`),
weight init, random sampling (`multinomial`/`randn`/gumbel), callback-driven
generation loops (beam/greedy/speculative), and gradient-flow tests. For those,
`check.py <id> --numpy` reports that the problem is torch-only.

`frameworks.problem_frameworks(pid)` returns the supported set for any problem;
`solutions_numpy.NUMPY_SUPPORTED` is the source of truth. The web app filters and
badges by framework (a dual-support problem shows `pt+np`).
