"""Per-parent hint lists shown by `check.py NN --hint`.

Hints are graduated: each `--hint` invocation shows the next hint in the list.
Child problems (e.g. p04b) inherit their parent's hints (p04).

The hint progress is tracked in `.hint_state.json` (gitignored, per-machine).
"""

HINTS = {
    "01": [
        "All four are one-liners using built-in torch creation helpers.",
        "`torch.zeros(shape, dtype=...)` takes shape as a tuple/list, NOT as a tensor.",
        "Use `torch.arange(start, end, step)`. To move/cast: `x.to(dtype=..., device=...)`.",
        "Last hint: `tensor_info` should return tuple(x.shape) (not torch.Size), x.dtype, x.device, x.numel().",
    ],
    "02": [
        "All indexing is direct: `x[i]`, `x[:, ::2]`, `x[idx]`, `x[:k, :k]`.",
        "`torch.diag(x)` (or `x.diagonal()`) gets the diagonal of a square matrix.",
        "`select_rows(x, idx)` with a LongTensor uses fancy indexing: `x[idx]`.",
    ],
    "03": [
        "`x.reshape(B, -1)` for flatten; `x.transpose(-1, -2)` for swap; `x.permute(...)` for arbitrary reorder.",
        "`permute` arguments are the SOURCE indices in OUTPUT order. (B,C,H,W) -> (B,H,W,C) is permute(0, 2, 3, 1).",
        "`.contiguous()` returns the same tensor if already contiguous, or a contiguous copy otherwise.",
    ],
    "04": [
        "Always reach for `keepdim=True` when the reduction result will be used in broadcasting.",
        "For pairwise distances, expand: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b. Use A @ B.T for the dot products.",
        "For per-row normalization, use `x.mean(dim=1, keepdim=True)` and `x.std(dim=1, unbiased=False, keepdim=True)`.",
        "Outer product of u (M,) and v (N,) -> use `torch.outer(u, v)` OR `u[:, None] * v[None, :]`.",
    ],
    "05": [
        "`dim=` accepts a tuple, e.g. `x.mean(dim=(0, 2, 3))` reduces over multiple dims at once.",
        "For `(B, *)` -> `(B,)` reductions, `x.flatten(1).max(dim=1).values` works for any rank.",
        "Tensor.max returns a named tuple `(values, indices)`. Use `.values` when you want just the numbers.",
        "`x.argmax(dim=-1)` returns the index of the max along the last dim — no `.values/.indices` (just one return value).",
    ],
    "06": [
        "`torch.cat` joins along an EXISTING dim; `torch.stack` creates a NEW dim.",
        "`torch.split(x, k)` splits into pieces OF SIZE k. To split into k pieces, use `torch.chunk(x, k)`.",
        "Interleave: `torch.stack([a, b], dim=1).reshape(-1, D)` because reshape walks the rightmost dims first.",
    ],
    "07": [
        "`torch.where(condition, value_if_true, value_if_false)` — signature in that order. Easy to flip backwards.",
        "Counts of booleans: `(bool_tensor).sum().item()` gives the count of True values as a Python int.",
        "`torch.nan_to_num(x, nan=value)` replaces NaN with value. Or `torch.where(torch.isnan(x), value, x)`.",
        "`x[mask].mean()` selects elements where the bool mask is True, then averages.",
    ],
    "08": [
        "`gather` READS values by index along a dim. The index tensor's shape must match the input except on the gather dim.",
        "`gather_per_row(x, idx)`: `x.gather(1, idx.unsqueeze(1)).squeeze(1)` — unsqueeze idx to (N, 1), then squeeze the result.",
        "one_hot is `zeros + scatter_(1, labels.unsqueeze(1), 1.0)`. scatter_ writes values at index positions.",
        "For segment-sum (sum rows by index), `out.index_add_(0, index, values)` is cleaner than scatter_add_.",
    ],
    "09": [
        "Every letter in the einsum string is one dim. Letters not in the OUTPUT after `->` get summed away.",
        "matmul: `'ik,kn->in'` (k contracts). Adapt: for `(M,K) @ (K,N) -> (M,N)`, letters are arbitrary as long as they connect.",
        "Use `...` for any-leading-dims: `'...ij,...jk->...ik'` is a matmul that works for any batch shape.",
        "For attention scores: `q @ k.transpose(-1, -2)` or `'...id,...jd->...ij'` (d contracts; q's T and k's T need different letters i, j).",
    ],
    "10": [
        "`x.topk(k)` returns (values, indices) named tuple; defaults are dim=-1, largest=True, sorted=True.",
        "For k SMALLEST values' indices, pass `largest=False` to topk and take `.indices`.",
        "For 2D global argmax: `x.argmax()` returns a flat index; recover (row, col) with `// x.shape[1]` and `% x.shape[1]`.",
        "k-th largest: `x.flatten().topk(k).values[-1]` is O(N log k), faster than a full sort.",
    ],
    # ----- Tiers 2+ have generic guidance; the user can fill in per-problem hints later.
    "11": ["Use torch.autograd.grad for functional gradients (returns Tensor, doesn't accumulate)."],
    "12": ["Custom Function needs @staticmethod forward and backward. Save tensors with ctx.save_for_backward."],
    "13": ["`detach()` shares data but no autograd. `no_grad()` disables grad recording inside a block."],
    "14": ["nn.Linear stores weight as (out, in). Forward is `x @ self.weight.t() + self.bias` for any-rank x."],
    "15": ["Stable sigmoid: branch on x sign to avoid exp() overflow."],
    "16": ["The max-shift trick: subtract max before exp to prevent overflow. log-sum-exp uses this."],
    "17": ["Both MSE and Huber are elementwise then reduced. Huber: 0.5*r^2 if |r|<=delta else delta*(|r|-0.5*delta)."],
    "18": ["Use the stable form: max(x, 0) - x*y + log(1 + exp(-|x|)). Avoid sigmoid + BCE separately — fuse it."],
    "19": ["log_softmax (stable) + NLL = cross_entropy. Use `.gather(1, target.unsqueeze(1)).squeeze(1)` for NLL."],
    "20": ["Label-smoothed CE = (1-eps)*nll + eps*(-mean(log_softmax))."],
    "21": ["Focal: `-(1-p_t)^gamma * log(p_t)`. p_t comes from gathering log_softmax at the target index, then exp()."],
    "22": ["Triplet: `(d(a,p) - d(a,n) + margin).clamp_min(0)`. Hardest-triplet uses cdist + per-row max-pos / min-neg."],
}


def get_hints(problem_id):
    """Return the list of hints for this problem id (looking up the parent)."""
    import re as _re
    m = _re.match(r"^(\d+)", str(problem_id))
    if not m:
        return []
    return HINTS.get(m.group(1), [])
