"""Reference solutions + benchmark configurations.

Each problem id (e.g. '01a') maps to a self-contained reference implementation
(as a Python source string). The string is exec'd into a fresh namespace by
`debug_tools.show_solution` / `show_time`.

Add entries here as you write canonical solutions for more problems. To enable
the `--time` benchmark for a problem, also add a BENCHMARKS[id] entry that
specifies the function name + an input-generator function.
"""

SOLUTIONS = {
    # --- Tier 1: Tensor Fundamentals ---
    "01a": '''import torch

def make_zeros(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype)
''',
    "01b": '''import torch

def make_range(start, end, step=1):
    return torch.arange(start, end, step)
''',
    "01c": '''import torch

def to_dtype_device(x, dtype, device):
    return x.to(dtype=dtype, device=device)
''',
    "01d": '''import torch

def tensor_info(x):
    return {
        "shape": tuple(x.shape),
        "dtype": x.dtype,
        "device": x.device,
        "numel": x.numel(),
    }
''',
    "02a": '''import torch

def get_row(x, i):
    return x[i]
''',
    "02b": '''import torch

def get_diagonal(x):
    return x.diagonal()
''',
    "02c": '''import torch

def every_other_col(x):
    return x[:, ::2]
''',
    "02d": '''import torch

def select_rows(x, idx):
    return x[idx]
''',
    "02e": '''import torch

def top_left_block(x, k):
    return x[:k, :k]
''',
    "03a": '''import torch

def flatten_batch(x):
    return x.reshape(x.shape[0], -1)
''',
    "03b": '''import torch

def swap_last_two(x):
    return x.transpose(-1, -2)
''',
    "03c": '''import torch

def to_channels_last(x):
    return x.permute(0, 2, 3, 1)
''',
    "03d": '''import torch

def require_contiguous(x):
    return x.contiguous()
''',
    "04a": '''import torch

def center_rows(X):
    return X - X.mean(dim=1, keepdim=True)
''',
    "04b": '''import torch

def normalize_per_sample(X, eps=1e-6):
    mu  = X.mean(dim=1, keepdim=True)
    std = X.std(dim=1, unbiased=False, keepdim=True)
    return (X - mu) / (std + eps)
''',
    "04c": '''import torch

def pairwise_squared_distances(A, B):
    a2 = (A * A).sum(dim=1, keepdim=True)
    b2 = (B * B).sum(dim=1, keepdim=True).T
    return (a2 + b2 - 2 * A @ B.T).clamp(min=0)
''',
    "04d": '''import torch

def outer_product(u, v):
    return torch.outer(u, v)
''',
    "05a": '''import torch

def per_channel_mean(x):
    return x.mean(dim=(0, 2, 3))
''',
    "05b": '''import torch

def per_batch_max(x):
    return x.flatten(1).max(dim=1).values
''',
    "05c": '''import torch

def normalize_along_dim(x, dim):
    mu  = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True)
    return (x - mu) / std
''',
    "05d": '''import torch

def rowwise_argmax(x):
    return x.argmax(dim=-1)
''',
    "06a": '''import torch

def cat_features(a, b):
    return torch.cat([a, b], dim=1)
''',
    "06b": '''import torch

def stack_batch(items):
    return torch.stack(list(items), dim=0)
''',
    "06c": '''import torch

def split_evenly(x, k):
    return torch.chunk(x, k, dim=0)
''',
    "06d": '''import torch

def interleave(a, b):
    return torch.stack([a, b], dim=1).reshape(-1, a.shape[1])
''',
    "07a": '''import torch

def clip_negatives(x):
    return torch.where(x < 0, torch.zeros_like(x), x)
''',
    "07b": '''import torch

def count_above(x, threshold):
    return int((x > threshold).sum().item())
''',
    "07c": '''import torch

def replace_nan(x, value):
    return torch.nan_to_num(x, nan=value)
''',
    "07d": '''import torch

def masked_mean(x, mask):
    return x[mask].mean()
''',
    "08a": '''import torch

def gather_per_row(x, idx):
    return x.gather(1, idx.unsqueeze(1)).squeeze(1)
''',
    "08b": '''import torch

def one_hot(labels, num_classes):
    out = torch.zeros(labels.shape[0], num_classes)
    out.scatter_(1, labels.unsqueeze(1), 1.0)
    return out
''',
    "08c": '''import torch

def scatter_sum_rows(values, index, num_rows):
    out = torch.zeros(num_rows, values.shape[1])
    out.index_add_(0, index, values)
    return out
''',
    "09a": '''import torch

def matmul(A, B):
    return torch.einsum("ik,kj->ij", A, B)
''',
    "09b": '''import torch

def batch_matmul(A, B):
    return torch.einsum("bik,bkj->bij", A, B)
''',
    "09c": '''import torch

def bilinear(x, W, y):
    return torch.einsum("bi,ij,bj->b", x, W, y)
''',
    "09d": '''import torch

def attention_scores(q, k):
    return torch.einsum("bhid,bhjd->bhij", q, k)
''',
    "10a": '''import torch

def top_k_per_row(x, k):
    return x.topk(k, dim=-1, largest=True, sorted=True)
''',
    "10b": '''import torch

def sort_then_take_indices(x, k):
    return x.topk(k, dim=-1, largest=False, sorted=True).indices
''',
    "10c": '''import torch

def argmax_2d(x):
    flat = x.argmax()
    return torch.stack([flat // x.shape[1], flat % x.shape[1]]).long()
''',
    "10d": '''import torch

def kth_largest(x, k):
    return x.flatten().topk(k).values[-1]
''',
}


# Benchmark setups: per-id, define the function name + an inputs generator.
# The benchmark runs the function on fresh inputs from `setup()` repeatedly.
import torch as _torch


def _setup_2d():
    return (_torch.randn(64, 64),)


def _setup_2d_AB():
    return (_torch.randn(128, 256), _torch.randn(256, 64))


def _setup_pairwise():
    return (_torch.randn(512, 32), _torch.randn(512, 32))


def _setup_2d_x_with_dim():
    return (_torch.randn(64, 128), 1)


def _setup_4d():
    return (_torch.randn(8, 16, 32, 32),)


def _setup_einsum_attn():
    q = _torch.randn(2, 8, 64, 16)
    k = _torch.randn(2, 8, 64, 16)
    return (q, k)


def _setup_2d_for_topk():
    return (_torch.randn(128, 1000), 10)


def _setup_1d_for_kth():
    return (_torch.randn(10000), 100)


BENCHMARKS = {
    "04c": {"fn": "pairwise_squared_distances", "setup": _setup_pairwise, "n_runs": 25},
    "09a": {"fn": "matmul", "setup": _setup_2d_AB, "n_runs": 30},
    "09d": {"fn": "attention_scores", "setup": _setup_einsum_attn, "n_runs": 25},
    "05a": {"fn": "per_channel_mean", "setup": _setup_4d, "n_runs": 30},
    "10a": {"fn": "top_k_per_row", "setup": _setup_2d_for_topk, "n_runs": 30},
    "10d": {"fn": "kth_largest", "setup": _setup_1d_for_kth, "n_runs": 30},
}


def get_solution(pid):
    return SOLUTIONS.get(pid)


def get_benchmark(pid):
    return BENCHMARKS.get(pid)
