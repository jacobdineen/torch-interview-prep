"""NumPy reference solutions for problems that support a NumPy variant.

Mirrors solutions.PARENT_SOLUTIONS but in NumPy. A problem supports NumPy iff its
function is a pure tensor->tensor map (no autograd / nn / device): those are
graded against the existing torch test through np_bridge. Torch-specific problems
(.backward(), requires_grad, nn.Module, .device) have no NumPy variant.

NUMPY_PARENTS: parent-id -> full numpy source. NUMPY_SUPPORTED: the child ids
with a numpy variant (a parent may be only partially supported). verify_numpy.py
is the gate that each claimed variant passes its torch test through the bridge.
"""

NUMPY_PARENTS = {

"01": '''import numpy as np

def make_zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)

def make_range(start, end, step=1):
    return np.arange(start, end, step)
''',

"02": '''import numpy as np

def get_row(x, i):
    return x[i]

def get_diagonal(x):
    return np.diagonal(x)

def every_other_col(x):
    return x[:, ::2]

def select_rows(x, idx):
    return x[idx]

def top_left_block(x, k):
    return x[:k, :k]
''',

"03": '''import numpy as np

def flatten_batch(x):
    return x.reshape(x.shape[0], -1)

def swap_last_two(x):
    return np.swapaxes(x, -1, -2)

def to_channels_last(x):
    return np.transpose(x, (0, 2, 3, 1))

def require_contiguous(x):
    return np.ascontiguousarray(x)
''',

"04": '''import numpy as np

def center_rows(X):
    return X - X.mean(axis=1, keepdims=True)

def normalize_per_sample(X, eps=1e-6):
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mu) / (std + eps)

def pairwise_squared_distances(A, B):
    a2 = (A * A).sum(axis=1, keepdims=True)
    b2 = (B * B).sum(axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2 * A @ B.T, 0.0)

def outer_product(u, v):
    return np.outer(u, v)
''',

"05": '''import numpy as np

def per_channel_mean(x):
    return x.mean(axis=(0, 2, 3))

def per_batch_max(x):
    return x.reshape(x.shape[0], -1).max(axis=1)

def normalize_along_dim(x, dim):
    mu = x.mean(axis=dim, keepdims=True)
    std = x.std(axis=dim, keepdims=True)
    return (x - mu) / std

def rowwise_argmax(x):
    return x.argmax(axis=-1)
''',

"06": '''import numpy as np

def cat_features(a, b):
    return np.concatenate([a, b], axis=1)

def stack_batch(items):
    return np.stack(list(items), axis=0)

def split_evenly(x, k):
    return tuple(np.array_split(x, k, axis=0))

def interleave(a, b):
    return np.stack([a, b], axis=1).reshape(-1, a.shape[1])
''',

"07": '''import numpy as np

def clip_negatives(x):
    return np.where(x < 0, 0.0, x)

def count_above(x, threshold):
    return int((x > threshold).sum())

def replace_nan(x, value):
    return np.nan_to_num(x, nan=value)

def masked_mean(x, mask):
    return x[mask].mean()
''',

"08": '''import numpy as np

def gather_per_row(x, idx):
    return x[np.arange(x.shape[0]), idx]

def one_hot(labels, num_classes):
    return np.eye(num_classes, dtype=np.float32)[labels]

def scatter_sum_rows(values, index, num_rows):
    out = np.zeros((num_rows, values.shape[1]), dtype=values.dtype)
    np.add.at(out, index, values)
    return out
''',

"09": '''import numpy as np

def matmul(A, B):
    return np.einsum("ik,kj->ij", A, B)

def batch_matmul(A, B):
    return np.einsum("bik,bkj->bij", A, B)

def bilinear(x, W, y):
    return np.einsum("bi,ij,bj->b", x, W, y)

def attention_scores(q, k):
    return np.einsum("bhid,bhjd->bhij", q, k)
''',

"10": '''import numpy as np

def argmax_2d(x):
    flat = int(np.argmax(x))
    return np.array([flat // x.shape[1], flat % x.shape[1]])

def kth_largest(x, k):
    return np.sort(x.reshape(-1))[::-1][k - 1]
''',

"15": '''import numpy as np
from math import erf as _erf

def relu(x):
    return np.where(x > 0, x, 0.0)

def leaky_relu(x, negative_slope=0.01):
    return np.where(x > 0, x, negative_slope * x)

def sigmoid(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def tanh(x):
    e = np.exp(-2 * np.abs(x))
    out = (1 - e) / (1 + e)
    return np.where(x >= 0, out, -out)

def gelu(x):
    return (x * 0.5 * (1.0 + np.vectorize(_erf)(x / np.sqrt(2.0)))).astype(x.dtype)
''',

"16": '''import numpy as np

def logsumexp(x, dim):
    mx = np.max(x, axis=dim, keepdims=True)
    return np.log(np.sum(np.exp(x - mx), axis=dim)) + np.squeeze(mx, axis=dim)

def softmax(x, dim):
    mx = np.max(x, axis=dim, keepdims=True)
    e = np.exp(x - mx)
    return e / np.sum(e, axis=dim, keepdims=True)

def log_softmax(x, dim):
    mx = np.max(x, axis=dim, keepdims=True)
    return (x - mx) - np.log(np.sum(np.exp(x - mx), axis=dim, keepdims=True))
''',

"17": '''import numpy as np

def mse_loss(pred, target, reduction="mean"):
    d = (pred - target) ** 2
    if reduction == "mean": return d.mean()
    if reduction == "sum": return d.sum()
    return d

def huber_loss(pred, target, delta=1.0, reduction="mean"):
    r = pred - target; a = np.abs(r)
    out = np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    if reduction == "mean": return out.mean()
    if reduction == "sum": return out.sum()
    return out
''',

"18": '''import numpy as np

def bce_with_logits(logits, targets, pos_weight=None, reduction="mean"):
    max_val = np.clip(-logits, 0, None)
    log_one_plus = max_val + np.log(np.exp(-max_val) + np.exp(-logits - max_val))
    if pos_weight is None:
        loss = (1 - targets) * logits + log_one_plus
    else:
        log_sig = -log_one_plus
        log_one_minus_sig = -logits - log_one_plus
        loss = -(pos_weight * targets * log_sig + (1 - targets) * log_one_minus_sig)
    if reduction == "mean": return loss.mean()
    if reduction == "sum": return loss.sum()
    return loss
''',

"19": '''import numpy as np

def cross_entropy(logits, targets, reduction="mean", ignore_index=None):
    mx = np.max(logits, axis=-1, keepdims=True)
    log_probs = (logits - mx) - np.log(np.sum(np.exp(logits - mx), axis=-1, keepdims=True))
    mask = (targets != ignore_index) if ignore_index is not None else np.ones_like(targets, dtype=bool)
    safe = np.array(targets); safe[~mask] = 0
    nll = -log_probs[np.arange(len(safe)), safe] * mask.astype(log_probs.dtype)
    if reduction == "mean": return nll.sum() / max(int(mask.sum()), 1)
    if reduction == "sum": return nll.sum()
    return nll
''',

"20": '''import numpy as np

def label_smoothing_ce(logits, targets, smoothing=0.1, reduction="mean"):
    mx = np.max(logits, axis=-1, keepdims=True)
    log_probs = (logits - mx) - np.log(np.sum(np.exp(logits - mx), axis=-1, keepdims=True))
    nll = -log_probs[np.arange(len(targets)), targets]
    smooth = -log_probs.mean(axis=-1)
    out = (1 - smoothing) * nll + smoothing * smooth
    if reduction == "mean": return out.mean()
    if reduction == "sum": return out.sum()
    return out
''',

"21": '''import numpy as np

def focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    mx = np.max(logits, axis=-1, keepdims=True)
    log_probs = (logits - mx) - np.log(np.sum(np.exp(logits - mx), axis=-1, keepdims=True))
    log_pt = log_probs[np.arange(len(targets)), targets]
    pt = np.exp(log_pt)
    focal = -((1 - pt) ** gamma) * log_pt
    if alpha is not None:
        a = alpha[targets] if (isinstance(alpha, np.ndarray) and alpha.ndim > 0) else float(alpha)
        focal = a * focal
    if reduction == "mean": return focal.mean()
    if reduction == "sum": return focal.sum()
    return focal
''',

"22": '''import numpy as np

def triplet_margin_loss(anchor, positive, negative, margin=1.0, reduction="mean"):
    dp = np.linalg.norm(anchor - positive, axis=-1)
    dn = np.linalg.norm(anchor - negative, axis=-1)
    loss = np.maximum(dp - dn + margin, 0.0)
    if reduction == "mean": return loss.mean()
    if reduction == "sum": return loss.sum()
    return loss

def hardest_triplet_loss(embeddings, labels, margin=1.0):
    N = embeddings.shape[0]
    d = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)
    same = labels[None, :] == labels[:, None]
    pos_mask = same & (~np.eye(N, dtype=bool))
    hp = np.where(pos_mask, d, -np.inf).max(axis=-1)
    hn = np.where(same, np.inf, d).min(axis=-1)
    return np.maximum(hp - hn + margin, 0.0).mean()
''',

}

NUMPY_SUPPORTED = {
    "15a",
    "15b",
    "15c",
    "15d",
    "15e",
    "15f",
    "16a",
    "16b",
    "16c",
    "16d",
    "16e",
    "16f",
    "17a",
    "17b",
    "17c",
    "17d",
    "17e",
    "18a",
    "18b",
    "18c",
    "19a",
    "19b",
    "19c",
    "20a",
    "20b",
    "20c",
    "20d",
    "21a",
    "21b",
    "21c",
    "21d",
    "22a",
    "22b",
    "22c",
    "22d",
    "22e",
    "01a", "01b",
    "02a", "02b", "02c", "02d", "02e",
    "03a", "03b", "03c", "03d",
    "04a", "04b", "04c", "04d",
    "05a", "05b", "05c", "05d",
    "06a", "06b", "06c", "06d",
    "07a", "07b", "07c", "07d",
    "08a", "08b", "08c",
    "09a", "09b", "09c", "09d",
    "10c", "10d",
}
