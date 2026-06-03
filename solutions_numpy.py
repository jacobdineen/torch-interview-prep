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

}

NUMPY_SUPPORTED = {
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
