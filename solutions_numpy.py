"""NumPy reference solutions for problems that support a NumPy variant.

Mirrors solutions.PARENT_SOLUTIONS but in NumPy. A problem supports NumPy iff its
function is a pure tensor->tensor map (no autograd / nn / device): those are
graded against the existing torch test through np_bridge. Torch-specific problems
(e.g. anything touching .backward(), requires_grad, nn.Module, or .device) have
no NumPy variant and stay torch-only.

NUMPY_PARENTS: parent-id -> full numpy source (every numpy-supported function of
that parent). NUMPY_SUPPORTED: the set of child ids that have a numpy variant
(a parent may be only partially supported — e.g. device-dependent children are
torch-only).
"""

NUMPY_PARENTS = {

# Tier 1 — Tensor Fundamentals
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

}

# Child ids that have a numpy variant (verified by verify_numpy via the bridge).
NUMPY_SUPPORTED = {
    "02a", "02b", "02c", "02d", "02e",
}
