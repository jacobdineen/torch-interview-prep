"""Reference solutions + benchmark configurations.

Each PARENT (e.g. '04') has one canonical implementation containing every
function/class the parent's tests reference. For a CHILD id (e.g. '04b'),
`get_solution` looks up the child's stub, identifies which symbols it defines,
and returns a slimmed reference containing only those.

Add/edit a parent entry once and every child of that parent picks it up.
"""
import ast
import glob
import os
import re

import torch as _torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_SOLUTIONS = {

# ============================================================================
# TIER 1 — Tensor Fundamentals
# ============================================================================

"01": '''import torch

def make_zeros(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype)

def make_range(start, end, step=1):
    return torch.arange(start, end, step)

def to_dtype_device(x, dtype, device):
    return x.to(dtype=dtype, device=device)

def tensor_info(x):
    return {"shape": tuple(x.shape), "dtype": x.dtype, "device": x.device, "numel": x.numel()}
''',

"02": '''import torch

def get_row(x, i):
    return x[i]

def get_diagonal(x):
    return x.diagonal()

def every_other_col(x):
    return x[:, ::2]

def select_rows(x, idx):
    return x[idx]

def top_left_block(x, k):
    return x[:k, :k]
''',

"03": '''import torch

def flatten_batch(x):
    return x.reshape(x.shape[0], -1)

def swap_last_two(x):
    return x.transpose(-1, -2)

def to_channels_last(x):
    return x.permute(0, 2, 3, 1)

def require_contiguous(x):
    return x.contiguous()
''',

"04": '''import torch

def center_rows(X):
    return X - X.mean(dim=1, keepdim=True)

def normalize_per_sample(X, eps=1e-6):
    mu = X.mean(dim=1, keepdim=True)
    std = X.std(dim=1, unbiased=False, keepdim=True)
    return (X - mu) / (std + eps)

def pairwise_squared_distances(A, B):
    a2 = (A * A).sum(dim=1, keepdim=True)
    b2 = (B * B).sum(dim=1, keepdim=True).T
    return (a2 + b2 - 2 * A @ B.T).clamp(min=0)

def outer_product(u, v):
    return torch.outer(u, v)
''',

"05": '''import torch

def per_channel_mean(x):
    return x.mean(dim=(0, 2, 3))

def per_batch_max(x):
    return x.flatten(1).max(dim=1).values

def normalize_along_dim(x, dim):
    mu = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True)
    return (x - mu) / std

def rowwise_argmax(x):
    return x.argmax(dim=-1)
''',

"06": '''import torch

def cat_features(a, b):
    return torch.cat([a, b], dim=1)

def stack_batch(items):
    return torch.stack(list(items), dim=0)

def split_evenly(x, k):
    return torch.chunk(x, k, dim=0)

def interleave(a, b):
    return torch.stack([a, b], dim=1).reshape(-1, a.shape[1])
''',

"07": '''import torch

def clip_negatives(x):
    return torch.where(x < 0, torch.zeros_like(x), x)

def count_above(x, threshold):
    return int((x > threshold).sum().item())

def replace_nan(x, value):
    return torch.nan_to_num(x, nan=value)

def masked_mean(x, mask):
    return x[mask].mean()
''',

"08": '''import torch

def gather_per_row(x, idx):
    return x.gather(1, idx.unsqueeze(1)).squeeze(1)

def one_hot(labels, num_classes):
    out = torch.zeros(labels.shape[0], num_classes)
    out.scatter_(1, labels.unsqueeze(1), 1.0)
    return out

def scatter_sum_rows(values, index, num_rows):
    out = torch.zeros(num_rows, values.shape[1])
    out.index_add_(0, index, values)
    return out
''',

"09": '''import torch

def matmul(A, B):
    return torch.einsum("ik,kj->ij", A, B)

def batch_matmul(A, B):
    return torch.einsum("bik,bkj->bij", A, B)

def bilinear(x, W, y):
    return torch.einsum("bi,ij,bj->b", x, W, y)

def attention_scores(q, k):
    return torch.einsum("bhid,bhjd->bhij", q, k)
''',

"10": '''import torch

def top_k_per_row(x, k):
    return x.topk(k, dim=-1, largest=True, sorted=True)

def sort_then_take_indices(x, k):
    return x.topk(k, dim=-1, largest=False, sorted=True).indices

def argmax_2d(x):
    flat = x.argmax()
    return torch.stack([flat // x.shape[1], flat % x.shape[1]]).long()

def kth_largest(x, k):
    return x.flatten().topk(k).values[-1]
''',

# ============================================================================
# TIER 2 — Autograd & Linear
# ============================================================================

"11": '''import torch

def grad_of_sum_of_squares(x):
    x = x.detach().clone().requires_grad_(True)
    (x ** 2).sum().backward()
    return x.grad

def jacobian_diag(f, x):
    x = x.detach().clone().requires_grad_(True)
    y = f(x)
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]

def second_derivative(x):
    x = x.detach().clone().requires_grad_(True)
    L = (x ** 4).sum()
    g = torch.autograd.grad(L, x, create_graph=True)[0]
    return torch.autograd.grad(g.sum(), x)[0]
''',

"12": '''import torch
from torch.autograd import Function

class StableSoftplus(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0) + torch.log1p(torch.exp(-x.abs()))

    @staticmethod
    def backward(ctx, go):
        (x,) = ctx.saved_tensors
        return go * torch.sigmoid(x)

def stable_softplus(x):
    return StableSoftplus.apply(x)
''',

"13": '''import torch

def param_norm(model):
    with torch.no_grad():
        return torch.cat([p.flatten() for p in model.parameters()]).norm()

def stop_gradient_at(x, threshold):
    return torch.where(x > threshold, x.detach(), x)

def is_leaf_and_requires_grad(t):
    return (t.is_leaf, t.requires_grad)
''',

"14": '''import math
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out
''',

"15": '''import math
import torch

def relu(x):
    return torch.where(x > 0, x, torch.zeros_like(x))

def leaky_relu(x, negative_slope=0.01):
    return torch.where(x > 0, x, negative_slope * x)

def sigmoid(x):
    return torch.where(x >= 0,
                       1.0 / (1.0 + torch.exp(-x)),
                       torch.exp(x) / (1.0 + torch.exp(x)))

def tanh(x):
    e = torch.exp(-2 * x.abs())
    out = (1 - e) / (1 + e)
    return torch.where(x >= 0, out, -out)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
''',

"16": '''import torch

def logsumexp(x, dim):
    mx = x.max(dim=dim, keepdim=True).values
    return (x - mx).exp().sum(dim=dim).log() + mx.squeeze(dim)

def softmax(x, dim):
    mx = x.max(dim=dim, keepdim=True).values
    e = (x - mx).exp()
    return e / e.sum(dim=dim, keepdim=True)

def log_softmax(x, dim):
    mx = x.max(dim=dim, keepdim=True).values
    return (x - mx) - (x - mx).exp().sum(dim=dim, keepdim=True).log()
''',

# ============================================================================
# TIER 3 — Loss Functions
# ============================================================================

"17": '''import torch

def mse_loss(pred, target, reduction="mean"):
    d = (pred - target) ** 2
    if reduction == "mean": return d.mean()
    if reduction == "sum":  return d.sum()
    return d

def huber_loss(pred, target, delta=1.0, reduction="mean"):
    r = pred - target
    abs_r = r.abs()
    out = torch.where(abs_r <= delta, 0.5 * r * r, delta * (abs_r - 0.5 * delta))
    if reduction == "mean": return out.mean()
    if reduction == "sum":  return out.sum()
    return out
''',

"18": '''import torch

def bce_with_logits(logits, targets, pos_weight=None, reduction="mean"):
    max_val = (-logits).clamp(min=0)
    log_one_plus = max_val + torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val))
    if pos_weight is None:
        loss = (1 - targets) * logits + log_one_plus
    else:
        log_sig = -log_one_plus
        log_one_minus_sig = -logits - log_one_plus
        loss = -(pos_weight * targets * log_sig + (1 - targets) * log_one_minus_sig)
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss
''',

"19": '''import torch

def cross_entropy(logits, targets, reduction="mean", ignore_index=None):
    mx = logits.max(dim=-1, keepdim=True).values
    log_probs = (logits - mx) - (logits - mx).exp().sum(dim=-1, keepdim=True).log()
    if ignore_index is not None:
        mask = targets != ignore_index
    else:
        mask = torch.ones_like(targets, dtype=torch.bool)
    safe = targets.clone()
    safe[~mask] = 0
    nll = -log_probs.gather(1, safe.unsqueeze(1)).squeeze(1)
    nll = nll * mask.float()
    if reduction == "mean":
        return nll.sum() / mask.float().sum().clamp_min(1)
    if reduction == "sum":
        return nll.sum()
    return nll
''',

"20": '''import torch
import torch.nn.functional as F

def label_smoothing_ce(logits, targets, smoothing=0.1, reduction="mean"):
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    out = (1 - smoothing) * nll + smoothing * smooth
    if reduction == "mean": return out.mean()
    if reduction == "sum":  return out.sum()
    return out
''',

"21": '''import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    focal = -((1 - pt) ** gamma) * log_pt
    if alpha is not None:
        if torch.is_tensor(alpha) and alpha.dim() > 0:
            a = alpha[targets]
        else:
            a = float(alpha)
        focal = a * focal
    if reduction == "mean": return focal.mean()
    if reduction == "sum":  return focal.sum()
    return focal
''',

"22": '''import torch

def triplet_margin_loss(anchor, positive, negative, margin=1.0, reduction="mean"):
    dp = (anchor - positive).norm(dim=-1)
    dn = (anchor - negative).norm(dim=-1)
    loss = (dp - dn + margin).clamp_min(0.0)
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss

def hardest_triplet_loss(embeddings, labels, margin=1.0):
    N = embeddings.shape[0]
    d = torch.cdist(embeddings, embeddings)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same & (~torch.eye(N, dtype=torch.bool))
    hp = d.masked_fill(~pos_mask, float("-inf")).max(dim=-1).values
    hn = d.masked_fill(same, float("inf")).min(dim=-1).values
    return (hp - hn + margin).clamp_min(0.0).mean()
''',

# ============================================================================
# TIER 4 — Normalization
# ============================================================================

"23": '''import torch
import torch.nn as nn

class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.weight = None
            self.bias = None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(
                    x.var(dim=0, unbiased=True), alpha=self.momentum)
            out = (x - mean) / torch.sqrt(var + self.eps)
        else:
            out = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        if self.affine:
            out = out * self.weight + self.bias
        return out
''',

"24": '''import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.shape))
            self.bias = nn.Parameter(torch.zeros(self.shape))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        dims = tuple(range(-len(self.shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out
''',

"25": '''import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.shape))

    def forward(self, x):
        dims = tuple(range(-len(self.shape), 0))
        rms = torch.sqrt(x.pow(2).mean(dim=dims, keepdim=True) + self.eps)
        return x / rms * self.weight
''',

"26": '''import torch
import torch.nn as nn

class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.G = num_groups
        self.C = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        N = x.shape[0]
        spatial = x.shape[2:]
        xr = x.reshape(N, self.G, self.C // self.G, *spatial)
        dims = tuple(range(2, xr.dim()))
        mean = xr.mean(dim=dims, keepdim=True)
        var = xr.var(dim=dims, unbiased=False, keepdim=True)
        xr = (xr - mean) / torch.sqrt(var + self.eps)
        out = xr.reshape(N, self.C, *spatial)
        if self.affine:
            shape = (1, self.C) + (1,) * len(spatial)
            out = out * self.weight.view(shape) + self.bias.view(shape)
        return out
''',

"27": '''import torch
import torch.nn as nn

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * mask / (1.0 - self.p)
''',

# ============================================================================
# TIER 5 — Embeddings & Init
# ============================================================================

"28": '''import torch
import torch.nn as nn

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.padding_idx = padding_idx
        w = torch.randn(num_embeddings, embedding_dim)
        if padding_idx is not None:
            with torch.no_grad():
                w[padding_idx].zero_()
        self.weight = nn.Parameter(w)
        if padding_idx is not None:
            pidx = padding_idx
            self.weight.register_hook(lambda g: g.clone().index_fill_(0, torch.tensor([pidx]), 0.0))

    def forward(self, idx):
        return self.weight[idx]
''',

"29": '''import math
import torch

def xavier_normal_(tensor, gain=1.0):
    fo, fi = tensor.shape
    std = gain * math.sqrt(2.0 / (fi + fo))
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor

def kaiming_normal_(tensor, mode="fan_in", nonlinearity="relu"):
    fo, fi = tensor.shape
    fan = fi if mode == "fan_in" else fo
    gain = math.sqrt(2.0) if nonlinearity == "relu" else 1.0
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor
''',

# ============================================================================
# TIER 6 — Optimizers
# ============================================================================

"30": '''import torch

class MySGD:
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.mu = momentum
        self.wd = weight_decay
        self.buf = {id(p): None for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            g = p.grad
            if self.wd: g = g + self.wd * p
            if self.mu:
                b = self.buf[id(p)]
                b = g.clone() if b is None else self.mu * b + g
                self.buf[id(p)] = b
                g = b
            p.add_(g, alpha=-self.lr)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none: p.grad = None
            elif p.grad is not None: p.grad.zero_()
''',

"31": '''import torch

class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.state = {id(p): {"step": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p)}
                       for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            g = p.grad + (self.wd * p if self.wd else 0)
            s = self.state[id(p)]
            s["step"] += 1
            s["m"] = self.b1 * s["m"] + (1 - self.b1) * g
            s["v"] = self.b2 * s["v"] + (1 - self.b2) * g * g
            mh = s["m"] / (1 - self.b1 ** s["step"])
            vh = s["v"] / (1 - self.b2 ** s["step"])
            p.add_(-self.lr * mh / (vh.sqrt() + self.eps))

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none: p.grad = None
            elif p.grad is not None: p.grad.zero_()
''',

"32": '''import torch

class MyAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.state = {id(p): {"step": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p)}
                       for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            g = p.grad
            s = self.state[id(p)]
            s["step"] += 1
            if self.wd:
                p.mul_(1 - self.lr * self.wd)
            s["m"] = self.b1 * s["m"] + (1 - self.b1) * g
            s["v"] = self.b2 * s["v"] + (1 - self.b2) * g * g
            mh = s["m"] / (1 - self.b1 ** s["step"])
            vh = s["v"] / (1 - self.b2 ** s["step"])
            p.add_(-self.lr * mh / (vh.sqrt() + self.eps))

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none: p.grad = None
            elif p.grad is not None: p.grad.zero_()
''',

"33": '''import math
import torch

def linear_warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    if torch.is_tensor(step):
        step = step.float()
        warmup_t = torch.tensor(float(warmup_steps))
        total_t = torch.tensor(float(total_steps))
        warm = base_lr * step / warmup_t
        progress = ((step - warmup_t) / max(1.0, float(total_steps - warmup_steps))).clamp(0.0, 1.0)
        cos = min_lr + 0.5 * (base_lr - min_lr) * (1 + torch.cos(math.pi * progress))
        out = torch.where(step < warmup_t, warm, cos)
        out = torch.where(step > total_t, torch.full_like(out, float(min_lr)), out)
        return out
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
''',

"34": '''import torch

def clip_grad_norm_(params, max_norm, eps=1e-6):
    params = [p for p in params if p.grad is not None]
    total = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params))
    if total > max_norm:
        scale = max_norm / (total + eps)
        for p in params:
            p.grad.detach().mul_(scale)
    return total
''',

# ============================================================================
# TIER 7 — Training Infrastructure
# ============================================================================

"35": '''import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, num_samples, dim, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(num_samples, dim, generator=g)
        self.y = (self.x ** 2).sum(dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def epoch_mean(loader):
    total, n = 0.0, 0
    for _, y in loader:
        total += y.sum().item()
        n += y.shape[0]
    return torch.tensor(total / n)
''',

"36": '''import torch

def pad_collate(batch, pad_value=0):
    seqs = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    T = max(s.shape[0] for s in seqs)
    B = len(seqs)
    padded = torch.full((B, T), pad_value, dtype=seqs[0].dtype)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
        mask[i, :s.shape[0]] = True
    return padded, mask, labels
''',

"37": '''import torch
import torch.nn as nn

def build_mlp(in_dim, hidden_dims, out_dim):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

def train_one_epoch(model, optimizer, x, y, batch_size=32):
    losses = []
    idx = torch.randperm(x.shape[0])
    for s in range(0, x.shape[0], batch_size):
        b = idx[s:s + batch_size]
        optimizer.zero_grad()
        loss = ((model(x[b]) - y[b]) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

def fit(model, x, y, epochs, lr, batch_size=32):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = 0.0
    for _ in range(epochs):
        last = train_one_epoch(model, opt, x, y, batch_size)
    return last
''',

"38": '''def train_step_with_accumulation(model, optimizer, micro_batches, loss_fn, accum_steps):
    optimizer.zero_grad()
    total = 0.0
    for x, y in micro_batches:
        l = loss_fn(model, x, y)
        (l / accum_steps).backward()
        total += l.item()
    optimizer.step()
    return total
''',

"39": '''import torch

def forward_autocast(model, x, dtype=torch.bfloat16, device_type="cpu"):
    with torch.autocast(device_type=device_type, dtype=dtype):
        return model(x)

def amp_train_step(model, optimizer, x, y, dtype=torch.bfloat16, device_type="cpu"):
    optimizer.zero_grad()
    with torch.autocast(device_type=device_type, dtype=dtype):
        out = model(x)
        loss = ((out - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    return loss.item()
''',

"40": '''def collect_activations(model, x, layer_names):
    named = dict(model.named_modules())
    handles = []
    out = {}
    def make_hook(n):
        def hook(_mod, _inp, output):
            out[n] = output
        return hook
    try:
        for n in layer_names:
            handles.append(named[n].register_forward_hook(make_hook(n)))
        model(x)
    finally:
        for h in handles:
            h.remove()
    return out
''',

"41": '''import torch

def save_checkpoint(path, model, optimizer, epoch, best_metric):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }, path)

def load_checkpoint(path, model, optimizer):
    ck = torch.load(path, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    return ck["epoch"], ck["best_metric"]
''',

"42": '''import copy
import torch

@torch.no_grad()
def ema_update(ema_model, model, decay=0.999):
    for p_e, p_m in zip(ema_model.parameters(), model.parameters()):
        p_e.mul_(decay).add_(p_m.detach(), alpha=1 - decay)
    for b_e, b_m in zip(ema_model.buffers(), model.buffers()):
        b_e.copy_(b_m)

class EMAWrapper:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self):
        ema_update(self.ema_model, self.model, self.decay)

    def ema_state_dict(self):
        return self.ema_model.state_dict()
''',

"43": '''import torch

@torch.no_grad()
def average_gradients(models):
    param_lists = [list(m.parameters()) for m in models]
    for params_tuple in zip(*param_lists):
        grads = [p.grad for p in params_tuple]
        nones = [g is None for g in grads]
        if all(nones): continue
        if any(nones):
            raise ValueError("Some replicas have None grad, others don't.")
        mean = torch.stack(grads, dim=0).mean(dim=0)
        for p in params_tuple:
            p.grad.copy_(mean)
''',

# ============================================================================
# TIER 8 — Convolutions
# ============================================================================

"44": '''import torch
import torch.nn.functional as F

def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    B, Cin, H, W = x.shape
    Cout, _, kH, kW = weight.shape
    cols = F.unfold(x, kernel_size=(kH, kW), padding=padding, stride=stride)
    out = weight.reshape(Cout, -1) @ cols
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    Hout = (H + 2 * padding - kH) // stride + 1
    Wout = (W + 2 * padding - kW) // stride + 1
    return out.reshape(B, Cout, Hout, Wout)
''',

"45": '''import torch
import torch.nn.functional as F

def my_max_pool2d(x, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size
    B, C, H, W = x.shape
    if padding > 0:
        x = F.pad(x, [padding] * 4, value=float("-inf"))
    cols = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    return cols.contiguous().view(B, C, cols.shape[2], cols.shape[3], -1).max(dim=-1).values
''',

"46": '''def conv_out_shape(h_in, kernel, stride=1, padding=0, dilation=1):
    return (h_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

def conv_chain_shape(h_in, layers):
    h = h_in
    for l in layers:
        h = conv_out_shape(h, l["kernel"], l.get("stride", 1),
                            l.get("padding", 0), l.get("dilation", 1))
    return h

def transposed_conv_out_shape(h_in, kernel, stride=1, padding=0, output_padding=0, dilation=1):
    return (h_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
''',

"47": '''import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.b2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.b2(self.b1(x)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
''',

# ============================================================================
# TIER 9 — Sequences
# ============================================================================

"48": '''import math
import torch
import torch.nn as nn

class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        nn.init.kaiming_uniform_(self.W_ih, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_hh, a=math.sqrt(5))

    def forward(self, x, h_prev):
        return torch.tanh(x @ self.W_ih.t() + self.b_ih + h_prev @ self.W_hh.t() + self.b_hh)

def run_rnn(cell, X, h0):
    outs = []
    h = h0
    for t in range(X.shape[1]):
        h = cell(X[:, t], h)
        outs.append(h)
    return torch.stack(outs, dim=1), h
''',

"49": '''import math
import torch
import torch.nn as nn

class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        nn.init.kaiming_uniform_(self.W_ih, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_hh, a=math.sqrt(5))

    def forward(self, x, state):
        h_p, c_p = state
        gates = x @ self.W_ih.t() + self.b_ih + h_p @ self.W_hh.t() + self.b_hh
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c_p + i * g
        return o * torch.tanh(c), c
''',

"50": '''import torch

def pad_and_mask(seqs):
    lengths = torch.tensor([s.shape[0] for s in seqs])
    T = int(lengths.max().item())
    D = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), T, D)
    mask = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
        mask[i, :s.shape[0]] = True
    return padded, mask, lengths

def masked_mean_pool(padded, mask):
    m_ = mask.unsqueeze(-1).float()
    return (padded * m_).sum(dim=1) / m_.sum(dim=1).clamp_min(1)

def last_real_state(padded, lengths):
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, padded.shape[-1])
    return padded.gather(1, idx).squeeze(1)
''',

"51": '''import torch

def encode_packed(lstm, padded, lengths):
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
    out_packed, (h_n, c_n) = lstm(packed)
    out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=padded.shape[1])
    sent = torch.cat([h_n[0], h_n[1]], dim=-1)
    return out, sent
''',

# ============================================================================
# TIER 10 — Attention Primitives
# ============================================================================

"52": '''import torch

def sinusoidal_positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).float()
    i = torch.arange(d_model // 2).float()
    div = torch.pow(10000.0, 2 * i / d_model)
    angles = pos.unsqueeze(1) / div.unsqueeze(0)
    out = torch.zeros(seq_len, d_model)
    out[:, 0::2] = torch.sin(angles)
    out[:, 1::2] = torch.cos(angles)
    return out

def add_positional_encoding(x):
    return x + sinusoidal_positional_encoding(x.shape[1], x.shape[2]).to(x.dtype).to(x.device).unsqueeze(0)
''',

"53": '''import math
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn
''',

"54": '''import math
import torch

def causal_mask(T, device=None):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

def apply_causal_mask(scores):
    T = scores.shape[-1]
    return scores.masked_fill(causal_mask(T, scores.device), float("-inf"))

def causal_attention(q, k, v):
    d = q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    scores = apply_causal_mask(scores)
    return torch.softmax(scores, dim=-1) @ v
''',

"55": '''import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, bias=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.H = num_heads
        self.D = d_model // num_heads
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.H, self.D).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.H, self.D).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.H, self.D).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.D)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[None, None, :, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(out)
''',

"56": '''import torch

def rope_freqs(seq_len, d, base=10000.0, device=None):
    i = torch.arange(d // 2, device=device).float()
    theta = 1.0 / (base ** (2 * i / d))
    pos = torch.arange(seq_len, device=device).float()
    angles = pos.unsqueeze(1) * theta.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    new1 = x1 * cos - x2 * sin
    new2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = new1
    out[..., 1::2] = new2
    return out
''',

"57": '''import torch

def alibi_slopes(num_heads):
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / num_heads) for i in range(num_heads)])

def alibi_bias(num_heads, seq_len, causal=True):
    s = alibi_slopes(num_heads)
    i = torch.arange(seq_len).view(-1, 1)
    j = torch.arange(seq_len).view(1, -1)
    dist = (i - j).abs().float()
    return -s.view(num_heads, 1, 1) * dist.unsqueeze(0)
''',

"58": '''import math
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_groups, bias=False):
        super().__init__()
        assert d_model % num_q_heads == 0
        assert num_q_heads % num_kv_groups == 0
        self.H = num_q_heads
        self.G = num_kv_groups
        self.D = d_model // num_q_heads
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, num_q_heads * self.D, bias=bias)
        self.w_k = nn.Linear(d_model, num_kv_groups * self.D, bias=bias)
        self.w_v = nn.Linear(d_model, num_kv_groups * self.D, bias=bias)
        self.w_o = nn.Linear(num_q_heads * self.D, d_model, bias=bias)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.H, self.D).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.G, self.D).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.G, self.D).transpose(1, 2)
        rep = self.H // self.G
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.D)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[None, None, :, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.H * self.D)
        return self.w_o(out)
''',

# ============================================================================
# TIER 11 — Transformer Architecture
# ============================================================================

"59": '''import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x
''',

"60": '''import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
''',

"61": '''import torch
import torch.nn as nn

class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x
''',

"62": '''import math
import torch
import torch.nn as nn

class CachedMHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.H = num_heads
        self.D = d_model // num_heads
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x_new, cache=None):
        B, Tn, _ = x_new.shape
        q = self.w_q(x_new).view(B, Tn, self.H, self.D).transpose(1, 2)
        k_new = self.w_k(x_new).view(B, Tn, self.H, self.D).transpose(1, 2)
        v_new = self.w_v(x_new).view(B, Tn, self.H, self.D).transpose(1, 2)
        if cache is None:
            k_all, v_all, Tpast = k_new, v_new, 0
        else:
            k_all = torch.cat([cache["k"], k_new], dim=2)
            v_all = torch.cat([cache["v"], v_new], dim=2)
            Tpast = cache["k"].shape[2]
        Ttot = Tpast + Tn
        scores = (q @ k_all.transpose(-1, -2)) / math.sqrt(self.D)
        qi = torch.arange(Tpast, Ttot).view(-1, 1)
        ki = torch.arange(Ttot).view(1, -1)
        scores = scores.masked_fill((ki > qi)[None, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v_all
        out = out.transpose(1, 2).contiguous().view(B, Tn, self.d_model)
        return self.w_o(out), {"k": k_all, "v": v_all}
''',

"63": '''import math
import torch

def sliding_window_causal_mask(T, window_size):
    i = torch.arange(T).view(-1, 1)
    j = torch.arange(T).view(1, -1)
    return ~((j <= i) & (i - j < window_size))

def sliding_window_attention(q, k, v, window_size):
    T = q.shape[-2]
    D = q.shape[-1]
    mask = sliding_window_causal_mask(T, window_size).to(q.device)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
''',

"64": '''import torch
import torch.nn as nn

class TiedLM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        h = self.embed(token_ids)
        return h @ self.embed.weight.t()
''',

# ============================================================================
# TIER 12 — Efficient Attention
# ============================================================================

"65": '''import math
import torch

def flash_attention_tiled(q, k, v, block_size_q=32, block_size_k=32):
    B, H, T, D = q.shape
    Tk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)
    for q_start in range(0, T, block_size_q):
        q_end = min(q_start + block_size_q, T)
        qb = q[..., q_start:q_end, :]
        Bq = q_end - q_start
        m_run = torch.full((B, H, Bq, 1), float("-inf"), device=q.device, dtype=q.dtype)
        l_run = torch.zeros((B, H, Bq, 1), device=q.device, dtype=q.dtype)
        o = torch.zeros((B, H, Bq, D), device=q.device, dtype=q.dtype)
        for k_start in range(0, Tk, block_size_k):
            k_end = min(k_start + block_size_k, Tk)
            kb = k[..., k_start:k_end, :]
            vb = v[..., k_start:k_end, :]
            s = (qb @ kb.transpose(-1, -2)) * scale
            m_new = torch.maximum(m_run, s.max(dim=-1, keepdim=True).values)
            p = torch.exp(s - m_new)
            alpha = torch.exp(m_run - m_new)
            o = o * alpha + p @ vb
            l_run = l_run * alpha + p.sum(dim=-1, keepdim=True)
            m_run = m_new
        out[..., q_start:q_end, :] = o / l_run
    return out
''',

# ============================================================================
# TIER 13 — Generation
# ============================================================================

"66": '''import torch

def top_k_filter(logits, k):
    v, _ = logits.topk(k, dim=-1)
    thr = v[..., -1:].expand_as(logits)
    return torch.where(logits >= thr, logits, torch.full_like(logits, float("-inf")))

def top_p_filter(logits, p):
    sl, si = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sl, dim=-1)
    cum = probs.cumsum(dim=-1)
    keep = cum <= p
    keep[..., 0] = True
    filt = torch.where(keep, sl, torch.full_like(sl, float("-inf")))
    out = torch.empty_like(logits)
    out.scatter_(-1, si, filt)
    return out

def sample_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    l = logits / max(temperature, 1e-12)
    if top_k is not None:
        l = top_k_filter(l, top_k)
    if top_p is not None:
        l = top_p_filter(l, top_p)
    probs = torch.softmax(l, dim=-1)
    flat = probs.view(-1, probs.shape[-1])
    return torch.multinomial(flat, 1, generator=generator).view(*probs.shape[:-1])
''',

"67": '''import torch

def apply_repetition_penalty(logits, generated_ids, penalty=1.0):
    out = logits.clone()
    if penalty == 1.0:
        return out
    B = out.shape[0]
    for b in range(B):
        for tok in generated_ids[b].tolist():
            v = out[b, tok]
            out[b, tok] = v / penalty if v > 0 else v * penalty
    return out

def penalize_unique(logits, generated_ids, penalty=1.0):
    out = logits.clone()
    if penalty == 1.0:
        return out
    B = out.shape[0]
    for b in range(B):
        for tok in set(generated_ids[b].tolist()):
            v = out[b, tok]
            out[b, tok] = v / penalty if v > 0 else v * penalty
    return out
''',

"68": '''import torch

def beam_search(step_fn, start_tokens, beam_size, max_steps, end_token, length_penalty=0.0):
    beams = [(start_tokens.clone(), 0.0)]
    finished = []
    for _ in range(max_steps):
        if not beams: break
        cand = []
        for tokens, lp in beams:
            logp = step_fn(tokens)
            topv, topi = logp.topk(beam_size)
            for v, idx in zip(topv.tolist(), topi.tolist()):
                cand.append((torch.cat([tokens, torch.tensor([idx])]), lp + v, idx))
        cand.sort(key=lambda c: -c[1])
        new = []
        for tok, lp, idx in cand:
            if idx == end_token:
                score = lp / (len(tok) ** length_penalty) if length_penalty else lp
                finished.append((tok, score))
            else:
                new.append((tok, lp))
            if len(new) >= beam_size: break
        beams = new
        if not beams: break
    if not finished:
        tok, lp = max(beams, key=lambda b: b[1])
        return tok, lp / (len(tok) ** length_penalty) if length_penalty else lp
    finished.sort(key=lambda c: -c[1])
    return finished[0]
''',

"69": '''import torch

def greedy_generate(step_fn, prompt_ids, max_new_tokens, eos_token=None):
    ids = prompt_ids.clone()
    cache = None
    B = prompt_ids.shape[0]
    finished = torch.zeros(B, dtype=torch.bool)
    logits, cache = step_fn(prompt_ids, cache)
    next_tok = logits[:, -1].argmax(dim=-1)
    for _ in range(max_new_tokens):
        if eos_token is not None:
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_token), next_tok)
            finished = finished | (next_tok == eos_token)
        ids = torch.cat([ids, next_tok.unsqueeze(1)], dim=1)
        if eos_token is not None and finished.all():
            break
        logits, cache = step_fn(next_tok.unsqueeze(1), cache)
        next_tok = logits[:, -1].argmax(dim=-1)
    while eos_token is not None and ids.shape[1] < prompt_ids.shape[1] + max_new_tokens:
        ids = torch.cat([ids, torch.full((B, 1), eos_token, dtype=ids.dtype)], dim=1)
    return ids
''',

"70": '''import torch

def speculative_decode(target_step, draft_step, prompt_ids, max_new_tokens, K=4):
    ids = prompt_ids.clone()
    target_len_start = prompt_ids.shape[1]
    while ids.shape[1] - target_len_start < max_new_tokens:
        draft_seq = ids.clone()
        for _ in range(K):
            logits = draft_step(draft_seq)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            draft_seq = torch.cat([draft_seq, nxt], dim=1)
        target_logits = target_step(draft_seq)
        T0 = ids.shape[1]
        accepted = 0
        for i in range(K):
            target_argmax = target_logits[:, T0 - 1 + i].argmax(dim=-1)
            draft_token = draft_seq[:, T0 + i]
            if torch.equal(target_argmax, draft_token):
                accepted += 1
            else:
                break
        new_tokens = draft_seq[:, T0:T0 + accepted]
        if accepted < K:
            bonus = target_logits[:, T0 - 1 + accepted].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, new_tokens, bonus], dim=1)
        else:
            bonus = target_logits[:, T0 - 1 + K].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, new_tokens, bonus], dim=1)
    return ids[:, :target_len_start + max_new_tokens]
''',

# ============================================================================
# TIER 14 — Auxiliary Losses
# ============================================================================

"71": '''import torch
import torch.nn.functional as F

def info_nce_loss(a, b, tau=0.1):
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    logits = a_n @ b_n.T / tau
    labels = torch.arange(a.shape[0])
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
''',

"72": '''import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5):
    ce = F.cross_entropy(student_logits, targets)
    t_logp = F.log_softmax(teacher_logits / T, dim=-1)
    s_logp = F.log_softmax(student_logits / T, dim=-1)
    kld = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1).mean()
    return alpha * ce + (1 - alpha) * (T * T) * kld
''',

"73": '''import torch

def reparameterize(mu, logvar, generator=None):
    eps = torch.randn(mu.shape, generator=generator)
    return mu + torch.exp(0.5 * logvar) * eps

def kl_divergence_standard_normal(mu, logvar, reduction="batchmean"):
    per = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
    if reduction == "none":      return per
    if reduction == "sum":       return per.sum()
    if reduction == "mean":      return per.mean()
    if reduction == "batchmean": return per.sum() / mu.shape[0]
    raise ValueError(reduction)
''',

"74": '''import torch

def gumbel_noise(shape, eps=1e-20, generator=None):
    u = torch.rand(shape, generator=generator)
    return -torch.log(-torch.log(u + eps) + eps)

def gumbel_softmax(logits, tau=1.0, hard=False, generator=None):
    g = gumbel_noise(logits.shape, generator=generator)
    y_soft = torch.softmax((logits + g) / tau, dim=-1)
    if hard:
        idx = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft
''',

# ============================================================================
# TIER 15 — LLM Specifics & Capstone
# ============================================================================

"75": '''import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear, r, alpha=1.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * x @ self.lora_A.t() @ self.lora_B.t()

def merge_lora(lora_linear):
    new = nn.Linear(lora_linear.base.in_features, lora_linear.base.out_features,
                    bias=(lora_linear.base.bias is not None))
    with torch.no_grad():
        new.weight.copy_(lora_linear.base.weight
                          + lora_linear.scale * (lora_linear.lora_B @ lora_linear.lora_A))
        if lora_linear.base.bias is not None:
            new.bias.copy_(lora_linear.base.bias)
    return new
''',

"76": '''import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedSequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for b in self.blocks:
            if self.training and x.requires_grad:
                x = checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return x
''',

"77": '''import torch
import torch.nn as nn
import torch.nn.functional as F

class _Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.0):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        return x @ self.tok_embed.weight.t()

def causal_lm_loss(logits, targets, ignore_index=-100):
    V = logits.shape[-1]
    return F.cross_entropy(logits[:, :-1].reshape(-1, V),
                            targets[:, 1:].reshape(-1),
                            ignore_index=ignore_index)
''',

}  # end PARENT_SOLUTIONS


# --------------------------------------------------------------------------
#  Child-level extraction
# --------------------------------------------------------------------------

def _parent_of(child_id):
    m = re.match(r"^(\d+)([a-z]?)$", str(child_id))
    if not m: return None, None
    return m.group(1), m.group(2)


def get_solution(child_id):
    """Return a reference implementation containing only the symbols this child's
    stub defines, drawn from the parent's full reference."""
    parent, letter = _parent_of(child_id)
    if parent is None: return None
    parent_src = PARENT_SOLUTIONS.get(parent)
    if not parent_src:
        return None
    if not letter:
        return parent_src
    child_files = glob.glob(os.path.join(HERE, "problems", f"p{parent}{letter}_*.py"))
    if not child_files:
        return parent_src
    child_src = open(child_files[0]).read()
    try:
        child_tree = ast.parse(child_src)
    except SyntaxError:
        return parent_src
    child_symbols = {
        n.name for n in child_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if not child_symbols:
        return parent_src
    parent_tree = ast.parse(parent_src)
    out_nodes = []
    for n in parent_tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out_nodes.append(n)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name in child_symbols:
                out_nodes.append(n)
    if not out_nodes:
        return parent_src
    new_tree = ast.Module(body=out_nodes, type_ignores=[])
    return ast.unparse(new_tree)


# --------------------------------------------------------------------------
#  Benchmark configurations
# --------------------------------------------------------------------------

def _setup_2d_AB():
    return (_torch.randn(128, 256), _torch.randn(256, 64))

def _setup_pairwise():
    return (_torch.randn(512, 32), _torch.randn(512, 32))

def _setup_4d_BCHW():
    return (_torch.randn(8, 16, 32, 32),)

def _setup_einsum_attn():
    return (_torch.randn(2, 8, 64, 16), _torch.randn(2, 8, 64, 16))

def _setup_2d_for_topk():
    return (_torch.randn(128, 1000), 10)

def _setup_1d_for_kth():
    return (_torch.randn(10000), 100)

def _setup_outer():
    return (_torch.randn(256), _torch.randn(256))

def _setup_2d_logits_targets():
    return (_torch.randn(128, 100), _torch.randint(0, 100, (128,)))

def _setup_lse():
    return (_torch.randn(64, 1000), -1)

def _setup_softmax_3d():
    return (_torch.randn(32, 128, 256), -1)

def _setup_conv2d():
    return (_torch.randn(4, 3, 32, 32), _torch.randn(16, 3, 3, 3), None, 1, 1)

def _setup_maxpool2d():
    return (_torch.randn(8, 16, 32, 32), 2, 2, 0)

def _setup_attn_sdpa():
    q = _torch.randn(2, 4, 128, 32)
    k = _torch.randn(2, 4, 128, 32)
    v = _torch.randn(2, 4, 128, 32)
    return (q, k, v)

def _setup_flash():
    q = _torch.randn(1, 4, 256, 32)
    k = _torch.randn(1, 4, 256, 32)
    v = _torch.randn(1, 4, 256, 32)
    return (q, k, v)


BENCHMARKS = {
    "04c": {"fn": "pairwise_squared_distances", "setup": _setup_pairwise, "n_runs": 25},
    "04d": {"fn": "outer_product", "setup": _setup_outer, "n_runs": 30},
    "05a": {"fn": "per_channel_mean", "setup": _setup_4d_BCHW, "n_runs": 30},
    "09a": {"fn": "matmul", "setup": _setup_2d_AB, "n_runs": 30},
    "09d": {"fn": "attention_scores", "setup": _setup_einsum_attn, "n_runs": 25},
    "10a": {"fn": "top_k_per_row", "setup": _setup_2d_for_topk, "n_runs": 30},
    "10d": {"fn": "kth_largest", "setup": _setup_1d_for_kth, "n_runs": 30},
    "16a": {"fn": "logsumexp", "setup": _setup_lse, "n_runs": 30},
    "16b": {"fn": "softmax", "setup": _setup_softmax_3d, "n_runs": 30},
    "16c": {"fn": "log_softmax", "setup": _setup_softmax_3d, "n_runs": 30},
    "19a": {"fn": "cross_entropy", "setup": _setup_2d_logits_targets, "n_runs": 25},
    "44a": {"fn": "my_conv2d", "setup": _setup_conv2d, "n_runs": 15},
    "45a": {"fn": "my_max_pool2d", "setup": _setup_maxpool2d, "n_runs": 20},
    "53a": {"fn": "scaled_dot_product_attention", "setup": _setup_attn_sdpa, "n_runs": 20},
    "54c": {"fn": "causal_attention", "setup": _setup_attn_sdpa, "n_runs": 20},
    "65a": {"fn": "flash_attention_tiled", "setup": _setup_flash, "n_runs": 10},
}


def get_benchmark(pid):
    if pid in BENCHMARKS:
        return BENCHMARKS[pid]
    parent, _ = _parent_of(pid)
    return BENCHMARKS.get(f"{parent}a")
