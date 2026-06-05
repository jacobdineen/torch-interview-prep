"""Build spec for the flash-attention-from-scratch project."""

TITLE = 'Build Flash Attention from Scratch'

PARTS = [
    ("Standard Attention", "The baseline we optimize: scaled dot-product scores, a numerically stable row softmax, and full materialized attention."),
    ("Online Softmax", "The streaming-softmax core: a running max + running denominator that fold one key/value block into the result without ever seeing all scores at once."),
    ("Flash Forward (Tiled)", "Tile over key/value blocks, run the online-softmax update per block, normalize, and save the log-sum-exp — exact attention with O(N) extra memory."),
    ("Causal Flash Attention", "Mask future keys per block and tile a causal (autoregressive) attention that matches a masked reference."),
    ("Flash Backward (Recomputation)", "The backward pass that recomputes the softmax from the saved log-sum-exp instead of storing the N x M probabilities."),
]

STEPS = [
    ("attention_scores", 0), ("stable_rowmax", 0), ("softmax_rows", 0), ("attention_reference", 0),
    ("correction", 1), ("update_running_max", 1), ("block_exp_scores", 1), ("update_normalizer", 1), ("online_softmax_step", 1),
    ("make_blocks", 2), ("finalize", 2), ("logsumexp", 2), ("flash_attention_forward", 2),
    ("causal_block_mask", 3), ("masked_attention_reference", 3), ("flash_attention_causal", 3),
    ("attention_backward_D", 4), ("flash_attention_backward", 4),
]


def step_id(i):
    return f"{i:04d}"


PRIMER = '''
Conventions: single-head, no batch. Q (N,d), K (M,d), V (M,d), output O (N,d); scores S (N,M);
per-row stats m, l, L, D are shape (N,). `scale` is the scalar the caller passes (1/sqrt(d)).
The flash forward tiles over key/value blocks and keeps an UNNORMALIZED running output that is
divided by the running denominator l only at the end.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py flash-attention-from-scratch` (or the outline drawer) to see all signatures.'''
