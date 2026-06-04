"""Build spec for the einops project.

PARTS: (title, description) in order.
STEPS: (name, part_index) in order. The step id is its 1-based position,
zero-padded to 4 digits (0001, 0002, ...).
"""

TITLE = 'Tensor Ops with einops'

PARTS = [
    ("Rearrange — moving axes around",
     "Transpose, flatten, split and merge axes with rearrange. No numbers "
     "change, only the layout."),
    ("Reduce — pooling over axes",
     "Mean/max/sum-pool whole axes or fixed-size patches with reduce."),
    ("Repeat — broadcasting and tiling",
     "Copy data along new or existing axes with repeat."),
    ("Einsum — contractions",
     "Batched matmuls, attention scores, weighted sums and traces with einsum."),
    ("Pack & unpack — variadic axes",
     "Flatten and concatenate ragged-rank tensors with pack."),
]

# (function_or_class_name, part_index) in solve order.
STEPS = [
    # Part 1 — Rearrange
    ("transpose_2d", 0),
    ("flatten_image", 0),
    ("merge_batch_and_time", 0),
    ("split_into_heads", 0),
    ("channels_last_to_first", 0),
    # Part 2 — Reduce
    ("global_average_pool", 1),
    ("max_pool_2x2", 1),
    ("sequence_mean", 1),
    # Part 3 — Repeat
    ("gray_to_rgb", 2),
    ("tile_rows", 2),
    ("upsample_nearest", 2),
    # Part 4 — Einsum
    ("batched_matmul", 3),
    ("attention_scores", 3),
    ("weighted_token_sum", 3),
    ("trace_batch", 3),
    # Part 5 — Pack & unpack
    ("flatten_keep_batch", 4),
    ("concat_features", 4),
]


def step_id(i):
    return f"{i:04d}"
