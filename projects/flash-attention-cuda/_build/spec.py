"""Build spec for the flash-attention-cuda project."""

TITLE = 'Build Flash Attention in CUDA from Scratch'

PARTS = [
    ("CUDA Primitives Warm-up", "Elementwise and reduction kernels — the building blocks for matrix and attention operations."),
    ("Matrix Operations", "The matmul, transpose, and dot-product utilities needed for attention scoring."),
    ("Naive Attention Baseline", "Compose QK^T scoring, row-wise softmax, and PV multiplication into a straightforward attention pipeline."),
    ("Online Softmax Math", "The running-max and running-sum updates that let softmax be computed incrementally across tiles."),
    ("Tiled Attention Building Blocks", "Per-tile routines for loading, scoring, reducing, exponentiating, and accumulating PV."),
    ("Fused Flash Attention Kernel", "Assemble the building blocks into the full online-softmax Flash Attention kernel and its host launcher."),
    ("Causal Flash Attention", "Extend the kernel with a causal mask for autoregressive models."),
]

STEPS = [
    ("vector_add", 0), ("scale_array", 0), ("elementwise_exp", 0), ("row_max", 0), ("row_sum", 0),
    ("dot_product", 1), ("matmul", 1), ("transpose", 1),
    ("qk_scores", 2), ("softmax_rows", 2), ("pv_matmul", 2), ("naive_attention", 2),
    ("online_max", 3), ("correction_factor", 3), ("update_running_sum", 3), ("rescale_output", 3),
    ("load_tile", 4), ("tile_scores", 4), ("tile_rowmax", 4), ("tile_exp", 4), ("tile_rowsum", 4), ("accumulate_pv", 4),
    ("flash_attention_kernel", 5), ("flash_attention_launcher", 5),
    ("causal_mask", 6), ("flash_attention_causal_kernel", 6),
]


def step_id(i):
    return f"{i:04d}"


PRIMER = '''
This is a CUDA project: each step is a Python file holding ONE CUDA source string,
named exactly for the step (e.g. the `vector_add` variable holds the CUDA for the
vector_add step). You edit the CUDA inside that string. At grade time the source
is JIT-compiled with nvcc (CUDA 12.3, g++-12 host compiler, set up for you) and
the host function — which MUST be named exactly like the step and take/return
torch tensors — is run on the GPU and compared to a PyTorch oracle.

Tensors are contiguous float32 on CUDA; index a 2-D (R,C) tensor as ptr[r*C + c].
Compiling takes ~30-60s the first time (cached after). Conventions are in each
step's docstring. Run `uv run python projects.py flash-attention-cuda` for the
full outline.
'''
