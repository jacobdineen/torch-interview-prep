"""Hidden tests: compile each step's CUDA + check it against a torch oracle.
Each delegates to grade() (in reference.py); grade no-ops without a GPU."""


def test_0001_vector_add(ns):
    ns["grade"](ns, "vector_add")

def test_0002_scale_array(ns):
    ns["grade"](ns, "scale_array")

def test_0003_elementwise_exp(ns):
    ns["grade"](ns, "elementwise_exp")

def test_0004_row_max(ns):
    ns["grade"](ns, "row_max")

def test_0005_row_sum(ns):
    ns["grade"](ns, "row_sum")

def test_0006_dot_product(ns):
    ns["grade"](ns, "dot_product")

def test_0007_matmul(ns):
    ns["grade"](ns, "matmul")

def test_0008_transpose(ns):
    ns["grade"](ns, "transpose")

def test_0009_qk_scores(ns):
    ns["grade"](ns, "qk_scores")

def test_0010_softmax_rows(ns):
    ns["grade"](ns, "softmax_rows")

def test_0011_pv_matmul(ns):
    ns["grade"](ns, "pv_matmul")

def test_0012_naive_attention(ns):
    ns["grade"](ns, "naive_attention")

def test_0013_online_max(ns):
    ns["grade"](ns, "online_max")

def test_0014_correction_factor(ns):
    ns["grade"](ns, "correction_factor")

def test_0015_update_running_sum(ns):
    ns["grade"](ns, "update_running_sum")

def test_0016_rescale_output(ns):
    ns["grade"](ns, "rescale_output")

def test_0017_load_tile(ns):
    ns["grade"](ns, "load_tile")

def test_0018_tile_scores(ns):
    ns["grade"](ns, "tile_scores")

def test_0019_tile_rowmax(ns):
    ns["grade"](ns, "tile_rowmax")

def test_0020_tile_exp(ns):
    ns["grade"](ns, "tile_exp")

def test_0021_tile_rowsum(ns):
    ns["grade"](ns, "tile_rowsum")

def test_0022_accumulate_pv(ns):
    ns["grade"](ns, "accumulate_pv")

def test_0023_flash_attention_kernel(ns):
    ns["grade"](ns, "flash_attention_kernel")

def test_0024_flash_attention_launcher(ns):
    ns["grade"](ns, "flash_attention_launcher")

def test_0025_causal_mask(ns):
    ns["grade"](ns, "causal_mask")

def test_0026_flash_attention_causal_kernel(ns):
    ns["grade"](ns, "flash_attention_causal_kernel")
