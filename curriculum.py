"""Shared curriculum definitions — used by both runner.py and run_all.py."""

# (tier_name, first_problem_inclusive, last_problem_inclusive)
TIERS = [
    ("Tensor Fundamentals",        1, 10),
    ("Autograd & Linear",         11, 16),
    ("Loss Functions",            17, 22),
    ("Normalization",             23, 27),
    ("Embeddings & Init",         28, 29),
    ("Optimizers",                30, 34),
    ("Training Infrastructure",   35, 43),
    ("Convolutions",              44, 47),
    ("Sequences",                 48, 51),
    ("Attention Primitives",      52, 58),
    ("Transformer Architecture",  59, 64),
    ("Efficient Attention",       65, 65),
    ("Generation",                66, 70),
    ("Auxiliary Losses",          71, 74),
    ("LLM Specifics & Capstone",  75, 77),
]

TOTAL_PROBLEMS = TIERS[-1][2]


def find_tier(num):
    """Return (tier_index, name, lo, hi) for the tier containing this problem number,
    or None if the number is outside the curriculum."""
    n = int(num)
    for idx, (name, lo, hi) in enumerate(TIERS):
        if lo <= n <= hi:
            return idx, name, lo, hi
    return None


def tier_members(lo, hi):
    """Zero-padded problem numbers in a tier."""
    return [f"{n:02d}" for n in range(lo, hi + 1)]
