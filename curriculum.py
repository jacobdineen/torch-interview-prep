"""Shared curriculum definitions — used by runner.py, run_all.py, reset.py.

Problem IDs use the format ``NN<letter>`` (e.g. ``01a``, ``55c``). Tier ranges
are defined in terms of the PARENT integer (the first two digits); membership
is computed at runtime by scanning the problems/ directory.
"""
import glob
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBLEMS_DIR = os.path.join(_HERE, "problems")
_ID_RE = re.compile(r"^p(\d+[a-z]?)_")

# (tier_name, first_parent_inclusive, last_parent_inclusive)
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


def _parent_of(problem_id):
    """Integer parent number for a problem id like '01a' or '55c'."""
    m = re.match(r"^(\d+)", str(problem_id))
    return int(m.group(1)) if m else None


def all_problem_ids():
    """Scan problems/ and return all problem IDs (e.g., ['01a', '01b', ...]),
    sorted by (parent_int, letter)."""
    ids = []
    for fname in os.listdir(_PROBLEMS_DIR):
        m = _ID_RE.match(fname)
        if m:
            ids.append(m.group(1))
    ids.sort(key=lambda pid: (_parent_of(pid), pid))
    return ids


def find_tier(problem_id):
    """Return (tier_index, name, lo, hi) for the tier containing this problem id."""
    n = _parent_of(problem_id)
    if n is None:
        return None
    for idx, (name, lo, hi) in enumerate(TIERS):
        if lo <= n <= hi:
            return idx, name, lo, hi
    return None


def tier_members(lo, hi):
    """Return all problem IDs whose parent number is in [lo, hi]."""
    out = []
    for pid in all_problem_ids():
        n = _parent_of(pid)
        if n is not None and lo <= n <= hi:
            out.append(pid)
    return out


# Total problem count is the total number of split children, computed at import time.
TOTAL_PROBLEMS = len(all_problem_ids())
