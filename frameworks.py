"""Single source of truth for an item's framework: 'torch' or 'numpy'.

Problems and projects mix the two; this classifies each by what it imports, so
the CLI flow (prep.py / projects.py), the web catalog, and the nvim picker can
tag and filter by framework consistently.
"""
import glob
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_TORCH = re.compile(r"^\s*import\s+torch\b", re.M)
_NUMPY = re.compile(r"^\s*import\s+numpy\b", re.M)

TORCH = "torch"
NUMPY = "numpy"


def framework_of_source(src):
    """torch if it imports torch (torch wins over numpy when both appear),
    numpy if it imports numpy, else torch — the standalone problems are a torch
    track, so an import-less stub defaults there."""
    if _TORCH.search(src):
        return TORCH
    if _NUMPY.search(src):
        return NUMPY
    return TORCH


def framework_of_file(path):
    try:
        with open(path) as f:
            return framework_of_source(f.read())
    except OSError:
        return TORCH


def problem_framework(pid):
    """Framework of a standalone problem, from the stub it edits."""
    hits = sorted(glob.glob(os.path.join(_HERE, "problems", f"p{pid}_*.py")))
    return framework_of_file(hits[0]) if hits else TORCH


def project_framework(project_dir):
    """Framework of a project: an explicit project.json 'framework' wins, else
    inferred from the hidden reference implementation (authoritative)."""
    try:
        with open(os.path.join(project_dir, "project.json")) as f:
            fw = json.load(f).get("framework")
        if fw in (TORCH, NUMPY):
            return fw
    except Exception:
        pass
    return framework_of_file(os.path.join(project_dir, "tests", "_ref", "reference.py"))
