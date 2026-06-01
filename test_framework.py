"""Tests for the project framework itself (not the curriculum content).

verify_problems.py / the project verify.py scripts guard the *content* — that
every reference solution passes its tests. Nothing guards the *tooling*: the
namespace-swap grader in project_runner.py, reference injection, class-step
handling, and the solution.py assembler. A regression there could silently pass
everything (or break every project), and the content checks wouldn't notice.

This builds a tiny throwaway project in a tempdir and drives the real
project_runner against it, so it's fast, dependency-free, and never touches any
real project's steps or progress files.

  python test_framework.py        # exit 0 only if all framework tests pass
"""
import ast
import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import project_runner as pr  # noqa: E402


REFERENCE_SRC = '''\
def add_one(x):
    return x + 1


def use_helper(x):
    # Depends on add_one being injected into scope by the grader.
    return add_one(x) * 2


class Counter:
    def __init__(self, start=0):
        self.n = start

    def inc(self):
        self.n += 1
        return self.n
'''

TESTS_SRC = '''\
def test_0001_add_one(ns):
    assert ns["add_one"](3) == 4, "add_one(3) should be 4"


def test_0002_use_helper(ns):
    assert ns["use_helper"](3) == 8, "use_helper(3) should be 8"


def test_0003_make_counter(ns):
    c = ns["Counter"](10)
    assert c.inc() == 11
    assert c.inc() == 12
'''

STEP_SRCS = {
    "0001_add_one.py": "def add_one(x):\n    return x + 1\n",
    # Calls add_one — only passes if the reference funcs are injected into scope.
    "0002_use_helper.py": "def use_helper(x):\n    return add_one(x) * 2\n",
    "0003_make_counter.py": (
        "class Counter:\n"
        "    def __init__(self, start=0):\n"
        "        self.n = start\n\n"
        "    def inc(self):\n"
        "        self.n += 1\n"
        "        return self.n\n"
    ),
}

WRONG_ADD_ONE = "def add_one(x):\n    return x + 2\n"  # off by one -> must FAIL


def _make_project(root):
    os.makedirs(os.path.join(root, "tests", "_ref"))
    os.makedirs(os.path.join(root, "steps"))
    with open(os.path.join(root, "tests", "_ref", "reference.py"), "w") as f:
        f.write(REFERENCE_SRC)
    with open(os.path.join(root, "tests", "_ref", "tests.py"), "w") as f:
        f.write(TESTS_SRC)
    steps_meta = [
        {"id": "0001", "name": "add_one", "part": 0, "points": 5},
        {"id": "0002", "name": "use_helper", "part": 0, "points": 5},
        {"id": "0003", "name": "make_counter", "part": 0, "points": 5},
    ]
    manifest = {
        "name": "_framework_test", "title": "Framework Test",
        "parts": [{"title": "Part One", "description": "throwaway"}],
        "steps": steps_meta,
    }
    with open(os.path.join(root, "project.json"), "w") as f:
        json.dump(manifest, f)
    for fname, src in STEP_SRCS.items():
        with open(os.path.join(root, "steps", fname), "w") as f:
            f.write(src)


def _run_step_quiet(step_path):
    """run_step but with its PASS/FAIL chatter suppressed; returns the exit code."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = pr.run_step(step_path)
    return rc


# ---------- the checks ----------

def check_reference_funcs(root):
    pr._ensure_compiled(root)  # run_step does this before loading; mirror it here
    funcs = pr._reference_funcs(root)
    assert "add_one" in funcs, "reference loader dropped a plain function"
    assert "use_helper" in funcs, "reference loader dropped a plain function"
    assert "Counter" in funcs, "reference loader dropped a class (class steps would break)"


def check_correct_steps_pass(root):
    for fname in STEP_SRCS:
        rc = _run_step_quiet(os.path.join(root, "steps", fname))
        assert rc == 0, f"correct step {fname} should grade PASS, got exit {rc}"


def check_reference_injection(root):
    # 0002 calls add_one with no import; it can only work if the grader injected
    # the reference functions into the step's globals.
    rc = _run_step_quiet(os.path.join(root, "steps", "0002_use_helper.py"))
    assert rc == 0, "step calling an earlier reference fn failed — injection broke"


def check_wrong_step_fails(root):
    path = os.path.join(root, "steps", "0001_add_one.py")
    good = STEP_SRCS["0001_add_one.py"]
    try:
        with open(path, "w") as f:
            f.write(WRONG_ADD_ONE)
        rc = _run_step_quiet(path)
        assert rc != 0, "an INCORRECT step graded PASS — the grader isn't catching errors"
    finally:
        with open(path, "w") as f:
            f.write(good)


def check_progress_recorded(root):
    _run_step_quiet(os.path.join(root, "steps", "0001_add_one.py"))
    progress = pr._load_progress(root)
    assert progress.get("0001", {}).get("ever_passed") is True, \
        "passing a step didn't record progress"


def check_assembler(root):
    # Pass all steps, then assemble and verify the generated solution.py is valid
    # and contains each solved def (including the class step).
    for fname in STEP_SRCS:
        _run_step_quiet(os.path.join(root, "steps", fname))
    sol_path = pr.assemble_solution(root)
    with open(sol_path) as f:
        src = f.read()
    tree = ast.parse(src)  # must be syntactically valid
    top = {n.name for n in tree.body
           if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for expected in ("add_one", "use_helper", "Counter"):
        assert expected in top, f"assembled solution.py is missing {expected}"
    assert "TODO: solve" not in src, "all steps solved but assembler left a TODO marker"


CHECKS = [
    check_reference_funcs,
    check_correct_steps_pass,
    check_reference_injection,
    check_wrong_step_fails,
    check_progress_recorded,
    check_assembler,
]


def main():
    passed, failed = 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "proj")
        _make_project(root)
        for check in CHECKS:
            try:
                check(root)
                print(f"PASS  {check.__name__}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL  {check.__name__}: {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR {check.__name__}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\nframework: {passed} pass, {failed} fail")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
