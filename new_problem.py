"""Scaffold a new single-task problem: stub + pristine snapshot + test skeleton.

Adding a problem by hand means creating four things that must agree on the id,
slug, and function name (the stub, its .stubs/ pristine copy, the tests/_src test,
and the reference in solutions.py). This tool writes the three mechanical files
for you so the only thing left to think about is the actual math: the test
assertions and the reference implementation.

  python new_problem.py 78a clip_to_unit                  # fn defaults to the slug, sig to "(x)"
  python new_problem.py 78a clip_to_unit --sig "(x, lo, hi)"
  python new_problem.py 78a clip_to_unit --desc "clamp x into [lo, hi]"

It never overwrites an existing file. After it runs:
  1. fill in the assertions in tests/_src/test_p<id>_<slug>.py
  2. add the reference to PARENT_SOLUTIONS["<NN>"] in solutions.py
  3. python rebuild.py            # compile the test
  4. python verify_problems.py    # reference must pass its own test

See docs/adding-a-problem.md for the full walkthrough.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.join(HERE, "problems")
STUBS = os.path.join(HERE, ".stubs")
SRC = os.path.join(HERE, "tests", "_src")

_ID_RE = re.compile(r"^(\d{1,2})([a-z]?)$")

STUB_TEMPLATE = '''"""
Problem {id}: {desc}
"""
import torch


def {sig}:
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
'''

TEST_TEMPLATE = '''import contextlib


@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError with the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {{label!r}}: {{msg}}") from e
    except Exception as e:
        raise AssertionError(f"step {{label!r}} crashed with {{type(e).__name__}}: {{e}}") from e


import torch
from p{id}_{slug} import *


def test_p{id}_{slug}():
    # Example shape — replace with real inputs and a real expected result:
    #   x = torch.tensor([...])
    #   with step("{desc}"):
    #       assert torch.equal({fn}(x), torch.tensor([...]))
    with step("{desc}"):
        raise AssertionError(
            "TODO: write the test for {fn} in tests/_src/test_p{id}_{slug}.py")
'''


def _norm_id(raw):
    m = _ID_RE.match(raw.strip().lstrip("p"))
    if not m:
        return None
    return f"{int(m.group(1)):02d}{m.group(2)}"


def _write(path, content, label):
    if os.path.exists(path):
        print(f"  exists, not overwriting: {label}")
        return False
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote: {label}")
    return True


def _tier_for(parent_int):
    try:
        from curriculum import TIERS
    except Exception:
        return None
    for name, lo, hi in TIERS:
        if lo <= parent_int <= hi:
            return name
    return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("id", help="problem id, e.g. '78a' or '78'")
    p.add_argument("slug", help="snake_case name, e.g. clip_to_unit (also the function name)")
    p.add_argument("--sig", help="function signature without the name, e.g. '(x, lo, hi)'")
    p.add_argument("--fn", help="function name (defaults to the slug)")
    p.add_argument("--desc", help="one-line description (defaults to the slug)")
    args = p.parse_args()

    pid = _norm_id(args.id)
    if not pid:
        print(f"bad id {args.id!r} — expected NN or NN<letter>, e.g. 78 or 78a")
        sys.exit(2)
    slug = args.slug
    if not re.match(r"^[a-z_][a-z0-9_]*$", slug):
        print(f"bad slug {slug!r} — use snake_case (letters, digits, underscores)")
        sys.exit(2)
    fn = args.fn or slug
    sig = f"{fn}{args.sig}" if args.sig else f"{fn}(x)"
    desc = args.desc or slug

    base = f"p{pid}_{slug}"
    parent_int = int(pid[:2])
    tier = _tier_for(parent_int)
    print(f"\nProblem {pid}  ({base})  parent {parent_int:02d} -> "
          f"{tier or 'NO TIER (add a range to curriculum.TIERS)'}\n")

    stub = STUB_TEMPLATE.format(id=pid, desc=desc, sig=sig)
    test = TEST_TEMPLATE.format(id=pid, slug=slug, fn=fn, desc=desc)
    wrote_any = False
    wrote_any |= _write(os.path.join(PROBLEMS, f"{base}.py"), stub, f"problems/{base}.py")
    wrote_any |= _write(os.path.join(STUBS, f"{base}.py"), stub, f".stubs/{base}.py")
    wrote_any |= _write(os.path.join(SRC, f"test_{base}.py"), test, f"tests/_src/test_{base}.py")

    print("\nNext:")
    print(f"  1. write the assertions in tests/_src/test_{base}.py")
    print(f"  2. add the reference for {fn} to PARENT_SOLUTIONS[\"{pid[:2]}\"] in solutions.py")
    print("  3. python rebuild.py            # compile the test")
    print("  4. python verify_problems.py    # the reference must pass its own test")
    if tier is None:
        print(f"  !  parent {parent_int:02d} is in no tier — add it to curriculum.TIERS "
              "or the problem won't show in the dashboard")
    return 0 if wrote_any else 1


if __name__ == "__main__":
    sys.exit(main())
