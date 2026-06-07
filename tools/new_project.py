"""Scaffold a new multi-step project: the full _build/steps/tests tree.

Adding a project by hand means copying a _build/ dir and remembering to edit the
hardcoded name/title in gen.py (forget, and project.json points at the wrong
project — silent). This mirrors new_problem.py: it lays down a working skeleton
so the only thing left is the content (reference.py, tests.py, and the step
list in spec.py).

  python new_project.py my-cool-project --title "My Cool Project"
  python new_project.py my-cool-project --title "My Cool Project" --torch

It refuses to overwrite an existing project. After it runs:
  1. write the reference implementations in projects/<name>/tests/_ref/reference.py
  2. write a test per step in projects/<name>/tests/_ref/tests.py
  3. list the steps + parts in projects/<name>/_build/spec.py
  4. python projects/<name>/_build/gen.py       # stubs + project.json + compiled tests
  5. python projects/<name>/_build/verify.py    # every reference must pass its test

See docs/adding-a-project.md for the full walkthrough.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS = os.path.join(HERE, "projects")
TEMPLATE = os.path.join(PROJECTS, "tic-tac-toe-rl", "_build")  # most capable (handles class steps)

SPEC_TEMPLATE = '''"""Build spec for the {name} project.

PARTS: (title, description) in order.
STEPS: (name, part_index) in order. The step id is its 1-based position,
zero-padded to 4 digits (0001, 0002, ...).
"""

TITLE = {title!r}

PARTS = [
    # ("Part Title", "One-line description of the part."),
]

# (function_or_class_name, part_index) in solve order.
STEPS = [
    # ("first_fn", 0),
]


def step_id(i):
    return f"{{i:04d}}"
'''

REFERENCE_TEMPLATE = '''"""Hidden reference implementations for {name}.

One top-level def (or class) per step, named exactly as in spec.STEPS, each with
a one-line docstring (it becomes the stub's prompt). Steps may call earlier
reference functions directly — they share this namespace.
"""
{import_line}


# def first_fn(x):
#     """One-line description shown to the learner."""
#     return x
'''

TESTS_TEMPLATE = '''"""Hidden tests for {name}. One test_<id>_<name>(ns) per step; assert on
ns["<name>"]. Keep each fast (they run in CI)."""


# def test_0001_first_fn(ns):
#     assert ns["first_fn"](2) == 2
'''

SCAFFOLD_TEMPLATE = '''"""{title} — end-to-end demo.

Run once you've solved (enough of) the steps:

    python projects.py {name} --scaffold

Imports your assembled solution.py and runs the whole thing end-to-end. Keep it
small enough to finish in well under a minute on CPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution import *  # noqa: E402,F401,F403


def main():
    print("TODO: wire up the end-to-end demo from your solved steps.")


if __name__ == "__main__":
    main()
'''

README_TEMPLATE = '''# {title}

TODO: one-paragraph description of what you're building.

Each step is one function (or class) in `steps/NNNN_<fn>.py`, graded in isolation
against a hidden reference. See [../../docs/adding-a-project.md](../../docs/adding-a-project.md)
for how this project is built.

## Working through it

```bash
uv run python projects.py {name}              # parts + steps, [x]/[ ] solved
uv run python projects.py {name} --next       # jump to the next unsolved step
uv run python projects.py {name} --scaffold   # end-to-end demo (once solved)
```
'''


def _write(path, content, label):
    if os.path.exists(path):
        print(f"  exists, not overwriting: {label}")
        return False
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote: {label}")
    return True


def _gen_from_template(torch_stub):
    """Copy the template gen.py, but derive name/title from the dir + spec (so a
    new project never has a hardcoded name to forget), and pick the stub import."""
    with open(os.path.join(TEMPLATE, "gen.py")) as f:
        src = f.read()
    src = re.sub(r'"name":\s*"[^"]*",', '"name": os.path.basename(PROJECT),', src)
    src = re.sub(r'"title":\s*"[^"]*",',
                 '"title": getattr(spec, "TITLE", os.path.basename(PROJECT)),', src)
    if torch_stub:
        src = src.replace("import numpy as np  # noqa: F401", "import torch  # noqa: F401")
    return src


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="kebab-case project name, e.g. my-cool-project")
    p.add_argument("--title", help="human title (defaults to the name)")
    p.add_argument("--torch", action="store_true",
                   help="generate torch step stubs (default: numpy)")
    args = p.parse_args()

    name = args.name
    if not re.match(r"^[a-z0-9][a-z0-9-]*$", name):
        print(f"bad name {name!r} — use kebab-case (lowercase, digits, dashes)")
        sys.exit(2)
    root = os.path.join(PROJECTS, name)
    if os.path.exists(root):
        print(f"project '{name}' already exists at {root}")
        sys.exit(2)
    title = args.title or name

    for d in ("_build", "steps", os.path.join("tests", "_ref")):
        os.makedirs(os.path.join(root, d), exist_ok=True)

    print(f"\nScaffolding project '{name}' ({title})\n")
    imp = "import torch" if args.torch else "import numpy as np"
    _write(os.path.join(root, "_build", "spec.py"),
           SPEC_TEMPLATE.format(name=name, title=title), f"projects/{name}/_build/spec.py")
    _write(os.path.join(root, "_build", "gen.py"),
           _gen_from_template(args.torch), f"projects/{name}/_build/gen.py")
    with open(os.path.join(TEMPLATE, "verify.py")) as f:
        verify_src = f.read()
    _write(os.path.join(root, "_build", "verify.py"), verify_src, f"projects/{name}/_build/verify.py")
    _write(os.path.join(root, "tests", "_ref", "reference.py"),
           REFERENCE_TEMPLATE.format(name=name, import_line=imp),
           f"projects/{name}/tests/_ref/reference.py")
    _write(os.path.join(root, "tests", "_ref", "tests.py"),
           TESTS_TEMPLATE.format(name=name), f"projects/{name}/tests/_ref/tests.py")
    _write(os.path.join(root, "scaffold.py"),
           SCAFFOLD_TEMPLATE.format(name=name, title=title), f"projects/{name}/scaffold.py")
    _write(os.path.join(root, "README.md"),
           README_TEMPLATE.format(name=name, title=title), f"projects/{name}/README.md")

    print("\nNext:")
    print(f"  1. fill in tests/_ref/reference.py (one def/class per step)")
    print(f"  2. fill in tests/_ref/tests.py     (a test_<id>_<name> per step)")
    print(f"  3. list parts + steps in _build/spec.py")
    print(f"  4. python projects/{name}/_build/gen.py")
    print(f"  5. python projects/{name}/_build/verify.py")
    print(f"  6. uv run python verify_all.py     # picks up the new project automatically")
    return 0


if __name__ == "__main__":
    main()
