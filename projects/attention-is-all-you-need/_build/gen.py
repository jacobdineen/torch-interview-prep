"""Generate project.json, step stubs, and compiled hidden tests from spec.py
and the reference implementations.

  python _build/gen.py

Idempotent. Only (re)writes a step stub if it does not yet exist OR is still a
pristine stub, so it never clobbers your in-progress work. Stubs are generated
only for steps whose reference function exists (supports staged building).
"""
import ast
import importlib.util
import inspect
import json
import os
import py_compile
import sys

BUILD = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BUILD)
REF_SRC = os.path.join(PROJECT, "tests", "_ref", "reference.py")
TESTS_SRC = os.path.join(PROJECT, "tests", "_ref", "tests.py")
COMPILED = os.path.join(PROJECT, "tests", "_compiled")
STEPS_DIR = os.path.join(PROJECT, "steps")

sys.path.insert(0, BUILD)
import spec  # noqa: E402


def _load_module(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


STUB_TEMPLATE = '''"""
Step {id}: {name}

Part {pnum} — {ptitle}
{doc}
"""
import torch  # noqa: F401


def {name}{sig}:
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
'''


CLASS_STUB_TEMPLATE = '''"""
Step {id}: {name}

Part {pnum} — {ptitle}
{doc}
"""
import torch  # noqa: F401


class {name}:
{methods}


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
'''


def _class_methods_src(cls):
    out = []
    for mname, m in cls.__dict__.items():
        if inspect.isfunction(m):
            out.append(f"    def {mname}{inspect.signature(m)}:")
            out.append("        raise NotImplementedError")
            out.append("")
    return "\n".join(out).rstrip() or "    pass"


def _raises_not_implemented(fdef):
    body = [n for n in fdef.body if not (isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant))]  # drop docstring
    return (len(body) == 1 and isinstance(body[0], ast.Raise)
            and getattr(getattr(body[0].exc, "func", body[0].exc), "id", None)
            == "NotImplementedError")


def _is_pristine(path, name):
    """True if the step file is still an untouched stub. For a function: body is
    just `raise NotImplementedError`. For a class: every method raises it."""
    try:
        with open(path) as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return _raises_not_implemented(node)
        if isinstance(node, ast.ClassDef) and node.name == name:
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            return bool(methods) and all(_raises_not_implemented(m) for m in methods)
    return False


def main():
    ref = _load_module(REF_SRC, "reference_src")
    os.makedirs(STEPS_DIR, exist_ok=True)
    os.makedirs(COMPILED, exist_ok=True)

    manifest_steps = []
    written, skipped = 0, 0
    for i, (name, part) in enumerate(spec.STEPS, start=1):
        sid = spec.step_id(i)
        fn = getattr(ref, name, None)
        sig = str(inspect.signature(fn)) if fn else "(*args)"
        doc = (inspect.getdoc(fn) or "").strip() if fn else "TODO: not yet built."
        manifest_steps.append({
            "id": sid, "name": name, "part": part, "points": 5,
            "signature": f"{name}{sig}", "doc": doc,
        })
        if fn is None:
            continue  # no reference yet -> no stub (staged build)
        path = os.path.join(STEPS_DIR, f"{sid}_{name}.py")
        if os.path.exists(path) and not _is_pristine(path, name):
            skipped += 1  # user has work here; leave it
            continue
        ptitle = spec.PARTS[part][0]
        with open(path, "w") as f:
            if inspect.isclass(fn):
                f.write(CLASS_STUB_TEMPLATE.format(id=sid, name=name, doc=doc,
                        pnum=part + 1, ptitle=ptitle, methods=_class_methods_src(fn)))
            else:
                f.write(STUB_TEMPLATE.format(id=sid, name=name, sig=sig, doc=doc,
                                             pnum=part + 1, ptitle=ptitle))
        written += 1

    manifest = {
        "name": os.path.basename(PROJECT),
        "title": getattr(spec, "TITLE", os.path.basename(PROJECT)),
        "parts": [{"title": t, "description": d} for t, d in spec.PARTS],
        "steps": manifest_steps,
    }
    with open(os.path.join(PROJECT, "project.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Compile hidden reference + tests.
    py_compile.compile(REF_SRC, cfile=os.path.join(COMPILED, "reference.pyc"), doraise=True)
    py_compile.compile(TESTS_SRC, cfile=os.path.join(COMPILED, "tests.pyc"), doraise=True)

    print(f"manifest: {len(manifest_steps)} steps")
    print(f"stubs written: {written}, preserved (had work): {skipped}")
    print("compiled: reference.pyc, tests.pyc")


if __name__ == "__main__":
    main()
