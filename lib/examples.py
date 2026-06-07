"""Worked-example extraction for problem descriptions.

Show a concrete Input -> Output example (like a good problem statement) WITHOUT
revealing the solution. Strategy:

  * Find the first real assertion in the test that calls the problem's function.
  * If its inputs are small literals (no randomness), RUN the hidden reference on
    those inputs and show Input -> the actual Output value. Output *values* reveal
    nothing about the algorithm.
  * If the inputs are random, fall back to a spec hint: the input shape plus, when
    the test compares against a torch builtin, "should match torch.<fn>(...)".

Everything is best-effort: any failure returns None (no example shown).
"""
import ast
import glob
import os
import re

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_HERE, "tests", "_src")
_PROBLEMS = os.path.join(_HERE, "problems")


def _fn_name(pid, slug):
    """(name, is_class) for the symbol the problem defines (the slug differs for
    variant problems, e.g. p16d_softmax2 defines `softmax`). Class problems get no
    worked example — instantiating an nn.Module isn't an input->output example."""
    hits = sorted(glob.glob(os.path.join(_PROBLEMS, f"p{pid}_{slug}.py")))
    if not hits:
        return None, False
    try:
        tree = ast.parse(open(hits[0]).read())
    except (OSError, SyntaxError):
        return None, False
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            return n.name, isinstance(n, ast.ClassDef)
    return None, False


def _param_names(pid, slug):
    hits = sorted(glob.glob(os.path.join(_PROBLEMS, f"p{pid}_{slug}.py")))
    if not hits:
        return []
    try:
        tree = ast.parse(open(hits[0]).read())
    except (OSError, SyntaxError):
        return []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            return [a.arg for a in n.args.args]
    return []


def _shape_fmt(v):
    """Shape-only rendering for random inputs (values would just be noise)."""
    try:
        import torch
    except Exception:
        torch = None
    try:
        import numpy as np
    except Exception:
        np = None
    if torch is not None and isinstance(v, torch.Tensor):
        return f"shape {tuple(v.shape)}"
    if np is not None and isinstance(v, np.ndarray):
        return f"shape {tuple(v.shape)}"
    if isinstance(v, tuple):
        return "(" + ", ".join(_shape_fmt(x) for x in v) + ")"
    if isinstance(v, float):
        return f"{v:.4g}"
    return _fmt(v)


def _clean(src):
    src = re.sub(r"\btorch\.", "", src)
    # tensor([...]) -> [...] (bare list, like a problem statement); single-level only.
    src = re.sub(r"\btensor\((\[[^\[\]]*\])\)", r"\1", src)
    return src


def _call_to(name, node):
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name:
            return n
    return None


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


_CONSTRUCTORS = {"tensor", "Tensor", "as_tensor", "zeros", "ones", "full", "arange",
                 "eye", "randn", "rand", "randint", "empty", "zeros_like", "ones_like",
                 "tensor_split", "linspace"}


def _builtin_match(node):
    """A torch/F builtin the output should match — the spec, not the solution.
    Excludes plain constructors (those are literals, shown as the output)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = node.func.value
        if (isinstance(base, ast.Name) and base.id in ("torch", "F", "nn")
                and node.func.attr not in _CONSTRUCTORS):
            return ast.unparse(node)
    return None


def _fmt(v):
    try:
        import torch
    except Exception:
        torch = None
    try:
        import numpy as np
    except Exception:
        np = None
    if torch is not None and isinstance(v, torch.Tensor):
        if v.numel() == 0:
            return "[]"
        if v.numel() <= 16 and v.dim() <= 2:
            return f"{_round(v.tolist())}"   # bare list, like a problem statement
        return f"shape {tuple(v.shape)}"
    if np is not None and isinstance(v, np.ndarray):
        if v.size == 0:
            return "[]"
        if v.size <= 16 and v.ndim <= 2:
            return f"{_round(v.tolist())}"
        return f"shape {tuple(v.shape)}"
    if np is not None and isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, tuple):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_fmt(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _round(x):
    if isinstance(x, list):
        return [_round(e) for e in x]
    if isinstance(x, float):
        return round(x, 4)
    return x


def _find_assert(fn, name):
    """Find the first step block that asserts on `name`, anywhere (including inside
    for/if). Returns (assigns, block_assigns, call_node, expected_node)."""
    assigns = {}
    result = []

    def step_asserts(with_body):
        block, asserts = {}, []
        def scan(body):
            for s in body:
                if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                    block[s.targets[0].id] = s.value
                elif isinstance(s, ast.Assert):
                    asserts.append(s.test)
                elif isinstance(s, (ast.For, ast.If, ast.With)):
                    scan(s.body)
        scan(with_body)
        return block, asserts

    def visit(body):
        for stmt in body:
            if result:
                return
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                assigns[stmt.targets[0].id] = stmt.value
            elif isinstance(stmt, ast.With):
                block, asserts = step_asserts(stmt.body)
                both = {**assigns, **block}
                for a in asserts:
                    call = _call_to(name, a)
                    if call is None:
                        for nm in _names(a):
                            if nm in both and _call_to(name, both[nm]):
                                call = _call_to(name, both[nm]); break
                        if call is None:
                            continue
                    expected = None
                    if isinstance(a, ast.Compare) and len(a.comparators) == 1:
                        expected = a.comparators[0] if _call_to(name, a.left) else a.left
                    elif isinstance(a, ast.Call):
                        others = [x for x in a.args if _call_to(name, x) is None
                                  and not (isinstance(x, ast.Name) and x.id in both and _call_to(name, both[x.id]))]
                        expected = others[0] if others else None
                    if isinstance(expected, ast.Name) and expected.id in both:
                        expected = both[expected.id]
                    result.append((dict(assigns), block, call, expected))
                    return
            elif isinstance(stmt, (ast.For, ast.If)):
                visit(stmt.body)

    visit(fn.body)
    return result[0] if result else None


def example_for(pid, slug):
    hits = sorted(glob.glob(os.path.join(_SRC, f"test_p{pid}_{slug}.py")))
    if not hits:
        return None
    try:
        tree = ast.parse(open(hits[0]).read())
    except (OSError, SyntaxError):
        return None
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == f"test_p{pid}_{slug}"), None)
    if fn is None:
        return None
    name, is_class = _fn_name(pid, slug)
    if is_class:        # nn.Module / class problems don't have a value example
        return None
    name = name or slug   # the real function name (slug differs for variants)
    found = _find_assert(fn, name)
    if not found:
        return None
    assigns, block, call, expected = found
    both = {**assigns, **block}

    referenced = _names(call) & set(both)
    matches = _builtin_match(expected) if expected is not None else None
    is_random = any(re.search(r"\brand|manual_seed", _clean(ast.unparse(both[v]))) for v in referenced)

    # Build the "Input" as named arguments (param = value), like a problem statement:
    #   x = arange(20).reshape(4, 5),  i = 2
    params = _param_names(pid, slug)
    parts = []
    for idx, a in enumerate(call.args):
        pname = params[idx] if idx < len(params) else f"arg{idx}"
        if isinstance(a, ast.Name) and a.id in both:
            parts.append(f"{pname} = {_clean(ast.unparse(both[a.id]))}")
        else:
            parts.append(f"{pname} = {_clean(ast.unparse(a))}")
    for kw in call.keywords:
        parts.append(f"{kw.arg} = {_clean(ast.unparse(kw.value))}")
    inputs = ",  ".join(parts)

    # Run the hidden reference on the test's (seeded) inputs. Literal inputs -> real
    # VALUES (a clean Input->Output); random inputs -> only the SHAPE (values would
    # be noise, and a shape reveals nothing about the algorithm).
    out = _run_reference(pid, name, fn, call)
    output = (_shape_fmt(out) if is_random else _fmt(out)) if out is not None else None
    # A no-argument function's "output" IS the answer (e.g. choose-a-hyperparameter
    # steps) — never show that. Need real inputs to be a non-giveaway example.
    if not inputs.strip() or (output is None and matches is None):
        return None
    return {"inputs": inputs, "output": output, "matches": matches, "random": is_random}


def _run_reference(pid, name, fn, call):
    """Execute the reference on the test's literal inputs; return the output value."""
    try:
        import torch  # noqa: F401
        import torch.nn.functional as F  # noqa: F401
        if _HERE not in __import__("sys").path:
            __import__("sys").path.insert(0, _HERE)
        from lib.solutions import PARENT_SOLUTIONS
        parent = f"{int(re.match(r'[0-9]+', pid).group()):02d}"
        ref = PARENT_SOLUTIONS.get(parent)
        if not ref:
            return None
        ns = {"torch": torch, "F": F}
        exec(compile(ref, "ref", "exec"), ns)
        # replay the test's setup statements (assigns + seeds) up to the call.
        for stmt in fn.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                exec(compile(ast.Module([stmt], []), "s", "exec"), ns)
            elif isinstance(stmt, ast.Expr):  # e.g. torch.manual_seed(0)
                try:
                    exec(compile(ast.Module([stmt], []), "s", "exec"), ns)
                except Exception:
                    pass
            elif isinstance(stmt, ast.With):
                for s in stmt.body:
                    if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name):
                        try:
                            exec(compile(ast.Module([s], []), "s", "exec"), ns)
                        except Exception:
                            pass
                break
        return eval(compile(ast.Expression(call), "c", "eval"), ns)
    except Exception:
        return None


# ---------- project steps (tests call ns["name"](...) against a per-project reference) ----------

_PROJECTS = os.path.join(_HERE, "projects")


def _ns_call(name, node):
    """Find an ns["name"](...) call anywhere in an expression."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript):
            sub = n.func
            if (isinstance(sub.value, ast.Name) and sub.value.id == "ns"):
                key = sub.slice.value if isinstance(sub.slice, ast.Index) else sub.slice
                if isinstance(key, ast.Constant) and key.value == name:
                    return n
    return None


def _step_param_names(project, sid, name):
    hits = sorted(glob.glob(os.path.join(_PROJECTS, project, "steps", f"{sid}_*.py")))
    if not hits:
        return []
    try:
        tree = ast.parse(open(hits[0]).read())
    except (OSError, SyntaxError):
        return []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return [a.arg for a in n.args.args]
        if isinstance(n, ast.ClassDef):
            return ["__class__"]
    return []


def example_for_step(project, sid, name):
    """Worked Input->Output example for a project step, run against its reference.
    None for class steps or when not cleanly extractable."""
    params = _step_param_names(project, sid, name)
    if params == ["__class__"]:
        return None
    tpath = os.path.join(_PROJECTS, project, "tests", "_ref", "tests.py")
    rpath = os.path.join(_PROJECTS, project, "tests", "_ref", "reference.py")
    if not (os.path.exists(tpath) and os.path.exists(rpath)):
        return None
    try:
        tree = ast.parse(open(tpath).read())
    except (OSError, SyntaxError):
        return None
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == f"test_{sid}_{name}"), None)
    if fn is None:
        return None

    assigns, call = {}, None
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            c = _ns_call(name, stmt.value)
            if c is not None and call is None:
                call = c
            assigns[stmt.targets[0].id] = stmt.value
        elif call is None:
            c = _ns_call(name, stmt)
            if c is not None:
                call = c
    if call is None:
        return None

    referenced = {n.id for n in ast.walk(call) if isinstance(n, ast.Name)} & set(assigns)
    is_random = any(re.search(r"\brand|randn|randint|seed|shuffle|choice", _clean(ast.unparse(assigns[v])))
                    for v in referenced)

    parts = []
    for idx, a in enumerate(call.args):
        pname = params[idx] if idx < len(params) else f"arg{idx}"
        if isinstance(a, ast.Name) and a.id in assigns:
            parts.append(f"{pname} = {_clean(ast.unparse(assigns[a.id]))}")
        else:
            parts.append(f"{pname} = {_clean(ast.unparse(a))}")
    for kw in call.keywords:
        parts.append(f"{kw.arg} = {_clean(ast.unparse(kw.value))}")
    inputs = ",  ".join(parts)

    out = _run_step_reference(rpath, fn, name, call)
    output = (_shape_fmt(out) if is_random else _fmt(out)) if out is not None else None
    # No real inputs -> the output is just the answer; don't show it.
    if not inputs.strip() or output is None:
        return None
    return {"inputs": inputs, "output": output, "matches": None, "random": is_random}


def _run_step_reference(rpath, fn, name, call):
    try:
        ns = {}
        exec(compile(open(rpath).read(), "ref", "exec"), ns)
        # replay the test's setup assigns up to the call
        for stmt in fn.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                if _ns_call(name, stmt.value) is not None:
                    break
                try:
                    exec(compile(ast.Module([stmt], []), "s", "exec"), ns)
                except Exception:
                    pass
        # rewrite ns["name"](...) -> name(...) for eval against the reference
        new_call = ast.Call(func=ast.Name(id=name, ctx=ast.Load()),
                            args=call.args, keywords=call.keywords)
        ast.fix_missing_locations(new_call)
        return eval(compile(ast.Expression(new_call), "c", "eval"), ns)
    except Exception:
        return None
