"""Graduated hints for project steps, generated on the fly.

Unlike standalone problems (hand-authored hints in lib/hints.py), project steps
have no authored hint content — but each project ships READABLE sources for its
grader (tests/_ref/tests.py) and reference (tests/_ref/reference.py), which is
enough to build a useful 3-step ladder without giving the answer away:

  1. orient  — signature + what the hidden test actually verifies (its comment
               lines + how many assertions, which provided symbols it uses)
  2. shape   — the reference's structure: what it calls, its control flow, its
               size; no bodies
  3. opening — the first couple of lines of the reference body, then "..."

Hint state is persisted through lib.store's hint counter using the composite
key "proj:<project>:<step-id>" (the table is a plain TEXT key/value)."""
import ast
import json
import os


def _read(path):
    with open(path) as f:
        return f.read()


def _find_def(tree, src, name):
    """(node, source_segment) for a top-level def/class `name`, or (None, None)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == name:
            return node, ast.get_source_segment(src, node)
    return None, None


def _find_source_string(tree, name):
    """The string value of a top-level `name = '''...'''` assignment (CUDA-style
    projects store each step's reference as a source-code string), or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def _comment_lines(segment):
    out = []
    for ln in (segment or "").splitlines():
        st = ln.strip()
        if st.startswith("#") and len(st) > 2:
            out.append(st.lstrip("# "))
    return out


def _called_names(node):
    """Names this function calls, in first-use order (dedup)."""
    seen, out = set(), []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                name = f"{f.value.id}.{f.attr}"
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _flow_words(node):
    kinds = {ast.For: "a loop", ast.While: "a while-loop", ast.If: "branching",
             ast.FunctionDef: "an inner def (a closure)", ast.ListComp: "a comprehension",
             ast.DictComp: "a comprehension", ast.Try: "a try/except"}
    seen, out = set(), []
    for n in ast.walk(node):
        for k, w in kinds.items():
            if isinstance(n, k) and n is not node and w not in seen:
                seen.add(w)
                out.append(w)
    return out


def _body_lines(node, segment):
    """The reference body's source lines, minus its docstring line(s)."""
    lines = (segment or "").splitlines()
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return []
    start = body[0].lineno - node.lineno   # segment-relative
    return [l for l in lines[start:] if l.strip()]


def step_hints(project_dir, sid, name):
    """Up to 3 graduated hint strings for one step, or [] if sources are missing."""
    ref_path = os.path.join(project_dir, "tests", "_ref", "reference.py")
    test_path = os.path.join(project_dir, "tests", "_ref", "tests.py")
    man_path = os.path.join(project_dir, "project.json")
    if not (os.path.isfile(ref_path) and os.path.isfile(test_path)):
        return []
    try:
        man = json.load(open(man_path))
        step = next((s for s in man.get("steps", []) if s["id"] == sid), {})
        ref_src = _read(ref_path)
        test_src = _read(test_path)
        ref_tree = ast.parse(ref_src)
        test_tree = ast.parse(test_src)
    except (OSError, SyntaxError, json.JSONDecodeError):
        return []

    ref_node, ref_seg = _find_def(ref_tree, ref_src, name)
    test_node, test_seg = _find_def(test_tree, test_src, f"test_{sid}_{name}")
    hints = []

    # --- 1. orient: signature + what the hidden test verifies
    h = []
    if step.get("signature"):
        h.append(f"Signature: {step['signature']}.")
    if test_node is not None:
        n_asserts = sum(isinstance(n, ast.Assert) for n in ast.walk(test_node))
        pulls = sorted({n.slice.value for n in ast.walk(test_node)
                        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                        and n.value.id == "ns" and isinstance(n.slice, ast.Constant)
                        and isinstance(n.slice.value, str)})
        others = [p for p in pulls if p != name]
        h.append(f"The hidden test makes {n_asserts} assertion{'s' * (n_asserts != 1)}"
                 + (f" and also uses the provided symbol(s): {', '.join(others)}." if others else "."))
        notes = _comment_lines(test_seg)[:4]
        if notes:
            h.append("Facts the test's own comments state: " + " | ".join(notes))
    if h:
        hints.append(" ".join(h))

    # --- CUDA-style steps: the reference is a source STRING, not a def
    if ref_node is None:
        cuda_src = _find_source_string(ref_tree, name)
        if cuda_src is not None:
            lines = [l for l in cuda_src.splitlines() if l.strip()]
            notes = _comment_lines("\n".join(
                l.replace("//", "#", 1) if l.strip().startswith("//") else ""
                for l in lines))[:3]
            h = [f"The reference source is {len(lines)} lines of CUDA/C++."]
            if notes:
                h.append("Its own comments say: " + " | ".join(notes))
            hints.append(" ".join(h))
            if lines:
                hints.append("The reference begins:\n    "
                             + "\n    ".join(l.strip() for l in lines[:3])
                             + ("\n    ..." if len(lines) > 3 else ""))
            return hints

    # --- 2. shape: the reference's structure, no bodies
    if ref_node is not None:
        h = []
        doc = ast.get_docstring(ref_node)
        if doc:
            h.append(doc.strip().split("\n")[0])
        step_names = {s["name"] for s in man.get("steps", [])}
        calls = _called_names(ref_node)
        earlier = [c for c in calls if c in step_names and c != name]
        other = [c for c in calls if c not in step_names][:5]
        if earlier:
            h.append(f"The reference builds on earlier step(s): {', '.join(earlier)}.")
        if other:
            h.append(f"It calls: {', '.join(other)}.")
        flow = _flow_words(ref_node)
        n_lines = len(_body_lines(ref_node, ref_seg))
        h.append(f"Its body is {n_lines} line{'s' * (n_lines != 1)}"
                 + (f" and uses {', '.join(flow)}." if flow else "."))
        hints.append(" ".join(h))

    # --- 3. opening: the first lines of the reference body
    if ref_node is not None:
        body = _body_lines(ref_node, ref_seg)
        if body:
            shown = body[:2]
            tail = "\n    ..." if len(body) > 2 else ""
            hints.append("The reference begins:\n    "
                         + "\n    ".join(l.strip() for l in shown) + tail)

    return hints
