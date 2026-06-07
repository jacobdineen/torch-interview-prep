"""Compile-time assertion rewriter that captures values on failure.

For each `assert <expr>` in a test source, we transform it so the AssertionError
message includes (a) the source line text and (b) the actual values of the
operands. This gives pytest-style failure output without a runtime cost on PASS.

Patterns handled:
  assert a == b           -> shows left, right
  assert a != b           -> shows left, right
  assert a < b / <= / > / >=
  assert a is b / is not b
  assert torch.equal(a, b)
  assert torch.allclose(a, b, ...)
  assert torch.isclose(a, b, ...)
  assert not <expr>       -> shows <expr>'s value
  assert <anything>       -> falls back to source-line message

Skips assertions that already have a user-supplied message.
"""
import ast


_FMT_HELPER = '''def _fmt(v):
    """Format a value for assertion error messages."""
    try:
        import torch as _torch
        if isinstance(v, _torch.Tensor):
            shape = tuple(v.shape)
            if v.numel() == 0:
                return f"Tensor(shape={shape}, dtype={v.dtype}, empty)"
            if v.numel() <= 12:
                return f"Tensor(shape={shape}, dtype={v.dtype}, data={v.tolist()})"
            flat = v.detach().reshape(-1)
            return (f"Tensor(shape={shape}, dtype={v.dtype}, "
                    f"head={flat[:4].tolist()}, "
                    f"min={v.min().item():.4g}, max={v.max().item():.4g})")
    except Exception:
        pass
    if isinstance(v, (tuple, list)):
        if len(v) <= 8: return repr(v)
        return f"{type(v).__name__}(len={len(v)})"
    if isinstance(v, dict):
        if len(v) <= 6: return repr(v)
        return f"dict(len={len(v)})"
    return repr(v)


def _fail_msg(src_text, a, b, op_label="=="):
    """Multi-line failure message comparing two operands.

    Special cases:
      * two tensors with mismatched shapes  -> axis-by-axis diff
      * two shape-like tuples (or torch.Size vs tuple) with mismatched length/content
                                            -> axis-by-axis diff
      * two same-shape numeric tensors      -> values + max abs diff + first-mismatch index
      * mismatched dtypes                   -> dtype line + values
      * everything else                     -> values"""
    lines = [src_text]
    try:
        import torch as _torch
        if isinstance(a, _torch.Tensor) and isinstance(b, _torch.Tensor):
            sa, sb = tuple(a.shape), tuple(b.shape)
            if sa != sb:
                lines.append("  shape mismatch:")
                max_d = max(len(sa), len(sb))
                for i in range(max_d):
                    ax_a = sa[i] if i < len(sa) else "-"
                    ax_b = sb[i] if i < len(sb) else "-"
                    marker = "  <-- differs" if ax_a != ax_b else ""
                    lines.append(f"    axis {i}: {ax_a} vs {ax_b}{marker}")
                lines.append(f"    left :  shape={sa}, dtype={a.dtype}")
                lines.append(f"    right: shape={sb}, dtype={b.dtype}")
                return "\\n".join(lines)
            if a.dtype != b.dtype:
                lines.append(f"  dtype mismatch: {a.dtype} vs {b.dtype}")
                lines.append(f"    left :  {_fmt(a)}")
                lines.append(f"    right: {_fmt(b)}")
                return "\\n".join(lines)
            lines.append(f"  left  ({op_label}): {_fmt(a)}")
            lines.append(f"  right ({op_label}): {_fmt(b)}")
            if _torch.is_floating_point(a) and a.numel() > 0:
                diff = (a - b).abs()
                lines.append(f"  max |left - right| = {diff.max().item():.6g}")
                rel = (diff > 1e-6).nonzero(as_tuple=False)
                if rel.numel() > 0:
                    idx = tuple(rel[0].tolist())
                    lines.append(f"  first difference at index {idx}: "
                                  f"left={a[idx].item():.6g}, right={b[idx].item():.6g}")
            return "\\n".join(lines)
    except Exception:
        pass
    # Shape-like tuples / torch.Size: show axis-by-axis when they differ.
    try:
        is_shape = lambda x: isinstance(x, tuple) and all(isinstance(v, int) for v in x)
        if is_shape(a) and is_shape(b) and a != b:
            lines.append("  shape comparison:")
            max_d = max(len(a), len(b))
            for i in range(max_d):
                ax_a = a[i] if i < len(a) else "-"
                ax_b = b[i] if i < len(b) else "-"
                marker = "  <-- differs" if ax_a != ax_b else ""
                lines.append(f"    axis {i}: {ax_a} vs {ax_b}{marker}")
            return "\\n".join(lines)
    except Exception:
        pass
    lines.append(f"  left  ({op_label}): {_fmt(a)}")
    lines.append(f"  right ({op_label}): {_fmt(b)}")
    return "\\n".join(lines)
'''


def _name(id_, ctx=None):
    return ast.Name(id=id_, ctx=(ctx or ast.Load()))


def _store(id_):
    return ast.Name(id=id_, ctx=ast.Store())


def _fmt_call(varname):
    """Build `_fmt(varname)` as an AST expression."""
    return ast.Call(func=_name("_fmt"), args=[_name(varname)], keywords=[])


def _fail_msg_call(src_text, op_label="=="):
    """Build `_fail_msg(<src>, __l, __r, <op>)` AST."""
    return ast.Call(
        func=_name("_fail_msg"),
        args=[
            ast.Constant(value=src_text),
            _name("__l"),
            _name("__r"),
            ast.Constant(value=op_label),
        ],
        keywords=[],
    )


def _msg_single(prefix_src, varname, label):
    return ast.JoinedStr(values=[
        ast.Constant(value=f"{prefix_src}\n  {label}: "),
        ast.FormattedValue(value=_fmt_call(varname), conversion=-1),
    ])


_TORCH_COMPARE = {"equal", "allclose", "isclose"}


def _is_torch_compare_call(call):
    if not isinstance(call, ast.Call):
        return False
    f = call.func
    return (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
            and f.value.id == "torch" and f.attr in _TORCH_COMPARE)


class AssertRewriter(ast.NodeTransformer):
    def __init__(self, source):
        self.source = source

    def visit_Assert(self, node):
        # Respect user-supplied messages — don't clobber.
        if node.msg is not None:
            return node

        src_text = ast.get_source_segment(self.source, node) or "assert ..."
        if src_text.startswith("assert "):
            src_text = src_text[len("assert "):]

        test = node.test

        # `assert <left> <op> <right>` (single comparator).
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                                              ast.Gt, ast.GtE, ast.Is, ast.IsNot,
                                              ast.In, ast.NotIn))):
            return self._compare(src_text, test)

        # `assert torch.equal(a, b)` / `torch.allclose(...)` / `torch.isclose(...)`.
        if isinstance(test, ast.Call) and _is_torch_compare_call(test):
            return self._torch_call(src_text, test)

        # `assert not X`
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self._not_assert(src_text, test)

        # Fallback: source text only.
        return ast.copy_location(
            ast.Assert(test=test, msg=ast.Constant(value=src_text)),
            node,
        )

    def _compare(self, src_text, compare):
        left = compare.left
        op = compare.ops[0]
        right = compare.comparators[0]
        op_repr = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                   ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
                   ast.In: "in", ast.NotIn: "not in"}.get(type(op), "?")
        return [
            ast.Assign(targets=[_store("__l")], value=left),
            ast.Assign(targets=[_store("__r")], value=right),
            ast.Assert(
                test=ast.Compare(left=_name("__l"), ops=[op],
                                  comparators=[_name("__r")]),
                msg=_fail_msg_call(src_text, op_repr),
            ),
        ]

    def _torch_call(self, src_text, call):
        if len(call.args) < 2:
            return ast.Assert(test=call, msg=ast.Constant(value=src_text))
        a_arg = call.args[0]
        b_arg = call.args[1]
        new_call = ast.Call(
            func=call.func,
            args=[_name("__l"), _name("__r")] + list(call.args[2:]),
            keywords=call.keywords,
        )
        # Use the function's own name as the op label (e.g., "torch.allclose").
        op_label = "torch." + call.func.attr if isinstance(call.func, ast.Attribute) else "?"
        return [
            ast.Assign(targets=[_store("__l")], value=a_arg),
            ast.Assign(targets=[_store("__r")], value=b_arg),
            ast.Assert(test=new_call, msg=_fail_msg_call(src_text, op_label)),
        ]

    def _not_assert(self, src_text, unaryop):
        operand = unaryop.operand
        msg = _msg_single(src_text, "__v", "value")
        return [
            ast.Assign(targets=[_store("__v")], value=operand),
            ast.Assert(
                test=ast.UnaryOp(op=ast.Not(), operand=_name("__v")),
                msg=msg,
            ),
        ]


def rewrite_assertions(source):
    """Apply assertion rewriting to a Python source string. Returns the new
    source as a string (after ast.unparse), with the _fmt helper prepended."""
    tree = ast.parse(source)
    AssertRewriter(source).visit(tree)
    ast.fix_missing_locations(tree)
    # Inject the _fmt helper at the top of the module.
    helper_tree = ast.parse(_FMT_HELPER)
    tree.body = helper_tree.body + tree.body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
