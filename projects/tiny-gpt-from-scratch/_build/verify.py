"""Verify that every reference implementation passes its own hidden test.

  python _build/verify.py

For each step that has both a reference function and a test, run the test with
the reference namespace (user == reference). All should PASS; this is the gate
that the reference math + the tests agree (gradient checks included).
"""
import importlib.util
import os
import sys
import types

BUILD = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BUILD)
sys.path.insert(0, BUILD)
import spec  # noqa: E402


def _load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    ref = _load(os.path.join(PROJECT, "tests", "_ref", "reference.py"), "ref_src")
    tests = _load(os.path.join(PROJECT, "tests", "_ref", "tests.py"), "tests_src")
    ref_ns = {k: v for k, v in vars(ref).items()
              if isinstance(v, types.FunctionType) and not k.startswith("_")}

    npass, nfail, nskip = 0, 0, 0
    for i, (name, _part) in enumerate(spec.STEPS, start=1):
        sid = spec.step_id(i)
        test_fn = getattr(tests, f"test_{sid}_{name}", None)
        if test_fn is None or name not in ref_ns:
            nskip += 1
            continue
        try:
            test_fn(dict(ref_ns))
            npass += 1
        except Exception as e:
            nfail += 1
            print(f"FAIL {sid} {name}: {type(e).__name__}: {e}")
    print(f"\n{npass} pass, {nfail} fail, {nskip} not-yet-built")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
