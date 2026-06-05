"""End-to-end smoke test for the mle_prep web app — proves it ACTUALLY works.

Run it against a live server (start it with web/serve-app.sh first):

    python web/smoke_test.py                 # checks http://127.0.0.1:8000
    APIPORT=8000 python web/smoke_test.py

It exercises the real API + static assets + the nvim editor path and guards the
exact regressions we have hit: the CSRF token must be wired both ends, a graded
run must not clobber unrelated files (the old `:wa` bug), the JS must parse, and
every element id / fetch route the frontend uses must exist in the HTML / backend.
Exits non-zero if anything fails — so "is the app working?" is a command, not a guess.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)
STATIC = os.path.join(WEB, "static")
PORT = int(os.environ.get("APIPORT", "8000"))
BASE = f"http://127.0.0.1:{PORT}"
NVIM_SOCK = os.environ.get("NVIM_SOCK", "/tmp/mle_nvim.sock")

PASS, FAIL, SKIP = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mSKIP\033[0m"
_results = []


def check(name, ok, detail="", skipped=False):
    tag = SKIP if skipped else (PASS if ok else FAIL)
    print(f"  [{tag}] {name}" + (f" - {detail}" if detail else ""))
    if not skipped:
        _results.append(bool(ok))
    return ok


def _req(path, method="GET", token=None, body=None, timeout=620):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if token is not None:
        headers["X-MLE-Token"] = token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _json(path, **kw):
    code, raw = _req(path, **kw)
    try:
        return code, json.loads(raw or b"{}")
    except Exception:
        return code, {}


def main():
    print(f"smoke test -> {BASE}\n")
    try:
        code, cfg = _json("/api/config")
    except Exception as e:
        check("server reachable", False, f"{e} - is serve-app.sh running?")
        return _summary()
    token = cfg.get("token")
    check("GET /api/config returns a token", code == 200 and bool(token) and len(token or "") >= 16, f"http {code}")

    code, cat = _json("/api/catalog")
    items = cat.get("items") or []
    sources = cat.get("sources") or []
    check("GET /api/catalog populated", code == 200 and len(items) > 0 and "Problems" in sources,
          f"{len(items)} items, {len(sources)} sources")
    check("catalog exposes projects meta + frameworks",
          isinstance(cat.get("projects"), dict) and bool(cat.get("frameworks")))

    probs = [x for x in items if x.get("source") == "Problems"]
    a_prob = probs[0] if probs else None
    solved = next((x for x in probs if x.get("solved")), a_prob)

    if a_prob:
        code, meta = _json("/api/item?key=" + a_prob["key"])
        check("GET /api/item returns goal + signature",
              code == 200 and (meta.get("doc") is not None) and "signature" in meta, a_prob["id"])
    else:
        check("GET /api/item", False, "no problems in catalog")

    code_no, _ = _req("/api/open", "POST", token=None, body={"key": a_prob["key"] if a_prob else "prob:04a"})
    check("CSRF: /api/open without token -> 403", code_no == 403, f"http {code_no}")
    if a_prob:
        code_ok, openr = _json("/api/open", method="POST", token=token, body={"key": a_prob["key"]})
        check("/api/open with token -> ok:true (editor switches)",
              code_ok == 200 and openr.get("ok") is True, f"http {code_ok} {openr}")
    code_run_no, _ = _req("/api/run", "POST", token=None, body={"key": "prob:04a"})
    check("CSRF: /api/run without token -> 403", code_run_no == 403, f"http {code_run_no}")

    if solved:
        before = _hash("web/static/app.js")
        code, rr = _json("/api/run", method="POST", token=token, body={"key": solved["key"]})
        check("/api/run grades (valid JSON, status pass/fail/error)",
              code == 200 and rr.get("status") in ("pass", "fail", "error"),
              f"{solved['id']} -> {rr.get('status')}")
        after = _hash("web/static/app.js")
        check("running a problem does NOT modify web/static/app.js (:wa guard)",
              before == after and before is not None)
    else:
        check("/api/run grades", False, "no problem available", skipped=True)

    for f in ("index.html", "app.js", "style.css"):
        code, raw = _req("/" if f == "index.html" else f"/static/{f}")
        check(f"static {f} serves 200 non-empty", code == 200 and len(raw or b"") > 100)
    code, appjs = _req("/static/app.js")
    appjs = (appjs or b"").decode("utf-8", "replace")
    check("frontend wires the CSRF token (X-MLE-Token + TOKEN=cfg.token)",
          "X-MLE-Token" in appjs and "TOKEN = cfg.token" in appjs)

    if _have("node"):
        rc = subprocess.run(["node", "--check", os.path.join(STATIC, "app.js")], capture_output=True).returncode
        check("app.js passes node --check", rc == 0)
    else:
        check("app.js syntax", False, "node not found", skipped=True)

    html = _read("static/index.html")
    js = _read("static/app.js")
    ids_used = set(re.findall(r'\$\("([^"]+)"\)', js))
    ids_def = set(re.findall(r'id="([^"]+)"', html))
    missing_ids = sorted(ids_used - ids_def)
    check('every $("id") in app.js exists in index.html', not missing_ids,
          ("missing: " + ", ".join(missing_ids)) if missing_ids else f"{len(ids_used)} ids ok")

    py = _read("app.py")
    routes_used = set(re.findall(r'api\.(?:get|post)\("(/api/[^"?]+)"', js))
    routes_used |= set(re.findall(r'fetch\("(/api/[^"?]+)"', js))
    handled = set(re.findall(r'u\.path == "(/api/[^"]+)"', py))
    missing_routes = sorted(routes_used - handled)
    check("every /api route used by app.js is handled in app.py", not missing_routes,
          ("missing: " + ", ".join(missing_routes)) if missing_routes else f"{len(routes_used)} routes ok")

    if _have("nvim"):
        rc = subprocess.run(["nvim", "--server", NVIM_SOCK, "--remote-expr", "1"], capture_output=True, timeout=8)
        check("nvim editor server responds", rc.returncode == 0 and rc.stdout.strip() == b"1",
              "down - reconnect the terminal tab" if rc.returncode else "")
    else:
        check("nvim server", False, "nvim not found", skipped=True)

    return _summary()


def _hash(rel):
    try:
        return hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()
    except OSError:
        return None


def _read(rel):
    try:
        return open(os.path.join(WEB, rel)).read()
    except OSError:
        return ""


def _have(prog):
    from shutil import which
    return which(prog) is not None


def _summary():
    n, ok = len(_results), sum(_results)
    print(f"\n{'=' * 50}")
    if ok == n:
        print(f"  ALL {n} checks passed - the app is working.")
        return 0
    print(f"  {n - ok}/{n} checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
