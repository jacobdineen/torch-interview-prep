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
import time
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


def static_checks():
    """No-server checks (CI-safe): contract, syntax, token wiring, parse-ability."""
    print("static checks (no server)\n")
    html = _read("static/index.html"); js = _read("static/app.js"); py = _read("app.py")
    check("index.html / app.js / app.py readable", bool(html and js and py))
    ids_used = set(re.findall(r'\$\("([^"]+)"\)', js))
    ids_def = set(re.findall(r'id="([^"]+)"', html))
    missing_ids = sorted(ids_used - ids_def)
    check('every $("id") in app.js exists in index.html', not missing_ids,
          ("missing: " + ", ".join(missing_ids)) if missing_ids else f"{len(ids_used)} ids ok")
    routes_used = set(re.findall(r'(?:api\.(?:get|post)|fetch)\("(/api/[a-z_]+)', js))
    handled = set(re.findall(r'u\.path == "(/api/[^"]+)"', py))
    missing_routes = sorted(routes_used - handled)
    check("every /api route used by app.js is handled in app.py", not missing_routes,
          ("missing: " + ", ".join(missing_routes)) if missing_routes else f"{len(routes_used)} routes ok")
    check("frontend wires the CSRF token (X-MLE-Token + TOKEN=cfg.token)",
          "X-MLE-Token" in js and "TOKEN = cfg.token" in js)
    check("app.py never writes ALL buffers (:wa/:wqa clobber guard)",
          ":wa<CR>" not in py and ":wqa" not in py and "execute('silent! update')" in py)
    import ast
    try:
        ast.parse(py); check("app.py parses (python AST)", True)
    except SyntaxError as e:
        check("app.py parses (python AST)", False, str(e))
    if _have("node"):
        rc = subprocess.run(["node", "--check", os.path.join(STATIC, "app.js")], capture_output=True).returncode
        check("app.js passes node --check", rc == 0)
    else:
        check("app.js syntax", False, "node not found", skipped=True)
    for sh in ("serve-app.sh", "serve-nvim.sh", "mle-nvim-ensure.sh"):
        rc = subprocess.run(["bash", "-n", os.path.join(WEB, sh)], capture_output=True).returncode
        check(f"{sh} is valid bash", rc == 0)
    return _summary()


def main():
    if "--static" in sys.argv:
        return static_checks()
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
        openr = {}
        for _ in range(20):   # the persistent nvim server may still be loading config right after start
            _, openr = _json("/api/open", method="POST", token=token, body={"key": a_prob["key"]})
            if openr.get("ok"):
                break
            time.sleep(0.5)
        check("/api/open with token -> ok:true (editor switches)", openr.get("ok") is True, str(openr))
    for route in ("/api/run", "/api/solution", "/api/shutdown"):
        c, _ = _req(route, "POST", token=None, body={"key": "prob:04a"})
        check(f"CSRF: {route} without token -> 403", c == 403, f"http {c}")

    if a_prob:
        # open a sentinel file in nvim and dirty it, so a run that wrongly :wa-saved
        # would overwrite it -> this actually reproduces the clobber regression
        sentinel = "/tmp/mle_smoke_sentinel.txt"
        try:
            open(sentinel, "w").write("SENTINEL_ORIGINAL\n")
            _nvim_remote(sentinel)
            _nvim_send(":call append(0, 'DIRTY')\r")
        except Exception:
            sentinel = None
        before = _hash("web/static/app.js")
        code, rr = _json("/api/run", method="POST", token=token, body={"key": a_prob["key"]})
        check("/api/run grades (status pass/fail, NOT error)",
              code == 200 and rr.get("status") in ("pass", "fail"),
              f"{a_prob['id']} -> {rr.get('status')}: {(rr.get('message') or '')[:60]}")
        after = _hash("web/static/app.js")
        check("a run does NOT modify web/static/app.js", before == after and before is not None)
        if sentinel:
            kept = open(sentinel).read().strip() == "SENTINEL_ORIGINAL"
            check("a run does NOT write an unrelated dirty buffer (sentinel)", kept)
            try:
                _nvim_send(":bd! " + sentinel + "\r"); os.remove(sentinel)
            except Exception:
                pass
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
    routes_used = set(re.findall(r'(?:api\.(?:get|post)|fetch)\("(/api/[a-z_]+)', js))
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


def _nvim_remote(path):
    subprocess.run(["nvim", "--server", NVIM_SOCK, "--remote", path], timeout=8)


def _nvim_send(keys):
    subprocess.run(["nvim", "--server", NVIM_SOCK, "--remote-send", keys], timeout=8)


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
