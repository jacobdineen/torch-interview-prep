"""Web-app backend for the practice curriculum (stdlib only).

A thin HTTP API + static server that pairs a problem-description UI with your
REAL Neovim as the editor (embedded via ttyd in the page). The editor isn't an
emulation: the API drives one persistent nvim over its --listen socket, so
opening a problem is `nvim --remote <file>` and Run is a remote `:wa` + check.py.

  NVIM_SOCK=/tmp/mle_nvim.sock TTYD_PORT=7681 APIPORT=8000 python web/app.py

Started together with ttyd by web/serve-app.sh.
"""
import ast
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)
STATIC = os.path.join(WEB, "static")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NVIM_SOCK = os.environ.get("NVIM_SOCK", "/tmp/mle_nvim.sock")
TTYD_PORT = int(os.environ.get("TTYD_PORT", "7681"))
APIPORT = int(os.environ.get("APIPORT", "8000"))
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


# ---------- metadata ----------

def _problem_path(pid):
    hits = sorted(glob.glob(os.path.join(ROOT, "problems", f"p{pid}_*.py")))
    return hits[0] if hits else None


def _signature(tree):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            return f"def {node.name}({args})"
        if isinstance(node, ast.ClassDef):
            return f"class {node.name}"
    return ""


def _progress():
    pf = os.path.join(ROOT, ".progress.json")
    if not os.path.exists(pf):
        return {}
    try:
        with open(pf) as f:
            return json.load(f)
    except Exception:
        return {}


def _meta(pid):
    path = _problem_path(pid)
    if not path:
        return None
    with open(path) as f:
        src = f.read()
    try:
        tree = ast.parse(src)
        doc = (ast.get_docstring(tree) or "").strip()
    except SyntaxError:
        doc, tree = "", None
    lines = doc.splitlines()
    title = pid
    if lines:
        m = re.match(r"Problem\s+\S+:\s*(.+)", lines[0])
        if m:
            title = m.group(1).strip()
    # tier (shown as "difficulty"-style tag)
    tier_name = ""
    try:
        from curriculum import find_tier
        t = find_tier(pid)
        tier_name = t[1] if t else ""
    except Exception:
        pass
    concept = ""
    try:
        from concepts import get_concept
        concept = get_concept(pid) or ""
    except Exception:
        pass
    prog = _progress().get(pid, {})
    return {
        "id": pid,
        "title": title,
        "tier": tier_name,
        "signature": _signature(tree) if tree else "",
        "doc": doc,
        "concept": concept,
        "solved": bool(prog.get("ever_passed")),
        "last_status": prog.get("last_status"),
        "file": path,
    }


def _all_problems():
    out = []
    try:
        from curriculum import all_problem_ids
        ids = all_problem_ids()
    except Exception:
        ids = []
    prog = _progress()
    for pid in ids:
        path = _problem_path(pid)
        title = pid
        if path:
            try:
                with open(path) as f:
                    first = (ast.get_docstring(ast.parse(f.read())) or "").splitlines()
                if first:
                    m = re.match(r"Problem\s+\S+:\s*(.+)", first[0])
                    if m:
                        title = m.group(1).strip()
            except Exception:
                pass
        tier = ""
        try:
            from curriculum import find_tier
            t = find_tier(pid)
            tier = t[1] if t else ""
        except Exception:
            pass
        out.append({"id": pid, "title": title, "tier": tier,
                    "solved": bool(prog.get(pid, {}).get("ever_passed"))})
    return out


# ---------- nvim remote control ----------

def _nvim(*args, timeout=10):
    try:
        return subprocess.run(["nvim", "--server", NVIM_SOCK, *args],
                              capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return subprocess.CompletedProcess(args, 1, "", str(e))


def nvim_open(path):
    return _nvim("--remote", path).returncode == 0


def nvim_save_all():
    # Normal mode, then write all modified buffers, so Run tests the latest edits.
    _nvim("--remote-send", "<C-\\><C-N>:wa<CR>")


def teardown():
    """Stop everything: save + quit the editor, stop ttyd, drop the socket, then
    exit this API process. Idempotent and best-effort."""
    try:
        _nvim("--remote-send", "<C-\\><C-N>:wqa!<CR>", timeout=3)  # save then quit nvim
    except Exception:
        pass
    pid = os.environ.get("TTYD_PID")
    try:
        if pid:
            os.kill(int(pid), signal.SIGTERM)
        else:
            subprocess.run(["pkill", "-f", f"ttyd.*-p {TTYD_PORT}"], timeout=5)
    except Exception:
        pass
    try:
        if os.path.exists(NVIM_SOCK):
            os.remove(NVIM_SOCK)
    except Exception:
        pass
    # Exit the API shortly after the HTTP response has flushed.
    threading.Timer(0.4, lambda: os._exit(0)).start()


# ---------- run / debug via check.py ----------

def _run_check(pid, extra=()):
    env = dict(os.environ, PREP_JSON="1")
    proc = subprocess.run([PYTHON, os.path.join(ROOT, "check.py"), pid, *extra],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return {"status": "error", "message": (proc.stderr or proc.stdout or "no output").strip()[:4000]}


def _text_check(pid, extra):
    proc = subprocess.run([PYTHON, os.path.join(ROOT, "check.py"), pid, *extra],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    return (proc.stdout or "") + (proc.stderr or "")


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel):
        path = os.path.normpath(os.path.join(STATIC, rel.lstrip("/")))
        if not path.startswith(STATIC) or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = {".html": "text/html", ".js": "application/javascript",
                 ".css": "text/css"}.get(os.path.splitext(path)[1], "text/plain")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _body_json(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if u.path.startswith("/static/"):
            return self._serve_static(u.path[len("/static/"):])
        if u.path == "/api/config":
            return self._send(200, {"ttyd_port": TTYD_PORT})
        if u.path == "/api/problems":
            return self._send(200, _all_problems())
        if u.path == "/api/problem":
            pid = (q.get("id") or [""])[0]
            meta = _meta(pid)
            return self._send(200 if meta else 404, meta or {"error": "no such problem"})
        if u.path == "/api/hint":
            pid = (q.get("id") or [""])[0]
            return self._send(200, {"text": _text_check(pid, ["--hint"])})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        data = self._body_json()
        pid = str(data.get("id", ""))
        if u.path == "/api/open":
            if not _problem_path(pid):
                return self._send(404, {"error": "no such problem"})
            ok = nvim_open(_problem_path(pid))
            return self._send(200, {"ok": ok})
        if u.path == "/api/run":
            nvim_save_all()
            return self._send(200, _run_check(pid))
        if u.path == "/api/solution":
            extra = ["--solution"] + (["--i-give-up"] if data.get("give_up") else [])
            return self._send(200, {"text": _text_check(pid, extra)})
        if u.path == "/api/shutdown":
            self._send(200, {"ok": True})
            teardown()
            return
        return self._send(404, {"error": "not found"})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", APIPORT), Handler)
    print(f"API on http://127.0.0.1:{APIPORT}  (nvim socket {NVIM_SOCK}, ttyd :{TTYD_PORT})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
