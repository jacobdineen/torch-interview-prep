"""Web-app backend for the practice curriculum (stdlib only).

A thin HTTP API + static server that pairs a problem-description UI with your
REAL Neovim as the editor (embedded via ttyd in the page). The editor isn't an
emulation: the API drives one persistent nvim over its --listen socket, so
opening an item is `nvim --remote <file>` and Run is a remote `:wa` + the test.

The catalog spans BOTH standalone problems and every project's steps. Items are
addressed by a key: `prob:<id>` (e.g. prob:04a) or `proj:<name>:<sid>`
(e.g. proj:tiny-gpt-from-scratch:0012).

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


# ---------- shared ----------

def _signature(tree):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            return f"def {node.name}({args})"
        if isinstance(node, ast.ClassDef):
            return f"class {node.name}"
    return ""


def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------- problems ----------

def _problem_path(pid):
    hits = sorted(glob.glob(os.path.join(ROOT, "problems", f"p{pid}_*.py")))
    return hits[0] if hits else None


def _problem_progress():
    return _load_json(os.path.join(ROOT, ".progress.json"))


def _problem_title(pid):
    path = _problem_path(pid)
    if not path:
        return pid
    try:
        with open(path) as f:
            doc = (ast.get_docstring(ast.parse(f.read())) or "").splitlines()
        if doc:
            m = re.match(r"Problem\s+\S+:\s*(.+)", doc[0])
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return pid


def _tier(pid):
    try:
        from curriculum import find_tier
        t = find_tier(pid)
        return t[1] if t else ""
    except Exception:
        return ""


# ---------- projects ----------

def _project_dirs():
    return sorted(d for d in glob.glob(os.path.join(ROOT, "projects", "*"))
                  if os.path.isfile(os.path.join(d, "project.json")))


def _manifest(proj_dir):
    return _load_json(os.path.join(proj_dir, "project.json"))


def _project_progress(proj_dir):
    return _load_json(os.path.join(proj_dir, ".project_progress.json"))


def _step_path(proj_dir, sid, name):
    return os.path.join(proj_dir, "steps", f"{sid}_{name}.py")


def _part(man, idx):
    parts = man.get("parts", [])
    return parts[idx] if 0 <= idx < len(parts) else {"title": "", "description": ""}


# ---------- catalog + resolution ----------

def _catalog():
    """All selectable items (problems + built project steps), with a source list
    for grouping/filtering in the picker."""
    items, sources = [], ["Problems"]
    prog = _problem_progress()
    try:
        from curriculum import all_problem_ids
        ids = all_problem_ids()
    except Exception:
        ids = []
    import frameworks
    for pid in ids:
        items.append({"key": f"prob:{pid}", "kind": "problem", "source": "Problems",
                      "group": _tier(pid) or "Problems", "id": pid,
                      "title": _problem_title(pid), "framework": frameworks.problem_framework(pid),
                      "numpy": frameworks.problem_has_numpy(pid),
                      "solved": bool(prog.get(pid, {}).get("ever_passed"))})
    for d in _project_dirs():
        man = _manifest(d)
        name = man.get("name", os.path.basename(d))
        sources.append(name)
        fw = frameworks.project_framework(d)
        pp = _project_progress(d)
        for s in man.get("steps", []):
            if not os.path.exists(_step_path(d, s["id"], s["name"])):
                continue  # staged build: not yet generated
            items.append({"key": f"proj:{name}:{s['id']}", "kind": "project",
                          "source": name, "framework": fw,
                          "group": f"Part {s.get('part', 0) + 1}: {_part(man, s.get('part', 0))['title']}",
                          "id": s["id"], "title": s["name"],
                          "solved": bool(pp.get(s["id"], {}).get("ever_passed"))})
    return {"sources": sources, "items": items, "frameworks": [frameworks.NUMPY, frameworks.TORCH]}


def _resolve(key):
    if key.startswith("prob:"):
        pid = key[5:]
        path = _problem_path(pid)
        return {"kind": "problem", "pid": pid, "path": path} if path else None
    if key.startswith("proj:"):
        try:
            _, name, sid = key.split(":", 2)
        except ValueError:
            return None
        d = os.path.join(ROOT, "projects", name)
        if not os.path.isfile(os.path.join(d, "project.json")):
            return None
        man = _manifest(d)
        s = next((x for x in man.get("steps", []) if x["id"] == sid), None)
        if not s:
            return None
        path = _step_path(d, sid, s["name"])
        if not os.path.exists(path):
            return None
        return {"kind": "project", "name": name, "sid": sid, "step": s,
                "manifest": man, "dir": d, "path": path}
    return None


def _item_meta(key):
    r = _resolve(key)
    if not r:
        return None
    if r["kind"] == "problem":
        pid = r["pid"]
        with open(r["path"]) as f:
            src = f.read()
        try:
            tree = ast.parse(src)
            full = (ast.get_docstring(tree) or "").strip()
        except SyntaxError:
            tree, full = None, ""
        # Body = docstring minus the "Problem NN: title" line (shown as the H1)
        # and the internal "(Split from parent ...)" artifact.
        body = [ln for ln in full.splitlines()[1:]
                if not ln.strip().startswith("(Split from parent problem")]
        prog = _problem_progress().get(pid, {})
        concept = ""
        try:
            from concepts import get_concept
            concept = get_concept(pid) or ""
        except Exception:
            pass
        import frameworks
        return {"key": key, "kind": "problem", "id": pid, "title": _problem_title(pid),
                "source": "Problems", "group": _tier(pid),
                "framework": frameworks.framework_of_source(src),
                "numpy": frameworks.problem_has_numpy(pid),
                "signature": _signature(tree) if tree else "",
                "doc": "\n".join(body).strip(), "concept": concept,
                "solved": bool(prog.get("ever_passed")), "last_status": prog.get("last_status")}
    import frameworks
    man, s = r["manifest"], r["step"]
    part = _part(man, s.get("part", 0))
    pp = _project_progress(r["dir"]).get(s["id"], {})
    doc = s.get("doc", "")
    if part.get("description"):
        doc = (doc + "\n\n" + part["description"]).strip()
    return {"key": key, "kind": "project", "id": s["id"], "title": s["name"],
            "source": man.get("title", r["name"]),
            "group": f"Part {s.get('part', 0) + 1}: {part['title']}",
            "framework": frameworks.project_framework(r["dir"]),
            "signature": s.get("signature", ""), "doc": doc, "concept": "",
            "solved": bool(pp.get("ever_passed")), "last_status": pp.get("last_status")}


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
    _nvim("--remote-send", "<C-\\><C-N>:wa<CR>")


def teardown():
    """Stop everything: save + quit the editor, stop ttyd, drop the socket, then
    exit this API process. Idempotent and best-effort."""
    try:
        _nvim("--remote-send", "<C-\\><C-N>:wqa!<CR>", timeout=3)
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
    threading.Timer(0.4, lambda: os._exit(0)).start()


# ---------- run / debug ----------

def _parse_json_line(proc):
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return {"status": "error", "message": (proc.stderr or proc.stdout or "no output").strip()[:4000]}


def _run_item(key):
    r = _resolve(key)
    if not r:
        return {"status": "error", "message": "unknown item"}
    nvim_save_all()
    if r["kind"] == "problem":
        cmd = [PYTHON, os.path.join(ROOT, "check.py"), r["pid"]]
    else:
        cmd = [PYTHON, r["path"]]  # step file's __main__ -> project_runner, emits PREP_JSON
    env = dict(os.environ, PREP_JSON="1")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    return _parse_json_line(proc)


def _text(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    return (proc.stdout or "") + (proc.stderr or "")


def _hint_text(key):
    r = _resolve(key)
    if not r:
        return "unknown item"
    if r["kind"] == "problem":
        return _text([PYTHON, os.path.join(ROOT, "check.py"), r["pid"], "--hint"])
    return ("Project steps don't have graded hints — read the part description under "
            "the title, or use Solution.")


def _solution_text(key, give_up):
    r = _resolve(key)
    if not r:
        return "unknown item"
    extra = ["--solution"] + (["--i-give-up"] if give_up else [])
    if r["kind"] == "problem":
        return _text([PYTHON, os.path.join(ROOT, "check.py"), r["pid"], *extra])
    return _text([PYTHON, os.path.join(ROOT, "projects.py"), r["name"], r["sid"], *extra])


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
        if u.path == "/api/catalog":
            return self._send(200, _catalog())
        if u.path == "/api/item":
            meta = _item_meta((q.get("key") or [""])[0])
            return self._send(200 if meta else 404, meta or {"error": "no such item"})
        if u.path == "/api/hint":
            return self._send(200, {"text": _hint_text((q.get("key") or [""])[0])})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        data = self._body_json()
        key = str(data.get("key", ""))
        if u.path == "/api/open":
            r = _resolve(key)
            if not r:
                return self._send(404, {"error": "no such item"})
            return self._send(200, {"ok": nvim_open(r["path"])})
        if u.path == "/api/run":
            return self._send(200, _run_item(key))
        if u.path == "/api/solution":
            return self._send(200, {"text": _solution_text(key, bool(data.get("give_up")))})
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
