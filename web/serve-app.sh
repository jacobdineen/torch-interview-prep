#!/usr/bin/env bash
# The full web app: a problem-description UI with your REAL nvim as the editor.
#
# Starts two local services and ties them together:
#   - ttyd on $TTYD_PORT  -> your real nvim, with a --listen socket for remote control
#   - a stdlib API on $APIPORT -> problem metadata, run (check.py), and nvim remote-open
# Open http://127.0.0.1:$APIPORT in a browser. Both bind loopback only; reach a
# remote box by tunnelling BOTH ports (ssh -L $APIPORT:localhost:$APIPORT \
# -L $TTYD_PORT:localhost:$TTYD_PORT ...). See docs/web-nvim.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTYD_PORT="${TTYD_PORT:-7681}"
APIPORT="${APIPORT:-8000}"
SOCK="${NVIM_SOCK:-/tmp/mle_nvim.sock}"
PYTHON="$REPO/.venv/bin/python"; [ -x "$PYTHON" ] || PYTHON="python3"

command -v ttyd >/dev/null || { echo "ttyd not found: sudo apt-get install -y ttyd" >&2; exit 1; }

rm -f "$SOCK"

# 1) ttyd + nvim (with the remote socket). Loopback only.
NVIM_LISTEN="$SOCK" PORT="$TTYD_PORT" BIND=lo "$REPO/web/serve-nvim.sh" >/tmp/mle_ttyd.log 2>&1 &
TTYD_PID=$!

# Wait for nvim's socket so the first remote-open works.
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.1; done

# 2) API + static UI. Pass TTYD_PID so the in-app "Tear down" button can stop ttyd.
NVIM_SOCK="$SOCK" TTYD_PORT="$TTYD_PORT" APIPORT="$APIPORT" TTYD_PID="$TTYD_PID" \
  "$PYTHON" "$REPO/web/app.py" &
API_PID=$!

cleanup() { kill "$TTYD_PID" "$API_PID" 2>/dev/null || true; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM

echo
echo "  mle_prep web app:  http://127.0.0.1:$APIPORT"
echo "  (editor = real nvim via ttyd :$TTYD_PORT; loopback only)"
echo "  Ctrl-C to stop."
echo
wait
