#!/usr/bin/env bash
# Tear down the web app from the CLI: stop ttyd, the API, and clear the nvim
# socket. Same effect as the in-app "Tear down" button. Safe to run anytime.
#
#   ./web/teardown.sh
TTYD_PORT="${TTYD_PORT:-7681}"
APIPORT="${APIPORT:-8000}"
SOCK="${NVIM_SOCK:-/tmp/mle_nvim.sock}"

pkill -f "ttyd.*-p ${TTYD_PORT}" 2>/dev/null && echo "stopped ttyd (:${TTYD_PORT})" || echo "no ttyd on :${TTYD_PORT}"
pkill -f "web/app.py" 2>/dev/null && echo "stopped API (:${APIPORT})" || echo "no API process"
rm -f "$SOCK" && echo "removed socket $SOCK"
echo "torn down."
