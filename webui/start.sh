#!/usr/bin/env bash
# Build the React app and serve it with the local API on http://127.0.0.1:8601 (loopback only).
# For live-reload development run `npm run dev` in a second terminal (http://127.0.0.1:8600).
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d node_modules ]]; then npm install --no-fund --no-audit; fi
if [[ "${1:-}" != "--no-build" ]]; then npm run build; fi
cd ..
exec .venv/bin/python -m webui.api.main
