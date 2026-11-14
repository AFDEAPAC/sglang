#!/usr/bin/env bash
# Health + smoke check for the Kimi-K3 server (run on the master node, rank 0).
set -uo pipefail
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
BASE="http://${HOST}:${PORT}"

echo "== /health =="
curl -sS -m 10 -o /dev/null -w "http=%{http_code}\n" "${BASE}/health" || echo "unreachable"

echo "== /v1/models =="
curl -sS -m 10 "${BASE}/v1/models" || echo "unreachable"
echo

echo "== /v1/chat/completions =="
curl -sS -m 180 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":16,"temperature":0}'
echo
