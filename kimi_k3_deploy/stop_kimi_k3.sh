#!/usr/bin/env bash
# Stop every SGLang process on this node and free the GPUs.
set -uo pipefail
LOG_DIR="/opt/scripts/logs"

for f in "${LOG_DIR}"/sglang_node*.pid; do
  [[ -f "$f" ]] || continue
  pid="$(cat "$f" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
done
sleep 2
pkill -9 -f "pd_router.py" 2>/dev/null || true
pkill -9 -f "sglang" 2>/dev/null || true
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 3
echo "remaining sglang procs:"
pgrep -af sglang || echo "  none"
echo "vram in use:"
rocm-smi --showmeminfo vram 2>/dev/null | grep -i "Used Memory" | head -8
