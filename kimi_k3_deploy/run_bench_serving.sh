#!/usr/bin/env bash
# Serving bench: random 4096/512, concurrency 8, 32 prompts. Run on node0 against :30000.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
TAG="${1:-eval}"
LOG="${LOG_DIR}/bench_${TAG}_4096_512_c8_n32.log"

export PYTHONPATH=/opt/sglang/python:/opt/aiter/aiter/jit/utils
export PYTHONUNBUFFERED=1

pkill -f 'python3 -m sglang.bench_serving' 2>/dev/null || true
sleep 1

echo "BENCH_START $(date -u +%FT%TZ) tag=${TAG}" | tee "${LOG}"
set +e
python3 -u -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model kimi-k3 \
  --dataset-name random \
  --random-input-len 4096 --random-output-len 512 \
  --request-rate inf --max-concurrency 8 --num-prompts 32 \
  2>&1 | tee -a "${LOG}"
ec=${PIPESTATUS[0]}
echo "BENCH_EXIT:${ec} $(date -u +%FT%TZ)" | tee -a "${LOG}"
exit "${ec}"
