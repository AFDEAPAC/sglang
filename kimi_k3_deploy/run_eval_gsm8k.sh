#!/usr/bin/env bash
# GSM8K Chat API, 200 examples, 5-shot, temp=0, 8 threads. Run on node0 against :30000.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
TAG="${1:-eval}"
LOG="${LOG_DIR}/gsm8k_${TAG}.log"

export PYTHONPATH=/opt/sglang/python:/opt/aiter/aiter/jit/utils
export PYTHONUNBUFFERED=1

if [[ ! -f /tmp/test.jsonl ]]; then
  python3 - <<'PY'
from sglang.utils import download_and_cache_file
url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
print(download_and_cache_file(url))
PY
fi

echo "GSM8K_START $(date -u +%FT%TZ) tag=${TAG}" | tee "${LOG}"
set +e
python3 -m sglang.test.run_eval \
  --eval-name gsm8k --api chat \
  --host 127.0.0.1 --port 30000 --model kimi-k3 \
  --num-examples 200 --num-threads 8 \
  --num-shots 5 --max-tokens 512 --temperature 0 \
  --gsm8k-data-path /tmp/test.jsonl \
  2>&1 | tee -a "${LOG}"
ec=${PIPESTATUS[0]}
echo "GSM8K_EXIT:${ec} $(date -u +%FT%TZ)" | tee -a "${LOG}"
exit "${ec}"
