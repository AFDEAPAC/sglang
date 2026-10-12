#!/usr/bin/env bash
set -euo pipefail
C="${C:-20260910_tp8pp2}"
NODE2="${NODE2:-10.170.168.2}"
PORT="${PORT:-30000}"
SRV_PREFIX='ulimit -c 0;'
# SGLANG_AITER_FUSED_KDA_DECODE is deliberately ABSENT: it was adopted 2026-09-16
# (+2.59% TPOT) and now comes from /opt/scripts/common.env via
# tools/apply_adopt_kda_fused.py. Pinning it to 0 here would silently un-adopt it on
# every "restore production", which is what this comment exists to prevent.
SRV_ENV='HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 AITER_A16WMIX_FP8=1 AITER_A16WMIX_FP8_S2=1 TP_SIZE=8 PP_SIZE=2 EP_SIZE=1 DP_SIZE=1 ENABLE_DSPARK=0 ATTENTION_BACKEND=aiter CUDA_GRAPH_BACKEND_DECODE=full CUDA_GRAPH_MAX_BS_DECODE=32 SGLANG_ROCM_K3_FUSE_KDA_INPROJ=1 SGLANG_K3_RADIX4_TOPK=1 SGLANG_DECODE_MLA_ABSORB_V_BUF=1 SGLANG_USE_AITER_FP8_PER_TOKEN=1 SGLANG_PP_LAYER_PARTITION=47,46 MODEL_PATH=/mnt/md0/mh/models/Kimi-K3-fp8v3 AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=/opt/scripts/configs/a8w8_bpreshuffle_tuned_gfx942.csv EXTRA_ARGS=--disable-cuda-graph-padding SGLANG_AITER_MLA_GLUON=0'

docker exec "$C" bash -lc 'pkill -9 -f "python.*sglang" 2>/dev/null; /opt/scripts/stop_kimi_k3.sh 2>/dev/null' || true
ssh -o BatchMode=yes "$NODE2" "docker exec $C bash -lc 'pkill -9 -f python.*sglang 2>/dev/null; /opt/scripts/stop_kimi_k3.sh 2>/dev/null'" || true
sleep 20
docker exec "$C" bash -lc "$SRV_PREFIX $SRV_ENV /opt/scripts/start_node0.sh" >/dev/null 2>&1 &
ssh -o BatchMode=yes "$NODE2" "docker exec $C bash -lc \"$SRV_PREFIX $SRV_ENV /opt/scripts/start_node1.sh\"" >/dev/null 2>&1 &
for i in $(seq 1 150); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
  [[ -n "$code" ]] || code=000
  [[ "$code" == "200" ]] && { echo "health=200 after ${i}0s"; exit 0; }
  sleep 10
done
echo "FAIL health=$code"; exit 1
