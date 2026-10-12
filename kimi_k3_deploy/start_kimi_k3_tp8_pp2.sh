#!/usr/bin/env bash
# Kimi-K3 2-node launcher for AMD MI308X (gfx942), TP=8 x PP=2.
#
# Usage inside the container:
#   /opt/scripts/start_kimi_k3_tp8_pp2.sh 0     # on 10.170.168.1
#   /opt/scripts/start_kimi_k3_tp8_pp2.sh 1     # on 10.170.168.2
#
# MoE backend (gfx942 / CDNA3): Kimi-K3 is MXFP4 with the SiTU (SiTUv2) activation.
# CK 2stages has no situv2 ActOP, and the FlyDSL a8w4 / a4w4 variants are gfx950-only,
# so the only kernel that can run this MoE is the a16w4 (bf16 activation x MXFP4
# weight) pair from AFDEAPAC/aiter 3e7aa9214, cherry-picked into /opt/aiter. It
# dispatches only when activations stay bf16, so AITER_SITUV2_A8W4 / _A4W4 must
# remain unset, and only when there is no expert mask, hence EP_SIZE=1.
set -euo pipefail

NODE_RANK="${1:-${NODE_RANK:-}}"
if [[ -z "${NODE_RANK}" || ( "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ) ]]; then
  echo "ERROR: NODE_RANK must be 0 or 1" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.env"

# RDMA needs unlocked memory; container default memlock (8192) breaks NCCL_NET=IB.
ulimit -l unlimited 2>/dev/null || true
ulimit -n 1048576 2>/dev/null || true

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sglang_node${NODE_RANK}.log"
PID_FILE="${LOG_DIR}/sglang_node${NODE_RANK}.pid"

if [[ "${NODE_RANK}" == "0" ]]; then
  export SGLANG_HOST_IP="${NODE0_IP}"
else
  export SGLANG_HOST_IP="${NODE1_IP}"
fi

# --- IB / RCCL / NCCL (see common.env for HCA / GID rationale) ---
# Adapted from lab example + verified fix:
#   NCCL_NET=IB
#   NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
#   NCCL_IB_GID_INDEX=0
#   NCCL_SOCKET_IFNAME=ens5f0np0
#   NCCL_NET_GDR_LEVEL=0   # required: no peermem; DMAbuf ibv_reg_mr_iova2 fails
#   RCCL_AINIC_ROCE=0
ENABLE_NCCL_IB="${ENABLE_NCCL_IB:-1}"
export NCCL_SOCKET_IFNAME="${NETWORK_IFACE}"
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME="${NETWORK_IFACE}"
export RCCL_SOCKET_IFNAME="${NETWORK_IFACE}"
export RCCL_AINIC_ROCE="${RCCL_AINIC_ROCE}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"

if [[ "${ENABLE_NCCL_IB}" == "1" ]]; then
  export NCCL_NET="${NCCL_NET:-IB}"
  export NCCL_IB_HCA="${NCCL_IB_HCA}"
  export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"
  export NCCL_IB_DISABLE=0
else
  export NCCL_IB_HCA="${NCCL_IB_HCA}"
  export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"
  export NCCL_IB_DISABLE=1
  unset NCCL_NET || true
fi

# NCCL channel / buffer: serving default is 8–64 (best of 2026-08-20 A/B).
# UNSET_NCCL_NCHANNELS=1 is trial-only; do not use for serving.
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ -n "${NCCL_DEBUG_SUBSYS:-}" ]]; then
  export NCCL_DEBUG_SUBSYS
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
UNSET_NCCL_NCHANNELS="${UNSET_NCCL_NCHANNELS:-0}"
if [[ "${UNSET_NCCL_NCHANNELS}" == "1" ]]; then
  unset NCCL_MAX_NCHANNELS || true
  unset NCCL_MIN_NCHANNELS || true
  NCCL_CHANNEL_INFO="unset (RCCL default)"
else
  export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-64}"
  export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
  NCCL_CHANNEL_INFO="${NCCL_MIN_NCHANNELS}-${NCCL_MAX_NCHANNELS}"
fi

if [[ -n "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]]; then
  export CUDA_DEVICE_MAX_CONNECTIONS
fi
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-0}"

export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
export GPU_FORCE_BLIT_COPY_SIZE="${GPU_FORCE_BLIT_COPY_SIZE:-4090000}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"

export SGLANG_USE_AITER="1"
export SGLANG_AITER_K3_OPT="1"
export AITER_SITUV2_A8W4="0"
export AITER_SITUV2_A4W4="0"
export SGLANG_ROCM_FUSED_DECODE_MLA="0"
export ENABLE_CUSTOM_AR="${ENABLE_CUSTOM_AR:-1}"
export SGLANG_USE_AITER_AR="${SGLANG_USE_AITER_AR:-1}"
export ENABLE_AITER_AR_FUSION="${ENABLE_AITER_AR_FUSION:-1}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="0"
export SGLANG_K3_ATTN_RES_MODE="jit"
export SGLANG_MOE_FUSED_GATE_RADIX="1"
export SGLANG_MLA_DECODE_TUNE="${SGLANG_MLA_DECODE_TUNE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib64/libibverbs:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Per-shape hipBLASLt solution overrides; must be set before the process starts because
# decode solutions are bound at CUDA-graph capture. Silently ignored if the file is absent.
export HIPBLASLT_TUNING_OVERRIDE_FILE="${HIPBLASLT_TUNING_OVERRIDE_FILE:-/opt/hipblaslt_tuning.txt}"
[[ -f "${HIPBLASLT_TUNING_OVERRIDE_FILE}" ]] || unset HIPBLASLT_TUNING_OVERRIDE_FILE
# Required for rocprofv3 --attach / rocprof-attach on a live process.
export ROCP_TOOL_ATTACH="${ROCP_TOOL_ATTACH:-1}"
# One-shot dump of the a16wmix call invariants; cheap and only fires once per rank.
# Debug dumps are expensive and previously left max_eid garbage visible; keep off by default.
export AITER_A16WMIX_DEBUG="${AITER_A16WMIX_DEBUG:-0}"
export FLYDSL_A16WMIX_TUNED_CSV="${FLYDSL_A16WMIX_TUNED_CSV:-/opt/scripts/configs/a16wmix_tuned.csv}"
if [[ -n "${AITER_CONFIG_GEMM_BF16:-}" ]]; then
  export AITER_CONFIG_GEMM_BF16
fi

if [[ "${EP_SIZE}" != "1" ]]; then
  # The a16w4 kernel refuses an expert mask unless this guard is lifted.
  export AITER_A16W4_ALLOW_EP="1"
  # Sort global router IDs through the EP mask, then cap GEMM grids to local E.
  # Idempotent and must run before SGLang imports aiter.
  python3 "${SCRIPT_DIR}/patch_a16wmix_ep_global_experts.py"
fi

# Prefill-decode disaggregation: each node is its own nnodes=1 world.
# Rank 0 = prefill (API PD_PREFILL_PORT), rank 1 = decode (PD_DECODE_PORT).
PD_DISAGG="${PD_DISAGG:-0}"
SGLANG_NODE_RANK="${NODE_RANK}"
if [[ "${PD_DISAGG}" == "1" ]]; then
  NNODES=1
  PP_SIZE=1
  SGLANG_NODE_RANK=0
  DIST_INIT_ADDR="127.0.0.1:20000"
  ENABLE_DSPARK=0
  DP_SIZE=1
  ENABLE_DP_ATTENTION=0
  ENABLE_DP_LM_HEAD=0
  if [[ "${NODE_RANK}" == "0" ]]; then
    DISAGGREGATION_MODE="prefill"
    SERVE_PORT="${PD_PREFILL_PORT:-30000}"
  else
    DISAGGREGATION_MODE="decode"
    SERVE_PORT="${PD_DECODE_PORT:-30001}"
  fi
  DISAGGREGATION_TRANSFER_BACKEND="${PD_TRANSFER_BACKEND:-mooncake_tcp}"
  DISAGGREGATION_BOOTSTRAP_PORT="${PD_BOOTSTRAP_PORT:-8998}"
  echo "[INFO] PD disagg: mode=${DISAGGREGATION_MODE} backend=${DISAGGREGATION_TRANSFER_BACKEND} port=${SERVE_PORT} bootstrap=${DISAGGREGATION_BOOTSTRAP_PORT}"
  python3 - <<'PY' || true
import traceback
try:
    from mooncake.engine import TransferEngine
    print("[INFO] mooncake TransferEngine import OK")
except Exception as e:
    print("[WARN] mooncake TransferEngine import failed:", type(e).__name__, e)
    traceback.print_exc()
PY
fi

# DSPARK requires pp_size == 1 (SGLang server_args + speculative_hook).
ENABLE_DSPARK="${ENABLE_DSPARK:-0}"
if [[ "${ENABLE_DSPARK}" == "1" ]]; then
  DSPARK_LAYOUT="${DSPARK_LAYOUT:-2node}"
  PP_SIZE=1
  if [[ "${DSPARK_LAYOUT}" == "1node" ]]; then
    NNODES=1
    TP_SIZE=8
    if [[ "${NODE_RANK}" != "0" ]]; then
      echo "ERROR: DSpark 1node layout only starts NODE_RANK=0 (PP must be 1; TP=8 on node0)." >&2
      exit 1
    fi
  elif [[ "${DSPARK_LAYOUT}" == "2node" ]]; then
    NNODES=2
    TP_SIZE=16
  elif [[ "${DSPARK_LAYOUT}" == "keep" ]]; then
    # Keep TP/EP/DP/nnodes from env; only force PP=1.
    :
  else
    echo "ERROR: DSPARK_LAYOUT must be 1node, 2node, or keep (got ${DSPARK_LAYOUT})" >&2
    exit 1
  fi
  if [[ ! -f "${SPECULATIVE_DRAFT_MODEL_PATH}/config.json" ]]; then
    echo "ERROR: missing DSpark config ${SPECULATIVE_DRAFT_MODEL_PATH}/config.json" >&2
    exit 1
  fi
  # aiter MLA ASM has no qseqlen=8 heuristic (DSPARK verify = gamma+1).
  # Draft worker already falls back to triton; target verify must too.
  ATTENTION_BACKEND="triton"
  LINEAR_ATTN_VERIFY_BACKEND="${LINEAR_ATTN_VERIFY_BACKEND:-triton}"
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: missing ${MODEL_PATH}/config.json" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  old="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old}" ]]; then
    kill "${old}" 2>/dev/null || true
    sleep 1
    kill -9 "${old}" 2>/dev/null || true
  fi
fi
pkill -f "sglang.*(serve|launch_server).*Kimi-K3" 2>/dev/null || true
sleep 2

echo "[INFO] NODE_RANK=${NODE_RANK} DIST_INIT=${DIST_INIT_ADDR} TP=${TP_SIZE} PP=${PP_SIZE} EP=${EP_SIZE} DP=${DP_SIZE:-1} dp_attn=${ENABLE_DP_ATTENTION:-0} dp_lm=${ENABLE_DP_LM_HEAD:-0} a2a=${MOE_A2A_BACKEND:-none} nnodes=${NNODES}"
echo "[INFO] attention=${ATTENTION_BACKEND} linear_attn=${LINEAR_ATTN_BACKEND} moe=${MOE_RUNNER_BACKEND} mla_tune=${SGLANG_MLA_DECODE_TUNE}"
echo "[INFO] DSpark: enable=${ENABLE_DSPARK} layout=${DSPARK_LAYOUT:-} draft=${SPECULATIVE_DRAFT_MODEL_PATH:-} block=${SPECULATIVE_DSPARK_BLOCK_SIZE:-} replayssm_spec=${ENABLE_LINEAR_REPLAYSSM_SPEC:-0}"
echo "[INFO] AR: custom=${ENABLE_CUSTOM_AR:-0} aiter_ar=${SGLANG_USE_AITER_AR} fusion=${ENABLE_AITER_AR_FUSION:-0}"
echo "[INFO] alloc=${PYTORCH_CUDA_ALLOC_CONF:-} blit=${GPU_FORCE_BLIT_COPY_SIZE:-} sdma=${HSA_ENABLE_SDMA:-}"
echo "[INFO] cuda_graph decode=${CUDA_GRAPH_BACKEND_DECODE:-disabled} prefill=${CUDA_GRAPH_BACKEND_PREFILL:-disabled} max_bs_decode=${CUDA_GRAPH_MAX_BS_DECODE:-}"
echo "[INFO] radix_cache=disabled (--disable-radix-cache)"
echo "[INFO] IB: enable=${ENABLE_NCCL_IB} NET=${NCCL_NET:-(auto/socket)} HCA=${NCCL_IB_HCA} GID=${NCCL_IB_GID_INDEX} GDR=${NCCL_NET_GDR_LEVEL} SOCK=${NCCL_SOCKET_IFNAME} IB_DISABLE=${NCCL_IB_DISABLE} AINIC_ROCE=${RCCL_AINIC_ROCE}"
echo "[INFO] NCCL channels=${NCCL_CHANNEL_INFO} buff=${NCCL_BUFFSIZE} unset_flag=${UNSET_NCCL_NCHANNELS} max_conn=${CUDA_DEVICE_MAX_CONNECTIONS:-} nccl_hi=${TORCH_NCCL_HIGH_PRIORITY:-0}"
echo "[INFO] copy: GPU_FORCE_BLIT_COPY_SIZE=${GPU_FORCE_BLIT_COPY_SIZE:-} HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-}"
echo "[INFO] log=${LOG_FILE}"

if command -v sglang >/dev/null 2>&1; then
  LAUNCH=(sglang serve)
else
  LAUNCH=(python3 -m sglang.launch_server)
fi

ARGS=(
  --trust-remote-code
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tp-size "${TP_SIZE}"
  --pp-size "${PP_SIZE}"
  --nnodes "${NNODES}"
  --node-rank "${SGLANG_NODE_RANK}"
  --dist-init-addr "${DIST_INIT_ADDR}"
  --dist-timeout 3600
  --host "${SERVE_HOST}"
  --port "${SERVE_PORT}"
  --dtype bfloat16
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --attention-backend "${ATTENTION_BACKEND}"
  --linear-attn-backend "${LINEAR_ATTN_BACKEND}"
  --moe-runner-backend "${MOE_RUNNER_BACKEND}"
  --watchdog-timeout 3600
  --reasoning-parser kimi_k3
  --tool-call-parser kimi_k3
  # Keep every serving/eval arm comparable: never reuse prompt prefixes.
  --disable-radix-cache
  # Skip built-in warmup: first-request a16w4 JIT on PP stage1 can exceed the
  # 600s warmup HTTP timeout and then kill_process_tree the whole server.
  --skip-server-warmup
)
if [[ "${ENABLE_CUSTOM_AR:-0}" != "1" ]]; then
  ARGS+=(--disable-custom-all-reduce)
fi
if [[ "${ENABLE_AITER_AR_FUSION:-0}" == "1" ]]; then
  ARGS+=(--enable-aiter-allreduce-fusion)
fi
# Decode-only graphs by default; prefill stays disabled (PP + MoE JIT).
if [[ -n "${CUDA_GRAPH_BACKEND_DECODE:-}" ]]; then
  ARGS+=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
fi
if [[ -n "${CUDA_GRAPH_BACKEND_PREFILL:-}" ]]; then
  ARGS+=(--cuda-graph-backend-prefill "${CUDA_GRAPH_BACKEND_PREFILL}")
fi
if [[ -n "${CUDA_GRAPH_MAX_BS_DECODE:-}" && "${CUDA_GRAPH_BACKEND_DECODE:-disabled}" != "disabled" ]]; then
  ARGS+=(--cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}")
fi
if [[ -n "${TRITON_ATTENTION_NUM_KV_SPLITS}" ]]; then
  ARGS+=(--triton-attention-num-kv-splits "${TRITON_ATTENTION_NUM_KV_SPLITS}")
fi
if [[ -n "${TRITON_ATTENTION_SPLIT_TILE_SIZE}" ]]; then
  ARGS+=(--triton-attention-split-tile-size "${TRITON_ATTENTION_SPLIT_TILE_SIZE}")
fi
if [[ -n "${CHUNKED_PREFILL_SIZE}" ]]; then
  ARGS+=(--chunked-prefill-size "${CHUNKED_PREFILL_SIZE}")
fi
if [[ "${EP_SIZE}" != "1" ]]; then
  ARGS+=(--ep-size "${EP_SIZE}")
fi
if [[ "${DP_SIZE:-1}" != "1" ]]; then
  ARGS+=(--dp-size "${DP_SIZE}")
fi
if [[ "${ENABLE_DP_ATTENTION:-0}" == "1" ]]; then
  ARGS+=(--enable-dp-attention)
fi
if [[ "${ENABLE_DP_LM_HEAD:-0}" == "1" ]]; then
  ARGS+=(--enable-dp-lm-head)
fi
if [[ -n "${MOE_A2A_BACKEND:-}" ]]; then
  ARGS+=(--moe-a2a-backend "${MOE_A2A_BACKEND}")
fi
if [[ "${PD_DISAGG}" == "1" ]]; then
  ARGS+=(--disaggregation-mode "${DISAGGREGATION_MODE}")
  ARGS+=(--disaggregation-transfer-backend "${DISAGGREGATION_TRANSFER_BACKEND}")
  ARGS+=(--disaggregation-bootstrap-port "${DISAGGREGATION_BOOTSTRAP_PORT}")
  if [[ -n "${PD_IB_DEVICE:-}" ]]; then
    ARGS+=(--disaggregation-ib-device "${PD_IB_DEVICE}")
  fi
fi
if [[ "${ENABLE_DSPARK}" == "1" ]]; then
  ARGS+=(--speculative-algorithm DSPARK)
  ARGS+=(--speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}")
  ARGS+=(--speculative-dspark-block-size "${SPECULATIVE_DSPARK_BLOCK_SIZE}")
  if [[ "${ENABLE_LINEAR_REPLAYSSM_SPEC:-0}" == "1" ]]; then
    ARGS+=(--enable-linear-replayssm-spec)
  fi
  if [[ -n "${SPECULATIVE_DRAFT_ATTENTION_BACKEND:-}" ]]; then
    ARGS+=(--speculative-draft-attention-backend "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}")
  fi
  if [[ -n "${LINEAR_ATTN_VERIFY_BACKEND:-}" ]]; then
    ARGS+=(--linear-attn-verify-backend "${LINEAR_ATTN_VERIFY_BACKEND}")
  fi
  if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
    ARGS+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
  fi
  if [[ -n "${MAX_MAMBA_CACHE_SIZE:-}" ]]; then
    ARGS+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
  fi
fi

# Free-form passthrough for A/B experiments, e.g.
#   EXTRA_ARGS="--disable-cuda-graph-padding" /opt/scripts/start_node0.sh
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=( ${EXTRA_ARGS} )
  echo "[INFO] EXTRA_ARGS=${EXTRA_ARGS}"
fi

if [[ "${SGLANG_FOREGROUND:-0}" == "1" ]]; then
  echo "[INFO] foreground exec (for rocprofv3 wrap)"
  exec "${LAUNCH[@]}" "${ARGS[@]}"
fi

if [[ "${ROCPROF_KERNEL:-0}" == "1" ]]; then
  mkdir -p /opt/scripts/logs/rocprof
  echo "[INFO] wrapping launch with rocprofv3 kernel trace period=${ROCPROF_PERIOD:-110:55:1}"
  nohup rocprofv3 \
    --kernel-trace --stats --summary --summary-per-domain \
    --output-format csv \
    -d /opt/scripts/logs/rocprof -o kernel \
    --collection-period "${ROCPROF_PERIOD:-110:55:1}" \
    --collection-period-unit sec \
    -- "${LAUNCH[@]}" "${ARGS[@]}" >"${LOG_FILE}" 2>&1 &
else
  nohup "${LAUNCH[@]}" "${ARGS[@]}" >"${LOG_FILE}" 2>&1 &
fi

echo $! >"${PID_FILE}"
echo "[INFO] started pid $(cat "${PID_FILE}")"
echo "[INFO] tail -f ${LOG_FILE}"
