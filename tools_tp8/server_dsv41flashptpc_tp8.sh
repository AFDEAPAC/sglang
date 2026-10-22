#!/usr/bin/env bash
# DeepSeek-V4.1-Flash FP4PTPC TP8 launcher, best-version default (dsv41best_20261019).
# A bare launch runs the screening winner M3: colleague a16wmix FP8 MoE (SGLANG_GFX942_MXFP4_INT4=false,
# AITER_GFX942_DSV41_A16WMIX=1, AITER_A16WMIX_FP8=1, AITER_A16WMIX_FP8_S2=1) + our fq fusions + aiter mHC
# (SGLANG_DSV41_HIP_TORCH_MHC=0) + FP8 MQA prefill indexer (SGLANG_DSV41_FP8_MQA_INDEXER=1) + decode indexer
# tile 128 (SGLANG_DSV41_DECODE_INDEXER_BLOCK_L=128), with our prefill setting (PREFILL_CP=1 / CP4, GRAPH=full).
# Every switch stays overridable by exporting it before the launcher. Two documented fall-backs (see --help):
#   our image default (INT4 + fq7) and the exact FP4 fq7 variant.
# TP8 defaults differ from TP4: EP_SIZE=1, MEMFRAC=0.80, PREFILL_CP=0.
#   EP_SIZE=1 (all 384 routed experts per rank, intermediate tensor-sharded to 288)
#   runs the a16wmix FP8 MoE pair -- the same winning kernel the TP4 path uses -- and
#   is throughput-best on TP8 (measured 119-138% of EP8 at every sample point, GSM8K
#   194/200). It REQUIRES the env-gated TP8-EP1 source fix committed on this branch:
#   sglang fp8.py extends gfx942_exact_native_a16w4_ep1 to w13=(384,576,2560)/
#   w2=(384,5120,144) so the loader preshuffles (guinterleave + e8m0 w2 scale); aiter
#   fused_moe.py extends the a16wmix gate to (5120,384,384,6). Both behind
#   AITER_GFX942_DSV41_A16WMIX_TP8EP1 (default-ON here, off elsewhere -> TP4 unchanged).
#   Without that source fix EP1 aborts on gfx942 (fp4x2 quant-sort: no __Float4_e2m1fn_x2).
#   Fallback: EP_SIZE=8 routes MoE to the exact-a16w4 FlyDSL kernel and boots with ZERO
#   source change (slower, but needs no patch) -- use it on an unpatched image.
#   MEMFRAC=0.80 (vs TP4 0.90): 0.90 OOMs during prefill CUDA-graph capture; 0.80 leaves
#   ~14 GB headroom at EP1. PREFILL_CP=0 is the validated TP8 default (CP is a candidate).
# Defaults: DSPARK=0, PREFILL_CP=0, EP_SIZE=1, MEMFRAC=0.80. Set DSPARK=1 for spec decoding;
# set EP_SIZE=8 for the zero-source-change exact-a16w4 fallback.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: [DSPARK=0|1] [PREFILL_CP=0|1] bash /opt/scripts/server_dsv41flashptpc_tp8.sh
Bare invocation = best-version default (screening winner M3) on DSPARK=0 PREFILL_CP=1 PORT=8040
DEVS=0,1,2,3,4,5,6,7 NUMA_ARGS="0 0 0 0 1 1 1 1" CONTEXT=auto RADIX=off GRAPH=full CP_SIZE=8 CP_STRATEGY=interleave:
colleague a16wmix FP8 MoE + our fq fusions + aiter mHC + FP8 MQA prefill indexer + decode indexer tile 128.
Best default switch values: SGLANG_GFX942_MXFP4_INT4=false AITER_GFX942_DSV41_A16WMIX=1 AITER_A16WMIX_FP8=1
AITER_A16WMIX_FP8_S2=1 SGLANG_DSV41_HIP_TORCH_MHC=0 SGLANG_DSV41_FP8_MQA_INDEXER=1
SGLANG_DSV41_DECODE_INDEXER_BLOCK_L=128.

Switch back to our image default (INT4 + fq7):
  SGLANG_GFX942_MXFP4_INT4=true AITER_GFX942_DSV41_A16WMIX=0 AITER_A16WMIX_FP8=0 AITER_A16WMIX_FP8_S2=0 \
  SGLANG_DSV41_HIP_TORCH_MHC=1 SGLANG_DSV41_FP8_MQA_INDEXER=0 SGLANG_DSV41_DECODE_INDEXER_BLOCK_L=64 \
  bash /opt/scripts/server_dsv41flashptpc_tp8.sh
Switch to the exact FP4 fq7 variant (our exact FP4 MoE, no a16wmix, no FP8 MQA indexer):
  SGLANG_GFX942_MXFP4_INT4=false AITER_GFX942_DSV41_A16WMIX=0 AITER_A16WMIX_FP8=0 AITER_A16WMIX_FP8_S2=0 \
  SGLANG_DSV41_HIP_TORCH_MHC=1 SGLANG_DSV41_FP8_MQA_INDEXER=0 SGLANG_DSV41_DECODE_INDEXER_BLOCK_L=64 \
  bash /opt/scripts/server_dsv41flashptpc_tp8.sh

Optional controls: MODEL, DEVS, PORT, NUMA_ARGS, MEMFRAC, CONTEXT, MAXRUN,
CHUNKED_PREFILL_SIZE, MAX_PREFILL_TOKENS, BLOCK_SIZE, GRAPH, RADIX, EXTRA_ARGS.
Switches (best-version default in parentheses): AITER_GFX942_DSV41_A16WMIX=0|1 (1),
AITER_A16WMIX_FP8=0|1 (1), AITER_A16WMIX_FP8_S2=0|1 (1) (colleague a16wmix FP8 MoE),
SGLANG_GFX942_MXFP4_INT4=true|false (false; true selects our INT4 MoE with AITER_MOE_A16WI4_OPT=5),
SGLANG_DSV41_HIP_TORCH_MHC=1|0 (0 = colleague aiter mHC),
SGLANG_DSV41_FP8_MQA_INDEXER=0|1 (1 = colleague FP8 MQA prefill indexer),
SGLANG_DSV41_DECODE_INDEXER_BLOCK_L=64|128 (128; export is mandatory, the in-code default is 128).
SGLANG_LAUNCHER_DRY_RUN=1 prints the resolved command without touching a GPU.
HELP
  exit 0
fi

die() { printf 'FATAL: %s\n' "$*" >&2; exit 2; }
require_bool() { [[ "$2" == 0 || "$2" == 1 ]] || die "$1 must be 0 or 1 (got $2)"; }
prepare_runtime_dir() {
  local path=$1
  install -d -m 0755 "$path"
  [[ -d "$path" && -w "$path" && -x "$path" ]] || die "unusable runtime directory: $path"
  /usr/bin/python3 - "$path" <<'PY'
from pathlib import Path
import os, sys
root = Path(sys.argv[1]).resolve()
probe = root / (".write-probe.%d" % os.getpid())
renamed = root / (probe.name + ".renamed")
with probe.open("wb") as handle:
    handle.write(b"ok")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(str(probe), str(renamed))
renamed.unlink()
PY
}

MODEL="${MODEL:-/dev/shm/DeepSeek-V4.1-Flash-1017-FP4PTPC}"
DRAFT_CFG="${DRAFT_CFG:-${MODEL}/dspark_config.json}"
DEVS="${DEVS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8040}"
NUMA_ARGS="${NUMA_ARGS:-0 0 0 0 1 1 1 1}"
MEMFRAC="${MEMFRAC:-0.80}"
CONTEXT="${CONTEXT:-auto}"
MAXRUN="${MAXRUN:-136}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
DSPARK="${DSPARK:-0}"
PREFILL_CP="${PREFILL_CP:-0}"
CP_SIZE="${CP_SIZE:-8}"
CP_STRATEGY="${CP_STRATEGY:-interleave}"
EP_SIZE="${EP_SIZE:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-5}"
SPEC_MOE_BACKEND="${SPEC_MOE_BACKEND:-aiter}"
GRAPH="${GRAPH:-full}"
RADIX="${RADIX:-off}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SGLANG_LAUNCHER_DRY_RUN="${SGLANG_LAUNCHER_DRY_RUN:-0}"
MXFP4_INT4="${SGLANG_GFX942_MXFP4_INT4:-false}"
require_bool DSPARK "$DSPARK"
require_bool PREFILL_CP "$PREFILL_CP"
require_bool SGLANG_LAUNCHER_DRY_RUN "$SGLANG_LAUNCHER_DRY_RUN"
[[ "$MXFP4_INT4" == true || "$MXFP4_INT4" == false ]] || die "SGLANG_GFX942_MXFP4_INT4 must be true or false (got $MXFP4_INT4)"
[[ "$MODEL" == /* ]] || die "MODEL must be an absolute path to a local model directory (got $MODEL)"
[[ "$DRAFT_CFG" == "$MODEL"/* ]] || die "DRAFT_CFG must be inside MODEL"

SGLANG_ROOT=/opt/sglang
AITER_ROOT=/opt/aiter
PYTHON_BIN=/usr/bin/python3
[[ -x "$PYTHON_BIN" ]] || die "missing Python: $PYTHON_BIN"
[[ -f "$SGLANG_ROOT/python/sglang/srt/configs/deepseek_v41.py" ]] || die "missing canonical SGLang source"
[[ -d "$AITER_ROOT/aiter" ]] || die "missing canonical AITER source"
[[ -f "$AITER_ROOT/aiter/configs/model_configs/dsv41_fp4ptpc_a8w8_bpreshuffle_tuned_gemm.csv" ]] || die "missing built-in DSV4.1 A8W8 config"
if [[ "$SGLANG_LAUNCHER_DRY_RUN" == 0 ]]; then
  [[ -d "$MODEL" ]] || die "missing model directory: $MODEL"
  [[ -f "$MODEL/config.json" ]] || die "missing config.json in model directory: $MODEL"
  if [[ "$DSPARK" == 1 ]]; then
    [[ -f "$DRAFT_CFG" ]] || die "missing draft config: $DRAFT_CFG"
  fi
fi

for name in ${!AITER_CONFIG_@}; do unset "$name"; done
unset SGLANG_SRC AITER_SRC AITER_COMPAT_OVERLAY BEST_CONFIG LOAD_BEST_CONFIG
unset AITER_JIT_DIR AITER_META_DIR AITER_ASM_DIR AITER_TUNED_CONFIGS_JSON
unset HIPBLASLT_TUNING_OVERRIDE_FILE CK_DIR HIP_KITTENS_DIR

RUNTIME_ROOT="/tmp/dsv41-runtime/launcher-$$"
prepare_runtime_dir "$RUNTIME_ROOT/tmp"
prepare_runtime_dir "$RUNTIME_ROOT/home"
prepare_runtime_dir "$RUNTIME_ROOT/xdg"
prepare_runtime_dir "$RUNTIME_ROOT/triton"
prepare_runtime_dir "$RUNTIME_ROOT/torchinductor"
prepare_runtime_dir "$RUNTIME_ROOT/flashinfer"
export TMPDIR="$RUNTIME_ROOT/tmp"
export HOME="$RUNTIME_ROOT/home"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_ROOT/torchinductor"
export FLASHINFER_CACHEDIR="$RUNTIME_ROOT/flashinfer"
export PYTHONPATH="$AITER_ROOT:$SGLANG_ROOT/python"
unset PYTHONNOUSERSITE
export PYTHONDONTWRITEBYTECODE=1 GPU_ARCHS=gfx942
export AITER_CONFIG_STAMP=2 AITER_REBUILD=0
export HIP_VISIBLE_DEVICES="$DEVS" CUDA_VISIBLE_DEVICES="$DEVS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_ALLOC_CONF=expandable_segments:False
export SGLANG_USE_AITER=1 SGLANG_USE_ROCM700A=1 SGLANG_MOE_PADDING=1
export AITER_FLYDSL_FORCE_REDUCE=1 AITER_FORCE_A8W4=0
export AITER_GFX942_FP4_LUT_DEQUANT=1
export SGLANG_GFX942_FUSED_DSV41=1
export SGLANG_GFX942_LOWRATIO_INDEX_CACHE=1 SGLANG_GFX942_WOB_STRIDED_QUANT=1
export SGLANG_GFX942_SHARED_ACT_QUANT=1 SGLANG_GFX942_RMSNORM_ROW_STRIDED=1
export SGLANG_GFX942_RMSNORM_PTQ=1
if [[ "$MXFP4_INT4" == true ]]; then
  export AITER_MOE_A16WI4_OPT="${AITER_MOE_A16WI4_OPT:-5}"
fi
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=0 AITER_FLYDSL_FORCE=1 AITER_KSPLIT=0
export AITER_GFX942_DSV41_DIRECT_MXFP4_REPACK=1
export AITER_GFX942_DSV41_EXACT_MXFP4_A16W4=1
export AITER_GFX942_DSV41_EXACT_MXFP4_A16W4_EP1=0
export AITER_GFX942_DSV41_EP1_LARGEM_SPLITK_BF16=1
export AITER_GFX942_DSV41_EP1_TUNED_FLYDSL=1
# TP8-EP1 a16wmix fix (env-gated, default-ON for this best-version TP8 default):
#   A16WMIX_TP8EP1 extends the sglang fp8 preshuffle gate + aiter a16wmix dispatch
#   gate to the TP8-EP1 shape; TP8_KSPLIT_COMPILED clamps wint4 split-K to compiled
#   kb {2,4}. Both match only TP8 shapes on gfx942, so TP4/EP8 stay byte-identical.
export AITER_GFX942_DSV41_A16WMIX_TP8EP1="${AITER_GFX942_DSV41_A16WMIX_TP8EP1:-1}"
export AITER_GFX942_DSV41_TP8_KSPLIT_COMPILED="${AITER_GFX942_DSV41_TP8_KSPLIT_COMPILED:-1}"
export SGLANG_USE_AITER_MOE_GU_ITLV=0
export AITER_GFX942_REUSE_MOE_TMP=1 AITER_GFX942_MOE_TMP_TOKENS=512
export SGLANG_DSV41_HIP_TORCH_HC_COMBINE=0
# Best-version image (dsv41best_20261019): the screening winner M3 is the default. Each switch below defaults to
# M3's winning value but stays overridable by exporting it before the launcher (see --help for the two fall-backs).
#   AITER_GFX942_DSV41_A16WMIX / AITER_A16WMIX_FP8 / AITER_A16WMIX_FP8_S2 : colleague a16wmix FP8 MoE (on by default).
#   SGLANG_DSV41_HIP_TORCH_MHC : 0 uses colleague's aiter mHC kernel (default); set 1 for our hip-torch mHC.
#   SGLANG_DSV41_FP8_MQA_INDEXER : colleague's FP8 MQA prefill indexer (on by default).
#   SGLANG_DSV41_DECODE_INDEXER_BLOCK_L : decode indexer tile; the in-code default is 128, exported here explicitly.
#     Set 64 for our original tile when switching back to our image / fq7 defaults.
export AITER_GFX942_DSV41_A16WMIX="${AITER_GFX942_DSV41_A16WMIX:-1}"
export AITER_A16WMIX_FP8="${AITER_A16WMIX_FP8:-1}"
export AITER_A16WMIX_FP8_S2="${AITER_A16WMIX_FP8_S2:-1}"
export SGLANG_DSV41_HIP_TORCH_MHC="${SGLANG_DSV41_HIP_TORCH_MHC:-0}"
export SGLANG_DSV41_FP8_MQA_INDEXER="${SGLANG_DSV41_FP8_MQA_INDEXER:-1}"
export SGLANG_DSV41_DECODE_INDEXER_BLOCK_L="${SGLANG_DSV41_DECODE_INDEXER_BLOCK_L:-128}"
export SGLANG_GFX942_ROUTED_SHARED_ADD=1 SGLANG_DSV41_DISABLE_FUSED_AR_MHC=1
export SGLANG_GFX942_TRITON_BF16_TO_FP32=1 SGLANG_GFX942_TRITON_MHC_POST=0
export SGLANG_GFX942_FUSED_MHC_STATS=1 SGLANG_GFX942_WORKSPACE_TOKENS=512
export SGLANG_GFX942_ENGRAM_GATHER=1 SGLANG_GFX942_ENGRAM_KERNELS=1
export SGLANG_DSV41_TEXT_ONLY=1 SGLANG_GFX942_CPU_ATTN_SINK=1
export SGLANG_GFX942_TRITON_CONTIGUOUS=1 SGLANG_GFX942_TRITON_FLAT_COPY=1
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE SGLANG_GFX942_MXFP4_INT4="$MXFP4_INT4"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false SGLANG_HACK_FLASHMLA_BACKEND=triton
export SGLANG_DSV41_TORCH_DECODE_INDEXER=0 SGLANG_DSV41_TORCH_PREFILL_INDEXER=0
export SGLANG_DSV41_TRITON_FP4_INDEXER=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0 NCCL_DMABUF_ENABLE=0
export SGLANG_DISABLE_SYMMETRIC_ALLOCATIONS=1
export SGLANG_DSV41_AITER_PREFILL_SCHEDULE_COMPAT=1
if [[ "$DSPARK" == 1 ]]; then
  export SGLANG_DSV41_DSPARK_SPEC_MOE_CONTEXT=1
  export SGLANG_DSV41_DSPARK_TRITON_BF16_MOE=0
  export SGLANG_DSV41_DSPARK_DRAFT_INT4=1
else
  export SGLANG_DSV41_DSPARK_SPEC_MOE_CONTEXT=0
  export SGLANG_DSV41_DSPARK_TRITON_BF16_MOE=0
  export SGLANG_DSV41_DSPARK_DRAFT_INT4=0
fi
if [[ "$PREFILL_CP" == 1 ]]; then
  export SGLANG_DSV41_HIP_PREFILL_CP=1
else
  unset SGLANG_DSV41_HIP_PREFILL_CP
fi

GRAPH_ARGS=()
case "$GRAPH" in
  full) GRAPH_ARGS=(--cuda-graph-backend-decode full --cuda-graph-backend-prefill breakable) ;;
  decode_only) GRAPH_ARGS=(--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled) ;;
  off) GRAPH_ARGS=(--disable-cuda-graph) ;;
  *) die "GRAPH must be full, decode_only, or off" ;;
esac
RADIX_ARGS=()
case "$RADIX" in off) RADIX_ARGS=(--disable-radix-cache) ;; on) ;; *) die "RADIX must be on or off" ;; esac
CONTEXT_ARGS=()
[[ "$CONTEXT" == auto || -z "$CONTEXT" ]] || CONTEXT_ARGS=(--context-length "$CONTEXT")
SPEC_ARGS=()
if [[ "$DSPARK" == 1 ]]; then
  SPEC_ARGS=(--speculative-algorithm DSPARK --speculative-dspark-block-size "$BLOCK_SIZE" --speculative-draft-model-path "$MODEL" --decrypted-draft-config-file "$DRAFT_CFG" --speculative-moe-runner-backend "$SPEC_MOE_BACKEND")
fi
CP_ARGS=()
if [[ "$PREFILL_CP" == 1 ]]; then
  CP_ARGS=(--enable-prefill-cp --attn-cp-size "$CP_SIZE" --cp-strategy "$CP_STRATEGY")
fi
read -r -a NUMA_ARR <<< "$NUMA_ARGS"
RAW_EXTRA_ARGV=()
[[ -z "$EXTRA_ARGS" ]] || read -r -a RAW_EXTRA_ARGV <<< "$EXTRA_ARGS"
EXTRA_ARGV=()
for ((i=0; i<${#RAW_EXTRA_ARGV[@]}; i++)); do
  arg=${RAW_EXTRA_ARGV[$i]}
  case "$arg" in
    --enable-prefill-cp|--disable-prefill-cp) ;;
    --attn-cp-size|--cp-strategy) ((i + 1 < ${#RAW_EXTRA_ARGV[@]})) || die "$arg is missing its value"; i=$((i + 1)) ;;
    --attn-cp-size=*|--cp-strategy=*) ;;
    *) EXTRA_ARGV+=("$arg") ;;
  esac
done

CMD=("$PYTHON_BIN" -m sglang.launch_server
  --model-path "$MODEL" --trust-remote-code --tp 8 --ep-size "$EP_SIZE"
  "${CONTEXT_ARGS[@]}" "${GRAPH_ARGS[@]}" "${RADIX_ARGS[@]}" --disable-overlap-schedule
  --max-running-requests "$MAXRUN" --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --mem-fraction-static "$MEMFRAC" --disable-shared-experts-fusion
  --moe-runner-backend auto --reasoning-parser auto --tool-call-parser auto
  "${SPEC_ARGS[@]}" "${CP_ARGS[@]}" --host 0.0.0.0 --port "$PORT" --numa-node "${NUMA_ARR[@]}"
  "${EXTRA_ARGV[@]}" "$@")

if [[ "$SGLANG_LAUNCHER_DRY_RUN" == 1 ]]; then
  DSV41_DSPARK="$DSPARK" DSV41_PREFILL_CP="$PREFILL_CP" DSV41_GRAPH="$GRAPH" \
  "$PYTHON_BIN" - "${CMD[@]}" <<'PY'
import json, os, sys
keys = sorted(k for k in os.environ if k.startswith(("SGLANG_", "AITER_", "NCCL_")) or k in {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "TMPDIR", "HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_CACHEDIR"})
print(json.dumps({"launcher":"server_dsv41flashptpc_tp8.sh","topology":"tp8","dspark":int(os.environ["DSV41_DSPARK"]),"prefill_cp":int(os.environ["DSV41_PREFILL_CP"]),"graph":os.environ["DSV41_GRAPH"],"environment":{k:os.environ[k] for k in keys},"argv":sys.argv[1:]}, indent=2, sort_keys=True))
PY
  exit 0
fi
printf '[dsv41-tp8] model=%s devices=%s port=%s ep=%s dspark=%s prefill_cp=%s graph=%s mxfp4_int4=%s a16wi4_opt=%s a16wmix=%s a16wmix_fp8=%s a16wmix_fp8_s2=%s hip_torch_mhc=%s fp8_mqa_indexer=%s decode_indexer_block_l=%s\n' "$MODEL" "$DEVS" "$PORT" "$EP_SIZE" "$DSPARK" "$PREFILL_CP" "$GRAPH" "$MXFP4_INT4" "${AITER_MOE_A16WI4_OPT:-unset}" "$AITER_GFX942_DSV41_A16WMIX" "$AITER_A16WMIX_FP8" "$AITER_A16WMIX_FP8_S2" "$SGLANG_DSV41_HIP_TORCH_MHC" "$SGLANG_DSV41_FP8_MQA_INDEXER" "$SGLANG_DSV41_DECODE_INDEXER_BLOCK_L"
exec "${CMD[@]}"
