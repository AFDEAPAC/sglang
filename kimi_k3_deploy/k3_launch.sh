#!/usr/bin/env bash
# Start one PP rank with a validated profile:
#   k3_launch.sh <0|1> [dspark|base]
#   dspark (default): DSpark-PP gamma 3 with draft fold, D1 prefix attention, DCP8, HiCache
#   base:             the same without DSpark
# Start rank 0 on node 0 first, then rank 1 on node 1 with the same profile.
set -euo pipefail
RANK=${1:?usage: k3_launch.sh <0|1> [dspark|base]}
PROFILE=${2:-${K3_PROFILE:-dspark}}
case "$PROFILE" in dspark|base) ;; *) echo "unknown profile: $PROFILE (dspark|base)" >&2; exit 1;; esac
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ulimit -c 0
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
export K3_ENABLE_RADIX_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.6,max_split_size_mb:512,roundup_power2_divisions:4
export PYTORCH_HIP_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
export AITER_A16WMIX_FP8=1 AITER_A16WMIX_FP8_S2=1
unset SGLANG_K3_KV_CHUNK_TOKENS
set -a
# shellcheck disable=SC1091
source "$DIR/profile_base.env"
# shellcheck disable=SC1091
[ "$PROFILE" = dspark ] && source "$DIR/profile_dspark.env"
set +a
exec "$DIR/start_node$RANK.sh"
