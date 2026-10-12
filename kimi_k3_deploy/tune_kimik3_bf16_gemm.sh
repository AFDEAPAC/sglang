#!/bin/bash
# Tune kimik3 bf16 GEMM shapes on gfx942 (hipBLASLt / skinny / opus / asm).
set -euo pipefail
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
cd /opt/aiter
mkdir -p /opt/scripts/configs /opt/scripts/logs
IN=/tmp/kimik3_bf16_untuned_gemm_gfx942.csv
OUT=/opt/scripts/configs/kimik3_bf16_tuned_gemm_gfx942.csv
python3 - <<'PY'
import pandas as pd
from pathlib import Path
src = Path("/opt/aiter/aiter/configs/model_configs/kimik3_bf16_untuned_gemm.csv")
df = pd.read_csv(src)
extra_m = [101, 2048, 4096, 8192]
nk = df[["N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]].drop_duplicates()
rows = [df]
for m in extra_m:
    add = nk.copy()
    add.insert(0, "M", m)
    rows.append(add)
out = pd.concat(rows, ignore_index=True).drop_duplicates()
out.to_csv("/tmp/kimik3_bf16_untuned_gemm_gfx942.csv", index=False)
print("untuned rows", len(out), "M", sorted(out.M.unique().tolist()))
PY
echo "[tune] gemm_a16w16 -> $OUT"
python3 csrc/gemm_a16w16/gemm_tuner.py \
  -i "$IN" \
  -o "$OUT" \
  --mp 8 \
  --shape_grouped \
  --with-hipblaslt \
  --libtype hipblaslt,skinny,opus,asm \
  --warmup 5 \
  --iters 30 \
  --timeout 1800 \
  --errRatio 0.05
echo "[tune] gemm done $(wc -l < "$OUT") lines $OUT"
