#!/usr/bin/env python3
"""Land Wave J: install the tuned GEMM tables over the production ones, keeping .pre_wave_j backups.

common.env assigns AITER_CONFIG_GEMM_BF16 itself (line ~92), so an appended env block cannot redirect it;
the tables are the tuning artefacts, so they are replaced in place. Idempotent; --check / --revert.
"""
import hashlib
import shutil
import sys
from pathlib import Path

PAIRS = [
    (Path("/opt/hipblaslt_tuning.txt"), Path("/opt/hipblaslt_tuning_j.txt")),
    (Path("/opt/scripts/configs/kimik3_bf16_tuned_gemm_gfx942.csv"),
     Path("/opt/scripts/configs/kimik3_bf16_tuned_gemm_gfx942_j.csv")),
]


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def landed():
    return all(prod.exists() and md5(prod) == md5(j) for prod, j in PAIRS)


if "--check" in sys.argv:
    print("landed" if landed() else "not landed", *(f"{p.name}={md5(p)}" for p, _ in PAIRS))
    sys.exit(0 if landed() else 1)
if "--revert" in sys.argv:
    for prod, _ in PAIRS:
        bak = prod.with_name(prod.name + ".pre_wave_j")
        if bak.exists():
            shutil.copy2(bak, prod)
    print("reverted", *(f"{p.name}={md5(p)}" for p, _ in PAIRS))
    sys.exit(0)
for prod, j in PAIRS:
    bak = prod.with_name(prod.name + ".pre_wave_j")
    if not bak.exists():
        shutil.copy2(prod, bak)
    shutil.copy2(j, prod)
print("landed", *(f"{p.name}={md5(p)}" for p, _ in PAIRS))
