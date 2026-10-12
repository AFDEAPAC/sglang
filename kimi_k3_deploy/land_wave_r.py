#!/usr/bin/env python3
"""Land the adopted conversion-2 (stage-2 row quant) flag in /opt/scripts/common.env (idempotent; --check / --revert)."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> Wave N1 adopted 2026-10-02"
END = "# <<< Wave N1 adopted"
BLOCK = [
    BEGIN,
    "# Prefill stage-2 MoE intermediate quantized by k3_rowquant (16 rows per program, bit-identical to aiter's",
    "# per-token kernel, 4.8x at 8192 tokens); decode-sized calls keep aiter's kernel. Same session vs",
    "# bracketing landed baselines: see prof/final/VERDICTS.md (Wave N1). Patch /opt/scripts/patches/k3_s2_rowquant/.",
    'export AITER_K3_S2_ROWQUANT="${AITER_K3_S2_ROWQUANT:-1}"',
    END,
]

with open(PATH) as f:
    lines = f.read().split("\n")
present = BEGIN in lines
if "--check" in sys.argv:
    print("landed" if present else "not landed")
    sys.exit(0 if present else 1)
if present:
    i = lines.index(BEGIN)
    del lines[i : lines.index(END, i) + 1]
while lines and lines[-1] == "":
    lines.pop()
if "--revert" not in sys.argv:
    lines += BLOCK
with open(PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print("reverted" if "--revert" in sys.argv else "landed")
