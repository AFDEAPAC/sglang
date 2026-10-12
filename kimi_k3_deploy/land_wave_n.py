#!/usr/bin/env python3
"""Land the adopted Wave N2 flag in /opt/scripts/common.env (idempotent; --check / --revert)."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> Wave N2 adopted 2026-10-02"
END = "# <<< Wave N2 adopted"
BLOCK = [
    BEGIN,
    "# a16w4 MoE gemm2 LDS block sized max(A tile, epilogue accumulator) instead of their sum at BM >= 32",
    "# (16 KB vs 20 KB, one more resident CTA per CU): 1.08x fused MoE @8192 tokens. Same session vs bracketing",
    "# landed baselines: see prof/final/VERDICTS.md (Wave N2). Needs the patched gemm2.py (backup gemm2.py.pre_wave_n).",
    'export A16WMIX_GEMM2_LDS_OVERLAY="${A16WMIX_GEMM2_LDS_OVERLAY:-1}"',
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
