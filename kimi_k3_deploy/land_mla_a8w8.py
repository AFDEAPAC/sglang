#!/usr/bin/env python3
"""Land SGLANG_K3_MLA_DECODE_A8W8=1 in /opt/scripts/common.env (idempotent; --check / --revert)."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> MLA a8w8 decode adopted 2026-10-05"
END = "# <<< MLA a8w8 decode adopted"
BLOCK = [
    BEGIN,
    "# Decode MLA: fp8 q + aiter mla_a8w8_qh16_qseqlen1_gqaratio16_ps. Patch",
    "# /opt/scripts/patches/k3_mla_decode_a8w8/. Revert the patch before reverting fused prep.",
    'export SGLANG_K3_MLA_DECODE_A8W8="${SGLANG_K3_MLA_DECODE_A8W8:-1}"',
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
