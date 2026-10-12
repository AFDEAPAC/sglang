#!/usr/bin/env python3
"""Land SGLANG_K3_MLA_PREFILL_FP8=1 + SGLANG_K3_MLA_PREFILL_FP8_FUSED=1 in /opt/scripts/common.env
(idempotent; --check / --revert). Run inside the container on both nodes."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> MLA prefill fp8 attention (fused fp8 emission) adopted 2026-10-12"
END = "# <<< MLA prefill fp8 attention adopted"
BLOCK = [
    BEGIN,
    "# Patch /opt/scripts/patches/k3_mla_prefill_fp8attn/; evidence evo-kimi-k3-e2e/prof/prefill_fp8_sweep_20261012/.",
    'export SGLANG_K3_MLA_PREFILL_FP8="${SGLANG_K3_MLA_PREFILL_FP8:-1}"',
    'export SGLANG_K3_MLA_PREFILL_FP8_FUSED="${SGLANG_K3_MLA_PREFILL_FP8_FUSED:-1}"',
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
