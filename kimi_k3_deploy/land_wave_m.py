#!/usr/bin/env python3
"""Land the adopted Wave M flag in /opt/scripts/common.env (idempotent; --check / --revert)."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> Wave M adopted 2026-10-02"
END = "# <<< Wave M adopted"
BLOCK = [
    BEGIN,
    "# Prefill-only dynamic fp8 (per-token x per-channel, aiter gemm_a8w8_bpreshuffle) for the bf16-kept dense",
    "# GEMMs at >= 1024 tokens: KDA wide in-proj, MoE shared gate_up + routed up slice (token-parallel branch),",
    "# MLA output gate. Same session vs bracketing landed baselines: see prof/final/VERDICTS.md (Wave M).",
    "# Patch: /opt/scripts/patches/k3_prefill_fp8/. Kill switch: set to empty.",
    'export SGLANG_K3_PREFILL_FP8="${SGLANG_K3_PREFILL_FP8-kda,front,up,mla_gate}"',
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
