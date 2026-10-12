#!/usr/bin/env python3
"""Land the adopted Wave M-decode flag in /opt/scripts/common.env (idempotent; --check / --revert)."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> Wave M-decode adopted 2026-10-02"
END = "# <<< Wave M-decode adopted"
BLOCK = [
    BEGIN,
    "# Dynamic fp8 (per-token x per-channel, aiter gemm_a8w8_bpreshuffle) on the fused decode path: MoE merged",
    "# front, routed up_proj, KDA wide in-proj ([f_a|b] tail stays bf16). fp8 copies built in post_load_weights.",
    "# Same session vs bracketing landed baselines: see prof/final/VERDICTS.md (Wave M-decode). The kda site is",
    "# the borderline one (GSM8K 0.98, 0.98, 0.985 on rerun): set to front,up to drop it, or empty to disable.",
    "# Patch: /opt/scripts/patches/k3_decode_fp8/.",
    'export SGLANG_K3_DECODE_FP8="${SGLANG_K3_DECODE_FP8-front,up,kda}"',
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
