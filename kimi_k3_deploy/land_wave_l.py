#!/usr/bin/env python3
"""Land the adopted Wave L flag in /opt/scripts/common.env (idempotent; --check / --revert)."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> Wave L adopted 2026-10-03"
END = "# <<< Wave L adopted"
BLOCK = [
    BEGIN,
    "# Prefill: o_proj reduce-scattered and the MLP-side attention-residual aggregation run on this rank's token",
    "# rows (layers whose MoE takes the token-parallel branch). Same session vs bracketing landed baselines: see",
    "# prof/final/VERDICTS.md (Wave L). Patch /opt/scripts/patches/k3_sp_prefill_agg2/ (revert it before Wave M / I).",
    'export SGLANG_K3_SP_PREFILL_AGG2="${SGLANG_K3_SP_PREFILL_AGG2:-1}"',
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
