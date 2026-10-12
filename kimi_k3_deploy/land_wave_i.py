#!/usr/bin/env python3
"""Land the adopted Wave I flag in /opt/scripts/common.env (idempotent; --check / --revert)."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> Wave I adopted 2026-10-02"
END = "# <<< Wave I adopted"
BLOCK = [
    BEGIN,
    "# Token-parallel prefill for the replicated router + latent projections (>=1024 tokens, tokens % tp == 0):",
    "# router/down on the 1/tp token slice + all-gather, reduce-scatter tail, up-proj on the slice, all-gather.",
    "# Same session vs bracketing landed baselines: see prof/final/VERDICTS.md (Wave I). Kill switch: set to 0.",
    "# Patch: /opt/scripts/patches/k3_tp_prefill_latent/.",
    'export SGLANG_K3_TP_PREFILL_LATENT="${SGLANG_K3_TP_PREFILL_LATENT:-1}"',
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
