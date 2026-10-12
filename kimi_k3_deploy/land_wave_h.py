#!/usr/bin/env python3
"""Land the adopted Wave H flag in /opt/scripts/common.env (idempotent; --check / --revert)."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> Wave H adopted 2026-10-01"
END = "# <<< Wave H adopted"
BLOCK = [
    BEGIN,
    "# #3 MoE output straight into the [latent | shared] buffer (no latent.copy_ per MoE layer).",
    "# Same session, bracketed by landed baselines: decode TPOT -0.83%, mid TPOT -1.57%, TTFT -0.18%;",
    "# GSM8K 0.985. Kill switch: set to 0. Patch: /opt/scripts/patches/moe_out_direct/.",
    'export SGLANG_K3_MOE_OUT_DIRECT="${SGLANG_K3_MOE_OUT_DIRECT:-1}"',
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
