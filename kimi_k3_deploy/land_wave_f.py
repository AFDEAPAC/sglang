#!/usr/bin/env python3
"""Land the adopted Wave F flags in /opt/scripts/common.env (idempotent; --check / --revert)."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> Wave F adopted 2026-10-01"
END = "# <<< Wave F adopted"
BLOCK = [
    BEGIN,
    "# Same session vs the closing landed baseline: decode TPOT 53.545 -> 51.185 ms (-4.4%),",
    "# TTFT 5274.5 -> 5210.1 ms (-1.2%), mid TPOT 30.425 -> 28.675 ms (-5.8%); GSM8K 0.985, KL 0.142.",
    "# Each line is a kill switch (set to 0 to revert that item). Patches: /opt/scripts/patches/.",
    'export SGLANG_K3_MLA_FUSED_PREP="${SGLANG_K3_MLA_FUSED_PREP:-1}"',
    'export SGLANG_K3_FRONT_EPILOGUE="${SGLANG_K3_FRONT_EPILOGUE:-1}"',
    'export AITER_K3_FUSED_ROUTE_PREP="${AITER_K3_FUSED_ROUTE_PREP:-1}"',
    'export SGLANG_K3_AITER_ATTN_RES="${SGLANG_K3_AITER_ATTN_RES:-1}"',
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
