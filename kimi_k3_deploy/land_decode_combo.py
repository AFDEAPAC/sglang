#!/usr/bin/env python3
"""Land SGLANG_K3_MLA_DECODE_BALANCED=1 + SGLANG_K3_PP_RECV_AFTER_PREV_FWD=2 in /opt/scripts/common.env
(idempotent; --check / --revert). Run inside the container on both nodes."""
import sys

PATH = "/opt/scripts/common.env"
BEGIN = "# >>> MLA decode balanced metadata + PP recv-after-forward adopted 2026-10-08"
END = "# <<< MLA decode balanced metadata + PP recv-after-forward adopted"
BLOCK = [
    BEGIN,
    "# Patches /opt/scripts/patches/k3_mla_decode_balanced/ and k3_pp_recv_after_fwd/; evidence",
    "# evo-kimi-k3-e2e/prof/decode_itl_probe/FINDINGS.md. Both are only read at server start.",
    'export SGLANG_K3_MLA_DECODE_BALANCED="${SGLANG_K3_MLA_DECODE_BALANCED:-1}"',
    'export SGLANG_K3_PP_RECV_AFTER_PREV_FWD="${SGLANG_K3_PP_RECV_AFTER_PREV_FWD:-2}"',
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
