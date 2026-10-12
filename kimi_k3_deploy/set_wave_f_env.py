#!/usr/bin/env python3
"""Rewrite the managed Wave F arm block in /opt/scripts/common.env.

  set_wave_f_env.py NAME=VALUE ...   replace the block with these exports
  set_wave_f_env.py --clear          remove the block (landed state)
"""
import sys

PATH = "/opt/scripts/common.env"
BEGIN, END = "# >>> wave_f arm (managed by set_wave_f_env.py)", "# <<< wave_f arm"

with open(PATH) as f:
    lines = f.read().split("\n")
if BEGIN in lines:
    i = lines.index(BEGIN)
    j = lines.index(END, i)
    del lines[i : j + 1]
while lines and lines[-1] == "":
    lines.pop()
args = [a for a in sys.argv[1:] if a != "--clear"]
if "--clear" not in sys.argv and args:
    lines.append(BEGIN)
    for a in args:
        k, v = a.split("=", 1)
        lines.append(f'export {k}="${{{k}:-{v}}}"')
    lines.append(END)
with open(PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print("common.env arm block:", args if "--clear" not in sys.argv else "cleared")
