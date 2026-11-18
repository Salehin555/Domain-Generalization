"""Diff two numbers.tex files: python diff_macros.py OLD NEW (changed and missing)."""
import re
import sys

RX = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")


def load(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        m = RX.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


o, n = load(sys.argv[1]), load(sys.argv[2])
ch = {k: (o[k], n[k]) for k in o if k in n and o[k] != n[k]}
miss = [k for k in o if k not in n]
print(f"old {len(o)}  new {len(n)}  added {len(set(n) - set(o))}")
for k, (a, b) in sorted(ch.items()):
    print(f"  CHANGED {k}: {a} -> {b}")
for k in miss:
    print(f"  MISSING {k}")
sys.exit(1 if ch or miss else 0)
