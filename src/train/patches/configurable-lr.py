"""Patch openwakeword's train.py so the base learning rate comes from config["lr"].

Learning rate is the first hyperparameter anyone tunes, and upstream it is not
reachable: `lr = 0.0001` is hardcoded inside `auto_train` (the line that opens
the sequence-1 block), not a config key, not a CLI flag, not read from
anywhere. This repo's src/train/oww/train.py has never touched it - 0.0001 with
the per-sequence /10 decay (0.0001 -> 0.00001 -> 0.000001 across the three
sequences) is upstream's schedule, unchanged, so the default here preserves
that exact behaviour: a config without "lr" trains at 0.0001 as before.

The mechanism: auto_train is a method with no config parameter, but the
module-level `config` the script loads after argument parsing (the same line
seed-augment.py anchors on) is in the method's global scope, so a bare
`config.get("lr", ...)` inside auto_train resolves to it by the time training
runs. Only the base line moves; the two `lr = lr/10` lines scale FROM the
base and stay as they are.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

SENTINEL = 'lr = config.get("lr", 0.0001)'
if SENTINEL in content:
    print(f"Already patched: {path}")
    sys.exit(0)

# 8-space indent: the line inside auto_train's sequence-1 block. The optimizer
# constructor elsewhere uses `lr=0.0001` (no spaces) and a different indent, so
# requiring exactly one match of this exact line is a real anchor, not a guess.
anchor = "        lr = 0.0001\n"
if content.count(anchor) != 1:
    print(f"WARNING: expected exactly 1 auto_train base-lr line, found "
          f"{content.count(anchor)} in {path}")
    sys.exit(0)
replacement = (
    "        # PATCHED: base learning rate from config (default = upstream's);\n"
    "        # the /10 per-sequence lines below scale from it. See\n"
    '        # patches/configurable-lr.py.\n'
    '        lr = config.get("lr", 0.0001)\n'
)
content = content.replace(anchor, replacement, 1)

with open(path, 'w') as f:
    f.write(content)

print(f"Patched: {path} (auto_train base lr now reads config['lr'])")
