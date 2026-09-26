"""Patch openwakeword's train.py so every RNG it draws from is seeded from config["seed"].

The tuning loop this repo is building (improvement.md P0) only means anything if two
runs differing in one hyperparameter differ in nothing else. Upstream seeds NOTHING:
the model's initial weights come from an unseeded torch generator, the augmentation
draws background / RIR / gain from the unseeded global numpy RNG, and Python's hash
randomization is on. This repo has already measured 10 points of run-to-run variance
at an IDENTICAL config (SPEED.md, CLAUDE.md) - until this patch lands, a sweep point
measures the seed, not the hyperparameter.

The patch does ONE thing, keyed on config["seed"] (absent or 0 = untouched, so a
config written by unpatched code behaves exactly as before): right after the
training config is loaded, seed random / numpy / torch from it. That is all the
augmentation pipeline needs: augment_clips (openwakeword/data.py) takes no seed
parameter and draws from the GLOBAL RNGs only - Python `random` for the RIR /
background choices, `np.random` for gains and jitter, torch for torch_audiomentations
- so a global seed placed before the first call makes the four augment_clips
generators deterministic in sequence. (The earlier version of this patch also
appended `seed=config.get("seed", 0)` to the four call sites on the false premise
that augment_clips accepts one; it does not - that is what the heal pass below
removes. Caught the expensive way: a real run died in the augmentation stage with
TypeError: augment_clips() got an unexpected keyword argument 'seed'.)

The TTS engines are NOT seedable (Piper's VITS samples noise per call, Kokoro
exposes nothing), so a same-seed run still renders different audio - which is why
the corpus is frozen as a manifest instead (src/train/corpus/manifest.py). The seed
covers everything downstream of the corpus.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

SENTINEL = "# PATCHED: seed every RNG from the config"
BAD_KWARG = 'RIR_paths=rir_paths, seed=config.get("seed", 0))'

if SENTINEL in content:
    if BAD_KWARG not in content:
        print(f"Already patched: {path}")
        sys.exit(0)
    # Heal the earlier version: the seed= keyword argument is not a parameter of
    # augment_clips (data.py) and crashes the augmentation stage. The global RNG
    # seed above the call sites makes the kwarg redundant anyway.
    n = content.count(BAD_KWARG)
    content = content.replace(BAD_KWARG, "RIR_paths=rir_paths)")
    with open(path, 'w') as f:
        f.write(content)
    print(f"Repaired: {path} (removed {n} stray seed= kwargs from augment_clips call sites)")
    sys.exit(0)

# Seed the process RNGs once, right after the config is loaded. The load line is
# the anchor: it runs on both the --augment_clips and the --train_model paths, so
# one insertion covers both subprocess invocations this repo makes.
anchor = "    config = yaml.load(open(args.training_config, 'r').read(), yaml.Loader)"
if anchor not in content:
    print("WARNING: patch target not found (config load line) in", path)
    sys.exit(0)
replacement = (
    anchor
    + "\n"
    + "    # PATCHED: seed every RNG from the config so a sweep point measures the\n"
    + "    # hyperparameter, not the draw. 0/absent = untouched (upstream behaviour).\n"
    + '    _seed = int(config.get("seed", 0))\n'
    + "    if _seed:\n"
    + "        import random as _seeded_random\n"
    + "        _seeded_random.seed(_seed)\n"
    + "        np.random.seed(_seed)\n"
    + "        torch.manual_seed(_seed)\n"
)
content = content.replace(anchor, replacement, 1)

# Defensive: if a stale copy of the call-site kwarg is present, remove it.
if BAD_KWARG in content:
    content = content.replace(BAD_KWARG, "RIR_paths=rir_paths)")

with open(path, 'w') as f:
    f.write(content)

print(f"Patched: {path} (RNGs seeded from config['seed'])")
