"""Patch openwakeword's train.py so every RNG it draws from is seeded from config["seed"].

The tuning loop this repo is building (improvement.md P0) only means anything if two
runs differing in one hyperparameter differ in nothing else. Upstream seeds NOTHING:
the model's initial weights come from an unseeded torch generator, the augmentation
draws background / RIR / gain from the unseeded global numpy RNG, and Python's hash
randomization is on. This repo has already measured 10 points of run-to-run variance
at an IDENTICAL config (SPEED.md, CLAUDE.md) - until this patch lands, a sweep point
measures the seed, not the hyperparameter.

The patch does two things, both keyed on config["seed"] (absent or 0 = untouched, so
a config written by unpatched code behaves exactly as before):

  1. right after the training config is loaded: seed random / numpy / torch from it;
  2. at every augment_clips call: pass seed=... - augment_clips already TAKES a seed
     (openwakeword/data.py) and seeds its own draws with it, and this repo never
     passed one, so the four feature-computation draws stayed unseeded.

The TTS engines are NOT seedable (Piper's VITS samples noise per call, Kokoro
exposes nothing), so a same-seed run still renders different audio - which is why
the corpus is frozen as a manifest instead (train/corpus/manifest.py). The seed
covers everything downstream of the corpus.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

SEED_CALL = 'seed=config.get("seed", 0)'
if SEED_CALL in content:
    print(f"Already patched: {path}")
    sys.exit(0)

# 1. Seed the process RNGs once, right after the config is loaded. The load line is
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

# 2. augment_clips takes the seed (data.py:313); pass it at all four call sites.
n = content.count("RIR_paths=rir_paths)")
if n != 4:
    print(f"WARNING: expected 4 augment_clips call sites, found {n} in {path}")
    sys.exit(1)
content = content.replace("RIR_paths=rir_paths)", f"RIR_paths=rir_paths, {SEED_CALL})")

with open(path, 'w') as f:
    f.write(content)

print(f"Patched: {path} (RNGs seeded from config['seed'], {n} augment_clips call sites)")
