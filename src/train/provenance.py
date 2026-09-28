#!/usr/bin/env python3
"""Name a training run after the code, the audio, and the settings.

    b38715c-d3e9f14         no --target: the legacy format, kept byte-for-byte
    b38715c-d1a2b3c-h4d5e6f  --target {oww,mww} [plus --config-file]:
                             c: audio THIS trainer consumed
                             h: the resolved config (hyperparameters + seed)

The commit alone is not enough: the two inputs this repo deliberately does NOT
track - the real recordings and the TTS-generated corpus - are precisely the
ones that move.

- c half: data/corpus/<wake_word>/<target> - sha256 of the corpus.json manifest
  bytes if present (cheap to re-hash, names the audio across re-renders), else a
  tree digest over the scoped directory. Recordings are covered, not a separate
  half: the build copies real clips into the corpus. data/recordings/holdout/ is
  NOT hashed - recording more holdout must not change a model's identity. Fixed
  downloads (audioset, fma, rirs, ambient) and the derived feature caches are not
  hashed either.
- h half: sha over the resolved hyperparameters, seed included. The TTS engines
  are not seedable, so render-to-render audio variance lives in the c half.

Expect the data half to move even when nothing changed - the TTS is not
bit-reproducible. The tag names a RUN, not a configuration. Content, not mtime:
a tag that changed because of a file copy (rsync rewrites timestamps) would be
worse than no tag.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# The GIT root: data/ lives there, one level above the train/ package.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Audio only. The feature caches beside it are derived from these files by tracked
# code, so hashing them would add gigabytes of I/O for no extra information.
AUDIO_SUFFIXES = (".wav",)

SHORT = 7          # same length as a git short hash, for the same reason

TARGETS = ("oww", "mww")


def code_tag():
    """The git short commit, plus `-dirty` when the tree has uncommitted changes."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        tag = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--quiet"]).returncode != 0
        return tag + ("-dirty" if dirty else "")
    except Exception:                                                # noqa: BLE001
        return None


def digest_tree(root):
    """(hex, files, bytes) over the audio under `root`, or (None, 0, 0) if absent.

    The relative path goes into the hash with the bytes: which speaker a clip
    belongs to is training data, so moving it must change the digest.
    """
    root = Path(root)
    if not root.is_dir():
        return None, 0, 0

    entries = sorted(
        (p.relative_to(root).as_posix(), p)
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)

    sha, count, total = hashlib.sha256(), 0, 0
    for rel, path in entries:
        data = path.read_bytes()
        sha.update(rel.encode("utf-8"))
        sha.update(b"\0")
        sha.update(data)
        count += 1
        total += len(data)
    return sha.hexdigest(), count, total


def components(wake_word, target=None):
    """The untracked inputs that feed training, in a fixed order.

    With a target the corpus component is scoped to data/corpus/<safe>/<target>:
    the legacy whole-tree hash read the other trainer's corpus and a 1.5 GB backup
    dir and took a measured 13.9 s to detect changes that could not affect it.
    """
    safe = wake_word.replace(" ", "_").lower()
    corpus_root = REPO_ROOT / "data" / "corpus" / safe
    if target is not None:
        corpus_root = corpus_root / target
    return [
        ("recordings", REPO_ROOT / "data" / "recordings" / "samples"),
        ("corpus", corpus_root),
    ]


def corpus_tag(wake_word, target):
    """(short7, manifest_path, hexdigest, count, total_bytes) for ONE trainer's audio.

    corpus.json is the corpus identity when it exists: its file bytes carry the
    content digests, re-hashing is cheap, and the name survives a re-render. A
    tree digest over the scoped tree stands in without it.
    """
    safe = wake_word.replace(" ", "_").lower()
    root = REPO_ROOT / "data" / "corpus" / safe / target
    manifest = root / "corpus.json"
    if manifest.is_file():
        data = manifest.read_bytes()
        hexd = hashlib.sha256(data).hexdigest()
        return hexd[:SHORT], manifest, hexd, 1, len(data)
    hexd, count, total = digest_tree(root)
    if hexd is None:
        return "absent", None, None, 0, 0
    return hexd[:SHORT], None, hexd, count, total


def config_tag(resolved_config):
    """A short hash over the resolved hyperparameters.

    The seed MUST be a key of the dict: two runs differing only in seed must not
    share a tag. This half moves only when the caller changes a setting - the
    TTS engines are not seedable, so render variance lives in the corpus half.
    """
    return hashlib.sha256(
        json.dumps(resolved_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:SHORT]


def data_tag(wake_word):
    """One short hash over every untracked input, plus the per-component detail."""
    rows, combined = [], hashlib.sha256()
    for name, root in components(wake_word):
        hexd, count, total = digest_tree(root)
        rows.append((name, root, hexd, count, total))
        # NAME as well as digest: an empty corpus and an empty recordings dir
        # must not hash the same.
        combined.update(name.encode("utf-8"))
        combined.update((hexd or "absent").encode("utf-8"))
    return combined.hexdigest()[:SHORT], rows


def run_tag(wake_word, target=None, config=None, fallback=None):
    """The name a run is filed under.

    With a target: `<commit>[-dirty]-c<corpus7>` plus `-h<config7>` when a resolved
    config is given, so sweep points at the same corpus get distinct names. Without:
    the legacy `<commit>[-dirty]-d<...>` format, unchanged, so existing scripts
    keep working.

    Outside a git checkout the code half falls back to the caller's stamp: a tag
    with no code half is still worth having - the data half cannot be recovered
    from anywhere else.
    """
    code = code_tag() or fallback or "nogit"
    if target is None:
        data, _ = data_tag(wake_word)
        return f"{code}-d{data}"
    short, _, _, _, _ = corpus_tag(wake_word, target)
    tag = f"{code}-c{short}"
    if config is not None:
        tag += f"-h{config_tag(config)}"
    return tag


def main():
    p = argparse.ArgumentParser(description="Identify a training run by code + data (+ config)")
    p.add_argument("--wake-word", required=True)
    p.add_argument("--tag", action="store_true", help="print only the tag")
    p.add_argument("--target", choices=TARGETS, default=None,
                   help="scope the corpus half to this trainer's tree")
    p.add_argument("--config-file", default=None,
                   help="JSON file holding the resolved config dict (the h-half)")
    p.add_argument("--fallback", default=None,
                   help="stand-in for the code half outside a git checkout")
    args = p.parse_args()

    config = None
    if args.config_file:
        config = json.loads(Path(args.config_file).read_text(encoding="utf-8"))

    if args.tag:
        print(run_tag(args.wake_word, target=args.target, config=config,
                      fallback=args.fallback))
        return

    code = code_tag() or "(not a git checkout)"
    print(f"  {'code':<12}{code}")

    missing = []
    if args.target is None:
        # The legacy breakdown, untouched: scripts diff it.
        data, rows = data_tag(args.wake_word)
        for name, root, hexd, count, total in rows:
            rel = root.relative_to(REPO_ROOT)
            if hexd is None:
                print(f"  {name:<12}{'-':<10}  MISSING  {rel}")
                missing.append(name)
                continue
            print(f"  {name:<12}{hexd[:SHORT]:<10}  {count:>6} files  "
                  f"{total / 1e6:>8.1f} MB  {rel}")
        print(f"  {'data':<12}d{data}")
        print(f"  {'tag':<12}{run_tag(args.wake_word, fallback=args.fallback)}")
        if missing:
            print()
            print("  A MISSING component still produces a tag, deliberately - training")
            print("  without real recordings, or before the corpus is built, is a real")
            print("  state and the tag should record it rather than refuse to name it.")
        return

    # The targeted breakdown. Recordings are shown for information only - the
    # c-half covers them: the build copies real clips into the scoped corpus.
    _, recordings_root = components(args.wake_word, args.target)[0]
    rhexd, rcount, rtotal = digest_tree(recordings_root)
    if rhexd is None:
        print(f"  {'recordings':<12}{'-':<10}  MISSING  "
              f"{recordings_root.relative_to(REPO_ROOT)}")
        missing.append("recordings")
    else:
        print(f"  {'recordings':<12}{rhexd[:SHORT]:<10}  {rcount:>6} files  "
              f"{rtotal / 1e6:>8.1f} MB  "
              f"{recordings_root.relative_to(REPO_ROOT)}")

    short, manifest, chexd, ccount, ctotal = corpus_tag(args.wake_word, args.target)
    croot = REPO_ROOT / "data" / "corpus" / args.wake_word.replace(" ", "_").lower() / args.target
    if chexd is None:
        print(f"  {'corpus':<12}{'-':<10}  MISSING  {croot.relative_to(REPO_ROOT)}")
        missing.append("corpus")
    else:
        print(f"  {'corpus':<12}{short:<10}  {ccount:>6} files  "
              f"{ctotal / 1e6:>8.1f} MB  {croot.relative_to(REPO_ROOT)}")
        identity = (f"manifest: {manifest.relative_to(REPO_ROOT)}"
                    if manifest is not None else "tree digest")
        # Indented under the corpus line: which identity is actually in the tag.
        print(f"  {'':12}{'':10}  identity: {identity}")

    if config is not None:
        print(f"  {'config':<12}{config_tag(config)}")
    # Single-line f-string: src/eval/.venv is Python 3.11 and multi-line f-strings
    # are 3.12 (PEP 701); src/scripts/sweep.py imports this module under either venv.
    tag = run_tag(args.wake_word, target=args.target, config=config, fallback=args.fallback)
    print(f"  {'tag':<12}{tag}")

    if missing:
        print()
        print("  A MISSING component still produces a tag, deliberately - training")
        print("  without real recordings, or before the corpus is built, is a real")
        print("  state and the tag should record it rather than refuse to name it.")


if __name__ == "__main__":
    main()
