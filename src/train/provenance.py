#!/usr/bin/env python3
"""Name a training run after the code, the audio, and the settings.

    python -m train.provenance --wake-word "hey seeree"          # the breakdown
    python -m train.provenance --wake-word "hey seeree" --tag    # just the tag
    python -m train.provenance --wake-word "hey seeree" \
        --target mww [--config-file resolved.json]               # per-trainer tag

    b38715c-d3e9f14            no --target: the legacy format, kept byte-for-byte
    ^^^^^^^ ^^^^^^^
    code    the audio training actually consumed

    b38715c-d1a2b3c-h4d5e6f    --target {oww,mww} [plus --config-file]
    ^^^^^^^ ^^^^^^^ ^^^^^^^
    code    c: audio THIS trainer consumed
            h: the resolved config (hyperparameters + seed)

WHY THE COMMIT ALONE IS NOT ENOUGH. A git hash identifies the code and every file
git tracks - including wordlists/, docker/requirements.txt and the patches. It says
nothing about the two inputs this repo deliberately does NOT track, and those are
precisely the ones that move: the real recordings, and the synthetic corpus generated
from a TTS server. Two runs at the same commit, one with a third speaker recorded and
one without, produce different models and were until now indistinguishable by name.

WHY THE c/h SPLIT, AND WHY IT IS SCOPED TO THE TARGET. The c-half is THE AUDIO THIS
TRAINER CONSUMED: data/corpus/<wake_word>/<target>. The real recordings are copied
INTO the corpus at build time, so they are already inside that half - which is why a
targeted tag needs no separate recordings half. Scoping is also what the legacy
whole-tree hash got wrong in practice: it read the OTHER trainer's corpus and stray
backup directories (data/corpus/hey_seeree/oww.backup-preruonsplit/ alone is 1.5 GB)
and took a measured 13.9 s to detect changes that could not affect this target.
Where a corpus build writes a manifest at data/corpus/<wake_word>/<target>/corpus.json,
the c-half is the sha256 of the MANIFEST FILE BYTES: it carries the content digests
inside, re-hashing it is cheap, and it names the audio even after a re-render that
would make a raw tree digest differ. Without the manifest, a tree digest over the
scoped directory stands in, so the tag works before the manifest workstream lands.
The h-half is a sha over the resolved hyperparameters, seed included: two sweep
points at the same commit with the same corpus share the c-half and need the h-half
to get distinct names - without it, mww training refuses to run into the directory
the first sweep point already claimed. The TTS engines are not seedable, so the
render-to-render audio variance is carried by the c-half, not the h-half.

WHAT IS HASHED, AND WHAT IS NOT (the targeted tag; the legacy one hashes the whole
corpus tree in place of the scoped half):

    data/recordings/samples/    COVERED, not a separate half - real clips are
                                      copied into the scoped corpus at build time
    data/corpus/<wake_word>/    YES (corpus.json if present, else a tree digest)
    <target>/
    data/recordings/holdout/    NO  - never trained on. Recording more holdout must
                                      not change a model's identity, or the tag stops
                                      meaning "what went in" and starts meaning
                                      "what was on disk".
    data/audioset_16k, fma,     NO  - fixed downloads. Tens of GB, hashing them would
    mit_rirs, ambient sets            cost minutes per run to detect a change that
                                      does not happen.
    feature caches (.npy,       NO  - derived from the WAVs above by the code above,
    *_mmap)                           so they are already covered, transitively, and
                                      they are the large part of the corpus.

EXPECT THE DATA HALF TO MOVE EVEN WHEN YOU CHANGED NOTHING, and treat that as the
tool working. The corpus is re-rendered by a TTS server on every run and the audio
is not bit-identical between renders - measured here: regenerating the eval corpus
from an unchanged wordlist produced the same 100 filenames and shifted the scorecard
(one category 0/12 -> 1/12). So the tag identifies A RUN, not a configuration. That
is the useful reading anyway: two runs of an identical configuration have measured
77% and 67% on the same held-out clips, and calling them by the same name is how
that stops being visible.

Content, not mtime. A rsync to the training box rewrites every timestamp, and a tag
that changed because of a file copy would be worse than no tag at all.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# The GIT root: data/ lives there, one level above the (now nested) train/
# package - the pre-reorg parents[1] now points at src/.
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

    The relative path goes into the hash with the bytes, so moving a clip between
    speaker directories changes the digest. It should: which speaker a clip belongs
    to is training data, not filing.
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

    With a target the corpus component is scoped to data/corpus/<safe>/<target>
    instead of the whole tree: the legacy whole-tree hash read the other
    trainer's corpus and backup directories (oww.backup-preruonsplit/ alone is
    1.5 GB) and took a measured 13.9 s.
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

    The manifest at data/corpus/<safe>/<target>/corpus.json is the corpus identity
    when it exists: we hash its file bytes (they carry the content digests) rather
    than the tree, so re-hashing stays cheap and the identity names the audio even
    after a re-render that would not be bit-identical. Without it, a tree digest
    over the scoped tree stands in.
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

    The seed MUST be a key of the dict the caller passes: two runs differing only
    in seed produce different audio and must not share a tag. The TTS engines are
    not seedable, so the render-to-render variance lives in the corpus half, not
    this one - this half moves only when the caller changes a setting.
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
        # The NAME is folded in as well as the digest, so an empty corpus and an
        # empty recordings directory cannot hash to the same thing.
        combined.update(name.encode("utf-8"))
        combined.update((hexd or "absent").encode("utf-8"))
    return combined.hexdigest()[:SHORT], rows


def run_tag(wake_word, target=None, config=None, fallback=None):
    """The name a run is filed under.

    With a target: `<commit>[-dirty]-c<corpus7>` plus `-h<config7>` when a resolved
    config is given - the c-half is the audio this trainer consumed, the h-half the
    settings (seed included) so sweep points get distinct names. Without a target:
    the legacy `<commit>[-dirty]-d<...>` format, unchanged, so existing scripts
    keep working until they move over.

    Falls back to the caller's stamp for the code half outside a git checkout - a
    tag with no code half is still worth having, because the data half is the part
    that cannot be recovered from anywhere else.
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
    p.add_argument("--wake-word", default="hey seeree")
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
        # The legacy breakdown, untouched, so scripts that diff it keep working.
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
    # tag's c-half covers them, because the build copies real clips into the
    # scoped corpus.
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
        # Indented under the corpus line, so a reader can tell which identity is
        # actually in the tag without re-deriving it.
        print(f"  {'':12}{'':10}  identity: {identity}")

    if config is not None:
        print(f"  {'config':<12}{config_tag(config)}")
    # Single-line f-string deliberately: eval/.venv is Python 3.11 (the eval
    # image's base) and multi-line f-strings are a 3.12 feature (PEP 701) -
    # scripts/sweep.py imports this module and runs under either venv.
    tag = run_tag(args.wake_word, target=args.target, config=config, fallback=args.fallback)
    print(f"  {'tag':<12}{tag}")

    if missing:
        print()
        print("  A MISSING component still produces a tag, deliberately - training")
        print("  without real recordings, or before the corpus is built, is a real")
        print("  state and the tag should record it rather than refuse to name it.")


if __name__ == "__main__":
    main()
