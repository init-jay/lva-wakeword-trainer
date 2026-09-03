#!/usr/bin/env python3
"""Name a training run after the code AND the audio that produced it.

    python -m train.provenance --wake-word "hey seeree"          # the breakdown
    python -m train.provenance --wake-word "hey seeree" --tag    # just the tag

    705c23b-d3e9f14
    ^^^^^^^ ^^^^^^^
    code    the audio training actually consumed

WHY THE COMMIT ALONE IS NOT ENOUGH. A git hash identifies the code and every file
git tracks - including wordlists/, requirements.txt and the patches. It says nothing
about the two inputs this repo deliberately does NOT track, and those are precisely
the ones that move: the real recordings, and the synthetic corpus generated from a
TTS server. Two runs at the same commit, one with a third speaker recorded and one
without, produce different models and were until now indistinguishable by name.

So the split is exactly the .gitignore boundary. Everything git tracks is in the
first half; everything under data/ that git ignores is in the second. Nothing is
covered twice and nothing is missed.

WHAT IS HASHED, AND WHAT IS NOT:

    data/recordings/samples/    YES - the real clips training copies in
    data/corpus/<wake_word>/    YES - the synthetic corpus training consumes
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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Audio only. The feature caches beside it are derived from these files by tracked
# code, so hashing them would add gigabytes of I/O for no extra information.
AUDIO_SUFFIXES = (".wav",)

SHORT = 7          # same length as a git short hash, for the same reason


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


def components(wake_word):
    """The untracked inputs that feed training, in a fixed order."""
    safe = wake_word.replace(" ", "_").lower()
    return [
        ("recordings", REPO_ROOT / "data" / "recordings" / "samples"),
        ("corpus", REPO_ROOT / "data" / "corpus" / safe),
    ]


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


def run_tag(wake_word, fallback=None):
    """`<commit>[-dirty]-d<data>`, the name a run is filed under.

    Falls back to the caller's stamp for the code half outside a git checkout - a
    tag with no code half is still worth having, because the data half is the part
    that cannot be recovered from anywhere else.
    """
    code = code_tag() or fallback or "nogit"
    data, _ = data_tag(wake_word)
    return f"{code}-d{data}"


def main():
    p = argparse.ArgumentParser(description="Identify a training run by code + data")
    p.add_argument("--wake-word", default="hey seeree")
    p.add_argument("--tag", action="store_true", help="print only the tag")
    p.add_argument("--fallback", default=None,
                   help="stand-in for the code half outside a git checkout")
    args = p.parse_args()

    if args.tag:
        print(run_tag(args.wake_word, args.fallback))
        return

    code = code_tag()
    data, rows = data_tag(args.wake_word)
    print(f"  {'code':<12}{code or '(not a git checkout)'}")
    for name, root, hexd, count, total in rows:
        rel = root.relative_to(REPO_ROOT)
        if hexd is None:
            print(f"  {name:<12}{'-':<10}  MISSING  {rel}")
            continue
        print(f"  {name:<12}{hexd[:SHORT]:<10}  {count:>6} files  "
              f"{total / 1e6:>8.1f} MB  {rel}")
    # Shown with the `d` prefix it carries in the tag, so the two are obviously the
    # same number rather than two hashes that happen to look alike.
    print(f"  {'data':<12}d{data}")
    print(f"  {'tag':<12}{run_tag(args.wake_word, args.fallback)}")

    if any(h is None for _, _, h, _, _ in rows):
        print()
        print("  A MISSING component still produces a tag, deliberately - training")
        print("  without real recordings, or before the corpus is built, is a real")
        print("  state and the tag should record it rather than refuse to name it.")


if __name__ == "__main__":
    main()
