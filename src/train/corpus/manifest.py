#!/usr/bin/env python3
"""Name the corpus that training consumes, and refuse to reuse a stale one.

After a corpus stage completes, this writes data/corpus/<wake_word_safe>/<target>/
corpus.json: the wake word, the engines probed (name, URL, version), the voice
list actually used, the per-bucket clip counts, EVERY corpus-shaping flag the
build was run with, the recipe hash, the seed, the wall time, and a content
digest of the final wav tree. The file IS the corpus's identity.

    python -m train.corpus.manifest --corpus-dir data/corpus/<wake_word>/mww --show
    python -m train.corpus.manifest --corpus-dir ... --check-requested '{"seed": 42}'

DURING A SWEEP THE CORPUS IS A HELD-FIXED INDEPENDENT VARIABLE. The TTS engines
are NOT bit-reproducible (Piper's VITS samples noise per call), so regenerating
per run means every comparison carries a fresh draw of TTS noise on top of the
knob under test. The manifest makes the corpus an explicit, named, frozen
artefact: a run says "I consumed THIS corpus", and `--corpus reuse` becomes a
check, not a hope.

FREEZING DOES NOT MEAN REPRODUCING. The engines cannot be seeded (the recorded
`seed` seeds the drawing/sampling, not the rendering), so reuse means exactly
one thing: use the clips ALREADY ON DISK. A re-render wants a deleted tree and
a rebuild; the `content_digest` (sha256 over the final wav tree) is the number
that moves when it does, and `corpus_identity()` (sha256 of the manifest FILE
itself) is what src/train/provenance.py hashes into the run tag's data half.

`check_reuse()` compares the requested shaping flags against the manifest and
exits non-zero on ANY difference, printing the diff: a corpus that no longer
matches its request is rebuilt, never coaxed. The voice SET is a first-class
axis, not a shaping flag: the cf9c065b reuse — every shaping flag matched, but
the check never compared `voices`, so a post-reservation request matched the
pre-reservation corpus. A manifest recording a different voice set is refused
just like one shaped differently.

DIGEST SCOPE: digestting the whole <safe>/<target> tree in one call covers the
oww four-subdir and mww two-subdir layouts and any future one, while
`per_voice_counts` keeps the per-bucket breakdown. The digest must cover the
FINAL tree state (trimming and child-range copies included): the call site is
the only thing that keeps it honest.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# This package sits at src/train/corpus/, so the import root src/ is two
# levels up, not one.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.provenance import digest_tree  # noqa: E402

SCHEMA = 1
MANIFEST_NAME = "corpus.json"

SHORT = 7          # same length as a git short hash, for the same reason


def _normalize_engines(engines):
    """The engines as recorded in the manifest: a dict keyed by engine name.
    The caller may hand over the dict form or a flat list of entries each
    carrying an engine name ("kokoro" or "piper" key, else "name"); the
    manifest ALWAYS stores the dict form - one canonical shape to compare
    against later."""
    if engines is None:
        return None
    if isinstance(engines, dict):
        return {str(name): dict(info) for name, info in engines.items()}
    out = {}
    for entry in engines:
        name = entry.get("engine") or entry.get("name")
        out[str(name)] = {k: v for k, v in entry.items() if k not in ("engine", "name")}
    return out


def _recipe_hash(recipe_path):
    """sha256-of-bytes of the recipe yaml, or None when not provided / absent.
    The recipe is the text the corpus renders, so it is corpus identity: a
    different recipe under the same shaping flags is a different corpus even
    though every shaping key still matches."""
    if recipe_path is None:
        return None
    path = Path(recipe_path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(corpus_dir, wake_word, target, seed, shaping,
                   engines=None, voices=None, per_voice_counts=None,
                   recipe_path=None, wall_time_s=None):
    """Digest the corpus tree and write corpus.json beside it. Returns the path.
    corpus_dir is the <wake_word_safe>/<target> directory - one tree for oww,
    two sub-trees for mww; the digest covers the whole tree. Call this only
    AFTER the corpus stage completes: called mid-build it will happily record
    the current state without failing - the call site is the only thing that
    keeps it honest."""
    corpus_dir = Path(corpus_dir)
    digest, files, total_bytes = digest_tree(corpus_dir)

    manifest = {
        "schema": SCHEMA,
        "wake_word": wake_word,
        "target": target,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Seeds the drawing/sampling (which phrases, voices, speakers), not the
        # rendering (Piper's VITS samples noise per call, Kokoro cannot be
        # seeded at all), so this alone does not make a rebuild reproducible.
        # Recorded anyway: a same-seed, different-draw sweep is the kind of
        # false comparison that cannot otherwise be distinguished.
        "seed": seed,
        "engines": _normalize_engines(engines),
        # The voice set ACTUALLY used, post-exclusion — not the catalog the
        # engine offered. Piper entries are (voice, speaker) pairs,
        # json-serialised as [voice, speaker] lists; matches_requested
        # normalises both shapes before comparing (a tuple never equals a
        # list). This is the axis the cf9c065b reuse missed: recorded but
        # never diffed, so a post-reservation request matched the
        # pre-reservation corpus on every shaping flag. `piper: []` is honest,
        # not a hole: the build ran with piper_fraction 0 and rendered no Piper
        # clips at all (the oww default).
        "voices": {"kokoro": list(voices.get("kokoro", [])),
                   "piper": list(voices.get("piper", []))} if voices is not None else None,
        # oww: {"positive_train", "positive_test", "negative_train", "negative_test"};
        # mww: {"positives", "negatives"}. Taken as a dict, never hardcoded.
        "per_voice_counts": dict(per_voice_counts) if per_voice_counts is not None else None,
        # Every corpus-shaping flag the build was run with (samples_per_voice,
        # runon_fraction, piper_fraction / kokoro_fraction, real_copies, ...),
        # verbatim, so matches_requested() can diff the request against exactly
        # what the build had.
        "shaping": dict(shaping) if shaping is not None else {},
        "recipe_hash": _recipe_hash(recipe_path),
        "wall_time_s": wall_time_s,
        "content_digest": {"sha256": digest, "files": files, "bytes": total_bytes},
    }

    path = corpus_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"corpus manifest: {path}  ({target}, {files} clips, "
          f"digest {digest[:SHORT] if digest else 'absent'})")
    return path


def load_manifest(corpus_dir):
    """The corpus.json at <corpus_dir>, or None when no manifest exists there."""
    path = Path(corpus_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _canonical_voice(entry):
    """One voice entry in its comparison shape: (voice, speaker-or-None).
    Kokoro entries are bare voice strings; Piper entries are (voice, speaker)
    pairs that json hands back as [voice, speaker] LISTS - a tuple and a list
    never compare equal, so both sides of any diff must land on this one shape
    first, or a matching corpus would refuse itself."""
    if isinstance(entry, (list, tuple)):
        speaker = entry[1] if len(entry) > 1 and entry[1] is not None else None
        return (str(entry[0]), speaker)
    return (str(entry), None)


def _canonical_voices(voices):
    """A recorded or requested voice set in comparable form: sorted per engine.
    Sorted rather than order-preserving: catalog order belongs to the engine,
    and the comparison is about WHICH voices, not which slot each sat in. A
    bare non-dict value reads as the empty set; None (no set recorded at all)
    stays None so the caller can report the missing field."""
    if voices is None:
        return None
    if not isinstance(voices, dict):
        voices = {}
    return {engine: sorted(_canonical_voice(e) for e in (voices.get(engine) or []))
            for engine in ("kokoro", "piper")}


def _voice_set_diffs(requested, recorded):
    """Diff lines for every engine whose voice set moved between build and request.
    The line names the voices that MOVED, not the whole set - the moved
    names are what the operator needs. Both sides are order-insensitive and
    tuple/list-normalised via _canonical_voices."""
    req = _canonical_voices(requested)
    man = _canonical_voices(recorded)
    diffs = []
    for engine in ("kokoro", "piper"):
        r, m = set(req[engine]), set(man[engine])
        if r == m:
            continue
        parts = []
        if r - m:
            parts.append("new since the build: " + ", ".join(
                f"{v}:{s}" if s else v for v, s in sorted(r - m)))
        if m - r:
            parts.append("dropped from the build: " + ", ".join(
                f"{v}:{s}" if s else v for v, s in sorted(m - r)))
        diffs.append(f"requested voices.{engine}={len(req[engine])} voice(s), "
                     f"corpus has {len(man[engine])} voice(s) ({'; '.join(parts)})")
    return diffs


def matches_requested(manifest, requested):
    """Diff strings for every requested shaping key that the manifest disagrees
    with. `requested` carries only the flags the caller cares about, and those
    keys are exactly the ones compared — a manifest missing a requested key is
    reported, not skipped. `"seed"` is special-cased against manifest["seed"]
    (top level, not under "shaping"); so is `"voices"`, against the top-level
    set the build actually used. The voice set is a different axis from every
    shaping flag (the cf9c065b reuse matched ALL shaping flags because the
    holdout exclusion lives in the voice set, not in any flag). A manifest
    that records no `voices` field cannot be verified on that axis and is
    refused, not skipped. Empty list = match."""
    shaping = manifest.get("shaping") or {}
    diffs = []
    for key, value in requested.items():
        if key == "seed":
            actual, where = manifest.get("seed"), "seed"
        elif key == "voices":
            actual, where = manifest.get("voices"), "voices"
            if actual is None:
                diffs.append(f"requested {key} (a voice set), corpus manifest has "
                             f"no {where} field - the corpus predates voice "
                             f"recording, so its voice set cannot be verified and "
                             f"it cannot be reused")
                continue
            diffs.extend(_voice_set_diffs(value, actual))
            continue
        else:
            actual, where = shaping.get(key), "shaping"
            if key not in shaping:
                diffs.append(f"requested {key}={value!r}, corpus manifest has no {where}.{key}")
                continue
        if actual != value:
            diffs.append(f"requested {key}={value!r}, corpus has {actual!r}")
    return diffs


def check_reuse(corpus_dir, requested):
    """The ergonomic front door for `--corpus reuse`: exit(1) unless the corpus
    matches. Silent stale reuse — a comparison run built on a corpus it
    believes it asked for but did not get — is the failure mode this refuses;
    refusal is cheap (one JSON read), a corrupted sweep point is the expensive
    kind of bug."""
    corpus_dir = Path(corpus_dir)
    manifest = load_manifest(corpus_dir)
    if manifest is None:
        print(f"no manifest at {corpus_dir / MANIFEST_NAME}; the corpus was not built "
              f"by a version that writes one. Rebuild it.")
        sys.exit(1)

    diffs = matches_requested(manifest, requested)
    if diffs:
        for line in diffs:
            print(f"  {line}")
        print("refusing to reuse a corpus shaped differently than requested — rebuild it")
        sys.exit(1)
    print(f"corpus at {corpus_dir} matches the requested shaping — reusing it")
    return None


def corpus_identity(corpus_dir):
    """The short-7 identity for the run tag: sha256 of the manifest FILE BYTES.
    Not the content digest: the file's own bytes already incorporate the tree
    digest, the shaping and the engines — one number a run can consume instead
    of re-walking the tree. A re-render (TTS is not bit-reproducible) changes
    the tree digest and therefore this identity too. None when absent."""
    path = Path(corpus_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:SHORT]


def main():
    p = argparse.ArgumentParser(
        description=f"{__doc__ and 'Inspect or check a frozen-corpus manifest'}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", required=True,
                   help="the <wake_word_safe>/<target> corpus directory")
    p.add_argument("--show", action="store_true",
                   help="print the manifest prettily (the default without "
                        "--check-requested)")
    p.add_argument("--check-requested", default=None, metavar="JSON",
                   help='a JSON object of requested shaping flags (and "seed", '
                        'and "voices", a {"kokoro": [...], "piper": [[voice, speaker], ...]} '
                        "set); prints the diff lines, or MATCH when the corpus agrees")
    args = p.parse_args()

    manifest = load_manifest(args.corpus_dir)
    if args.check_requested is not None:
        requested = json.loads(args.check_requested)
        if manifest is None:
            print(f"no manifest at {Path(args.corpus_dir) / MANIFEST_NAME}; nothing to "
                  f"check against. Rebuild the corpus.")
            sys.exit(1)
        diffs = matches_requested(manifest, requested)
        if diffs:
            for line in diffs:
                print(f"  {line}")
            # Non-zero, matching check_reuse: a script can gate a reuse on the
            # exit code without parsing the diff lines.
            sys.exit(1)
        print("MATCH")
    elif manifest is None:
        print(f"no manifest at {Path(args.corpus_dir) / MANIFEST_NAME}; the corpus was "
              f"not built by a version that writes one.")
        sys.exit(1)
    else:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
