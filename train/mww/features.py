#!/usr/bin/env python3
"""Turn the WAV corpus into microWakeWord's RaggedMmap spectrogram features.

WHY THIS EXISTS, HAVING ONCE CONCLUDED IT DID NOT NEED TO. A feature set in the
training YAML can be `type: clips`, which reads a directory of WAVs and generates
spectrograms on the fly - so it looked as though the corpus could be consumed
directly and no conversion step was needed. That is true for TRAINING and false for
everything else:

    # data.py, ClipsHandlerWrapperGenerator
    def get_mode_size(self, mode):
        if mode == "training":
            return len(self.spectrogram_generation.clips.clips)
        else:
            return 0

Validation and testing therefore receive nothing from a `clips` set, and the
validation step fails with a shape error from whatever the ambient sets happen to
yield instead. `type: mmap` is the only way to supply validation and testing data.

WHAT IT WRITES

    <out>/training/<name>_mmap/      slide_frames=10
    <out>/validation/<name>_mmap/    slide_frames=10
    <out>/testing/<name>_mmap/       slide_frames=1

data.py globs <features_dir>/<split>/**/*_mmap/, so the split directory names are
load-bearing - a set outside them is silently invisible.

`slide_frames` > 1 yields several overlapping spectrograms per clip by dropping
frames from the end, which imitates the sequential inputs a streaming model sees.
Testing wants the real thing, so it uses 1. Those are upstream's notebook values.

THE SPLIT IS OURS, NOT Clips'. It used to be `Clips(random_split_seed, split_count)`
partitioning the directory, which is per FILE - and a person's real recordings enter the
corpus as N copies of each (`--real-copies`, and their shifted variants), so per-file
scattering put copy 3 of one utterance in training and copy 7 of the same utterance in
validation. The consequence is not only a flattering validation number: mWW SELECTS
checkpoints on `average_viable_recall` over that validation set, so the leak biases which
weights ship. `group_partition` below splits by the identity of the underlying recording
instead, which is what lets `--real-copies` be used at all on this target.

The budget is 2 * split_count of the RECORDINGS, one row of each held out: with a high
copy factor a ROW budget would hold out a fraction of the recordings and count each of
them many times, which is the wrong shape for the number that picks the weights. The
partition is a stable hash (hashlib, not `hash()` - that one is salted per process, which
would make the split move between runs of identical input) over those identities, with no
seed input at all, so the same corpus gives the same split every run and no RNG state is
consumed by the corpus order the way Clips' shuffle did. See train/mww/split.py for why
"no seed" is the point rather than an omission.

    python -m train.mww.features --wake-word "hey seeree"
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np

# The REPO ROOT. This package sits at train/mww/ since the reorg, so the root is
# two levels up, not one - the old value now points at train/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmap_ninja.ragged import RaggedMmap  # noqa: E402
from microwakeword.audio.augmentation import Augmentation  # noqa: E402
from microwakeword.audio.clips import Clips  # noqa: E402
from microwakeword.audio.spectrograms import SpectrogramGeneration  # noqa: E402

from train.mww import config as mww_config  # noqa: E402
# The partition rules live in their own dependency-free module so they can be
# tested without microwakeword; see train/mww/split.py.
from train.mww.split import group_partition, recording_identity  # noqa: E402, F401

# split -> (Clips generator mode, slide_frames). Upstream's notebook values.
SPLITS = {
    "training": ("train", 10),
    "validation": ("validation", 10),
    "testing": ("test", 1),
}

def build_split(clips_dir: Path, out_root: Path, name: str, impulse, background,
                split_count=0.1, step_ms=None, clean=False):
    step_ms = step_ms or mww_config.WINDOW_STEP_MS
    clips = Clips(
        input_directory=str(clips_dir),
        file_pattern="*.wav",
        # Already trimmed by corpus/augment.py, with a method calibrated on this
        # corpus - letting webrtcvad trim again stacks two silence definitions.
        remove_silence=False,
        # No random_split_seed: the partition below is identity-aware, and passing the
        # seed here would use upstream's per-FILE shuffle instead.
        random_split_seed=None,
        split_count=split_count,
    )

    # THE SPLIT, OURS. Clips leaves split_clips unset when random_split_seed is None, so
    # build it here in the shape SpectrogramGeneration expects: three HF subsets of the
    # same rows Clips already filtered (duration etc), selected by position.
    import datasets as hf_datasets          # noqa: PLC0415  (Clips pulls it in anyway)
    # Read the paths back with decoding OFF. `clips.clips["audio"]` on the column Clips
    # already cast to Audio(sampling_rate=16000) DECODES every clip to a numpy array -
    # thousands of them, to get a filename - and yields dicts, not paths, which is how this
    # first attempt failed. cast_column with decode=False returns the {path, bytes} rows
    # without touching the audio, and does not reorder, so the indices still address
    # clips.clips' rows.
    paths = [Path(row["path"]).name for row in
             clips.clips.cast_column("audio", hf_datasets.Audio(decode=False))["audio"]]
    assignment = group_partition(paths, split_count)
    by_mode = {"train": [], "validation": [], "test": [], "dropped": []}
    # Index by the ORIGINAL row order: clips.clips.select() addresses rows of that
    # dataset, while group_partition iterates groups in hash order. Enumerating its
    # values instead of `paths` would select the wrong rows - and every count would
    # still look right.
    for idx, p in enumerate(paths):
        by_mode[assignment[p]].append(idx)
    dropped = by_mode.pop("dropped")
    if not by_mode["validation"] or not by_mode["test"]:
        # Unreachable in practice - group_partition refuses first - but the cost of it
        # ever being wrong is a run that selects nothing, so keep the guard on both sides.
        raise SystemExit(f"ERROR: empty validation/test split for {clips_dir}")
    clips.split_clips = hf_datasets.DatasetDict(
        {mode: clips.clips.select(idxs) for mode, idxs in by_mode.items()})
    if dropped:
        # Rows that exist in the corpus and in no split. Silent would be the worst
        # property of this number - it is how a 10x corpus stops being 10x.
        print(f"  split: {len(by_mode['train'])} train / "
              f"{len(by_mode['validation'])} validation / {len(by_mode['test'])} test "
              f"rows, {len(dropped)} copy rows DROPPED so held-out recordings stay "
              f"shallow")

    augmenter = Augmentation(
        augmentation_duration_s=mww_config.CLIP_DURATION_MS / 1000.0,
        impulse_paths=[str(p) for p in impulse],
        background_paths=[str(p) for p in background],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    written = {}
    for split, (mode, slide_frames) in SPLITS.items():
        gen = SpectrogramGeneration(clips, augmenter, step_ms=step_ms,
                                    slide_frames=slide_frames)
        out = out_root / split / f"{name}_mmap"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if not clean:
                # Skipping makes this resumable, but it also means features built
                # from a PREVIOUS corpus survive a corpus rebuild - and nothing
                # downstream can tell stale spectrograms from fresh ones.
                print(f"  {split}/{name}_mmap exists, SKIPPING - stale if the corpus "
                      f"changed since. Use --clean to rebuild.")
                continue
            shutil.rmtree(out)
        print(f"  {split}/{name}_mmap (slide_frames={slide_frames}) ...", flush=True)
        RaggedMmap.from_generator(
            out_dir=str(out),
            sample_generator=gen.spectrogram_generator(split=mode),
            batch_size=100,
            verbose=True,
        )
        written[split] = out
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", default="hey seeree")
    p.add_argument("--corpus-root", default="data/corpus")
    # Joined with only the BASENAME of IMPULSE_DIRS/BACKGROUND_DIRS below, so this
    # is the single place that decides where the third-party corpora are read from.
    p.add_argument("--data-dir", default="data/external")
    p.add_argument("--split-count", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0,
                   help="seed the augmentation draws (RIR/background choice, "
                        "jitter) for the spectrogram pass (default: %(default)s = "
                        "unseeded). The train/validation/test partition takes no "
                        "seed at all - it is a hash of the recording identities "
                        "(train/mww/split.py) so that it cannot move when this one "
                        "does; this covers the augmentation, which was the one "
                        "unseeded draw between the corpus and the model.")
    p.add_argument("--clean", action="store_true",
                   help="rebuild features that already exist. Required after "
                        "regenerating the corpus - otherwise the old spectrograms "
                        "are kept and silently trained on.")
    args = p.parse_args()

    if args.seed:
        # The augmentation (microwakeword's Augmentation) draws RIR, background
        # and jitter from the numpy global RNG; seeding here makes a same-seed,
        # same-corpus features pass draw the same augmentation. The TF global seed
        # is set too, guarded: this stage is numpy-only today, but the venv carries
        # TensorFlow and a future import that initialises from it must not become
        # a silent variance source.
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(args.seed)
        except ImportError:
            pass
        print(f"[seed] {args.seed}")

    safe = args.wake_word.replace(" ", "_").lower()
    corpus = Path(args.corpus_root) / safe / "mww"
    data = Path(args.data_dir)
    impulse = [data / Path(x).name for x in mww_config.IMPULSE_DIRS]
    background = [data / Path(x).name for x in mww_config.BACKGROUND_DIRS]

    for label, clips_dir in (("positives", corpus / "positives"),
                             ("negatives", corpus / "negatives")):
        n = len(list(clips_dir.glob("*.wav"))) if clips_dir.is_dir() else 0
        if n == 0:
            sys.exit(f"no clips in {clips_dir} - run `python -m train.mww.corpus` first")
        print(f"\n[{label}] {n} clips -> {corpus / 'features' / label}")
        build_split(clips_dir, corpus / "features" / label, label,
                    impulse, background,
                    split_count=args.split_count, clean=args.clean)

    print(f"\nDONE  features under {corpus / 'features'}")
    print("\nNext:")
    print(f'  python -m train.mww.train --wake-word "{args.wake_word}" \\')
    print("      --ambient data/external/mww_ambient/speech data/external/mww_ambient/no_speech")


if __name__ == "__main__":
    main()
