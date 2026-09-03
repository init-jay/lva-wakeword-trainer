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

THE SPLIT COMES FROM Clips, NOT FROM US. Clips(random_split_seed, split_count)
partitions the directory, so the same seed gives the same partition every run and
training never sees its own validation clips.

    python -m mww.features --wake-word "hey seeree"
"""

import argparse
import shutil
import sys
from pathlib import Path

# The REPO ROOT. This package sits at train/mww/ since the reorg, so the root is
# two levels up, not one - the old value now points at train/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmap_ninja.ragged import RaggedMmap  # noqa: E402
from microwakeword.audio.augmentation import Augmentation  # noqa: E402
from microwakeword.audio.clips import Clips  # noqa: E402
from microwakeword.audio.spectrograms import SpectrogramGeneration  # noqa: E402

from train.mww import config as mww_config  # noqa: E402

# split -> (Clips generator mode, slide_frames). Upstream's notebook values.
SPLITS = {
    "training": ("train", 10),
    "validation": ("validation", 10),
    "testing": ("test", 1),
}


def build_split(clips_dir: Path, out_root: Path, name: str, impulse, background,
                split_seed=10, split_count=0.1, step_ms=None, clean=False):
    step_ms = step_ms or mww_config.WINDOW_STEP_MS
    clips = Clips(
        input_directory=str(clips_dir),
        file_pattern="*.wav",
        # Already trimmed by corpus/augment.py, with a method calibrated on this
        # corpus - letting webrtcvad trim again stacks two silence definitions.
        remove_silence=False,
        random_split_seed=split_seed,
        split_count=split_count,
    )
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
    p.add_argument("--corpus-root", default="my_custom_model")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-seed", type=int, default=10)
    p.add_argument("--split-count", type=float, default=0.1)
    p.add_argument("--clean", action="store_true",
                   help="rebuild features that already exist. Required after "
                        "regenerating the corpus - otherwise the old spectrograms "
                        "are kept and silently trained on.")
    args = p.parse_args()

    safe = args.wake_word.replace(" ", "_").lower()
    corpus = Path(args.corpus_root) / safe / "mww"
    data = Path(args.data_dir)
    impulse = [data / Path(x).name for x in mww_config.IMPULSE_DIRS]
    background = [data / Path(x).name for x in mww_config.BACKGROUND_DIRS]

    for label, clips_dir in (("positives", corpus / "positives"),
                             ("negatives", corpus / "negatives")):
        n = len(list(clips_dir.glob("*.wav"))) if clips_dir.is_dir() else 0
        if n == 0:
            sys.exit(f"no clips in {clips_dir} - run `python -m mww.corpus` first")
        print(f"\n[{label}] {n} clips -> {corpus / 'features' / label}")
        build_split(clips_dir, corpus / "features" / label, label,
                    impulse, background, split_seed=args.split_seed,
                    split_count=args.split_count, clean=args.clean)

    print(f"\nDONE  features under {corpus / 'features'}")
    print("\nNext:")
    print(f'  python -m mww.train --wake-word "{args.wake_word}" \\')
    print("      --ambient data/mww_ambient/speech data/mww_ambient/no_speech")


if __name__ == "__main__":
    main()
