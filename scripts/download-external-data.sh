#!/usr/bin/env bash
#
# Download every third-party corpus the two trainers need, into data/external/.
# Idempotent: anything already present is skipped.
#
#   ./scripts/download-external-data.sh            # all - the default
#   ./scripts/download-external-data.sh oww        # openWakeWord only
#   ./scripts/download-external-data.sh mww        # microWakeWord only
#
#   docker compose run --rm trainer ./download-external-data.sh
#   docker compose run --rm mww     ./download-external-data.sh mww
#
# WHY A TARGET AND NOT JUST "DOWNLOAD EVERYTHING". Almost none of this is actually
# shared, and each trainer's private half is large. Sizes measured on disk after a
# full run, not taken from the download sizes - which for the ambient sets are off by
# more than 4x:
#
#   shared    mit_rirs, audioset_16k, fma       ~730 MB  BOTH trainers augment with
#                                                        these. The small part.
#   oww       ACAV100M + validation features    ~17.2 GB openWakeWord only
#   mww       mww_ambient RaggedMmap sets       ~25 GB   microWakeWord only
#                                                        (~5.7 GB of zips, unpacked)
#
# So `all` costs ~43 GB to get ~730 MB of genuinely common data. Training only for
# the ESP32 with `all` means fetching a 17.2 GB array that nothing in the mWW path
# opens; training only for the server means unpacking 25 GB of ambient spectrograms
# in microWakeWord's own feature format, which the openWakeWord path cannot read.
# Either way the surplus is larger than everything the two actually share.
#
# THE TARGETS ALSO HAVE DIFFERENT TOOL REQUIREMENTS, which is the other reason they
# are separable. `shared` and `oww` resample audio through Python `datasets`, `scipy`
# and `tqdm` - only the trainer image carries those. `mww` needs nothing but curl and
# unzip, so `mww` alone runs in the mww image, or on a bare host.
#
# WHY data/external/ AND NOT data/. Everything under data/ is untracked, but not
# everything under it is the same KIND of thing. These are third-party downloads:
# fixed, enormous, and reproducible from a URL. The recordings beside them are
# irreplaceable, and the corpora are regenerated every run. Keeping downloads in
# their own subtree is what lets `du -sh data/*` and a backup rule tell those three
# apart - and it is why train/provenance.py can hash the other two without ever
# walking these.

set -euo pipefail

TARGET="${1:-all}"
case "$TARGET" in
    all|oww|mww) ;;
    -h|--help|help)
        sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
        exit 0 ;;
    *)
        echo "unknown target: $TARGET (expected all, oww or mww)" >&2
        exit 2 ;;
esac

# /app/data/external inside the container, ./data/external on the host.
DATA_DIR="${DATA_DIR:-./data/external}"
AMBIENT_DIR="$DATA_DIR/mww_ambient"
mkdir -p "$DATA_DIR"

want_shared=0; want_oww=0; want_mww=0
case "$TARGET" in
    all) want_shared=1; want_oww=1; want_mww=1 ;;
    oww) want_shared=1; want_oww=1 ;;
    # NOT want_shared. The audio corpora are shared bytes, and fetching them needs
    # the trainer image's Python stack, which the mww image does not have. The check
    # at the end of this script says so rather than failing at training time.
    mww) want_mww=1 ;;
esac

echo "=== target '$TARGET' -> $DATA_DIR"
echo ""

# --- idempotence -----------------------------------------------------------------
#
# "SKIP IF IT EXISTS" IS ONLY SAFE IF NOTHING HALF-DONE CAN EXIST. Every download
# here therefore lands somewhere temporary and is renamed into its final name only
# once it is complete, so an interrupted run leaves nothing that a later run will
# mistake for finished work.
#
# The failure this prevents is quiet and expensive. A 17 GB curl straight to its
# final path, stopped by Ctrl-C or a full disk at 16 GB, leaves a truncated .npy
# that `[ -f ]` calls present - so every later run skips it and openWakeWord mmaps a
# truncated array deep inside training. Same for the audio corpora: `mkdir` before
# the conversion loop means an interruption leaves an empty directory that `[ -d ]`
# calls done, and the model then trains with no background audio at all.
#
# `--fail` belongs to the same problem. Without it curl writes an HTTP error page to
# the output file and exits 0, and a 400-byte "404: Not Found" then satisfies the
# existence check forever.

fetch_file() {
    # fetch_file <final-path> <url> <description>
    local dest="$1" url="$2" desc="$3"
    if [ -f "$dest" ]; then
        echo "$desc already present, skipping."
        return 0
    fi
    echo "Downloading $desc..."
    # --fail: no error page written as content. -C -: resume a previous .part
    # rather than restarting 17 GB from zero.
    curl -L --fail -C - -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

build_dir() {
    # build_dir <final-dir> <description> -- <command...>
    # Runs the command with BUILD_DIR set to a scratch directory, and renames that
    # into place only on success. The command must write into "$BUILD_DIR".
    local dest="$1" desc="$2"; shift 2
    [ "${1:-}" = "--" ] && shift
    if [ -d "$dest" ]; then
        echo "$desc already present, skipping."
        return 0
    fi
    echo "Downloading $desc..."
    BUILD_DIR="$dest.building"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    "$@"
    mv "$BUILD_DIR" "$dest"
    unset BUILD_DIR
}

# --- openWakeWord-only: the pre-computed feature arrays --------------------------

if [ "$want_oww" = 1 ]; then
    fetch_file "$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" \
        'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy' \
        "ACAV100M features (17GB - this takes a while)"

    fetch_file "$DATA_DIR/validation_set_features.npy" \
        'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy' \
        "validation features"
fi

# --- shared: impulse responses and background audio, for BOTH trainers -----------
#
# openWakeWord reads these as config["rir_paths"] / config["background_paths"];
# microWakeWord as mww/config.py's IMPULSE_DIRS / BACKGROUND_DIRS. Same bytes, same
# directories, fetched once.

build_mit_rirs() {
    # The clone is scratch too, and removed FIRST: leaving it behind on failure is
    # what made a retry die with "destination path already exists".
    local clone="$DATA_DIR/.MIT_environmental_impulse_responses_tmp"
    rm -rf "$clone"
    git lfs install
    git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses "$clone"

    BUILD_DIR="$BUILD_DIR" CLONE="$clone" python3 << 'EOF'
import os
import datasets
import scipy.io.wavfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

out, clone = os.environ["BUILD_DIR"], os.environ["CLONE"]
rir_dataset = datasets.Dataset.from_dict({
    "audio": [str(i) for i in Path(clone, "16khz").glob("*.wav")]
}).cast_column("audio", datasets.Audio())

for row in tqdm(rir_dataset, desc="Processing RIRs"):
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(
        os.path.join(out, name), 16000,
        (row['audio']['array'] * 32767).astype(np.int16)
    )
EOF
    rm -rf "$clone"
}

build_audioset() {
    local staging="$DATA_DIR/.audioset_staging"
    rm -rf "$staging"; mkdir -p "$staging"
    curl -L --fail -C - -o "$staging/bal_train00.tar" \
        'https://huggingface.co/datasets/agkphysics/AudioSet/resolve/196c0900867eff791b8f4d4be57db277e9a5b131/bal_train00.tar'
    tar -xf "$staging/bal_train00.tar" -C "$staging"

    BUILD_DIR="$BUILD_DIR" STAGING="$staging" python3 << 'EOF'
import os
import datasets
import scipy.io.wavfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

out, staging = os.environ["BUILD_DIR"], os.environ["STAGING"]
audioset = datasets.Dataset.from_dict({
    "audio": [str(i) for i in Path(staging, "audio").glob("**/*.flac")]
}).cast_column("audio", datasets.Audio(sampling_rate=16000))

for row in tqdm(audioset, desc="Processing AudioSet"):
    name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
    scipy.io.wavfile.write(
        os.path.join(out, name), 16000,
        (row['audio']['array'] * 32767).astype(np.int16)
    )
EOF
    rm -rf "$staging"
}

build_fma() {
    BUILD_DIR="$BUILD_DIR" python3 << 'EOF'
import os
import datasets
import scipy.io.wavfile
import numpy as np
from tqdm import tqdm

out = os.environ["BUILD_DIR"]
fma = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
fma = iter(fma.cast_column("audio", datasets.Audio(sampling_rate=16000)))

for i in tqdm(range(120), desc="Processing FMA"):
    try:
        row = next(fma)
        name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            os.path.join(out, name), 16000,
            (row['audio']['array'] * 32767).astype(np.int16)
        )
    except StopIteration:
        break
EOF
}

if [ "$want_shared" = 1 ]; then
    build_dir "$DATA_DIR/mit_rirs"      "room impulse responses" -- build_mit_rirs
    build_dir "$DATA_DIR/audioset_16k"  "AudioSet background audio" -- build_audioset

    build_dir "$DATA_DIR/fma" "FMA music samples" -- build_fma
fi

# --- microWakeWord-only: the ambient negative sets --------------------------------
#
# Pre-computed RaggedMmap spectrograms in microWakeWord's own feature format, so
# unlike the audio corpora above they cannot be shared with the openWakeWord side.
# Unlike this repo's adversarial negatives they are large and general - they are what
# teaches the model that ordinary rooms, music and conversation are not the wake word.
#
# They also carry mWW's primary metric. Its model selection minimises false accepts
# per hour on ambient audio before maximising recall, and testing_ambient /
# validation_ambient are what that is measured on.

# name:approx-size, for the progress message only.
SETS=(
    "dinner_party:444MB"
    "dinner_party_eval:82MB"
    "no_speech:2.0GB"
    "speech:3.2GB"
)
BASE_URL="https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main"

if [ "$want_mww" = 1 ]; then
    mkdir -p "$AMBIENT_DIR"

    # Warn early rather than at training time. With target `mww` these are not
    # fetched on purpose - see the header.
    missing=""
    for d in mit_rirs audioset_16k fma; do
        [ -d "$DATA_DIR/$d" ] || missing="$missing $d"
    done
    if [ -n "$missing" ]; then
        echo "WARNING: missing from $DATA_DIR:$missing"
        echo "         These are the augmentation corpora, shared with the"
        echo "         openWakeWord side. Fetch them from the TRAINER image, which"
        echo "         has the Python stack that resamples them:"
        echo "           docker compose run --rm trainer ./download-external-data.sh oww"
        echo
    fi

    for entry in "${SETS[@]}"; do
        name="${entry%%:*}"
        size="${entry##*:}"
        target="$AMBIENT_DIR/$name"

        if [ -d "$target" ]; then
            # REPAIR the double-nesting left by earlier versions of this script, which
            # unzipped straight into $target and so kept the archive's wrapper folder.
            # Cheap: a rename within one filesystem, not a copy of several GB. Done here
            # rather than as a one-off command because the files are root-owned by the
            # container that made them, and because re-running this script is the
            # obvious thing to reach for.
            if [ -d "$target/$name" ]; then
                echo "=== $name is double-nested ($target/$name) - flattening"
                mv "$target/$name" "$target.flat"
                rmdir "$target" 2>/dev/null || rm -rf "$target"
                mv "$target.flat" "$target"
            else
                echo "=== $name already present, skipping"
            fi
            continue
        fi

        echo "=== downloading $name ($size)"
        # -C - resumes a part-downloaded zip from a previous interrupted run rather
        # than re-fetching several GB. The unpack below is already atomic, so a
        # leftover .zip is the only thing an interruption can strand.
        curl -L --fail -C - -o "$AMBIENT_DIR/$name.zip" "$BASE_URL/$name.zip"

        echo "=== unpacking $name"
        # STRIP THE WRAPPER DIRECTORY. These archives contain a single top-level folder
        # named after the set, so unzipping straight into $target yields
        # <target>/<name>/training/..., one level deeper than microWakeWord looks.
        # data.py globs <features_dir>/<split>/**/*_mmap and merely WARNS when it finds
        # nothing, so the mistake shows up as a model trained without ambient negatives
        # rather than as an error.
        tmp="$AMBIENT_DIR/.unpack_$name"
        rm -rf "$tmp"; mkdir -p "$tmp"
        unzip -q "$AMBIENT_DIR/$name.zip" -d "$tmp"

        entries="$(find "$tmp" -mindepth 1 -maxdepth 1)"
        if [ "$(printf '%s\n' "$entries" | wc -l)" -eq 1 ] && [ -d "$entries" ]; then
            mv "$entries" "$target"
        else
            mkdir -p "$target"
            find "$tmp" -mindepth 1 -maxdepth 1 -exec mv {} "$target"/ \;
        fi
        rm -rf "$tmp"
        rm -f "$AMBIENT_DIR/$name.zip"

        # Verify what landed is what the trainer will actually look for, rather than
        # assuming the strip was right for this archive.
        if ! find "$target" -mindepth 2 -maxdepth 3 -type d -name "*_mmap" \
                -path "*/training/*" -o -path "*/testing/*" -name "*_mmap" \
                | grep -q .; then
            echo "  WARNING: $target has no <split>/*_mmap after unpacking - check it"
            find "$target" -maxdepth 2 -type d | head -5
        fi
    done

    echo
    echo "=== RaggedMmap sets found:"
    # The config wants a features_dir whose children are the split directories
    # (training/, validation/, testing/, testing_ambient/, validation_ambient/), each
    # containing *_mmap/ directories. Report what actually landed rather than assuming
    # the archive layout - it is what the --ambient arguments have to point at.
    # Dedupe on the SET directory, not on the mmap path: a set holds one *_mmap per
    # split, so uniquing the raw find output still prints the same set several times.
    found=0
    while IFS= read -r parent; do
        echo "  $parent"
        found=1
    done < <(find "$AMBIENT_DIR" -type d -name "*_mmap" 2>/dev/null \
             | while IFS= read -r m; do dirname "$(dirname "$m")"; done \
             | sort -u | head -40)

    if [ "$found" -eq 0 ]; then
        echo "  NONE FOUND - the archives did not contain *_mmap/ directories."
        echo "  Inspect $AMBIENT_DIR before configuring a run; data.py globs"
        echo "  <features_dir>/{training,validation,testing,...}/**/*_mmap/"
        exit 1
    fi

    echo
    echo "=== which splits each set provides:"
    # THE SETS ARE NOT INTERCHANGEABLE, and which is which is not obvious from the
    # names. Only the *_eval archives carry validation_ambient/testing_ambient, and
    # those are what model selection runs on: the maximization metric is
    # average_viable_recall, computed from false accepts per hour on ambient audio.
    # Without them it reads 0.000 at every step, the best checkpoint never improves on
    # anything, and the exported model is whichever happened to be current - while
    # accuracy, recall and precision all still look excellent.
    have_eval=""
    for entry in "${SETS[@]}"; do
        name="${entry%%:*}"
        [ -d "$AMBIENT_DIR/$name" ] || continue
        splits=""
        for split in training validation testing validation_ambient testing_ambient; do
            if [ -d "$AMBIENT_DIR/$name/$split" ]; then
                splits="$splits $split"
                case "$split" in *_ambient) have_eval=1 ;; esac
            fi
        done
        printf "  %-20s%s\n" "$name" "${splits:- (none - check this set)}"
    done

    echo
    echo "Pass ALL of them to --ambient; they play different roles:"
    printf "  --ambient"
    for entry in "${SETS[@]}"; do
        name="${entry%%:*}"
        [ -d "$AMBIENT_DIR/$name" ] && printf " %s" "$AMBIENT_DIR/$name"
    done
    echo
    if [ -z "$have_eval" ]; then
        echo
        echo "WARNING: no set provides validation_ambient/testing_ambient. Model"
        echo "         selection cannot work without them - average_viable_recall will"
        echo "         be 0.000 at every step. The *_eval archives carry those splits."
    fi
fi

echo ""
echo "=== done ($TARGET) ==="
du -sh "$DATA_DIR"/* 2>/dev/null || true
