#!/usr/bin/env bash
#
# Download the microWakeWord ambient negative sets. ~5.7 GB.
#
# THIS DELIBERATELY DOWNLOADS NOTHING ELSE. mWW's augmentation wants impulse
# responses and background audio, and those are the SAME three corpora setup-data.sh
# already fetched for openWakeWord - MIT RIRs, AudioSet and FMA, from the same URLs.
# train/mww/config.py points `impulse_paths` and `background_paths` at them in place.
# Re-fetching would cost another ~10 GB for identical bytes.
#
# What is genuinely new is the ambient negative sets. They are pre-computed
# RaggedMmap spectrograms in microWakeWord's own feature format, so unlike the audio
# corpora they cannot be shared with the openWakeWord side, and unlike this repo's
# adversarial negatives they are large and general - they are what teaches the model
# that ordinary rooms, music and conversation are not the wake word.
#
# They also carry mWW's primary metric. Its model selection minimises false accepts
# per hour on ambient audio before maximising recall, and testing_ambient /
# validation_ambient are what that is measured on.
#
# Idempotent: each set is skipped if already unpacked.
#
#   ./setup-mww-data.sh
#   DATA_DIR=/mnt/big/data ./setup-mww-data.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
AMBIENT_DIR="$DATA_DIR/mww_ambient"
BASE_URL="https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main"

# name:approx-size, for the progress message only.
SETS=(
    "dinner_party:444MB"
    "dinner_party_eval:82MB"
    "no_speech:2.0GB"
    "speech:3.2GB"
)

mkdir -p "$AMBIENT_DIR"

# Warn early rather than at training time: mWW needs these and this script does not
# provide them on purpose.
missing=""
for d in mit_rirs audioset_16k fma; do
    [ -d "$DATA_DIR/$d" ] || missing="$missing $d"
done
if [ -n "$missing" ]; then
    echo "WARNING: missing from $DATA_DIR:$missing"
    echo "         These are the augmentation corpora, shared with the openWakeWord"
    echo "         side. Run ./setup-data.sh - this script does not duplicate them."
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
    curl -L --fail -o "$AMBIENT_DIR/$name.zip" "$BASE_URL/$name.zip"

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
echo "=== done. RaggedMmap sets found:"
# The config wants a features_dir whose children are the split directories
# (training/, validation/, testing/, testing_ambient/, validation_ambient/), each
# containing *_mmap/ directories. Report what actually landed rather than assuming
# the archive layout - it is what the --ambient arguments have to point at.
found=0
while IFS= read -r mmap; do
    parent="$(dirname "$(dirname "$mmap")")"
    echo "  $parent"
    found=1
done < <(find "$AMBIENT_DIR" -type d -name "*_mmap" 2>/dev/null | head -40 | sort -u)

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
