"""Refuse, in seconds, to spend the TTS budget on a missing external corpus.

A half-finished ./src/scripts/download-external-data.sh used to fail AFTER
corpus generation - the largest stage of either run - with a deep stack
trace and ~20 minutes gone. Both entry points now check before that spend
(oww: train/oww/train.py main() before any stage; mww: train/mww/features.py
beside its clips guard) and exit with this module's message, which names
the idempotent fix - the download script skips whatever is already
present, so naming it IS the repair.

EXISTENCE ONLY, never completeness: a half-finished download leaves the
directory present, and a size check that rejects a good tree is worse than
no check. Completeness is the download script's job (fetch into a temp
name, rename into place only when done).

The names below are the SINGLE copy: the oww constants used here are the
same ones create_config in train/oww/train.py interpolates into the
training config, and mww/config.py's IMPULSE_DIRS/BACKGROUND_DIRS are the
same directories as repo-relative paths. A retyped second copy is how this
list would silently stop matching when a download moves.
"""

import sys
from pathlib import Path

from train.mww import config as mww_config

# The openWakeWord corpora, by name under --data-dir (data/external by
# default); the download script's header lists them the same way.
OWW_RIR_DIR = "mit_rirs"
OWW_BACKGROUND_DIRS = ("audioset_16k", "fma")
OWW_VALIDATION_FEATURES_NPY = "validation_set_features.npy"
OWW_ACAV_FEATURES_NPY = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"

# The audio corpora SHARED by both trainers' augmentation: the oww ones
# above, and (as repo-relative paths) mww_config's IMPULSE_DIRS/
# BACKGROUND_DIRS. download-external-data.sh fetches them under the
# `all`/`oww` targets only, so a bare `mww` run can be missing them on
# purpose - the note in check() says how to get them.
SHARED_CORPORA = (OWW_RIR_DIR, *OWW_BACKGROUND_DIRS)

# The two ambient RaggedMmap sets the mww route trains against, under
# --data-dir. Named here, not in features.py, because features.py imports
# microwakeword at module level and cannot be imported by the tests: the
# mww guard and the run's "Next:" hint both read the same two names here.
AMBIENT_SETS = ("mww_ambient/speech", "mww_ambient/no_speech")


def oww_external_paths(data_dir) -> list:
    """The corpora the oww route reads, resolved under `data_dir`."""
    data_dir = Path(data_dir)
    return ([data_dir / OWW_RIR_DIR]
            + [data_dir / d for d in OWW_BACKGROUND_DIRS]
            + [data_dir / OWW_VALIDATION_FEATURES_NPY,
               data_dir / OWW_ACAV_FEATURES_NPY])


def mww_external_paths(data_dir) -> list:
    """The corpora the mww route reads, resolved under `data_dir`.

    IMPULSE_DIRS and BACKGROUND_DIRS come from the mww config, resolved
    the way features.py resolves them (by NAME, since those constants are
    repo-relative, so a custom --data-dir is honoured); AMBIENT_SETS is
    already data-dir-relative.
    """
    data_dir = Path(data_dir)
    return ([data_dir / Path(p).name for p in mww_config.IMPULSE_DIRS + mww_config.BACKGROUND_DIRS]
            + [data_dir / p for p in AMBIENT_SETS])


def check(data_dir, target, paths) -> None:
    """Exit, before the corpus/TTS stage, if any of `paths` is absent.

    `target` ("oww" or "mww") picks the download script's target the
    message names; `paths` is what each entry point derives.
    """
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return
    msg = (f"ERROR: missing external data for the {target} trainer under {data_dir}:\n"
           + "\n".join(f"  {p}" for p in missing)
           + f"\nRun ./src/scripts/download-external-data.sh {target} to fetch it "
             f"(idempotent - it skips what is already present).")
    if target == "mww" and any(p.name in SHARED_CORPORA for p in missing):
        msg += ("\nNote: the shared corpora listed are fetched by the `oww` target, "
                "not `mww` - the mww image lacks the Python stack that resamples "
                "them - so for those, run ./src/scripts/download-external-data.sh oww.")
    sys.exit(msg)
