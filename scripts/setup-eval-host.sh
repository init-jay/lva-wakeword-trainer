#!/usr/bin/env bash
#
# Prepare the HOST evaluation environment on Apple Silicon. Setup only - it
# scores nothing; the `eval/.venv/bin/python eval/src/<tool>.py` invocations do.
#
#     ./scripts/setup-eval-host.sh
#
# Idempotent: `uv sync` is a no-op once the lockfile is satisfied, the clone
# install re-applies the same editable, and the .pth is rewritten with exactly
# one absolute path.
#
# THE PINS IN eval/pyproject.toml MUST STAY EQUAL TO eval/Dockerfile's
# (pymicro-wakeword>=2,<3, pyopen-wakeword>=1,<2, numpy>=2,<3, scipy, PyYAML) or
# host and container stop scoring the same inference pipeline and a number
# from one cannot be checked against a number from the other.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
ENV_DIR="eval"
CLONE="openwakeword"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: this is the Apple Silicon host path; you are on $(uname -s)/$(uname -m)." >&2
    echo "       Everywhere else, use the container: cd eval && docker compose build" >&2
    exit 2
fi

command -v uv >/dev/null || { echo "ERROR: uv not found - https://docs.astral.sh/uv/" >&2; exit 2; }

# --- the shared openwakeword clone ----------------------------------------------
#
# The .onnx path (backends.OpenWakeWordOnnxBackend) imports openwakeword.model,
# and P2.4 reuses the clone setup-applesilicon-trainer.sh already makes rather
# than a second one: it is the same upstream code (all eight of that script's
# patches take train.py as their sole target, so the inference path is
# unpatched) with the same v0.5.1 embedding models the Dockerfile bakes in.
if [[ ! -d "$CLONE/openwakeword" ]]; then
    echo "ERROR: no $CLONE/ clone - the .onnx path imports openwakeword.model." >&2
    echo "       ./scripts/setup-applesilicon-trainer.sh creates it (with the" >&2
    echo "       embedding models); the mWW and .tflite paths need no clone." >&2
    exit 2
fi
for m in embedding_model melspectrogram; do
    if [[ ! -f "$CLONE/openwakeword/resources/models/$m.onnx" ]]; then
        echo "ERROR: $CLONE/openwakeword/resources/models/$m.onnx is missing." >&2
        echo "       ./scripts/setup-applesilicon-trainer.sh fetches both; without" >&2
        echo "       them the .onnx path fails deep inside a run, the way the" >&2
        echo "       Dockerfile's verify block exists to prevent." >&2
        exit 2
    fi
done

# --- the environment ------------------------------------------------------------
#
# Python 3.11, the image's base (eval/pyproject.toml): the wheels are py3-none
# so the minor cannot change the pipeline, but the same one keeps a host number
# and a container number apart by nothing at all.
echo "==> syncing $ENV_DIR"
( cd "$ENV_DIR" && VIRTUAL_ENV="$ENV_DIR/.venv" uv sync --quiet )

# openwakeword itself, from the shared clone WITHOUT its dependencies.
#
# --no-deps for the reason the pyproject records: its setup.py pins
# torchaudio>=0.13.1,<1, and resolving that would drag torch into a scoring
# env. Both trainer envs make the same move.
#
# MUST COME AFTER `uv sync`, AND MUST BE REDONE AFTER ANY LATER ONE: sync prunes
# whatever is not in the lockfile, and this package is deliberately not - so a
# bare `uv sync` in that directory silently uninstalls openwakeword and the
# .onnx path dies on `ModuleNotFoundError: No module named 'openwakeword'`.
# Re-running this script is the supported repair; it is idempotent.
#
# The VIRTUAL_ENV pin is load-bearing here, not decorative: `uv pip` resolves
# its target environment from VIRTUAL_ENV before the project's .venv, so with
# any other venv activated in the invoking shell the editable lands in THAT
# environment (incident of 2026-09-07, setup-applesilicon-trainer.sh).
( cd "$ENV_DIR" && VIRTUAL_ENV="$ENV_DIR/.venv" uv pip install --no-deps -e "../$CLONE" --quiet )

# uv's PEP 660 editable is FINDER-based (appended to sys.meta_path). A finder
# loses to a namespace package: when the caller's cwd is the repo root - which
# is how every eval invocation here is made - the clone ROOT directory is a
# namespace portion via the cwd path entry, the PathFinder records it and never
# consults the finder, and `import openwakeword` resolves to the clone root
# (.__file__ is None): verified against train-applesilicon/.venv, which has the
# same install. The Docker image is immune: pip's classic editable writes
# easy-install.pth, a real sys.path entry, which the PathFinder honours.
# Repair: also write a path .pth with the clone root. Idempotent - the content
# is exactly one absolute path.
SITE="$("$ENV_DIR/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$(cd "$CLONE" && pwd)" > "$SITE/openwakeword-clone.pth"

# --- prove it --------------------------------------------------------------------
#
# The same build-time checks eval/Dockerfile's verify block runs, for the same
# reason: a missing shared library or feature model surfaces deep inside a run
# as something that reads like a model problem. Loading a builtin model
# exercises the bundled libtensorflowlite_c, which merely importing does not.
# Two assertions beyond the image's, for two host-only failure modes: numpy
# must stay >=2 (this env is separate from train-applesilicon's <2 for that
# reason), and openwakeword must resolve to the real package, not the clone
# root's namespace portion (the .pth step above).
"$ENV_DIR/.venv/bin/python" - <<'PY'
import numpy, scipy, onnxruntime
import openwakeword
assert openwakeword.__file__, "openwakeword resolved as a namespace package - see the .pth step above"
assert numpy.__version__.split(".")[0] == "2", \
    f"numpy>=2 is the point of this env being separate from train-applesilicon, got {numpy.__version__}"
from openwakeword.model import Model   # noqa: F401
import pymicro_wakeword, pyopen_wakeword
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model as MModel
from pyopen_wakeword import OpenWakeWordFeatures
MicroWakeWordFeatures()
MicroWakeWord.from_builtin(MModel.OKAY_NABU).reset()
OpenWakeWordFeatures.from_builtin()
print("  eval host env OK: python", __import__("sys").version.split()[0],
      "| numpy", numpy.__version__,
      "| pymicro-wakeword",
      pymicro_wakeword.__version__ if hasattr(pymicro_wakeword, "__version__") else "?",
      "| pyopen-wakeword",
      pyopen_wakeword.__version__ if hasattr(pyopen_wakeword, "__version__") else "?",
      "| onnxruntime", onnxruntime.__version__)
PY

# And the invocation FORM itself, not just the imports: the harness scripts
# carry a try/except for exactly two import layouts (the image's `eval`
# package, the plain-path script directory), and a module whose imports pass
# in one layout can still fail in the other. --help runs the whole import
# chain and stops before any scoring.
echo "==> checking the plain-path invocation form"
"$ENV_DIR/.venv/bin/python" eval/src/eval_model.py --help >/dev/null
"$ENV_DIR/.venv/bin/python" eval/src/compare_models.py --help >/dev/null
echo "  eval_model.py / compare_models.py import OK in the plain-path form"

echo
echo "==> ready. Score (from the repo root, no Docker):"
echo "      $ENV_DIR/.venv/bin/python $ENV_DIR/src/eval_model.py --model output/hey_seeree/oww/hey_seeree.onnx"
echo "      $ENV_DIR/.venv/bin/python $ENV_DIR/src/compare_models.py --models <new> <previous-best>"
echo
echo "    The container remains the route on a CUDA box or an eval-only machine:"
echo "      cd $ENV_DIR && docker compose run --rm eval python -m eval.eval_model --model ..."
