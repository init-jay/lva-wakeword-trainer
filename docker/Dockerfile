FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-dev python3-pip \
    git git-lfs curl build-essential portaudio19-dev libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install OpenWakeWord
RUN git clone https://github.com/dscripka/openWakeWord openwakeword \
    && pip install --no-cache-dir -e ./openwakeword

# ONNX Runtime with CUDA, for feature computation (melspectrogram + embedding).
#
# Must come AFTER openwakeword: its setup.py pins 'onnxruntime>=1.10.0,<2', the
# CPU-only package, which would otherwise be installed over the top. The two are
# separate PyPI packages that both provide the `onnxruntime` module, so the CPU one
# is removed rather than left to win by import order.
#
# 1.19 is the first release whose default PyPI wheels are built against CUDA 12,
# which is what this base image provides. Earlier versions default to CUDA 11.8 and
# would load nothing.
RUN pip uninstall -y onnxruntime \
    && pip install --no-cache-dir "onnxruntime-gpu>=1.19,<2"

# onnxruntime-gpu needs cuDNN and cuBLAS, which this base image does not carry - it
# is the -devel variant, not -cudnn-devel. They are already present as the nvidia-*
# wheels that the torch cu121 install pulled in, so point the loader at those rather
# than adding a second copy. Discovered from the installed package rather than
# hardcoded, since the path moves with the Python version.
RUN python3 -c "\
import glob, os, site, sysconfig; \
roots = set(site.getsitepackages() + [sysconfig.get_paths()['purelib']]); \
libs = [p for r in roots for p in glob.glob(r + '/nvidia/*/lib/*.so*')]; \
dirs = sorted({os.path.dirname(p) for p in libs}); \
open('/etc/ld.so.conf.d/nvidia-pip.conf', 'w').write('\n'.join(dirs) + '\n'); \
print('nvidia library paths:', dirs or 'NONE FOUND - onnxruntime will fall back to CPU')" \
    && ldconfig \
    && python3 -c "\
import onnxruntime; \
providers = onnxruntime.get_available_providers(); \
print('onnxruntime', onnxruntime.__version__, 'providers:', providers); \
assert 'CUDAExecutionProvider' in providers, 'CUDA provider missing from the build'"

# Patch: make piper generate_samples import conditional (we use Kokoro, not Piper)
COPY patches/skip-piper-import.py /tmp/skip-piper-import.py
RUN python3 /tmp/skip-piper-import.py openwakeword/openwakeword/train.py

# Patch: make augmentation_rounds > 1 actually produce more data instead of
# augmenting N times and discarding all but the first pass
COPY patches/honour-augmentation-rounds.py /tmp/honour-augmentation-rounds.py
RUN python3 /tmp/honour-augmentation-rounds.py openwakeword/openwakeword/train.py

# Patch: choose the feature-computation device from onnxruntime's own providers.
# Upstream asks torch, so with the CPU build of onnxruntime (which openwakeword's
# setup.py pins) it requests a CUDA provider that does not exist AND sets ncpu=1,
# leaving feature computation single-threaded on CPU - 36 min of an 83 min run.
COPY patches/feature-device-selection.py /tmp/feature-device-selection.py
RUN python3 /tmp/feature-device-selection.py openwakeword/openwakeword/train.py

# Patch: take the corpus directory from the config. Upstream derives it from
# output_dir/model_name, which also names the exported model, so the openWakeWord
# and microWakeWord corpora could not sit side by side under one wake word without
# renaming the model. See patches/configurable-corpus-dir.py.
COPY patches/configurable-corpus-dir.py /tmp/configurable-corpus-dir.py
RUN python3 /tmp/configurable-corpus-dir.py openwakeword/openwakeword/train.py

# Patch: hold the training features in VRAM rather than mmap. The 17.28 GB
# ACAV100M array against 20 GB of RAM leaves training stalling on page faults -
# GPU at 14%, CPU 37% idle, 7.2 GB swapped. Requires `docker compose stop kokoro
# kokoro2` before training so their CUDA contexts are not holding ~2.4 GB.
COPY patches/gpu-resident-features.py /tmp/gpu-resident-features.py
RUN python3 /tmp/gpu-resident-features.py openwakeword/openwakeword/data.py \
    && python3 /tmp/gpu-resident-features.py openwakeword/openwakeword/train.py

# Download embedding models (small, safe to bake into image)
RUN mkdir -p openwakeword/openwakeword/resources/models \
    && curl -L -o openwakeword/openwakeword/resources/models/embedding_model.onnx \
        'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx' \
    && curl -L -o openwakeword/openwakeword/resources/models/melspectrogram.onnx \
        'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx'

# The onnx -> tflite toolchain, checked here rather than at conversion time. The
# conversion runs AFTER a ~16 minute training run, so a dependency set that only
# resolves on paper would be discovered at the worst possible moment. PATH is
# checked as well as the import because onnx2tflite.py shells out to onnx2tf as
# a CLI rather than importing it.
RUN python3 -c "import tensorflow as tf; print('tensorflow', tf.__version__)" \
    && python3 -c "import onnx2tf" \
    && command -v onnx2tf

# The whole train/ package: train.corpus is shared, train.oww is this trainer.
# Copied as a package rather than as loose scripts because train/oww/train.py imports
# train.corpus by that name - without the package root the container fails at import,
# before any of the expensive setup runs.
COPY train/ ./train/
COPY scripts/setup-data.sh .
RUN chmod +x setup-data.sh

# Fail the build rather than a training run if the package does not import.
RUN python3 -c "import train.corpus.augment, train.corpus.negatives, \
                       train.corpus.real, train.corpus.piper"
