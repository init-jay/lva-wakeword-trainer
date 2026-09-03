#!/bin/bash
# OpenWakeWord Trainer - Manual Setup (without Docker)
# For the recommended Docker setup, see README.md
#
# Downloads dependencies and ~17GB of training data

set -e

# The REPO ROOT, not this script's directory - it moved to scripts/ in the reorg and
# everything below writes into data/, venv/ and the tree at the top level.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== OpenWakeWord Trainer Setup ==="
echo ""

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Training will be very slow without GPU."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Using Python: $(which python)"
pip install --upgrade pip

echo ""
echo "=== Installing Python dependencies ==="

# PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install -r requirements.txt

# Clone OpenWakeWord
if [ ! -d "openwakeword" ]; then
    echo ""
    echo "=== Cloning OpenWakeWord ==="
    git clone https://github.com/dscripka/openWakeWord openwakeword
fi
pip install -e ./openwakeword

# Download embedding models
echo ""
echo "=== Downloading OpenWakeWord models ==="
mkdir -p openwakeword/openwakeword/resources/models
curl -L -o openwakeword/openwakeword/resources/models/embedding_model.onnx \
    'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx'
curl -L -o openwakeword/openwakeword/resources/models/melspectrogram.onnx \
    'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx'

# Download pre-computed features (~17GB)
echo ""
echo "=== Downloading training features (17GB - this takes a while) ==="
if [ ! -f "openwakeword_features_ACAV100M_2000_hrs_16bit.npy" ]; then
    curl -L -O 'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy'
fi

if [ ! -f "validation_set_features.npy" ]; then
    curl -L -O 'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy'
fi

# Download MIT Room Impulse Responses
echo ""
echo "=== Downloading room impulse responses ==="
if [ ! -d "mit_rirs" ]; then
    git lfs install
    git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses

    mkdir -p mit_rirs
    python3 << 'EOF'
import datasets
import scipy.io.wavfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

rir_dataset = datasets.Dataset.from_dict({
    "audio": [str(i) for i in Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav")]
}).cast_column("audio", datasets.Audio())

for row in tqdm(rir_dataset, desc="Processing RIRs"):
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(
        f"mit_rirs/{name}", 16000,
        (row['audio']['array'] * 32767).astype(np.int16)
    )
EOF
fi

# Download background audio (AudioSet subset)
echo ""
echo "=== Downloading background audio ==="
if [ ! -d "audioset_16k" ]; then
    mkdir -p audioset audioset_16k
    curl -L -o audioset/bal_train09.tar \
        'https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/bal_train09.tar'
    tar -xf audioset/bal_train09.tar -C audioset

    python3 << 'EOF'
import datasets
import scipy.io.wavfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

audioset = datasets.Dataset.from_dict({
    "audio": [str(i) for i in Path("audioset/audio").glob("**/*.flac")]
}).cast_column("audio", datasets.Audio(sampling_rate=16000))

for row in tqdm(audioset, desc="Processing AudioSet"):
    name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
    scipy.io.wavfile.write(
        f"audioset_16k/{name}", 16000,
        (row['audio']['array'] * 32767).astype(np.int16)
    )
EOF
fi

# Download FMA music samples
if [ ! -d "fma" ]; then
    mkdir -p fma
    python3 << 'EOF'
import datasets
import scipy.io.wavfile
import numpy as np
from tqdm import tqdm

fma = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
fma = iter(fma.cast_column("audio", datasets.Audio(sampling_rate=16000)))

# Get ~1 hour of music (30-second clips)
for i in tqdm(range(120), desc="Processing FMA"):
    try:
        row = next(fma)
        name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            f"fma/{name}", 16000,
            (row['audio']['array'] * 32767).astype(np.int16)
        )
    except StopIteration:
        break
EOF
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Start Kokoro TTS:"
echo "     docker run -d --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest"
echo ""
echo "  2. Record your voice samples (optional but recommended):"
echo "     source venv/bin/activate"
echo "     cd record && uv run record_samples.py --wake-word \"hey seeree\""
echo ""
echo "  3. Train your model:"
echo "     python -m train.oww.train --wake-word \"hey seeree\""
echo ""
