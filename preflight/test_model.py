#!/usr/bin/env python3
"""
Test a trained OpenWakeWord model with microphone input.

Capture goes through ffmpeg's avfoundation input, so no PortAudio/PyAudio or other
system packages are required. Only numpy and openwakeword are needed.

Usage:
    python test_model.py --list-devices
    python test_model.py --model my_custom_model/hey_seeree.onnx
    python test_model.py --model my_custom_model/hey_seeree.onnx --threshold 0.3
"""

import argparse
import os
import re
import subprocess
import time

import numpy as np
from openwakeword.model import Model

RATE = 16000
CHUNK = 1280                   # 80ms at 16kHz - the frame size OpenWakeWord expects
CHUNK_BYTES = CHUNK * 2        # 16-bit mono
WARMUP = 0.6                   # seconds avfoundation needs to open the mic


def list_devices():
    """Print avfoundation audio input devices and their numbers."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    ).stderr
    audio = out.split("AVFoundation audio devices:")
    if len(audio) < 2:
        print(out)
        return
    print("Audio input devices (use the number with --device):")
    for line in audio[1].splitlines():
        m = re.search(r"\[(\d+)\]\s+(.*)", line)
        if m:
            print(f"  {m.group(1)}: {m.group(2).strip()}")


def open_stream(device: str) -> subprocess.Popen:
    """Spawn ffmpeg streaming raw 16kHz mono 16-bit PCM to stdout."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "avfoundation", "-i", f":{device}",
        "-ar", str(RATE), "-ac", "1",
        "-f", "s16le", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_exact(pipe, nbytes: int):
    """Read exactly nbytes from a pipe. Returns None if the stream ends.

    A pipe read can come back short, so this loops - handing OpenWakeWord a
    partial frame would silently corrupt its rolling feature buffer.
    """
    buf = b""
    while len(buf) < nbytes:
        chunk = pipe.read(nbytes - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    parser = argparse.ArgumentParser(description="Test a wake word model with microphone")
    parser.add_argument("--model", help="Path to .onnx model file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0-1.0)")
    parser.add_argument("--device", default="0", help="avfoundation audio device number")
    parser.add_argument("--list-devices", action="store_true", help="List microphones and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return
    if not args.model:
        parser.error("--model is required (or pass --list-devices)")

    if not os.path.exists(args.model):
        raise SystemExit(f"model not found: {args.model}")

    print("Loading model...")
    start = time.time()
    # The argument is `wakeword_models`, NOT `wakeword_model_paths`. Model.__init__
    # takes **kwargs, so a wrong name is swallowed silently, leaving the list empty -
    # which openWakeWord treats as "load all pre-trained models" and then fails deep
    # inside prediction with a confusing tensor-shape error. The check below turns
    # that whole class of mistake into an immediate, obvious failure.
    oww_model = Model(wakeword_models=[args.model])
    print(f"Model loaded in {time.time() - start:.2f}s")

    loaded = list(oww_model.models.keys())
    if len(loaded) != 1:
        raise SystemExit(
            f"expected exactly 1 model, got {len(loaded)}: {loaded}\n"
            "openWakeWord fell back to its pre-trained models, which means the model "
            "argument did not reach it."
        )
    print(f"Loaded: {loaded[0]}")

    proc = open_stream(args.device)
    # open_stream always sets both pipes; bind them so they read as non-optional.
    stdout, stderr_pipe = proc.stdout, proc.stderr
    if stdout is None or stderr_pipe is None:
        raise SystemExit("failed to open ffmpeg pipes")

    # Give the device time to come up before claiming to listen, and surface a
    # clear error if ffmpeg died instead of opening the mic.
    time.sleep(WARMUP)
    if proc.poll() is not None:
        stderr = stderr_pipe.read().decode(errors="replace")
        raise SystemExit(
            "ffmpeg could not open the microphone. If this is the first run, macOS "
            "may need microphone permission for your terminal "
            "(System Settings > Privacy & Security > Microphone).\n\n" + stderr
        )

    print(f"\nListening (threshold: {args.threshold}) - Ctrl+C to stop")
    print("=" * 50)

    try:
        while True:
            raw = read_exact(stdout, CHUNK_BYTES)
            if raw is None:
                print("\nAudio stream ended.")
                break
            audio_array = np.frombuffer(raw, dtype=np.int16)

            start = time.time()
            prediction = oww_model.predict(audio_array)
            inference_ms = (time.time() - start) * 1000

            for model_name, score in prediction.items():
                if score > args.threshold:
                    print(f"DETECTED: {model_name} (score: {score:.3f}, inference: {inference_ms:.1f}ms)")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
