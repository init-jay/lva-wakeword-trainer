#!/usr/bin/env python3
"""
Step 4 - listen to a trained wake word live, through the microphone.

RUNS ON THE HOST, IN ITS OWN uv ENVIRONMENT, for the same reason recording does: it
needs the mic. Nothing here is containerised.

    cd src/preflight
    uv run test_model.py --list-devices
    uv run test_model.py --model ../../output/hey_seeree/oww/hey_seeree_705c23b.tflite
    uv run test_model.py --model ../../output/hey_seeree/mww/hey_seeree_705c23b.json


THE THRESHOLD IS THE POINT OF THE EXERCISE. Say the phrase ten or twenty times, at the
distance and volume you would really use, and watch `peak`. A model whose peaks sit
just under your threshold is not broken - it is telling you the operating point is
wrong for this room. For a microWakeWord .json the cutoff comes from the manifest, so
running it here also puts that manifest under test.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# The eval harness source, so `backends` imports. On the host it is not a package
# (the image is what mounts it as `eval`), so preflight puts the directory itself
# on sys.path and imports the module directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "eval" / "src"))

import backends  # noqa: E402

RATE = 16000
WARMUP = 0.6            # seconds avfoundation needs to open the mic
STATUS_EVERY = 2.0      # seconds between level/peak lines while nothing fires


def list_devices(backend: str):
    """Print input devices. Numbering is PER BACKEND - list with the one you record with."""
    if backend == "pyaudio":
        import pyaudio

        pa = pyaudio.PyAudio()
        print("Audio input devices (use the number with --device):")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(f"  {i}: {info['name']}")
        pa.terminate()
        return

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
        match = re.search(r"\[(\d+)\]\s+(.*)", line)
        if match:
            print(f"  {match.group(1)}: {match.group(2).strip()}")


class PyAudioSource:
    """PortAudio capture. Default because ffmpeg's produced audible clicks in src/record/."""

    def __init__(self, device, chunk_samples):
        import pyaudio

        self.chunk_samples = chunk_samples
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
            input_device_index=int(device) if device is not None else None,
            frames_per_buffer=chunk_samples)

    def read(self):
        # exception_on_overflow=False: an overflow costs a chunk, and raising would
        # end a session that is otherwise fine. The status line shows the level, so a
        # dead stream is still visible.
        return self._stream.read(self.chunk_samples, exception_on_overflow=False)

    def close(self):
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()


class FfmpegSource:
    """avfoundation fallback, for a host without PortAudio."""

    def __init__(self, device, chunk_samples):
        self.chunk_samples = chunk_samples
        self.nbytes = chunk_samples * 2
        self._proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-f", "avfoundation", "-i", f":{device}",
             "-ar", str(RATE), "-ac", "1", "-f", "s16le", "-"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(WARMUP)
        if self._proc.poll() is not None:
            stderr = (self._proc.stderr.read().decode(errors="replace")
                      if self._proc.stderr else "")
            raise SystemExit(
                "ffmpeg could not open the microphone. On a first run macOS may need "
                "microphone permission for your terminal (System Settings > Privacy & "
                "Security > Microphone).\n\n" + stderr)

    def read(self):
        """Exactly one chunk. A short read would desync the frontend's buffer."""
        buf = b""
        while len(buf) < self.nbytes:
            piece = self._proc.stdout.read(self.nbytes - len(buf)) if self._proc.stdout else b""
            if not piece:
                return None
            buf += piece
        return buf

    def close(self):
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def dbfs(pcm: np.ndarray) -> float:
    peak = float(np.abs(pcm).max()) / 32768.0
    return 20.0 * np.log10(peak) if peak > 0 else -99.0


def main():
    parser = argparse.ArgumentParser(
        description="Listen to a trained wake word model through the microphone",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help=".tflite or microWakeWord .json")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Detection threshold. Default: the manifest's cutoff for "
                             "a microWakeWord .json, else 0.5")
    parser.add_argument("--backend", choices=["pyaudio", "ffmpeg"], default="pyaudio",
                        help="Capture library (default: %(default)s)")
    parser.add_argument("--device", default=None,
                        help="Input device number; numbering is per --backend")
    parser.add_argument("--sliding-window-size", type=int, default=None,
                        help="microWakeWord only: override the manifest's window")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices(args.backend)
        return
    if not args.model:
        parser.error("--model is required (or pass --list-devices)")
    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")
    if Path(args.model).suffix == ".onnx":
        raise SystemExit(
            "preflight runs the DEPLOYMENT runtime, which has no ONNX path, and a "
            "live check of a pipeline the device does not run would not mean much.\n"
            "Convert it first with src/train/oww/onnx2tflite.py, which verifies the "
            "result, then preflight the .tflite.")

    backend = backends.load(args.model, sliding_window_size=args.sliding_window_size)
    print(f"{Path(args.model).name}")
    print(f"  {backend.describe()}")

    # For a manifest the cutoff is part of the model, so preflighting it tests the
    # manifest too. Falling back to 0.5 for a bare .tflite is a placeholder, not a
    # recommendation - see the eval skill on why 0.5 is the wrong default to compare on.
    threshold = args.threshold
    if threshold is None:
        threshold = float(getattr(backend, "model", None) and
                          getattr(backend.model, "probability_cutoff", 0.5) or 0.5)
        source = "manifest" if getattr(backend, "from_manifest", False) else "default"
    else:
        source = "--threshold"

    chunk = backend.chunk_samples
    Source = PyAudioSource if args.backend == "pyaudio" else FfmpegSource
    try:
        source_stream = Source(args.device, chunk)
    except SystemExit:
        raise
    except Exception as exc:                                          # noqa: BLE001
        raise SystemExit(
            f"could not open the microphone with --backend {args.backend}: "
            f"{type(exc).__name__}: {exc}\n"
            f"On a first run macOS may need microphone permission for your terminal "
            f"(System Settings > Privacy & Security > Microphone).")

    backend.start()
    print(f"\nListening at threshold {threshold} ({source}), "
          f"{chunk / RATE * 1000:.0f}ms chunks - Ctrl-C to stop")
    print("Say the wake word the way you actually would. Watch `peak`: if it sits")
    print("just under the threshold, the operating point is wrong for this room.")
    print("=" * 68)

    detections, window_peak, session_peak, level = 0, 0.0, 0.0, -99.0
    last_status = time.time()
    firing = False
    try:
        while True:
            raw = source_stream.read()
            if raw is None:
                print("\nAudio stream ended.")
                break
            level = dbfs(np.frombuffer(raw, dtype=np.int16))

            for prob in backend.feed(raw):
                window_peak = max(window_peak, prob)
                session_peak = max(session_peak, prob)
                # Edge-triggered: one utterance crosses the threshold for several
                # consecutive frames, and printing each one buries the signal.
                if prob >= threshold and not firing:
                    detections += 1
                    firing = True
                    print(f"  DETECTED  score {prob:.3f}   level {level:+6.1f} dBFS   "
                          f"(#{detections})")
                elif prob < threshold:
                    firing = False

            now = time.time()
            if now - last_status >= STATUS_EVERY:
                # Printed even when nothing fires - a preflight that stays silent on
                # failure tells you nothing about whether the mic is even live.
                quiet = "  (silent - check the mic)" if level < -60 else ""
                print(f"  ... level {level:+6.1f} dBFS   peak score {window_peak:.3f}"
                      f"{quiet}")
                window_peak, last_status = 0.0, now
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        source_stream.close()

    print("=" * 68)
    print(f"{detections} detection(s); highest score seen {session_peak:.3f} "
          f"against threshold {threshold}")
    if detections == 0 and session_peak >= threshold * 0.6:
        print("Nothing fired, but scores got close. That is a threshold/room problem")
        print("rather than a dead model - try --threshold just under the peak above.")
    elif detections == 0:
        print("Nothing came close. Check the level line above was moving while you")
        print("spoke, then confirm the model scores your holdout in src/eval/.")


if __name__ == "__main__":
    main()
