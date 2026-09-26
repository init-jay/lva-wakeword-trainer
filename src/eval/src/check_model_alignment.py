#!/usr/bin/env python3
"""
Measure where in the detection window a *trained* model wants the phrase to sit.

`check_alignment.py` inspects the training clips - the input side. This inspects
the model that came out, which is the only way to confirm the clips taught it what
you intended.

Method: place each positive clip in a fixed-size window so its end sits `gap` ms
before the window end, compute features exactly as training does
(`AudioFeatures.embed_clips`, the call `compute_features_from_generator` makes),
and score the window. Sweeping `gap` traces the alignment the model actually
learned. This is deliberately NOT the streaming path, so nothing in the streaming
feature pipeline can be blamed for the result.

What to expect: `trim_silence` leaves `pad_ms=30` and `create_fixed_size_clip` adds
`end_jitter` from U(0, 200) ms, so a model trained on trimmed clips should peak
around 100-200 ms. A peak up near 400 ms means the clips carried trailing silence
into training - untrimmed Kokoro output sits about there (a measured result).

The peak also *is* the latency: the model cannot fire until that much audio has
arrived after you stop speaking.

Takes either the .onnx or the .tflite. Prefer the .tflite when that is what you
deploy: a wrong-axis conversion loads cleanly and returns plausible scores while
detecting nothing, so the artifact that ships is the one worth measuring.

WHY THIS KEEPS ITS OWN LOADER INSTEAD OF USING `src/eval/src/backends.py`. Step 3
said to point all three eval tools at the shared backend layer; the other two are on
it and this one is not, deliberately. That contract is `score(pcm) -> scores`, fed 16
kHz audio and streamed. This tool never has PCM at the point it calls the model - it
has already computed one window of embeddings via `AudioFeatures.embed_clips` - and it
deliberately does NOT stream, so that nothing in the streaming feature pipeline can be
blamed for the result. Different input, different question. Forcing it through the
streaming contract would delete the property it exists to have, at a saving of about
fifteen lines of interpreter setup.

The consequence is that this tool is openWakeWord-only. microWakeWord has no alignment
measurement yet; its latency is measured by `eval_model.py`, which is the deployed
number, but not where in its window it wants the phrase.

Needs onnxruntime (and ai-edge-litert for .tflite) plus an importable openwakeword, so
unlike `check_alignment.py` this does not run on a bare host. The eval image has
everything the .onnx path needs and runs on the Mac:

    docker compose run --rm eval python -m eval.check_model_alignment \\
        --model output/hey_seeree/oww/hey_seeree_705c23b.onnx

A .tflite needs ai-edge-litert, which the eval image does not carry - it uses the
deployment runtime's bundled interpreter instead - so use the trainer image for that.

Usage:
    python -m eval.check_model_alignment \\
        --model output/hey_seeree/oww/hey_seeree_705c23b.onnx
    python -m eval.check_model_alignment --model M \\
        --positives data/recordings/holdout/speaker1
    python -m eval.check_model_alignment --model M --step 20 --max-gap 600

Run it as a MODULE, from the repo root (`python -m eval.check_model_alignment`, from
/app in the image). The plain-path form (`python src/eval/src/check_model_alignment.py`)
puts this directory on sys.path instead of the root; the try/except on the `paths`
import below is what keeps it working.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile

try:
    from eval import paths
except ImportError:
    # Plain-path form (`python src/eval/src/check_model_alignment.py`): this directory
    # is on sys.path, the package is not.
    import paths

SR = 16000

# Mirrors trim_silence() in train.py. Duplicated rather than imported for the same
# reason check_alignment.py duplicates its energy detection, plus one specific to
# this script: train.py calls os.chdir() at import time, which would silently break
# any relative path passed on the command line.
TOP_DB = 40.0
PAD_MS = 30.0
FRAME_MS = 10.0

# create_fixed_size_clip picks end_jitter from U(0, 200) ms, so even a perfectly
# trimmed clip is placed somewhere in [PAD_MS, PAD_MS + 200] ms from the window end.
END_JITTER_MS = 200.0


def trim_silence(data, sr=SR, top_db=TOP_DB, pad_ms=PAD_MS, frame_ms=FRAME_MS):
    """Trim leading/trailing silence by short-time RMS energy (see train.py)."""
    if data.size == 0:
        return data

    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = len(data) // frame
    if n_frames < 2:
        return data

    frames = data[:n_frames * frame].astype(np.float64).reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return data

    voiced = np.flatnonzero(rms > peak * (10 ** (-top_db / 20)))
    if voiced.size == 0:
        return data

    pad = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * frame - pad)
    end = min(len(data), (voiced[-1] + 1) * frame + pad)
    if end - start < int(sr * 0.2):
        return data

    return data[start:end]


def load_clips(directories, trim=True, limit=None):
    """Load 16 kHz clips. Searched recursively, so per-speaker subdirectories work."""
    clips, wrong_rate = [], []
    for directory in directories:
        for wav in sorted(Path(directory).rglob("*.wav")):
            sr, data = scipy.io.wavfile.read(wav)
            if data.ndim > 1:
                data = data[:, 0]
            if sr != SR:
                wrong_rate.append(wav.name)
                continue
            clips.append((wav.name, (trim_silence(data, sr) if trim else data).astype(np.int16)))
    if wrong_rate:
        print(f"WARNING: skipped {len(wrong_rate)} clip(s) not at {SR} Hz: "
              f"{', '.join(wrong_rate[:3])}{'...' if len(wrong_rate) > 3 else ''}")
    return clips[:limit] if limit else clips


def place(clip, gap_ms, total_length, noise_floor, rng):
    """One window with the clip's end `gap_ms` before the window end.

    create_fixed_size_clip pads with zeros, but augment_clips then mixes background
    audio across the whole window, so at training time the padding was never digital
    silence - and pure zeros are a pathological input to the melspectrogram. Hence
    the noise floor.
    """
    if noise_floor > 0:
        window = rng.normal(0, noise_floor, total_length).astype(np.int16)
    else:
        window = np.zeros(total_length, dtype=np.int16)

    end = total_length - int(SR * gap_ms / 1000)
    start = end - len(clip)
    if start < 0:                     # clip too long to fit at this gap
        return None
    window[start:end] = clip
    return window


class WakeWordModel:
    """One interface over an .onnx or .tflite wake-word model.

    Scoring the tflite matters because it is what actually ships. The ONNX and the
    tflite are not guaranteed to agree - a wrong-axis conversion loads cleanly,
    reports a plausible input shape and returns plausible 0-1 scores while detecting
    nothing (see src/train/oww/onnx2tflite.py). Measuring what you deploy removes that
    whole class of surprise, and skips a conversion step when iterating.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.kind = "tflite" if self.path.suffix == ".tflite" else "onnx"

        if self.kind == "tflite":
            # Same import openwakeword uses (model.py:114), so a model that loads
            # here loads there.
            from ai_edge_litert.interpreter import Interpreter
            self._interp = Interpreter(model_path=str(self.path))
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]
            self._out = self._interp.get_output_details()[0]
            self.shape = [int(d) for d in self._in["shape"]]
        else:
            self._session = ort.InferenceSession(
                str(self.path), providers=["CPUExecutionProvider"])
            self._name = self._session.get_inputs()[0].name
            self.shape = self._session.get_inputs()[0].shape

    @property
    def batched(self):
        """Whether the model accepts more than one example at a time."""
        first = self.shape[0]
        return not isinstance(first, int) or first != 1

    def predict(self, embeddings):
        """Score a stack of (frames, dim) embeddings, returning one value each."""
        if self.kind == "tflite":
            scores = []
            for e in embeddings:
                self._interp.set_tensor(self._in["index"],
                                        e[None, :, :].astype(np.float32))
                self._interp.invoke()
                scores.append(float(self._interp.get_tensor(self._out["index"]).item()))
            return np.array(scores)

        if self.batched:
            return self._session.run(
                None, {self._name: embeddings.astype(np.float32)})[0].reshape(-1)
        # Exported with a fixed batch dimension of 1: one at a time.
        return np.array([
            self._session.run(None, {self._name: e[None, :, :].astype(np.float32)})[0].item()
            for e in embeddings])


def score_windows(windows, features, model):
    embeddings = features.embed_clips(np.stack(windows), batch_size=64)
    return model.predict(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Measure the window alignment a trained model prefers",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Trained .onnx or .tflite model. Prefer the .tflite "
                             "if that is what you deploy - the two are not "
                             "guaranteed to agree.")
    parser.add_argument("--positives", nargs="+", default=None,
                        help="Directories of positive clips, searched recursively "
                             "(default: the held-out speaker directories under "
                             "data/recordings/holdout/). Point it at "
                             "data/recordings/samples/ to ask the same question of "
                             "the clips the model trained on.")
    parser.add_argument("--total-length", type=int, default=32000,
                        help="Window size in samples (default: %(default)s, what "
                             "OpenWakeWord derives for a short wake phrase)")
    parser.add_argument("--max-gap", type=float, default=840, help="Largest gap to test, ms")
    parser.add_argument("--step", type=float, default=40,
                        help="Gap step in ms (default: %(default)s, one embedding frame)")
    parser.add_argument("--noise-floor", type=float, default=30.0,
                        help="Std dev of the padding noise in 16-bit counts; 0 for "
                             "digital silence (default: %(default)s)")
    parser.add_argument("--no-trim", action="store_true",
                        help="Score clips as they are on disk, matching train.py --no-trim")
    parser.add_argument("--limit", type=int, help="Only use the first N clips")
    args = parser.parse_args()

    # Imported here so --help works without openwakeword installed.
    from openwakeword.utils import AudioFeatures

    model = WakeWordModel(args.model)
    model_input = model.shape
    features = AudioFeatures(device="cpu")

    expected = features.get_embedding_shape(args.total_length / SR)[0]
    if isinstance(model_input[1], int) and model_input[1] != expected:
        print(f"ERROR: model expects {model_input[1]} embedding frames but a "
              f"{args.total_length}-sample window produces {expected}.")
        print("       Pass --total-length to match the value training used.")
        sys.exit(1)

    positive_dirs = args.positives or [str(d) for d in paths.holdout_dirs(runon=False)]
    paths.warn_if_trained_on(positive_dirs)

    clips = load_clips(positive_dirs, trim=not args.no_trim, limit=args.limit)
    if not clips:
        print(f"No WAV files found in {paths.describe(positive_dirs)}")
        sys.exit(1)
    lengths = np.array([len(c) / SR * 1000 for _, c in clips])

    print("=" * 66)
    print(f"{Path(args.model).name}   {model.kind}, input {model_input}")
    print(f"{len(clips)} clips from {paths.describe(positive_dirs)}"
          f"{'' if args.no_trim else ' (trimmed as train.py does)'}")
    print(f"Window: {args.total_length} samples ({args.total_length / SR:.2f}s), "
          f"clip length median {np.median(lengths):.0f}ms")
    print("=" * 66)
    print(f"  {'gap':>6}  {'median':>7}  {'fired':>6}  {'n':>4}")

    rng = np.random.default_rng(0)
    results = {}
    gap = 0.0
    while gap <= args.max_gap:
        windows = [w for w in (place(c, gap, args.total_length, args.noise_floor, rng)
                               for _, c in clips) if w is not None]
        if windows:
            preds = score_windows(windows, features, model)
            median, fired = float(np.median(preds)), float((preds >= 0.5).mean())
            results[gap] = median
            print(f"  {gap:>4.0f}ms  {median:>7.3f}  {fired:>5.0%}  {len(windows):>4}"
                  f"  {'#' * int(median * 40)}")
        gap += args.step

    if not results:
        print("\nEvery clip is longer than the window - nothing could be placed.")
        sys.exit(1)

    peak = max(results, key=results.get)
    usable = [g for g, m in results.items() if m >= 0.5]
    print()

    # Without a gap where the model actually fires there is no alignment to report -
    # the argmax of a flat line near zero is noise. Usually means these are not
    # positives for this model, or the model is broken.
    if not usable:
        print(f"The model never reaches 0.5 at any alignment (best {results[peak]:.3f} "
              f"at {peak:.0f}ms).")
        print("Nothing to conclude about alignment. Are these clips positives for this model?")
        return

    print(f"Peak at {peak:.0f}ms (median {results[peak]:.3f}), "
          f"fires from {min(usable):.0f}ms to {max(usable):.0f}ms.")
    print(f"The lower edge is the latency floor: the model cannot fire until "
          f"{min(usable):.0f}ms of")
    print("audio has arrived after you stop speaking.")

    # A model trained on trimmed clips saw the phrase in [PAD_MS, PAD_MS + jitter].
    upper = PAD_MS + END_JITTER_MS
    if peak <= upper:
        print(f"OK: peak is within the {PAD_MS:.0f}-{upper:.0f}ms range trimmed clips "
              "are placed in.")
    else:
        print(f"Peak sits {peak - upper:.0f}ms beyond the {upper:.0f}ms that trimmed clips")
        print("can reach, so the training clips carried trailing silence into the window.")
        print("Check check_alignment.py on the positive_train directory, and whether the")
        print("model predates trimming or was trained with --no-trim.")


if __name__ == "__main__":
    main()
