#!/usr/bin/env python3
"""One streaming interface over the wake-word models this repo produces, running
THE INFERENCE CODE THE DEPLOYMENT TARGET RUNS.

The target is Linux Voice Assistant (OHF-Voice/linux-voice-assistant), which does not
implement inference itself - it drives two libraries from `__main__.py`:

    pymicro-wakeword   MicroWakeWordFeatures + MicroWakeWord   (the ESP32/mWW models)
    pyopen-wakeword    OpenWakeWordFeatures  + OpenWakeWord    (the openWakeWord ones)

Both are used here exactly as LVA uses them, down to the chunk size and the order of
calls. 

    backend = load("output/hey_seeree/mww/hey_seeree_705c23b.json")
    scores, offsets = backend.score(pcm_int16)

`offsets[k]` is how many samples of the clip had been fed when `scores[k]` came out -
counted, not derived, by feeding the runtime in its own chunk size and recording the
position each probability appears at. It is the anchor for latency, and it differs
between backends (80 ms chunks for openWakeWord, 10 ms for microWakeWord), which is
why it is returned rather than recomputed by callers from a constant they guessed.

WHAT USING THE DEPLOYMENT RUNTIME SETTLED, that reading the training repo did not:

* THE MANIFEST IS PART OF THE MODEL. `MicroWakeWord.from_config()` takes the JSON,
  not the .tflite, and reads `probability_cutoff` and `sliding_window_size` from it.
  Score the .json and the manifest is under test too; score a bare .tflite and it is
  not.

* RESETTING BETWEEN CLIPS REQUIRES RELOADING THE MODEL. `MicroWakeWord.reset()` calls
  `close()` and `_load_model()`, with the comment "Need to reload model to reset
  intermediary results". Independently measured here before that was read: an
  interpreter that has `reset_all_variables()` called and its inputs re-zeroed still
  scores one clip 1.00000 the first time and 0.99608 every time after. `OpenWakeWord
  .reset()` only clears a ring buffer, because that model is stateless - the state
  lives in Python.

* THE DEPLOYED PROBABILITY IS NOT THE ONE IN `tflite_streaming_roc.txt`. The training
  repo dequantizes with a hardcoded 1/255 (microwakeword/inference.py); the runtime
  uses the output tensor's own scale, 1/256 here. So a cutoff read off the ROC table
  is about 0.4% off the number the device compares against. The ROC file is a guide;
  this harness is the measurement.

* int8 COARSENS THE SCORES. The output tensor is uint8: 256 levels, 0.0039 apart,
  divided by the sliding window average of 5. Sweeping much below 0.01 measures
  quantization rather than the model. `--model` self-check prints the real resolution.

* "THRESHOLD" IS TWO PARAMETERS for microWakeWord: `probability_cutoff` and
  `sliding_window_size`. Only the first is swept here; the second is fixed per
  comparison and printed with every result, because a cutoff means nothing without it.

"""

import argparse
import collections
import sys
import zlib
from pathlib import Path

import numpy as np

SR = 16000

# LVA feeds audio in these units (linux_voice_assistant/__main__.py and the two
# libraries' own constants). Matching them matters: the frontends buffer internally,
# so a different chunk size changes where in the audio each probability lands.
MWW_CHUNK_SAMPLES = 160     # pymicro_wakeword.SAMPLES_PER_CHUNK, 10 ms
OWW_CHUNK_SAMPLES = 1280    # pyopen_wakeword.SAMPLES_PER_CHUNK, 80 ms

# microwakeword/test.py:301 and the manifests this repo emits. Only meaningful
# alongside a cutoff - see the docstring.
MWW_SLIDING_WINDOW = 5

# Placeholder for a bare .tflite scored without its manifest. The harness sweeps the
# cutoff itself and reads `process_streaming_prob`, so the value never gates anything;
# it exists because the constructor requires one.
UNUSED_CUTOFF = 0.5


class Backend:
    """Common surface: reset, stream a clip, get scores and their audio offsets."""

    kind = "?"

    def __init__(self, path):
        self.path = Path(path)
        self.label = self.path.stem

    def score(self, audio):
        """(scores, offsets) for one clip of int16 PCM, from a fully reset model."""
        audio = np.asarray(audio, dtype=np.int16)
        scores, offsets = self._stream(audio)
        return np.asarray(scores, dtype=np.float64), np.asarray(offsets, dtype=np.int64)

    # --- incremental interface ---------------------------------------------------
    #
    # `start` + `feed` are how the runtime is actually driven; `_stream` below is
    # just those two in a loop over a fixed clip. Split out so that LIVE detection
    # (preflight/test_model.py, microphone, no end) and OFFLINE scoring (eval, whole
    # clips) go through one implementation rather than two. That matters more here
    # than it usually would: an earlier version of this file reimplemented the
    # runtime and got it subtly wrong, and a preflight that streams differently from
    # the harness would be measuring a third thing again.

    def start(self):
        """Reset the model and rebuild the feature extractor. Call before feeding."""
        raise NotImplementedError

    def feed(self, pcm_bytes):
        """Probabilities produced by one `chunk_samples` chunk of int16 PCM bytes.

        Returns a list because a chunk yields zero, one or several probabilities
        depending on how the frontend buffers - never assume one in, one out.
        """
        raise NotImplementedError

    def _stream(self, audio):
        """Offline scoring: start once, then feed the clip a chunk at a time."""
        self.start()
        scores, offsets, consumed = [], [], 0
        raw = audio.tobytes()
        step = self.chunk_samples * 2
        for start in range(0, len(raw) - step + 1, step):
            consumed += self.chunk_samples
            for prob in self.feed(raw[start:start + step]):
                scores.append(prob)
                offsets.append(consumed)
        return scores, offsets

    def describe(self):
        raise NotImplementedError


class MicroWakeWordBackend(Backend):
    """microWakeWord through pymicro-wakeword, driven as LVA drives it.

    Takes the manifest .json where there is one - that is what the runtime loads, and
    it puts `sliding_window_size` under test with the model. A bare .tflite is also
    accepted, with the window supplied here.
    """

    kind = "mww"
    chunk_samples = MWW_CHUNK_SAMPLES

    def __init__(self, path, sliding_window_size=None):
        super().__init__(path)
        from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures

        self._Features = MicroWakeWordFeatures
        if self.path.suffix == ".json":
            self.model = MicroWakeWord.from_config(self.path)
            self.from_manifest = True
            if sliding_window_size is not None:
                # Overriding the manifest means the deque behind it has to move too -
                # its maxlen IS the window, and MicroWakeWord sizes it at construction.
                self.model.sliding_window_size = int(sliding_window_size)
                self.model._probabilities = collections.deque(
                    maxlen=int(sliding_window_size))
        else:
            self.model = MicroWakeWord(
                id=self.path.stem, wake_word=self.path.stem,
                tflite_model=self.path, probability_cutoff=UNUSED_CUTOFF,
                sliding_window_size=int(sliding_window_size or MWW_SLIDING_WINDOW),
                trained_languages=[], libtensorflowlite_c_path=_tflite_c_path())
            self.from_manifest = False
        self.label = self.path.stem
        self.sliding_window_size = self.model.sliding_window_size

    def start(self):
        # reset() reloads the model - see the module docstring. Without it a clip's
        # score depends on the clip before it, and nothing about the result looks wrong.
        self.model.reset()
        self._features = self._Features()

    def feed(self, pcm_bytes):
        out = []
        for window in self._features.process_streaming(pcm_bytes):
            # The probability the runtime compares against probability_cutoff:
            # the model's own sliding-window mean, already dequantized.
            prob = self.model.process_streaming_prob(window)
            if prob is not None:
                out.append(prob)
        return out

    def describe(self):
        source = "manifest" if self.from_manifest else "bare tflite"
        return (f"microWakeWord via pymicro-wakeword (LVA deployment runtime, "
                f"{source}), sliding_window_size {self.sliding_window_size}")


class OpenWakeWordBackend(Backend):
    """openWakeWord through pyopen-wakeword, driven as LVA drives it.

    TFLite only - the library is a ctypes wrapper around libtensorflowlite_c and has
    no ONNX path. The melspectrogram and embedding models are the ones bundled with
    the package, which is what the device would use.
    """

    kind = "oww"
    chunk_samples = OWW_CHUNK_SAMPLES

    def __init__(self, path):
        super().__init__(path)
        from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures

        self._Features = OpenWakeWordFeatures
        self.model = OpenWakeWord.from_model(self.path)

    def start(self):
        # Stateless model; reset() clears the embedding ring buffer, which is all the
        # state there is. The feature extractor is rebuilt per clip for the same
        # reason it is for mWW: it buffers audio across calls.
        self.model.reset()
        self._features = self._Features.from_builtin()

    def feed(self, pcm_bytes):
        out = []
        for embedding in self._features.process_streaming(pcm_bytes):
            out.extend(self.model.process_streaming(embedding))
        return out

    def describe(self):
        return "openWakeWord via pyopen-wakeword (LVA deployment runtime), tflite"


class OpenWakeWordOnnxBackend(Backend):
    """openWakeWord through the TRAINING repo's own inference, for .onnx models.

    NOT THE DEPLOYMENT RUNTIME. It exists because this repo's ship candidates are
    .onnx and pyopen-wakeword cannot read them, and because all seventeen runs in
    the tuning log were measured on exactly this path - so a number from here is
    comparable with the notebook and a number from the LVA backend is not.
    """

    kind = "oww-onnx"
    chunk_samples = OWW_CHUNK_SAMPLES

    def __init__(self, path):
        super().__init__(path)
        from openwakeword.model import Model

        framework = "tflite" if self.path.suffix == ".tflite" else "onnx"
        self.model = Model(wakeword_models=[str(self.path)], inference_framework=framework)
        self.framework = framework
        self.name = self.path.stem   # Model keys its predictions by the file stem

    def _stream(self, audio):
        self.model.reset()
        # padding=0 bypasses openWakeWord's own zero padding: digital silence is a
        # pathological input to the melspectrogram, and callers pad with room tone.
        frames = self.model.predict_clip(audio, padding=0)
        scores = [f[self.name] for f in frames]
        offsets = (np.arange(len(scores)) + 1) * self.chunk_samples
        return scores, offsets

    # predict_clip() wants a whole clip, so this backend has no incremental form.
    # No loss for preflight: a live check exists to exercise what the device runs,
    # and this path deliberately is not that. Convert with train/oww/onnx2tflite.py
    # and preflight the .tflite instead.
    def start(self):
        raise NotImplementedError(
            "the .onnx path scores whole clips and cannot stream live. It is also "
            "not the deployment runtime - convert with train/oww/onnx2tflite.py "
            ".tflite, which is what the device would run.")

    def feed(self, pcm_bytes):
        raise NotImplementedError(self.start.__doc__)

    def describe(self):
        return (f"openWakeWord via openwakeword.model.Model ({self.framework}) - "
                f"the tuning-run path, NOT the deployment runtime")


def _tflite_c_path():
    """The libtensorflowlite_c that pymicro-wakeword ships for this platform."""
    import pymicro_wakeword

    lib_dir = Path(pymicro_wakeword.__file__).parent / "lib"
    lib = next(iter(lib_dir.glob("*tensorflowlite_c.*")), None)
    if lib is None:
        raise SystemExit(f"no libtensorflowlite_c in {lib_dir}")
    return lib


def is_microwakeword(path):
    """True for a microWakeWord model, decided by the artifact rather than the name.

    Both trainers emit `.tflite`, so the extension settles nothing. A mWW streaming
    model takes (1, stride, 40) int8 features; an openWakeWord model takes a stack of
    96-dim embeddings. A `.json` manifest says which it is outright.
    """
    path = Path(path)
    if path.suffix == ".json":
        import json

        return json.loads(path.read_text()).get("type") == "micro"
    if path.suffix != ".tflite":
        return False

    from pymicro_wakeword.wakeword import TfLiteWakeWord

    probe = TfLiteWakeWord(_tflite_c_path())
    model = probe.lib.TfLiteModelCreateFromFile(str(path).encode("utf-8"))
    interp = probe.lib.TfLiteInterpreterCreate(model, None)
    probe.lib.TfLiteInterpreterAllocateTensors(interp)
    tensor = probe.lib.TfLiteInterpreterGetInputTensor(interp, 0)
    dims = probe.lib.TfLiteTensorNumDims(tensor)
    shape = [probe.lib.TfLiteTensorDim(tensor, i) for i in range(dims)]
    probe.lib.TfLiteInterpreterDelete(interp)
    probe.lib.TfLiteModelDelete(model)
    return len(shape) == 3 and shape[2] == 40


def load(path, sliding_window_size=None):
    """The right backend for a model, chosen by inspecting it.

    .json               microWakeWord, manifest and all - the deployment case
    .tflite (1,s,40)    microWakeWord without its manifest
    .tflite otherwise   openWakeWord on the deployment runtime
    .onnx               openWakeWord on the tuning-run path, which is not deployment
    """
    path = Path(path)
    if is_microwakeword(path):
        return MicroWakeWordBackend(path, sliding_window_size=sliding_window_size)
    if path.suffix == ".onnx":
        return OpenWakeWordOnnxBackend(path)
    return OpenWakeWordBackend(path)


# --- self-check -----------------------------------------------------------------

def _noise(name, seconds, sr=SR, std=30.0):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return rng.normal(0, std, int(sr * seconds)).astype(np.int16)


def self_check(backend, clips):
    """Order independence, and the score resolution a threshold sweep can rely on.

    ORDER INDEPENDENCE IS THE ONE THAT MATTERS. A streaming model whose state is not
    fully reset scores a clip differently depending on what preceded it, and nothing
    about the output looks wrong - the corpus order just quietly becomes a variable.
    `clip_rng` in eval_model.py exists because of an almost identical bug in the
    padding noise.

    FULL TRACES, NOT PEAKS. The leak this was written to catch moved fine scores by
    1/255 and left most peaks at exactly 1.0, so a peak comparison passed it.
    """
    forward = [backend.score(d)[0] for _, d in clips]
    backward = [backend.score(d)[0] for _, d in reversed(clips)][::-1]
    ok = all(np.array_equal(f, b) for f, b in zip(forward, backward))
    print(f"  order independence   {'PASS' if ok else 'FAIL'}  "
          f"({len(clips)} clips scored forwards and backwards, full traces compared)")
    if not ok:
        worst = max(np.abs(f - b).max() for f, b in zip(forward, backward))
        differing = sum(int((f != b).sum()) for f, b in zip(forward, backward))
        print(f"    {differing} scores differ, largest disagreement {worst:.6f} - "
              f"state is leaking between clips")

    every = np.concatenate(forward)
    distinct = np.unique(every)
    gap = float(np.diff(distinct).min()) if len(distinct) > 1 else float("nan")
    print(f"  score resolution     {len(distinct)} distinct values over {len(every)} "
          f"scores; smallest gap {gap:.5f}")
    print("    A threshold sweep finer than that gap measures quantization, "
          "not the model.")
    return ok


def main():
    p = argparse.ArgumentParser(
        description="Inspect and self-check an eval backend",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="manifest .json (preferred for microWakeWord), .tflite or .onnx")
    p.add_argument("--clips", default=None,
                   help="Directory used for the self-check (default: the first "
                        "held-out speaker directory under data/recordings/holdout/)")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--sliding-window-size", type=int, default=None)
    args = p.parse_args()

    import scipy.io.wavfile

    from eval import paths

    # One speaker is enough: this is a self-check on the backend, not a measurement
    # of the model, so it wants a handful of real clips rather than the whole holdout.
    if args.clips is None:
        held_out = paths.holdout_dirs(runon=False)
        if not held_out:
            sys.exit(f"no held-out recordings under {paths.HOLDOUT_DIR}; record some "
                     f"with `record_samples.py --holdout --speaker NAME`, or pass "
                     f"--clips")
        args.clips = str(held_out[0])

    backend = load(args.model, sliding_window_size=args.sliding_window_size)
    print(f"{Path(args.model).name}")
    print(f"  backend  {backend.kind}: {backend.describe()}")

    clips = []
    for wav in sorted(Path(args.clips).rglob("*.wav"))[:args.limit]:
        sr, data = scipy.io.wavfile.read(wav)
        if data.ndim > 1:
            data = data[:, 0]
        if sr == SR:
            # Padded as the harness pads: the model needs context before the phrase,
            # and an unpadded clip measures a cold model rather than a live one.
            clips.append((wav.name, np.concatenate([
                _noise(wav.name + "lead", 1.0), data.astype(np.int16),
                _noise(wav.name + "tail", 1.0)])))
    if not clips:
        sys.exit(f"no 16 kHz WAVs in {args.clips}")

    print(f"\nself-check on {len(clips)} clips from {args.clips}")
    sys.exit(0 if self_check(backend, clips) else 1)


if __name__ == "__main__":
    main()
