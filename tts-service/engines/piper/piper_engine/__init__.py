"""In-process Piper protocol engine: the Mac route.

Loads piper-tts models directly and serves them on the repo's TTS protocol
port - no Wyoming process at all:

    uv run --project tts-service/engines/piper python -m piper_engine \
        --voices data/external/piper/voices --port 8898

`PiperVoice.synthesize` returning an AudioChunk stream is the exact call
wyoming-piper wraps, so this is the same code the Wyoming route ran: same
models, same G2P (piper-phonemize), same 22050 Hz output, and the same
post-processing tail (16 kHz polyphase resample, then WSOLA for `speed`).
Piper is stochastic per call either way - VITS samples noise and durations per
synthesis, so the same (voice, speaker, text, speed) renders differently every
time (measured in audit_voices.py, where one speaker scored 0% then 100% across
passes). For a corpus that is free diversity, and it is why the corpus is not
reproducible from a seed - which is already true of the Kokoro path too. So the
calibrated property is the DISTRIBUTION of renderings, not the bytes. Measured on
the host (2026-09): warm synthesize 28-39 ms/clip, model load ~0.6 s, switching
speakers within one multi-speaker model ~28 ms (the session is reused; only a
different model file costs a reload).

PIN: piper-tts==1.7.0 with onnxruntime==1.29.0 - the same pair the Wyoming
host venv (scripts/start-piper-host.sh) and the Docker image install, because
1.8.0's G2P path differs and a mixed fleet would be a distribution shift, not
a version bump.

ONE MODEL AT A TIME, like the Wyoming server it replaces: that server held a
single module-global voice because 84 models do not fit in memory; this
engine keeps the same discipline - it holds only the current model and
reloads on a switch (0.6 s against the hundreds of clips a voice generates).
The corpus generator iterates voice-outer, so a whole voice's clips come off
one load. That is why batch_mode="serial" and why voice stays the outer loop
in generate_piper_samples - order is a correctness property, not a style
choice.
"""
import json
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig
from scipy.signal import resample_poly

from tts_protocol.audio import SR, time_stretch
from tts_protocol.engine import Engine


def _config_json(voices_dir, name):
    """The .onnx.json config that ships next to every model file."""
    path = Path(voices_dir) / f"{name}.onnx.json"
    if not path.exists():
        raise RuntimeError(f"model {name} in {voices_dir} has no config json")
    return json.loads(path.read_text())


def piper_voices(voices_dir, languages=("en_US", "en_GB"), max_speakers=0):
    """Enumerate (voice, speaker) pairs from a Piper voices directory.

    The layout is what download-external-data.sh writes:
    `en_US-lessac-medium.onnx` with a same-stem `.onnx.json` config. A
    single-speaker model contributes (name, None); a multi-speaker model
    (num_speakers > 1) contributes (name, speaker_name) per entry of its
    speaker_id_map, in config order, capped at `max_speakers` per model
    (0 = all). `languages` filters on the config's language.code, with the
    filename prefix as the fallback.
    """
    voices_dir = Path(voices_dir)
    out = []
    for onnx in sorted(voices_dir.glob("*.onnx")):
        cfg = _config_json(voices_dir, onnx.stem)
        lang = (cfg.get("language") or {}).get("code") or onnx.name.split("-")[0]
        if languages is not None and lang not in languages:
            continue
        name = onnx.stem
        if int(cfg.get("num_speakers", 1)) > 1:
            speakers = list(cfg.get("speaker_id_map", {}).items())
            if max_speakers:
                speakers = speakers[:max_speakers]
            out.extend((name, sp) for sp, _ in speakers)
        else:
            out.append((name, None))
    return out


# One loaded model at a time (module-global, like the wyoming server it
# replaces): a reload is 0.6 s and the generator is voice-outer, so a whole
# voice renders off one load. Holding all 84 would be ~8 GB for nothing.
_MODEL = None
_MODEL_NAME = None
_MODEL_LOCK = threading.Lock()


def _get_voice(voices_dir, name):
    global _MODEL, _MODEL_NAME
    with _MODEL_LOCK:
        if _MODEL_NAME == name:
            return _MODEL
        path = Path(voices_dir) / f"{name}.onnx"
        if not path.exists():
            raise RuntimeError(f"no piper model {path}")
        _MODEL = PiperVoice.load(str(path))
        _MODEL_NAME = name
        return _MODEL


def piper_render(voices_dir, voice, speaker, text, speed=1.0):
    """Synthesize one phrase, returned as 16 kHz mono int16.

    `speed` > 1 is faster, applied with time_stretch AFTER the resample -
    the same tail the Wyoming adapter ran, in the same order (WSOLA on the
    16 kHz data, which the implementation requires).

    Raises on failure (the caller reports which voice failed) - unlike the
    kokoro engines, which return None on a transient miss.
    """
    voice_obj = _get_voice(voices_dir, voice)
    syn_config = None
    if speaker is not None:
        sid = voice_obj.config.speaker_id_map.get(str(speaker))
        if sid is None:
            raise RuntimeError(f"voice {voice} has no speaker {speaker!r}")
        # speaker_id set, every other field None: phoneme_ids_to_audio falls
        # back to the voice config for length/noise scales, so this changes
        # the speaker and nothing else.
        syn_config = SynthesisConfig(speaker_id=sid)

    pcm = b"".join(c.audio_int16_bytes for c in voice_obj.synthesize(text, syn_config))
    if not pcm:
        return np.zeros(0, dtype=np.int16)

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    rate = int(voice_obj.config.sample_rate)
    if rate != SR:
        # resample_poly rather than resample: rational up/down, no FFT-length
        # sensitivity, and it is what vocal_tract_shift already uses.
        frac = Fraction(SR, rate).limit_denominator(1000)
        audio = resample_poly(audio, frac.numerator, frac.denominator)

    if abs(speed - 1.0) > 1e-3:
        # speed 1.6 = 1.6x faster = 1/1.6 the duration.
        audio = time_stretch(audio, 1.0 / float(speed), sr=SR)

    return np.clip(audio, -32768, 32767).astype(np.int16)


class PiperEngine(Engine):
    """In-process Piper as a registered engine: name is "piper"."""

    name = "piper"
    supports_timestamps = False
    batch_mode = "serial"
    speaker_voices = True

    def __init__(self, voices_dir):
        self.voices_dir = str(voices_dir)

    def available(self):
        d = Path(self.voices_dir)
        models = list(d.glob("*.onnx")) if d.is_dir() else []
        if not models:
            return False, f"no piper models in {d}"
        return True, f"{len(models)} models in {d}"

    def voices(self, **kwargs):
        return piper_voices(self.voices_dir, **kwargs)

    def timed_render(self, voice, text: str, speed: float = 1.0):
        if isinstance(voice, (list, tuple)):
            voice, speaker = voice[0], voice[1] if len(voice) > 1 else None
        else:
            speaker = None
        audio = piper_render(self.voices_dir, voice, speaker, text, speed)
        return audio, None
