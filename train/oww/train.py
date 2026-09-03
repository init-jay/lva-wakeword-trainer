#!/usr/bin/env python3
"""
Train OpenWakeWord model using Kokoro TTS synthetic voices + real recordings.

Usage:
    python train.py --wake-word "hey seeree"
    python train.py --wake-word "okay jarvis" --samples-per-voice 300 --training-steps 75000

Docker:
    docker compose run --rm trainer python train.py --wake-word "hey seeree" --data-dir /app/data
"""

import argparse
import base64
import concurrent.futures as cf
import io
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import requests
import scipy.io.wavfile
import yaml
from tqdm import tqdm

# THE REPO ROOT, NOT THIS FILE'S DIRECTORY. Every path in this module is relative -
# data/ and output/ - and the chdir below is what anchors them. While this file lived
# at the repo root the two were the same thing; since it moved to train/oww/ they are
# three levels apart, and getting this wrong does not raise. It builds a corpus under
# train/oww/ and trains on nothing.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Importable when run as `python train/oww/train.py` as well as `python -m
# train.oww.train`; the plain-path form puts train/oww/ on sys.path, not the root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The engine-agnostic half of corpus construction, shared with the microWakeWord
# trainer (see ARCHITECTURE.md). Moved out of this file without behaviour change; the
# reasoning that used to live here moved with it.
from train.corpus.augment import (CHILD_STRETCH, CHILD_STRETCH_FRACTION,  # noqa: E402
                                  add_child_range_copies, trim_directory,
                                  trim_silence)
from train.corpus.negatives import (MISPRONOUNCING_VOICES,  # noqa: E402
                                    TRAINING_COMMANDS, build_negative_phrases)
from train.corpus.piper import (generate_piper_samples,  # noqa: E402
                                select_piper_voices)
from train.corpus.positives import (PLAIN_SPEED_GRID, PLAIN_SPEEDS,  # noqa: E402
                                    plain_positive_texts)
from train.corpus.real import copy_real_samples  # noqa: E402

warnings.filterwarnings("ignore", message="Reached EOF prematurely")

WORK_DIR = REPO_ROOT
os.chdir(WORK_DIR)

# BASE_NEGATIVES, TRAINING_COMMANDS, MISPRONOUNCING_VOICES and
# CONFUSABLE_NEGATIVES now live in corpus/negatives.py, imported above.

# Speed coverage of the positives, widened at the top for run 9.
#
# The measured failure: a synthetic sweep of the run 4 model detected 6/6 up to
# 1.25x and then fell off a cliff - 3/6 at 1.40x, 2/6 at 1.60x. Training rendered
# nothing above 1.3x, so the model fails just outside the range it was shown, and
# is fine below it (0.55x still gave 6/6). The asymmetry says widen the top only.
#
# It matches a real failure too: four of the five held-out clips run 4 missed were
# the fast ones, and the shortest (300 ms) was shorter than every clip it detected.
# Kokoro at 1.6x renders "hey seeree" in 390-590 ms depending on voice, which is
# exactly that range - checked for intelligibility first, since training on
# degraded audio would be worse than not covering the speed at all.
#
# Both lists move together: they are one variable, "how fast can the phrase be".
# Covering fast phrase-alone renderings while leaving run-ons at 1.2x would leave
# fast run-on speech - the commonest real case - still untrained.
#
# RUNON_SPEEDS stays discrete and five long so the fallback path can keep caching
# its phrase-alone reference per (voice, speed).
RUNON_SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6]
# PLAIN_SPEEDS, PLAIN_SPEED_STEP and PLAIN_SPEED_GRID now live in
# corpus/positives.py, imported above - both trainers use the same grid.

# How much of the command's onset to keep after the wake word ends, in ms.
#
# The value that matters is where the phrase ends relative to the END OF THE ARRAY,
# because create_fixed_size_clip aligns that with the window. Plain positives sit at
# ~80 ms (30 ms trim pad + ~50 ms residual). Run-on positives must match, or the
# positive set is bimodal and the model learns the later mode.
#
# The boundary itself now comes from Kokoro's /dev/captioned_speech word timestamps,
# so this is the whole overshoot rather than a jitter added to an estimate. Two
# earlier attempts inferred the boundary from a phrase-alone rendering instead:
#
#   v1, cut at phrase_len + U(50,250): kept 270-470 ms of command. The alignment
#      peak moved 160 -> 480 ms, median latency 70 -> 130 ms, and extend false
#      accepts 4/32 -> 7/32, because a trailing region holding speech in BOTH
#      classes stops discriminating and the model learns to ignore it.
#   v2, correcting for the 30 ms trim pad: still a median +153 ms late, and
#      voice-dependent (af_bella ~0 ms, bf_lily +348..+459 ms). Worse, 2 of 18
#      sampled clips cut slightly INSIDE the wake word, removing the coarticulated
#      ending that is the entire reason for generating these clips.
#
# The timestamps remove both the bias and the variance. The fallback path still uses
# the v2 estimate, which is why it reports itself loudly.
#
# THE RANGE MUST NOT START AT ZERO, and how far above zero is the open question.
#
# The margin is not padding. It is what lets the model hear that the word ENDED
# rather than continued, which is the entire discrimination between "hey seeree"
# and "hey serious". Measured against held-out real recordings:
#
#   effective margin   held-out run-on   extend+hey_other FA   latency
#     ~50 ms (run 5)         28%              12/32             -20 ms
#    ~140 ms (run 6)         40%               8/32              48 ms
#    ~200 ms (run 4)         46%               6/32              77 ms
#    ~225 ms (run 7)         56%               7/32              83 ms
#
# Runs 6 and 7 share an identical real-sample corpus and differ only in this
# constant, so that pair is clean: +85 ms of margin bought +16 points of real
# run-on detection. That is the relationship this value exists to exploit.
#
# The false-accept column is NOT a gradient. It plateaus at 7-8/32 across a 3x
# range of margin; the early monotonic reading was mostly the margin escaping the
# pathological zero case, and run 4 also had half the real data. Do not raise this
# expecting fewer false accepts.
#
# The cost is latency, which tracks margin closely and is at 83 ms against a 120 ms
# gate. There is roughly one more step of headroom, for diminishing returns.
RUNON_TAIL_MS = (150.0, 300.0)

# CHILD_STRETCH and CHILD_STRETCH_FRACTION now live in corpus/augment.py,
# imported above, alongside the transforms that use them.


def report_onnx_providers():
    """Say plainly whether feature computation will run on the GPU.

    Worth doing because the failure is silent and expensive. onnxruntime falls back
    to CPU with a warning rather than erroring when the CUDA provider cannot load,
    and openwakeword picks its thread count from torch rather than onnxruntime - so
    a box with a working GPU and the CPU build of onnxruntime computes features
    single-threaded on CPU. That cost 36 minutes of an 83-minute run before anyone
    noticed the warning in the scrollback.

    Checking `get_available_providers()` alone is not enough: it reports what the
    build supports, not what will load. So open a real session against the model
    that will actually be used, and report what it came back with.
    """
    try:
        import onnxruntime
    except ImportError:
        print("  onnxruntime not importable - feature computation will fail")
        return

    available = onnxruntime.get_available_providers()
    melspec = WORK_DIR / "openwakeword/openwakeword/resources/models/melspectrogram.onnx"

    actual = "CPUExecutionProvider"
    if "CUDAExecutionProvider" in available and melspec.exists():
        try:
            session = onnxruntime.InferenceSession(
                str(melspec), providers=["CUDAExecutionProvider"])
            actual = session.get_providers()[0]
        except Exception as e:
            print(f"  Could not open a CUDA session: {e}")

    print(f"  onnxruntime {onnxruntime.__version__}, using {actual}")
    if actual != "CUDAExecutionProvider":
        print("  WARNING: features will be computed on CPU. This is the slowest stage")
        print("           of the run. Install onnxruntime-gpu (>=1.19 for CUDA 12) and")
        print("           make sure cuDNN is on the library path.")


class KokoroPool:
    """Hands out TTS server URLs round-robin, so several servers share the work.

    Worth the trouble because one Kokoro process is single-threaded: measured at
    101.8% CPU (exactly one core) while the GPU sat at 21% and VRAM at 1.5 of 24 GB.
    Extra client threads against one server therefore just queue - four workers
    against one server measured 15.0 it/s versus 14.1 sequential. Extra *processes*
    are what scale, until the cores run out.

    Round-robin per job rather than a static split by index, so a thread that draws
    a slow server holds it longer and completes fewer jobs. That balancing is weak,
    though - it does not make a slow server free.

    MEASURED: with batching on, adding two remote servers to the two local ones made
    the run SLOWER, so the local pair alone is the better configuration here.
    Batching amortises latency, not bandwidth: it pays the ~119 ms fixed cost once
    per batch instead of once per clip, but the bytes returned are unchanged, and a
    batch of 16 sends back ~640 KB instead of ~40 KB. The local containers talk over
    the Docker bridge with no physical network; a remote server over a
    bandwidth-limited link gains nothing from batching and then takes an equal share
    of the jobs. Measure before adding servers rather than assuming they help.
    """

    def __init__(self, urls):
        self.urls = [u.strip().rstrip("/") for u in urls if u.strip()]
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        if len(self.urls) == 1:
            return self.urls[0]
        with self._lock:
            url = self.urls[self._i % len(self.urls)]
            self._i += 1
            return url

    def __len__(self):
        return len(self.urls)


def probe_kokoro_servers(pool: KokoroPool) -> list:
    """Report each server, drop the dead ones, and return the voices they all share.

    Servers are only interchangeable if they agree on the voice list - asking a
    server for a voice it lacks fails that job - so the intersection is used.
    Timestamp support is probed too: a server without /dev/captioned_speech pushes
    run-on clips onto the degraded phrase-alone estimate, which is the exact bug
    that cost two training runs, and it would otherwise do so silently.
    """
    voices_per_server = []
    for url in pool.urls:
        try:
            r = requests.get(f"{url}/v1/audio/voices", timeout=10)
            voices = r.json().get("voices", [])
            voices = [v["id"] if isinstance(v, dict) else v for v in voices]
            english = {v for v in voices if v.startswith(('af_', 'am_', 'bf_', 'bm_'))}
        except Exception as e:
            print(f"  {url}: UNREACHABLE ({e})")
            voices_per_server.append(set())
            continue

        _, timestamps = kokoro_tts_timed(url, sorted(english)[0], "test", 1.0)
        timed = "timestamps" if timestamps else "NO timestamps (run-ons degraded)"
        print(f"  {url}: {len(english)} English voices, {timed}")
        voices_per_server.append(english)

    live = [(u, v) for u, v in zip(pool.urls, voices_per_server) if v]
    if not live:
        print("ERROR: no usable Kokoro servers")
        sys.exit(1)

    pool.urls = [u for u, _ in live]
    shared = set.intersection(*[v for _, v in live])
    union = set.union(*[v for _, v in live])
    if shared != union:
        print(f"  NOTE: servers disagree on {len(union - shared)} voice(s); "
              f"using the {len(shared)} they share")
    return sorted(shared)


def get_kokoro_voices(kokoro_url: str) -> list:
    """Get all available English voices from Kokoro."""
    try:
        r = requests.get(f"{kokoro_url}/v1/audio/voices", timeout=5)
        voices = r.json().get("voices", [])
        # Filter to English voices (a = American, b = British)
        voices = [v["id"] if isinstance(v, dict) else v for v in voices]
        english = [v for v in voices if v.startswith(('af_', 'am_', 'bf_', 'bm_'))]
        print(f"Kokoro voices available: {len(english)}")
        return english
    except Exception as e:
        print(f"ERROR: Cannot connect to Kokoro at {kokoro_url}: {e}")
        print("Make sure Kokoro is running:")
        print("  docker compose up -d kokoro kokoro2")
        print("  # on the training server, for the GPU image:")
        print("  docker compose -f docker-compose.yml -f docker-compose.gpu.yml \\")
        print("      up -d kokoro kokoro2")
        sys.exit(1)


def kokoro_tts(kokoro_url: str, voice: str, text: str, speed: float):
    """Render one utterance as 16 kHz int16 audio, or None on failure."""
    try:
        r = requests.post(
            f"{kokoro_url}/v1/audio/speech",
            json={
                "model": "kokoro",
                "voice": voice,
                "input": text,
                "response_format": "wav",
                "speed": speed
            },
            timeout=30
        )
        if r.status_code != 200:
            return None

        sr, data = scipy.io.wavfile.read(io.BytesIO(r.content))
        if data.ndim > 1:
            data = data[:, 0]

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy.signal import resample
            num_samples = int(len(data) * 16000 / sr)
            data = resample(data, num_samples)

        return np.clip(data, -32768, 32767).astype(np.int16)
    except Exception:
        return None


def kokoro_tts_timed(kokoro_url: str, voice: str, text: str, speed: float):
    """Render an utterance and return (16 kHz int16 audio, word timestamps).

    Kokoro-FastAPI's /dev/captioned_speech returns per-word start/end times
    alongside the audio, which is what makes an exact cut possible for the run-on
    positives: it says precisely where the wake word ends inside the utterance,
    instead of that having to be inferred from a separate phrase-alone rendering.

    Returns (None, None) if the endpoint is unavailable, so callers can fall back.
    """
    try:
        r = requests.post(
            f"{kokoro_url}/dev/captioned_speech",
            json={
                "model": "kokoro",
                "voice": voice,
                "input": text,
                "response_format": "wav",
                "speed": speed,
                "stream": False,
                "return_timestamps": True,
            },
            timeout=60
        )
        if r.status_code != 200:
            return None, None

        payload = r.json()
        sr, data = scipy.io.wavfile.read(io.BytesIO(base64.b64decode(payload["audio"])))
        if data.ndim > 1:
            data = data[:, 0]
        if sr != 16000:
            from scipy.signal import resample
            data = resample(data, int(len(data) * 16000 / sr))
        return np.clip(data, -32768, 32767).astype(np.int16), payload.get("timestamps")
    except Exception:
        return None, None


def kokoro_tts_batch(kokoro_url: str, voice: str, texts: list, speed: float):
    """Render several utterances in ONE request and split them apart.

    Measured against a Kokoro-FastAPI server: a single "hey seeree" request costs
    ~119 ms of fixed overhead plus ~42 ms per second of audio, so for a phrase under
    a second, THREE QUARTERS of the request is overhead. Batching amortises it.

    In isolation a batch of 16-32 reaches ~37 ms/clip against 182 individually (5x),
    but real batches are smaller: buckets hold `samples_per_voice / len(grid)` clips,
    about 9.5 for plain positives at the defaults. Measured end to end at that shape
    it is 137 -> 42 ms/clip, a 3.3x speedup. A coarser speed grid would enlarge the
    buckets but measured no faster (39 ms/clip at 0.10 steps), so the finer grid is
    kept for the extra speed diversity.

    The split is exact, not energy-based: /dev/captioned_speech returns per-word
    start and end times, so each utterance is cut at its own word boundaries. The
    texts are joined with ". " and the punctuation tokens are filtered back out of
    the timestamp list.

    Every text in a batch shares one voice and one speed - that is what makes it a
    single forward pass - so callers must group by (voice, speed) before calling.

    Returns a list of (audio, timestamps) in the order given, with None for any
    utterance whose words could not be located. Falls back to nothing: a caller
    seeing None should render that one individually.
    """
    if not texts:
        return []

    joined = ". ".join(t.rstrip(".") for t in texts) + "."
    data, timestamps = kokoro_tts_timed(kokoro_url, voice, joined, speed)
    if data is None or not timestamps:
        return [(None, None)] * len(texts)

    # Punctuation arrives as its own token; drop it so word indices line up.
    words = [t for t in timestamps if str(t.get("word", "")).strip(".,!?;:")]

    out, cursor = [], 0
    pad = int(16000 * 30 / 1000)
    for text in texts:
        n_words = len(text.split())
        if cursor + n_words > len(words):
            out.append((None, None))
            continue
        span = words[cursor:cursor + n_words]
        cursor += n_words

        start = max(0, int(span[0]["start_time"] * 16000) - pad)
        end = min(len(data), int(span[-1]["end_time"] * 16000) + pad)
        if end <= start:
            out.append((None, None))
            continue

        # Re-base the timestamps so they read as if this clip were rendered alone -
        # phrase_end_sample and the run-on cut both index from the clip's own start.
        rebased = [{"word": t.get("word"),
                    "start_time": t["start_time"] - start / 16000,
                    "end_time": t["end_time"] - start / 16000} for t in span]
        out.append((data[start:end], rebased))
    return out


def phrase_end_sample(timestamps, wake_word: str, sr: int = 16000):
    """Sample index where the wake word ends, or None if the words do not line up.

    Verified rather than assumed: the timestamps are matched against the words of
    the wake phrase before their times are used. A mismatch (different tokenisation,
    a normalisation rule splitting a word) would otherwise cut at the wrong place
    silently, and a wrong cut here is what broke the alignment last time.
    """
    if not timestamps:
        return None

    strip = str.maketrans("", "", ".,!?;:\"'")
    expected = [w.translate(strip).lower() for w in wake_word.split()]
    got = [str(t.get("word", "")).translate(strip).lower()
           for t in timestamps[:len(expected)]]
    if got != expected:
        return None

    end = timestamps[len(expected) - 1].get("end_time")
    return int(end * sr) if end else None


def generate_kokoro_sample(kokoro_url: str, voice: str, text: str, output_dir: Path,
                           speed: float = None) -> bool:
    """Generate a single Kokoro TTS sample."""
    if speed is None:
        speed = np.random.uniform(*PLAIN_SPEEDS)
    data = kokoro_tts(kokoro_url, voice, text, speed)
    if data is None:
        return False
    filename = f"kokoro_{uuid.uuid4().hex}.wav"
    scipy.io.wavfile.write(str(output_dir / filename), 16000, data)
    return True


def run_jobs(jobs, worker, desc: str, workers: int, weights=None):
    """Run `worker` over `jobs` in a thread pool, with a progress bar.

    Threads rather than processes because every job is a blocking HTTP request to
    the TTS server - the work happens there, not here. The pool size is really a
    concurrency limit on the server: measured throughput rises from ~7 to ~11.5
    calls/s going from 1 to 4 workers and is flat at 8, so the server saturates
    early and more workers would only queue.
    """
    def guarded(job):
        # One transient failure must not abandon a batch that takes tens of minutes.
        # pool.map re-raises on iteration, so swallow here and count it as a miss;
        # the caller already reports success against the job total.
        try:
            return worker(job)
        except Exception:
            return 0

    # Workers return either a bool (one clip) or a count (a batch of clips);
    # int() covers both, since int(True) is 1.
    #
    # `weights` is how many clips each job produces. Without it the bar would count
    # REQUESTS once batching is on, so "12 it/s" would mean 12 batches - roughly 190
    # clips/s - and look like a slowdown against the pre-batching 28 clips/s. The bar
    # counts clips either way.
    weights = weights or [1] * len(jobs)
    success = 0
    with tqdm(total=sum(weights), desc=desc, unit="clip") as pbar:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for weight, result in zip(weights, pool.map(guarded, jobs)):
                success += int(result)
                pbar.update(weight)
    return success


def generate_kokoro_samples(pool: "KokoroPool", voices: list, output_dir: Path,
                            samples_per_voice: int, texts: list, desc: str,
                            workers: int = 2, batch: int = 16):
    """Generate Kokoro samples for all voices.

    Clips are rendered in batches sharing one voice and speed, which is a ~5x
    speedup (182 ms/clip individually, ~37 ms at a batch of 16-32) because three
    quarters of a short request is fixed overhead. `batch=1` renders each clip in
    its own request, which is the pre-batching behaviour.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # The job list is built up front, in one thread. Drawing the speed here rather
    # than inside the worker keeps the corpus a function of the seed alone, instead
    # of depending on the order threads happen to run in.
    from collections import defaultdict
    buckets = defaultdict(list)
    total = 0
    for v, voice in enumerate(voices):
        for i in range(samples_per_voice):
            # Offset each voice's starting point in the wordlist. Without it every
            # voice renders texts[0:samples_per_voice], so a list longer than
            # samples_per_voice never gets past its own beginning - which the test
            # sets hit as soon as the negative list grew past samples_per_voice//10.
            text = texts[(v * samples_per_voice + i) % len(texts)]
            speed = float(np.random.choice(PLAIN_SPEED_GRID))
            buckets[(voice, speed)].append(text)
            total += 1

    # One job per batch: a (voice, speed) group sliced into chunks.
    jobs = []
    for (voice, speed), group in buckets.items():
        for i in range(0, len(group), batch):
            jobs.append((voice, speed, group[i:i + batch]))

    def render(job):
        voice, speed, group = job
        url = pool.next()
        if batch == 1:
            data = kokoro_tts(url, voice, group[0], speed)
            results = [(data, None)]
        else:
            results = kokoro_tts_batch(url, voice, group, speed)

        written = 0
        for data, _ in results:
            if data is None:
                continue
            # The voice goes in the name so add_child_range_copies can pick a
            # stretch ratio from its sex, and so a bad voice can be traced later.
            scipy.io.wavfile.write(
                str(output_dir / f"kokoro_{voice}_{uuid.uuid4().hex}.wav"), 16000, data)
            written += 1
        return written

    written = run_jobs(jobs, render, desc, workers * len(pool),
                       weights=[len(g) for _, _, g in jobs])

    print(f"  Generated {written}/{total} samples "
          f"({len(jobs)} request(s), batch {batch})")
    return written


# build_negative_phrases now lives in corpus/negatives.py, imported above.


def generate_runon_samples(pool: "KokoroPool", voices: list, output_dir: Path,
                           per_voice: int, wake_word: str, desc: str,
                           reference: dict = None, workers: int = 2,
                           batch: int = 16):
    """Positives where the phrase runs straight into a command.

    The model measured in the tuning log detects 97% of "hey seeree, what's the time?"
    (comma, so the TTS puts a pause in) but only 83% of "hey seeree what's the time?"
    spoken as one breath. Splicing a command onto a separately-recorded phrase does
    not reproduce that, because the phrase keeps its isolated ending; the final
    syllable has to actually be coarticulated into the next word, which means
    rendering the whole thing as one utterance.

    The clip is then CUT shortly after the phrase, and that is the part to be careful
    about. create_fixed_size_clip aligns the END OF THE ARRAY with the end of the
    detection window, so a whole "hey seeree what's the time" would land the wake word
    ~1.5s before the window end - outside the window once it is truncated to 2s, and
    the opposite of the alignment the rest of the pipeline works to produce. Cutting
    just past the phrase leaves the command's onset as trailing context, which is the
    thing being taught, and keeps the phrase where the window expects it.

    The cut point comes from a phrase-alone rendering at the same voice and speed,
    cached per (voice, speed). Coarticulation makes the phrase slightly shorter inside
    the run-on than it is alone, so the cut lands a little way into the command -
    which is the intent, and the jitter on top of it is deliberate.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = {} if reference is None else reference
    # The fallback cache is read and written from several threads, and a miss costs
    # a TTS call, so guard it rather than racing to make the same call twice.
    reference_lock = threading.Lock()
    fallbacks = []

    # Group by (voice, speed) so a batch can share one request, then chunk. Each
    # clip keeps its own tail jitter, drawn here so the corpus stays a function of
    # the seed rather than of thread scheduling.
    from collections import defaultdict
    buckets = defaultdict(list)
    total = 0
    for voice in voices:
        for i in range(per_voice):
            speed = RUNON_SPEEDS[i % len(RUNON_SPEEDS)]
            command = TRAINING_COMMANDS[(i // len(RUNON_SPEEDS)) % len(TRAINING_COMMANDS)]
            tail = int(16000 * np.random.uniform(*RUNON_TAIL_MS) / 1000)
            buckets[(voice, speed)].append((command, tail))
            total += 1

    jobs = []
    for (voice, speed), group in buckets.items():
        for i in range(0, len(group), batch):
            jobs.append((voice, speed, group[i:i + batch]))

    def cut_and_write(data, timestamps, tail, voice, speed, kokoro_url, text):
        """Cut one run-on clip just past the wake word and write it."""
        cut = phrase_end_sample(timestamps, wake_word) if data is not None else None

        if cut is None:
            # Fallback for a server without /dev/captioned_speech: infer the
            # boundary from a phrase-alone rendering, cached per (voice, speed).
            # Measured at a median +153 ms late and voice-dependent, so it is a
            # degraded mode rather than an equivalent one.
            if data is None:
                data = kokoro_tts(kokoro_url, voice, text, speed)
            key = (voice, speed)
            with reference_lock:
                if key not in reference:
                    alone = kokoro_tts(kokoro_url, voice, wake_word, speed)
                    reference[key] = len(trim_silence(alone)) if alone is not None else None
                phrase_len = reference[key]
            if data is not None and phrase_len:
                data = trim_silence(data)
                cut = phrase_len - int(16000 * 30.0 / 1000)   # drop the trim pad
                fallbacks.append(1)

        if data is None or not cut:
            return 0
        data = data[:cut + tail] if cut + tail < len(data) else data
        scipy.io.wavfile.write(
            str(output_dir / f"runon_{voice}_{uuid.uuid4().hex}.wav"), 16000, data)
        return 1

    def render(job):
        voice, speed, group = job
        kokoro_url = pool.next()
        texts = [f"{wake_word} {command}" for command, _ in group]

        if batch == 1:
            results = [kokoro_tts_timed(kokoro_url, voice, texts[0], speed)]
        else:
            results = kokoro_tts_batch(kokoro_url, voice, texts, speed)

        written = 0
        for (data, timestamps), (command, tail), text in zip(results, group, texts):
            written += cut_and_write(data, timestamps, tail, voice, speed,
                                     kokoro_url, text)
        return written

    success = run_jobs(jobs, render, desc, workers * len(pool),
                       weights=[len(g) for _, _, g in jobs])

    print(f"  Generated {success}/{total} run-on samples "
          f"({len(jobs)} request(s), batch {batch})")
    if fallbacks:
        print(f"  NOTE: {len(fallbacks)} clip(s) fell back to the phrase-alone estimate "
              f"({len(reference)} reference renderings).")
        print("        /dev/captioned_speech was unavailable or its words did not match")
        print("        the wake phrase, so those cuts sit later than they should.")
    return success


# copy_real_samples, time_stretch, vocal_tract_shift,
# add_child_range_copies, trim_silence and trim_directory now live in
# corpus/real.py and corpus/augment.py, imported above.


def setup_training_dirs(wake_word: str) -> Path:
    """Set up training directory structure.

    data/corpus/<wake_word>/oww/ - beside the microWakeWord corpus at .../mww/,
    without either pipeline reaching into the other's directory.

    THE CORPUS AND THE MODELS ARE NOW SEPARATE TREES, and this function is why. It
    rmtree's its base directory on every run. That base used to sit under
    one tree, the same tree run-training.sh copies the commit-tagged models
    into, so a run could delete the archive of previous runs - it did exactly that
    until the corpus was nested a level deeper. With the corpus under data/ and the
    models under output/, the destructive path cannot reach a model at all: this
    rmtree is confined to generated audio that the next run would rebuild anyway.
    """
    safe_name = wake_word.replace(" ", "_").lower()
    base_dir = WORK_DIR / "data" / "corpus" / safe_name / "oww"

    if base_dir.exists():
        print("Clearing previous training outputs...")
        shutil.rmtree(base_dir)

    for subdir in ["positive_train", "positive_test", "negative_train", "negative_test"]:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    return base_dir


def create_config(wake_word: str, n_samples: int, training_steps: int,
                  layer_size: int, data_dir: str, augmentation_rounds: int = 3,
                  max_negative_weight: int = 2000):
    """Create training configuration."""
    safe_name = wake_word.replace(" ", "_").lower()

    # Load default config from OpenWakeWord
    default_path = WORK_DIR / "openwakeword/examples/custom_model.yml"
    with open(default_path, 'r') as f:
        config = yaml.load(f.read(), yaml.Loader)

    config["target_phrase"] = [safe_name]
    config["model_name"] = safe_name
    config["n_samples"] = n_samples
    config["n_samples_val"] = max(1000, n_samples // 10)
    config["steps"] = training_steps
    config["layer_size"] = layer_size
    config["target_accuracy"] = 0.7
    config["target_recall"] = 0.5
    config["target_false_positives_per_hour"] = 0.1
    # MODELS OUT, CORPUS ELSEWHERE. Upstream uses output_dir for exactly three
    # things (openwakeword/train.py:652, 905, 909): the .onnx export, the .tflite
    # conversion beside it, and an empty <output_dir>/<model_name>/ it creates
    # unconditionally - that last one is where the corpus WOULD have gone, and its
    # being empty is the visible sign that corpus_dir took over. Everything else
    # that used to derive from output_dir is re-pointed by
    # patches/configurable-corpus-dir.py, which is what makes corpus_dir exist.
    config["output_dir"] = f"./output/{safe_name}/oww"
    config["corpus_dir"] = f"./data/corpus/{safe_name}/oww"

    # End of a linear ramp: the negative-class loss weight grows from 1 to this
    # over training (openwakeword/train.py:274), so higher penalises false
    # positives harder.
    #
    # Run 8 tried 4000. It does reduce false accepts (7/32 -> 3/32 at threshold
    # 0.5) but costs detection (held-out plain 89% -> 77%, run-on 56% -> 37%).
    # Compared at MATCHED false-accept counts rather than a fixed threshold the
    # two models trade places without either dominating - so the weight mostly
    # moved the operating point along the same curve, which the detection
    # threshold does for free and without a retrain. Back at 2000 for that reason.
    #
    # Raise this only if the deployment cannot tune its threshold.
    config["max_negative_weight"] = max_negative_weight

    # Each round re-augments every clip with a different impulse response,
    # background and gain, so this multiplies the distinct feature vectors without
    # any extra TTS. It matters because training draws 50 positives per step for
    # 50,000 steps - 2.5M draws against ~14k clips, so every clip is revisited
    # ~180 times, and at one round those are 180 views of an identical vector.
    #
    # Needs patches/honour-augmentation-rounds.py: upstream multiplies the clip
    # list by this value but sizes the output array from the unmultiplied
    # directory, so without the patch the extra rounds are computed and discarded.
    config["augmentation_rounds"] = augmentation_rounds
    config["rir_paths"] = [f'{data_dir}/mit_rirs']
    config["background_paths"] = [f'{data_dir}/audioset_16k', f'{data_dir}/fma']
    config["false_positive_validation_data_path"] = f"{data_dir}/validation_set_features.npy"
    config["feature_data_files"] = {"ACAV100M_sample": f"{data_dir}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}
    config.pop("piper_sample_generator_path", None)  # We use Kokoro, not Piper

    config_path = WORK_DIR / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Config saved: {config_path}")
    return config


def run_augmentation():
    """Run OpenWakeWord augmentation pipeline."""
    print("\n" + "=" * 60)
    print("Running augmentation pipeline...")
    print("=" * 60)

    train_script = str(WORK_DIR / "openwakeword/openwakeword/train.py")
    subprocess.run([
        sys.executable, train_script,
        "--training_config", "training_config.yaml",
        "--augment_clips"
    ], check=True)


def run_training():
    """Run OpenWakeWord model training."""
    print("\n" + "=" * 60)
    print("Training model...")
    print("=" * 60)

    train_script = str(WORK_DIR / "openwakeword/openwakeword/train.py")
    result = subprocess.run([
        sys.executable, train_script,
        "--training_config", "training_config.yaml",
        "--train_model"
    ])
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train a custom OpenWakeWord model")
    parser.add_argument("--wake-word", default="hey seeree", help="Wake word/phrase to train")
    parser.add_argument("--samples-per-voice", type=int, default=300,
                        help="Samples per Kokoro voice (default: %(default)s). Raised from\n                             200 when the wordlists grew: 59 negative phrases and 60\n                             run-on command/speed combinations need more renderings each\n                             to keep per-item density up.")
    parser.add_argument("--training-steps", type=int, default=50000,
                        help="Steps for the first training sequence (default: "
                             "%(default)s). openwakeword scales warmup, hold and the "
                             "negative-weight ramp from this, so raising it also "
                             "slows how fast the negative weight climbs. Run 11 tried "
                             "100k: detection looked better at threshold 0.5, but at "
                             "MATCHED false-accept rates it was worse everywhere. It "
                             "moved the operating point, it did not improve the "
                             "model. Measured in tuning run 11.")
    parser.add_argument("--layer-size", type=int, default=64, choices=[32, 64, 128], help="Network layer size")
    parser.add_argument("--kokoro-url", default=os.environ.get("KOKORO_URL", "http://localhost:8880"),
                        help="Kokoro TTS URL. Comma-separate several to split the work "
                             "across them: one Kokoro process is single-threaded and "
                             "saturates one core, so more PROCESSES scale where more "
                             "client threads do not.")
    parser.add_argument("--data-dir", default=".", help="Directory containing training data (features, audioset, fma, mit_rirs)")
    parser.add_argument("--no-trim", action="store_true",
                        help="Skip silence trimming before augmentation (not recommended)")
    parser.add_argument("--negatives-file",
                        help="Text file of confusable negative phrases, one per line "
                             "(# comments allowed). Overrides the built-in list for "
                             "this wake word; base negatives are always included.")
    parser.add_argument("--tts-workers", type=int, default=2,
                        help="Concurrent requests PER SERVER (default: %(default)s). "
                             "A Kokoro process handles one at a time, so this only "
                             "covers the gap between responses; total concurrency is "
                             "this times the number of servers.")
    parser.add_argument("--tts-batch", type=int, default=16,
                        help="Utterances per Kokoro request (default: %(default)s). "
                             "A short request is ~75%% fixed overhead, so batching "
                             "is ~3x faster at realistic bucket sizes; clips are split apart exactly, on the "
                             "server's word timestamps. 1 disables it. Every clip in "
                             "a batch shares a voice and speed, which is why plain "
                             "speeds come from a grid rather than a continuous draw.")
    parser.add_argument("--real-copies", type=int, default=10,
                        help="How many times each real recording is duplicated into "
                             "the positive set (default: %(default)s). Weighting, "
                             "not augmentation - watch the held-out set for "
                             "overfitting to the specific clips.")
    parser.add_argument("--max-negative-weight", type=int, default=2000,
                        help="How hard false positives are penalised by the end of "
                             "training (default: %(default)s). Higher trades "
                             "detection for precision - but so does the detection "
                             "threshold, for free. Measured in tuning run 8. Recommend to leave at default.")
    parser.add_argument("--augmentation-rounds", type=int, default=3,
                        help="How many differently-augmented copies of each clip to "
                             "compute features for (default: %(default)s). Multiplies "
                             "training data at no TTS cost.")
    parser.add_argument("--runon-fraction", type=float, default=0.4,
                        help="Fraction of positives where the phrase runs straight "
                             "into a command instead of being followed by quiet "
                             "(default: %(default)s). 0 disables them.")
    parser.add_argument("--exclude-voices", default="",
                        help="Comma-separated Kokoro voices to skip, added to the "
                             "built-in MISPRONOUNCING_VOICES list for this wake "
                             "word. Use for a wake word with no built-in entry.")
    parser.add_argument("--child-fraction", type=float,
                        default=CHILD_STRETCH_FRACTION,
                        help="Fraction of Kokoro positives that get an ADDITIONAL "
                             "pitch/formant-shifted copy, to cover child and "
                             "adolescent voices (default: %(default)s). 0 disables "
                             "it, restoring the pre-run-12 adult-only corpus.")
    parser.add_argument("--piper-fraction", type=float, default=0.0,
                        help="Fraction of the PHRASE-ALONE positive budget rendered "
                             "by Piper instead of Kokoro (default: %(default)s = "
                             "off). SUBSTITUTES rather than adds, so corpus size, "
                             "the plain/run-on split and real-clip density are all "
                             "unchanged - which is what makes a comparison against "
                             "the previous run mean anything. Run-ons stay Kokoro: "
                             "their cut point comes from Kokoro's word timestamps, "
                             "and Wyoming has no equivalent, so Piper would fall "
                             "back to inferring it from a phrase-alone rendering - "
                             "measured at a median +153 ms late, against a "
                             "RUNON_TAIL_MS of 150-300 ms. That is run 14's "
                             "alignment regression waiting to happen.")
    parser.add_argument("--piper-url",
                        default=os.environ.get("PIPER_URL", "piper:10200"),
                        help="Wyoming TTS host:port for Piper (default: %(default)s)")
    parser.add_argument("--piper-speakers", type=int, default=12,
                        help="Speakers to sample per multi-speaker Piper voice, "
                             "evenly spaced (default: %(default)s). "
                             "en_US-libritts_r-medium alone carries 904, which "
                             "would swamp the corpus with one model's g2p.")
    parser.add_argument("--piper-languages", default="en_US,en_GB",
                        help="Language prefixes to include (default: %(default)s)")
    args = parser.parse_args()

    wake_word = args.wake_word
    safe_name = wake_word.replace(" ", "_").lower()

    print("=" * 60)
    print("OpenWakeWord Training")
    print("=" * 60)
    print(f"Wake word: {wake_word}")
    print(f"Samples per voice: {args.samples_per_voice}")
    print(f"Training steps: {args.training_steps}")
    print(f"Layer size: {args.layer_size}")
    print()

    print("[Compute]")
    report_onnx_providers()

    # Get Kokoro voices
    print("\n[Kokoro servers]")
    pool = KokoroPool(args.kokoro_url.split(","))
    kokoro_voices = probe_kokoro_servers(pool)
    if not kokoro_voices:
        print("ERROR: No Kokoro voices available!")
        sys.exit(1)
    print(f"  {len(pool)} server(s), {len(kokoro_voices)} shared English voices")

    excluded = set(MISPRONOUNCING_VOICES.get(safe_name, []))
    excluded.update(v.strip() for v in args.exclude_voices.split(",") if v.strip())
    if excluded:
        present = sorted(v for v in kokoro_voices if v in excluded)
        kokoro_voices = [v for v in kokoro_voices if v not in excluded]
        print(f"  Excluding {len(present)} voice(s) that mispronounce the wake word: "
              f"{', '.join(present)}")
        print(f"  {len(kokoro_voices)} voices remain")
        missing = sorted(excluded - set(present))
        if missing:
            print(f"  NOTE: {', '.join(missing)} not offered by these servers anyway")
        if not kokoro_voices:
            print("ERROR: every available voice is excluded!")
            sys.exit(1)

    # Setup directories
    base_dir = setup_training_dirs(wake_word)
    pos_train = base_dir / "positive_train"
    pos_test = base_dir / "positive_test"
    neg_train = base_dir / "negative_train"
    neg_test = base_dir / "negative_test"

    # Text variations for positive samples.
    #
    # NO UPPERCASE. `wake_word.upper()` was in this list for the first twelve runs
    # and it renders the invented word as SPELLED-OUT LETTERS - "hey S-E-E-R-E-E" -
    # which was then labelled as the wake word. A sixth of the plain positives were
    # mislabelled that whole time. Caught by ear; the measurements that were
    # supposed to catch it both failed, and how they failed is the point:
    #
    #   * duration: 1083 ms against 965 ms, only +12%. Spelling six letters should
    #     have doubled it. Too weak a signal to conclude anything from, and it was
    #     read as "emphatic delivery" instead.
    #   * embedding distance: 0.535 from plain, about the same as a DIFFERENT VOICE
    #     (0.70). That was read as "lots of diversity" when it was really "this is
    #     not the same phrase".
    #
    # A large distance from the plain rendering cannot distinguish useful variety
    # from a different utterance. Anything added here must be LISTENED to.
    #
    # It is uppercase on the invented word specifically: "HEY seeree" measures 0.031
    # from plain (nothing happens), while "hey SEEREE" measures 0.030 from
    # "HEY SEEREE" (both spell it). A real word in caps is fine; the wake word is
    # not a real word, which is the whole reason it makes a good wake word.
    #
    # What is left is punctuation, which changes prosody without touching
    # pronunciation. Distances from plain (af_bella / am_adam):
    #
    #   hey seeree,    0.437 / 0.344
    #   hey seeree!    0.291 / 0.201
    #   hey seeree...  0.264 / 0.523
    #   hey seeree!!   0.158 / 0.232
    #   hey seeree?    0.105 / 0.304
    #   hey seeree.    0.086 / 0.294
    #   Hey Seeree     0.027 / 0.053   <- dropped, indistinguishable from plain
    #
    # `.lower()` is also gone: it is the same STRING as `wake_word` for a lowercase
    # wake word, so it was a literal duplicate slot.
    #
    # The phrase-alone texts, and the tuned speed grid, live in
    # corpus/positives.py so the microWakeWord corpus renders the same thing.
    positive_texts = plain_positive_texts(wake_word)

    # Negative phrases - see build_negative_phrases for why the confusable ones
    # (near-misses of the wake word) are the important half of this list.
    print("\n[Negative wordlist]")
    negative_phrases = build_negative_phrases(wake_word, args.negatives_file,
                                             with_commands=args.runon_fraction > 0)
    print(f"  Total negative phrases: {len(negative_phrases)}")

    # === POSITIVE SAMPLES ===
    print("\n" + "=" * 60)
    print("Generating POSITIVE samples...")
    print("=" * 60)

    # Split the positive budget between the phrase alone and the phrase running into
    # a command. The total is unchanged, so the balance against the negatives is too.
    runon_train = int(args.samples_per_voice * args.runon_fraction)
    plain_train = args.samples_per_voice - runon_train
    runon_test = int(args.samples_per_voice // 10 * args.runon_fraction)
    plain_test = args.samples_per_voice // 10 - runon_test

    # Piper SUBSTITUTES for part of the phrase-alone budget rather than adding to it.
    #
    # Adding would move three things at once: engine diversity, total corpus size,
    # and - because real clips are a FRACTION of the positive set - real-clip
    # density, which run 10 measured as the largest single lever here (run-on
    # 53% -> 77%). A naive "also generate Piper" over all 84 voices would have taken
    # real clips from ~17% of positives to ~6%, and the result would have measured
    # dilution rather than diversity.
    #
    # Substituting holds the total, the plain/run-on split, and real-clip density
    # fixed, leaving one variable: where a share of the phrase-alone clips came from.
    # Run-ons stay entirely Kokoro - see the --piper-fraction help for why.
    piper_voices = []
    kokoro_plain_train, kokoro_plain_test = plain_train, plain_test
    if args.piper_fraction > 0:
        host, _, port = args.piper_url.rpartition(":")
        piper_voices = select_piper_voices(
            host, port, wake_word,
            languages=tuple(args.piper_languages.split(",")),
            max_speakers=args.piper_speakers)
        if piper_voices:
            kokoro_plain_train = int(round(plain_train * (1 - args.piper_fraction)))
            kokoro_plain_test = int(round(plain_test * (1 - args.piper_fraction)))
            # Budget in TOTAL clips, then spread over however many Piper voices
            # there are - the two engines do not have the same voice count, so a
            # per-voice figure would not substitute one-for-one.
            piper_total_train = (plain_train - kokoro_plain_train) * len(kokoro_voices)
            piper_total_test = (plain_test - kokoro_plain_test) * len(kokoro_voices)
            piper_per_voice_train = max(1, piper_total_train // len(piper_voices))
            piper_per_voice_test = max(1, piper_total_test // len(piper_voices))

    print("\n[Kokoro TTS]")
    print(f"  Per voice: {kokoro_plain_train} phrase-alone, {runon_train} run-on "
          f"({args.runon_fraction:.0%})")
    generate_kokoro_samples(pool, kokoro_voices, pos_train,
                            kokoro_plain_train, positive_texts, "Kokoro positive train",
                            args.tts_workers, args.tts_batch)
    generate_kokoro_samples(pool, kokoro_voices, pos_test,
                            kokoro_plain_test, positive_texts, "Kokoro positive test",
                            args.tts_workers, args.tts_batch)

    if piper_voices:
        host, _, port = args.piper_url.rpartition(":")
        print(f"\n[Piper TTS]  {len(piper_voices)} voices, "
              f"{piper_per_voice_train} phrase-alone each "
              f"(~{args.piper_fraction:.0%} of the phrase-alone budget)")
        generate_piper_samples(host, int(port), piper_voices, pos_train,
                               piper_per_voice_train, positive_texts,
                               PLAIN_SPEED_GRID, "Piper positive train")
        generate_piper_samples(host, int(port), piper_voices, pos_test,
                               piper_per_voice_test, positive_texts,
                               PLAIN_SPEED_GRID, "Piper positive test")

    if runon_train:
        # One reference cache across both sets: the phrase-alone lengths are the
        # same, and rebuilding it would cost a few hundred needless TTS calls.
        reference = {}
        generate_runon_samples(pool, kokoro_voices, pos_train,
                               runon_train, wake_word, "Kokoro run-on train",
                               reference, args.tts_workers, args.tts_batch)
        generate_runon_samples(pool, kokoro_voices, pos_test,
                               runon_test, wake_word, "Kokoro run-on test",
                               reference, args.tts_workers, args.tts_batch)

    # Before the real clips are copied in, so the shift only ever sees Kokoro
    # output - and before trimming, so the shifted copies are trimmed like the rest.
    if args.child_fraction > 0:
        print("\n[Child-range copies]")
        print(f"  Shifting {args.child_fraction:.0%} of Kokoro clips: "
              f"female {CHILD_STRETCH['f'][0]}-{CHILD_STRETCH['f'][1]}x, "
              f"male {CHILD_STRETCH['m'][0]}-{CHILD_STRETCH['m'][1]}x")
        add_child_range_copies(pos_train, "VTLP positive train", args.child_fraction)
        add_child_range_copies(pos_test, "VTLP positive test", args.child_fraction)

    print("\n[Real Voice]")
    # The training half of the recordings. data/recordings/holdout/ is a SIBLING and
    # is never read here - copy_real_samples globs this tree recursively, so a
    # holdout nested inside it would be trained on and every eval number after that
    # would be measuring memorisation. See eval/paths.py, which enforces the pair.
    real_samples_dir = WORK_DIR / "data" / "recordings" / "samples"
    real_count = copy_real_samples(real_samples_dir, pos_train, args.real_copies)
    if real_count > 5:
        copy_real_samples(real_samples_dir, pos_test, args.real_copies)

    # === NEGATIVE SAMPLES ===
    print("\n" + "=" * 60)
    print("Generating NEGATIVE samples...")
    print("=" * 60)

    print("\n[Kokoro TTS]")
    generate_kokoro_samples(pool, kokoro_voices, neg_train,
                            args.samples_per_voice, negative_phrases,
                            "Kokoro negative train", args.tts_workers, args.tts_batch)
    generate_kokoro_samples(pool, kokoro_voices, neg_test,
                            args.samples_per_voice // 10, negative_phrases,
                            "Kokoro negative test", args.tts_workers, args.tts_batch)

    # === COUNT SAMPLES ===
    n_pos_train = len(list(pos_train.glob("*.wav")))
    n_pos_test = len(list(pos_test.glob("*.wav")))
    n_neg_train = len(list(neg_train.glob("*.wav")))
    n_neg_test = len(list(neg_test.glob("*.wav")))

    print("\n" + "=" * 60)
    print("Sample counts:")
    print(f"  Positive: {n_pos_train} train, {n_pos_test} test")
    print(f"  Negative: {n_neg_train} train, {n_neg_test} test")
    print("=" * 60)

    # === TRIM SILENCE ===
    # Must run before augmentation: OpenWakeWord places the end of each array at the
    # end of the detection window, so silence on the clip displaces the speech.
    # Negatives are trimmed too - treating both classes identically keeps clip length
    # from becoming a cue the model can learn instead of the phrase itself.
    if not args.no_trim:
        print("\n" + "=" * 60)
        print("Trimming silence (aligns speech with the detection window)...")
        print("=" * 60)
        for directory, label in [(pos_train, "positive train"), (pos_test, "positive test"),
                                 (neg_train, "negative train"), (neg_test, "negative test")]:
            n_trimmed, mean_ms = trim_directory(directory, f"Trim {label}")
            print(f"  {label}: trimmed {n_trimmed} clips (mean {mean_ms:.0f}ms removed)")

    # Create config and run training
    create_config(wake_word, n_pos_train, args.training_steps, args.layer_size,
                  args.data_dir, args.augmentation_rounds, args.max_negative_weight)
    run_augmentation()

    # Note the existing model before training. setup_training_dirs clears the corpus
    # but NOT the exported model, so a previous run's model survives here - and if
    # training fails, an unchanged file would be reported as this run's output. That
    # happened: a CUDA OOM during validation killed training at 75%, the script still
    # printed "TRAINING COMPLETE!", and the stale model was copied off the box and
    # evaluated twice before the identical checksums gave it away. Now that the two
    # trees are separate the model ALWAYS survives a run, so this check matters more
    # than it did, not less.
    model_path = WORK_DIR / "output" / safe_name / "oww" / f"{safe_name}.onnx"
    before = model_path.stat().st_mtime if model_path.exists() else None

    returncode = run_training()

    # Whether the model was WRITTEN is the real signal, not the exit code.
    # openwakeword saves the .onnx and then tries to convert it to tflite, which
    # fails on this image and exits 1. Its conversion goes through onnx_tf, which
    # is deliberately not installed - onnx_tf 1.10 is abandoned and never worked
    # here anyway (tensorflow-cpu 2.8.1 against protobuf >= 3.20), so the image
    # carries onnx2tf and onnx2tflite.py in its place. Treating that exit as
    # failure would discard a perfectly good run.
    fresh = model_path.exists() and (before is None
                                     or model_path.stat().st_mtime != before)

    print("\n" + "=" * 60)
    if not fresh:
        print("TRAINING FAILED")
        print("=" * 60)
        if not model_path.exists():
            print(f"{model_path} does not exist.")
        else:
            print(f"{model_path} was not rewritten - it is still the PREVIOUS run's")
            print("model. Do not evaluate or deploy it as though it were this run's.")
        if returncode != 0:
            print(f"\nopenwakeword's train.py exited {returncode}; traceback above.")
        sys.exit(returncode or 1)

    if returncode != 0:
        print(f"NOTE: openwakeword's train.py exited {returncode}, but the model was")
        print("written. This is normally the tflite conversion failing after the .onnx")
        print("is already saved - see 'TFLite conversion error at end' in the README.")
        print("Convert with onnx2tflite.py instead, which verifies the result.")
        print("=" * 60)

    print("TRAINING COMPLETE!")
    print("=" * 60)
    size_kb = model_path.stat().st_size / 1024
    print(f"Model: {model_path} ({size_kb:.0f}KB)")
    print(f"\nTest with: python test_model.py --model {model_path}")


if __name__ == "__main__":
    main()
