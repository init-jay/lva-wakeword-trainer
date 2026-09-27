"""
Kokoro voice model for the generated corpus.

Transport is a thin TCP client over `tcp://` URLs (src/tts-service/
tts-protocol): both engine families - Kokoro-FastAPI (Docker) and kokoro-mlx
(Apple Silicon) - run a tts-protocol server in front of the engine, and this
is the ONLY transport code left in the trainer. Non-`tcp://` URLs are
deliberately rejected by the client: they used to mean different backends
with different audio, and a silent misread would render a corpus from the
wrong engine.

  KOKORO_URL=tcp://127.0.0.1:8899,tcp://127.0.0.1:8901 python src/train/oww/train.py ...

`probe` verifies each URL is a tts-protocol server that supports word
timestamps, and prints which engine is behind it.

`kokoro_tts` renders without word timestamps; `kokoro_tts_timed` returns
(audio, timestamps) - the timestamps are what lets run-on samples be cut at
the exact end of the wake word. One batch of text shares one voice and one
speed (a single forward pass); `kokoro_tts_batch` splits the joined rendering
back into per-utterance clips on those timestamps, so coarticulation between
texts does not leak across utterance boundaries in the saved files.

Kokoro returns 16 kHz mono int16, exactly what openWakeWord's feature
pipeline expects - no resampling here.
"""

import concurrent.futures as cf
import sys
import threading
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

from tts_protocol.audio import SR, phrase_end_sample  # noqa: F401
from tts_protocol.client import TtsClient

__all__ = [
    "SR",
    "phrase_end_sample",
    "KokoroPool",
    "probe_kokoro_servers",
    "run_jobs",
    "generate_kokoro_samples",
    "get_kokoro_voices",
    "kokoro_tts",
    "kokoro_tts_timed",
    "kokoro_tts_batch",
]


# ---------------------------------------------------------------------------
# TtsClient instance cache (one per tcp:// URL, shared across pool threads)
# ---------------------------------------------------------------------------

_CLIENTS = {}
_CLIENTS_LOCK = threading.Lock()


def _client(url: str) -> TtsClient:
    """One TtsClient per `tcp://` URL; created on first use, cached."""
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(url)
        if c is None:
            c = TtsClient(url)
            _CLIENTS[url] = c
        return c


# ---------------------------------------------------------------------------
# URL-named render helpers (speaking the protocol)
# ---------------------------------------------------------------------------

def get_kokoro_voices(url: str):
    """The English voice names a `tcp://` Kokoro server offers (cached)."""
    return _client(url).voices()


def kokoro_tts(url: str, voice: str, text: str, speed: float = 1.0):
    """Render a phrase with `voice` at `speed` (a multiplier on delivery
    rate). Returns a 16 kHz mono int16 numpy array, or None on a transient
    miss (the KOKORO convention: null clip, no raise)."""
    audio, _ = kokoro_tts_timed(url, voice, text, speed)
    return audio


def kokoro_tts_timed(url: str, voice: str, text: str, speed: float = 1.0):
    """Render a phrase and return (audio, word_timestamps), or (None, None)
    on a transient miss."""
    try:
        return _client(url).timed_render(voice, text, speed)
    except Exception:
        # The KOKORO convention: a miss (wedged server, OOM, bad voice)
        # surfaces as a null clip, not a raise - the corpus generator
        # retries the failing clip alone.
        return None, None


def kokoro_tts_batch(url: str, voice: str, texts: list, speed: float):
    """Render several utterances as ONE joined request and split them back
    apart on word timestamps (Engine.batch - the measurement that built it:
    a short Kokoro request is ~3/4 fixed overhead, so a 9-clip bucket went
    137 -> 42 ms/clip). Returns a list of (audio, timestamps), in order,
    with (None, None) for any utterance whose words could not be located.

    Every text in a batch shares one voice and one speed - that is what
    makes it a single forward pass - so callers must group by (voice, speed)
    before calling.
    """
    if not texts:
        return []
    return _client(url).batch(voice, speed, texts)


# ---------------------------------------------------------------------------
# Pool / probe / parallel runner
# ---------------------------------------------------------------------------

class KokoroPool:
    """Round-robin over several KOKORO_URL entries, so N Kokoro processes (Docker
    services, or a Mac host running several mlx processes) split the corpus
    generation. Every entry is a `tcp://` tts-protocol server; which engine
    it wraps is the server's business, not this module's."""

    def __init__(self, urls: list):
        if not urls:
            raise ValueError("KokoroPool: no URLs")
        self._urls = list(urls)
        self._idx = 0
        self._lock = threading.Lock()

    @property
    def urls(self) -> list:
        return self._urls

    def __len__(self) -> int:
        return len(self._urls)

    def next(self) -> str:
        with self._lock:
            u = self._urls[self._idx % len(self._urls)]
            self._idx += 1
            return u


def probe_kokoro_servers(urls, max_speakers: int = 4,
                         min_english_voices: int = 5):
    """Probe the `tcp://` tts-protocol servers behind KOKORO_URL and exit with a
    clear message if any is unreachable, or if none of them exposes the word
    timestamps the corpus actually needs. `urls` is a list of `tcp://` specs
    or a KokoroPool (iterated via .urls).

    Runs at the top of train.py BEFORE any corpus work: without it, a dead
    port, or a tts-protocol server fronting an engine with no word timestamps
    (a Piper server, say), is discovered at the end of a multi-hour render -
    zero run-on samples, and no explanation.

    The probe checks reachability and declared capabilities, not a test
    render: the `voices` op declares the engine's capabilities in the
    response envelope (wire.py - `timestamps`, `speaker`), and a miswired or
    stale server answers that op with an error. It also prints which engine
    each server reports - the first line of defense against pointing the
    corpus at the wrong engine by accident.
    """
    urls = getattr(urls, "urls", urls)  # accept a KokoroPool
    print(f"[probe] {len(urls)} KOKORO_URL server(s):")
    usable, mismatches, catalogs = [], [], []
    for url in urls:
        client = TtsClient(url)
        try:
            voices = client.voices()
        except Exception as e:
            print(f"  {url}: UNREACHABLE ({type(e).__name__}: {e})")
            continue
        engine = getattr(client, "server_engine", None) or "?"
        n = len(voices)
        caps_ts = client.supports_timestamps
        print(f"  {url}: OK (engine={engine}, {n} voices, "
              f"word_timestamps={'yes' if caps_ts else 'NO'})")
        if not caps_ts:
            mismatches.append(url)
            continue
        # English-filter the catalog before the cross-server intersection,
        # same as the old per-engine catalog: Kokoro voice ids are gendered
        # (af_, am_, bf_, bm_), not language-prefixed like Piper's en_US-*.
        en = [v for v in voices
              if isinstance(v, str) and v.startswith(('af_', 'am_', 'bf_', 'bm_'))]
        usable.append((url, en))
        catalogs.append(set(en))

    if mismatches:
        print(f"[probe] WARNING: {len(mismatches)} server(s) have no word "
              f"timestamps: {mismatches}")

    if not usable:
        print("[probe] ERROR: no reachable KOKORO_URL server with word "
              "timestamps. Nothing here can produce run-on samples.")
        sys.exit(1)

    common = set.intersection(*catalogs) if catalogs else set()
    common = sorted(common)
    print(f"[probe] voices in common across {len(usable)} usable server(s): "
          f"{len(common)}")
    if len(common) < min_english_voices:
        print(f"[probe] WARNING: fewer than {min_english_voices} voices in "
              f"common ({len(common)}); the corpus will use what is there "
              f"and the per-voice counts will be low.")
    return common


def run_jobs(jobs, job_func, desc: str = "kokoro", workers: int = 2,
             weights=None):
    """Run a list of independent jobs with a thread pool, in order of completion,
    with a progress bar. `jobs` is an iterable of opaque job arguments;
    `job_func(job)` is called for each, and a returned int is treated as the
    number of items the job produced (the sum is returned). `weights` (same
    order as jobs) scales the progress bar per job. Exceptions inside
    `job_func` are caught and reported - one bad voice must not abort the
    whole corpus run. The pool is deliberately small: each Kokoro request
    already occupies the whole GPU/CPU for the render, so extra workers just
    queue at the server side and add no throughput (measured: 2 workers ==
    4 workers against a single Kokoro-FastAPI process)."""
    jobs = list(jobs)
    if not jobs:
        return 0
    weights = list(weights) if weights is not None else [1] * len(jobs)
    if len(weights) != len(jobs):
        raise ValueError(f"run_jobs: {len(weights)} weights for {len(jobs)} jobs")
    ok, failed, produced = 0, 0, 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job_func, j): (j, w) for j, w in zip(jobs, weights)}
        with tqdm(total=sum(weights), desc=desc) as bar:
            for fut in cf.as_completed(futs):
                j, w = futs[fut]
                try:
                    r = fut.result()
                    ok += 1
                    if isinstance(r, int):
                        produced += r
                except Exception as e:
                    failed += 1
                    print(f"[{desc}] job {j!r} failed: {type(e).__name__}: {e}")
                bar.update(w)
    if failed:
        print(f"[{desc}] {failed}/{len(jobs)} job(s) failed; "
              f"{ok} succeeded.")
    return produced


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

def generate_kokoro_samples(pool: KokoroPool, voices, output_dir: Path,
                            samples_per_voice: int, texts, desc: str = "kokoro",
                            speeds=(1.0,), workers: int = 2,
                            batch: int = 16):
    """Render the TTS corpus for `voices` from `texts`, round-robin across the
    pool's `tcp://` servers, into `output_dir`.

    Each (voice, speed) pair is rendered `samples_per_voice` times by cycling
    through `texts` (the extra delivery rates are what makes the negatives
    harder). Clips shorter than ~0.5 s are discarded - too short for the
    feature pipeline, and Kokoro occasionally produces a runt on a very short
    text at a high speed.

    Batching: texts for the same (voice, speed) are rendered in groups of
    `batch` via `kokoro_tts_batch` (one joined request, split on word
    timestamps) instead of one request per clip. A short Kokoro request is
    ~3/4 fixed overhead, so at the ~9.5-clip buckets the default grid
    produces, this is a 3.3x speedup (137 -> 42 ms/clip, measured 2026-08).
    Any utterance the split could not locate (None) is re-rendered alone.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = list(texts)
    if not texts:
        raise ValueError("generate_kokoro_samples: no texts")

    # (voice, speed) -> list of (text, sample_index) to render
    per_vs = defaultdict(list)
    for voice in voices:
        for speed in speeds:
            for i in range(samples_per_voice):
                per_vs[(voice, speed)].append(texts[i % len(texts)])

    # Build jobs: one job per (voice, speed, batch-of-texts).
    jobs = []
    for (voice, speed), text_list in per_vs.items():
        for i in range(0, len(text_list), batch):
            chunk = text_list[i:i + batch]
            jobs.append((voice, speed, chunk))

    lock = threading.Lock()
    counts = defaultdict(int)

    def _job(vs):
        voice, speed, chunk = vs
        url = pool.next()
        out = kokoro_tts_batch(url, voice, chunk, speed)
        for text, (audio, _ts) in zip(chunk, out):
            if audio is None:
                # The split could not locate this utterance's words. Re-render
                # it alone; if that fails too, skip it (a transient miss, not
                # a data problem).
                audio2, _ = kokoro_tts_timed(url, voice, text, speed)
                if audio2 is None:
                    continue
                audio = audio2
            if len(audio) < int(0.5 * SR):
                continue  # runt, see docstring
            name = f"{uuid.uuid4().hex}.wav"
            scipy.io.wavfile.write(output_dir / name, SR, audio)
            with lock:
                # Only the counter needs the lock: the network re-render and
                # the wav write ran under it before, and at workers=2 one
                # miss stalled the other worker for a full render.
                counts[voice] += 1

    run_jobs(jobs, _job, desc=desc, workers=workers)

    print(f"[{desc}] rendered {sum(counts.values())} clips across "
          f"{len(counts)} voices:")
    for voice in voices:
        print(f"    {voice}: {counts.get(voice, 0)}")
