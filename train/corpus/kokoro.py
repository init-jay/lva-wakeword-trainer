"""
Kokoro voice model for the generated corpus.

The Kokoro-FastAPI server (https://github.com/gabrimatic/kokoro-fastapi) has
two endpoints that produce *different audio* for the same text:

  /dev/speech             - plain PCM, no word timestamps
  /dev/captioned_speech   - PCM + word timestamps (start/end in seconds)

This module keeps both: `kokoro_tts` renders without timestamps (used for
the negative/positional-audio paths where word timing is irrelevant), and
`kokoro_tts_timed` returns (audio, timestamps) for the wake-word path, where
the timestamps are what lets run-on samples be cut at the exact end of the
wake word.

TRANSPORT (since the tts-service split, 2026-09-08): this module no longer
talks to any HTTP API itself. Both server families - Kokoro-FastAPI (Docker,
cuda/cpu box) and kokoro-mlx (Apple Silicon, in-process MLX) - run a
tts-protocol server (tts-service/) in front of the actual engine and speak
one small line-based protocol over a plain TCP port. This module is a thin
TCP client over `tcp://` URLs, and it is the ONLY transport code left in the
trainer: no `requests`, no OpenAI-compatible JSON, no engine-specific
response shapes.

The URL is therefore a `tcp://` spec, not an HTTP URL:

  KOKORO_URL=tcp://127.0.0.1:8899,tcp://127.0.0.1:8900 python train/oww/train.py ...

The old `http://...` / `mlx://...` forms are deliberately rejected by the
client - they used to mean different backends with different audio, and a
silent misread would render a corpus from the wrong engine. The `probe`
function verifies that a `tcp://` URL is actually a tts-protocol server and
that it supports word timestamps, and it prints which engine is behind it.

One batch of text shares one voice and one speed - that is what makes it a
single forward pass. The `batch` helper groups jobs accordingly and splits
the joined rendering back into per-utterance clips using the word timestamps
(`kokoro_tts_timed`), so coarticulation between texts does not leak across
utterance boundaries in the saved files.

Kokoro returns 16 kHz mono int16, which is exactly what openWakeWord's
feature pipeline expects, so no resampling happens here.
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
# URL-named render helpers (same signatures as the old HTTP/MLX adapters,
# now speaking the protocol)
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
# Pool / probe / parallel runner (verbatim from the 2026-08-20 generator;
# see probe's docstring for what the protocol probe checks instead of the
# old per-server test render)
# ---------------------------------------------------------------------------

class KokoroPool:
    """Round-robin over several KOKORO_URL entries, so N Kokoro-FastAPI
    processes (Docker services, or a Mac host running several mlx
    processes) split the corpus generation.

    The old per-entry engine dispatch (mlx:// -> in-process kokoro-mlx,
    anything else -> requests) is gone: every entry is now a `tcp://`
    tts-protocol server, and the engines (which one wraps FastAPI, which
    wraps kokoro-mlx) are the server's business, not this module's.
    """

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
    """Probe the `tcp://` tts-protocol servers behind KOKORO_URL and exit
    with a clear message if any is unreachable, or if none of them exposes
    the word timestamps the corpus actually needs.

    `urls` is a list of `tcp://` specs or a KokoroPool (iterated via .urls).

    This runs at the top of train.py BEFORE any corpus work. Without it, an
    operator who pointed KOKORO_URL at a dead port, or at a tts-protocol
    server fronting an engine with no word timestamps (a Piper server, say),
    would discover the problem at the end of a multi-hour corpus render:
    zero run-on samples, and no explanation.

    The old probe did a test render against each reachable server to confirm
    the timestamps came back shaped correctly. The protocol makes that
    redundant: the `voices` op declares the engine's capabilities in the
    response envelope (wire.py - `timestamps`, `speaker`), and a miswired or
    stale server answers that op with an error, which is exactly the failure
    the test render was catching. So the probe now checks reachability and
    declared capabilities, and prints which engine the server reports -
    which is also the first line of defense against pointing the corpus at
    the wrong engine by accident.
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
    """Run a list of independent jobs with a thread pool, in order of
    completion, with a progress bar.

    `jobs` is an iterable of opaque job arguments; `job_func(job)` is called
    for each. If `job_func` returns an int it is treated as the number of
    items the job produced and the sum is returned - callers report
    "N/M written" from it. `weights` (same order as jobs) scales the
    progress bar per job, so a job covering 20 clips advances it 20 rather
    than 1. Exceptions inside `job_func` are caught and reported - one bad
    voice must not abort the whole corpus run. The pool is deliberately
    small: each Kokoro request already occupies the whole GPU/CPU for the
    duration of the render, so extra workers just queue at the server side
    and add no throughput (measured: 2 workers == 4 workers against a single
    Kokoro-FastAPI process).
    """
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
    """Render the TTS corpus for `voices` from `texts`, distributed round-
    robin across the pool's `tcp://` servers, into `output_dir`.

    Each (voice, speed) pair is rendered `samples_per_voice` times by
    cycling through `texts` (so the same text is rendered several times at
    slightly different delivery rates when `speeds` has more than one
    entry - the extra diversity is what makes the negatives harder).

    Clips shorter than ~0.5 s are discarded: they are too short for the
    feature pipeline to be meaningful, and Kokoro occasionally produces a
    runt when a very short text is rendered at a high speed.

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
        with lock:
            for text, (audio, _ts) in zip(chunk, out):
                if audio is None:
                    # The split could not locate this utterance's words.
                    # Re-render it alone - a single-utterance render either
                    # works or it does not, and if it does not we simply
                    # skip it (a transient miss, not a data problem).
                    audio2, _ = kokoro_tts_timed(url, voice, text, speed)
                    if audio2 is None:
                        continue
                    audio = audio2
                if len(audio) < int(0.5 * SR):
                    continue  # runt, see docstring
                counts[voice] += 1
                name = f"{uuid.uuid4().hex}.wav"
                scipy.io.wavfile.write(output_dir / name, SR, audio)

    run_jobs(jobs, _job, desc=desc, workers=workers)

    print(f"[{desc}] rendered {sum(counts.values())} clips across "
          f"{len(counts)} voices:")
    for voice in voices:
        print(f"    {voice}: {counts.get(voice, 0)}")
