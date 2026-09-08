"""Kokoro TTS integration (thin layer over tts_service).

Everything that was code here moved to tts-service on 2026-09-08 - the URL
functions to tts_service/engines/kokoro_http.py, the pool/probe/batcher to
tts_service/client.py, and the in-process MLX backend to
tts_service/engines/kokoro_mlx.py. The measured-value notes that were scattered
through those functions travelled with them; read them there before changing the
behaviour, because sixteen tuning runs are calibrated against it.

This module keeps every name and signature the rest of the trainer imports, so
call sites are unchanged:

    import train.corpus.kokoro as kokoro
    pool = kokoro.KokoroPool(KOKORO_URL)
    kokoro.probe_kokoro_servers(pool)
    kokoro.generate_kokoro_samples(pool, VOICES, out, n, texts, "positives")

`generate_kokoro_samples` now delegates to a tts_service.Client, which runs the
identical bucketing and the same batch algorithm on whichever engines the URL
points at - two Kokoro-FastAPI servers in Docker, or the in-process MLX model when
the URL is `mlx://` (see engines/kokoro_mlx.py).
"""
from .positives import PLAIN_SPEED_GRID

from tts_service.client import (
    KokoroPool,
    Client,
    probe_kokoro_servers,
    run_jobs,
)
from tts_service.engines.kokoro_http import (
    get_kokoro_voices,
    kokoro_tts,
    kokoro_tts_timed,
    kokoro_tts_batch,
    phrase_end_sample,
)

__all__ = [
    "KokoroPool",
    "Client",
    "probe_kokoro_servers",
    "run_jobs",
    "get_kokoro_voices",
    "kokoro_tts",
    "kokoro_tts_timed",
    "kokoro_tts_batch",
    "phrase_end_sample",
    "generate_kokoro_sample",
    "generate_kokoro_samples",
]


def generate_kokoro_sample(kokoro_url: str, voice: str, text: str, output_dir, speed: float = None) -> bool:
    """Generate a single Kokoro TTS sample (kept for reference only).

    Note: like the version this replaced, it names PLAIN_SPEEDS without importing
    it - it has never been called on a live path; the batched path is
    generate_kokoro_samples. Kept as documentation of the shape, not as an API.
    """
    import uuid

    import numpy as np
    import scipy.io.wavfile

    if speed is None:
        speed = np.random.uniform(*PLAIN_SPEEDS)  # noqa: F821 - see note above
    data = kokoro_tts(kokoro_url, voice, text, speed)
    if data is None:
        return False
    filename = f"kokoro_{uuid.uuid4().hex}.wav"
    scipy.io.wavfile.write(str(output_dir / filename), 16000, data)
    return True


def generate_kokoro_samples(pool, voices, output_dir, samples_per_voice, texts, desc,
                            workers=2, batch=16):
    """Render the `desc` stage of the corpus across the pool's Kokoro servers.

    Bucketed per (voice, speed); a batch is a group of texts, joined and rendered
    in ONE request, then split on word timestamps (tts_service.engine.Engine.batch)
    because a short request to Kokoro-FastAPI is ~3/4 fixed overhead. Batch
    16-32 measured 37 ms/clip vs 182 individually; at the ~9-clip buckets the
    default grid actually forms, 137 -> 42 end to end (3.3x).

    The body of tts_service.client.Client.generate is the git-HEAD version of this
    function verbatim: one np.random.choice(PLAIN_SPEED_GRID) per (voice,
    sample-index), voice-major, drawn before any thread starts - that draw order
    is what keeps the corpus a function of the seed alone.
    """
    client = Client.from_pool(pool)
    return client.generate(
        voices=voices,
        output_dir=output_dir,
        samples_per_voice=samples_per_voice,
        texts=texts,
        desc=desc,
        speeds=PLAIN_SPEED_GRID,
        batch=batch,
        workers=workers,
    )
