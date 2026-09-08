"""In-process Kokoro via MLX (moved to tts-service, 2026-09-08).

The implementation lives at tts-service/tts_service/engines/kokoro_mlx.py -
including the measurements of what MLX buys (single-clip 57 ms vs the FastAPI
server's batched 88; the +0.2 ms/clip cost of word timestamps; the 28-voice
catalog against the server's 42; the ~430 ms of leading silence) and the note
that its audio is a different runtime of the same model, not the server's bytes.
Read it there before deciding this path is good enough for a production corpus.

`mlx://` is the URL that selects it; KokoroPool, generate_kokoro_samples and
generate_runon_samples dispatch on it without knowing.
"""
from tts_service.engines.kokoro_mlx import (  # noqa: F401
    SAMPLE_RATE,
    URL_SCHEME,
    available,
    is_mlx_url,
    render,
    render_timed,
    voices,
)
