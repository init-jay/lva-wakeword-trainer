"""tts_service - the shared TTS layer of the wakeword trainer.

One package, three engines, one interface. A trainer or an eval generator asks
for engines by SPEC, not by class:

    from tts_service import Client
    client = Client.from_spec("kokoro-mlx")          # in-process MLX, Apple Silicon
    client = Client.from_spec("http://kokoro:8880")  # Kokoro-FastAPI, any machine
    client = Client.from_spec("127.0.0.1:10200")     # Piper over Wyoming, any machine
    client = Client.from_spec("tcp://127.0.0.1:8899")  # a tts-service (server.py), any engine behind it

    client.generate(voices, out_dir, per_voice, texts, desc, name_fn)  # batched
    engine.render(voice, text, speed)                                    # one clip

The engines live in tts_service/engines/ and each is also a module you can import
directly if you only need one (the train/corpus shims do). Adding a fourth engine
is a new engines/ module plus a registry row - see engines/__init__.py.
"""

from .audio import SR, time_stretch, to_int16
from .engine import Engine, split_joined
from .engines import engines_from_spec, is_mlx_url
from .engines.kokoro_mlx import KokoroMlxEngine
from .engines.kokoro_http import (
    KokoroHttpEngine,
    get_kokoro_voices,
    kokoro_tts,
    kokoro_tts_timed,
    kokoro_tts_batch,
    phrase_end_sample,
)
from .engines.piper import PiperWyomingEngine, piper_voices, piper_render
from .engines.tcp import TtsServiceTcpEngine, TtsServiceError
from .client import KokoroPool, probe_kokoro_servers, run_jobs, Client

__all__ = [
    "SR", "time_stretch", "to_int16",
    "Engine", "split_joined",
    "engines_from_spec", "is_mlx_url",
    "KokoroMlxEngine", "KokoroHttpEngine", "PiperWyomingEngine", "TtsServiceTcpEngine",
    "TtsServiceError",
    "get_kokoro_voices", "kokoro_tts", "kokoro_tts_timed", "kokoro_tts_batch",
    "phrase_end_sample",
    "piper_voices", "piper_render",
    "KokoroPool", "probe_kokoro_servers", "run_jobs", "Client",
]
