"""The engine registry: turn a spec string into Engine objects.

    tts_service.engines_from_spec("kokoro-mlx")
    tts_service.engines_from_spec("http://kokoro:8880,http://kokoro2:8880")
    tts_service.engines_from_spec("piper://127.0.0.1:10200")
    tts_service.engines_from_spec("127.0.0.1:10200")   # bare host:port = piper
    tts_service.engines_from_spec("tcp://127.0.0.1:8899")  # a tts-service (server.py)

This is where a fourth engine would be registered: a module defining an Engine
subclass and one row in _BY_NAME below. Nothing upstream - the trainers, the eval
generators - has to change, because they never name a class, only a spec.
"""

from ..engine import Engine
from .piper import PiperWyomingEngine
from .kokoro_http import KokoroHttpEngine
from .kokoro_mlx import KokoroMlxEngine, is_mlx_url
from .tcp import TtsServiceTcpEngine

_BY_NAME = {
    "kokoro-mlx": lambda: KokoroMlxEngine(),
    "mlx": lambda: KokoroMlxEngine(),
}


def engines_from_spec(spec: str) -> list:
    """Spec string -> list of Engine, in spec order.

    Kokoro: "kokoro-mlx"/"mlx"/"mlx://..." for the in-process MLX backend (Apple
    Silicon only, checked by available() at use time), a bare "kokoro-http", or one
    or more http(s) URLs comma-separated - the pool the corpus generators have
    always threaded around, as "http://kokoro:8880,http://kokoro2:8880".

    Piper: "piper://host:port", "wyoming://host:port", or a bare "host:port". A
    bare host:port means Piper because that is the shape of every PIPER_URL this
    repo has ever passed ("127.0.0.1:10200", "piper:10200" after splitting the
    scheme off) - a scheme-less Kokoro URL is not a thing the callers construct.

    The tts service: "tcp://host:port", one or more comma-separated. The client
    never sees which engine the service hosts (server.py forwards), which is
    what makes a new machine a port number rather than a dependency list.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty engine spec")

    if is_mlx_url(spec) or spec.lower() in ("mlx", "kokoro-mlx"):
        return [KokoroMlxEngine()]

    if spec.startswith(("piper://", "wyoming://")):
        spec = spec.split("://", 1)[1]

    if "://" in spec:
        urls = [u.strip() for u in spec.split(",") if u.strip()]
        out = []
        for u in urls:
            if u.startswith("tcp://"):
                rest = u[len("tcp://"):]
                host, _, port = rest.rpartition(":")
                if not port.isdigit():
                    raise ValueError(f"unsupported engine URL {u!r} - tcp://host:port")
                out.append(TtsServiceTcpEngine(host, int(port)))
            elif u.startswith(("http://", "https://")):
                out.append(KokoroHttpEngine(u))
            else:
                raise ValueError(f"unsupported engine URL {u!r} - expected "
                                 f"http(s), tcp, piper://, wyoming://, or mlx://")
        return out

    if ":" in spec:
        host, _, port = spec.rpartition(":")
        if not port.isdigit():
            raise ValueError(f"unsupported engine spec {spec!r}")
        return [PiperWyomingEngine(host, int(port))]

    raise ValueError(f"unsupported engine spec {spec!r} - expected a URL, a "
                     f"host:port (Piper), or one of {sorted(_BY_NAME)}")
