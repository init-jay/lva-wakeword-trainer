"""tts-protocol: the TTS wire protocol of this repo (wire.py) and its two
sides.

Server side: Engine (engine.py) is what an engine server implements and
serve() (server.py) is the generic TCP server that hosts one of them.

Client side: TtsClient (client.py) speaks the protocol to a server. The
trainers' corpus code and the eval generators are clients only - this is the
only TTS code the training images carry.
"""
from .audio import SR, phrase_end_sample, time_stretch, to_int16
from .client import TtsClient, TtsProtocolError
from .engine import Engine, split_joined
from .server import serve
from .wire import MAX_LINE

__all__ = [
    "Engine",
    "TtsClient",
    "TtsProtocolError",
    "serve",
    "split_joined",
    "SR",
    "time_stretch",
    "to_int16",
    "phrase_end_sample",
    "MAX_LINE",
]
