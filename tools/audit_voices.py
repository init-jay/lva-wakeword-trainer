#!/usr/bin/env python3
"""
Find TTS voices that say the wrong thing, before they poison the training corpus.

WHY THIS EXISTS
---------------
A wake word worth having is not a dictionary word, so a TTS engine's grapheme-to-
phoneme has to guess at it - and voices guess differently. Six of Kokoro's 42 English
voices do not say "hey seeree" (af_alloy, am_echo, bf_alice, bf_lily, bm_daniel,
bm_fable). Every clip such a voice produces is a mislabelled positive, and at 1/42 of
the voice list each that was ~14% of the synthetic corpus for the six together.

They were found by listening to all 42, which is the only method that had worked.
Duration is not a proxy: bm_fable sits at exactly the median length and is wrong,
while af_v0bella is 18% below median and is fine. That does not scale to a Piper
voice list with hundreds of speakers, which is what this script is for.

HOW IT WORKS
------------
Render the phrase at several speeds per voice, transcribe each with ASR, and compare
the FINAL TOKEN of each transcript against the consensus across the whole voice set.

The final token is the discriminator, not string similarity. ASR normalises an
invented word toward a real one - every good voice here transcribes as "hey siri" -
so whole-string similarity puts bf_lily's "hey sorry" at 0.82 against "hey siri" and
cannot separate it from af_sarah's "ahay siri" at 0.88. Comparing only the last token
(sorry != siri, siri == siri) separates them cleanly, and ignores the leading filler
ASR invents on slow renderings.

SEVERAL SPEEDS ARE REQUIRED. At 1.0x alone, bf_lily transcribes as "Hey, siri." and
bm_fable as "Hey Siri." - both pass. At 0.75x they are "Hey, sorry." and
"Hey, Sairee." A marginal pronunciation only separates from the real word when the
ASR's language model has less room to smooth it over.

SO ARE REPEATS, because Piper is stochastic. VITS samples noise and durations per
call, so one speaker genuinely says the phrase differently each time - measured:
en_US-libritts_r-medium:1383 scored 0% in one pass and 100% in the next. The ASR is
not the variable; transcribing an identical WAV four times gives an identical string
every time. `--repeats` samples each speed more than once so a single wobble does not
read as a verdict.

That stochasticity also changes what the score MEANS for Piper. It is not "is this
voice correct", it is "how often does this voice get it right" - which is directly
the contamination rate that voice would contribute. A speaker at 67% would put a bad
clip in the corpus one time in three, and belongs on the exclusion list even though
it is sometimes fine.

Validated against the six known-bad Kokoro voices: all six flagged, and the known-good
ones matched consensus at every speed.

THIS IS A SCREEN, NOT A VERDICT. It produces a ranked shortlist to check by ear. A
voice at 100% is probably fine; anything below it needs listening to before it is
trusted or excluded. It also cannot tell a mispronunciation from a strong accent -
en_US-l2arctic-medium is a non-native-speaker corpus and flags heavily, but accented
renderings of a correct phrase are GOOD training data, since real users have accents.
Listen before excluding those.

Usage:
    python audit_voices.py --wake-word "hey seeree" \
        --tts tcp://192.168.2.26:8899 --asr 192.168.2.14:10300
    python audit_voices.py --wake-word "hey seeree" \
        --tts tcp://127.0.0.1:8898 --asr 192.168.2.14:10300

The TTS server speaks the repo protocol (tts-service/) - the same per-engine
servers the corpus runs against, so what this audits is exactly what the corpus
gets. Speaker voices (piper-style) and plain voices (kokoro-style) are both
handled; the server's catalog answer decides which.

    # after: paste the printed block into MISPRONOUNCING_VOICES in train.py
"""

import argparse
import json
import re
import socket
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io.wavfile

SR = 16000
DEFAULT_SPEEDS = (0.75, 1.0, 1.3)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tts-service" / "tts_protocol"))

from tts_protocol import TtsClient  # noqa: E402


# --------------------------------------------------------------------------
# Wyoming ASR client
#
# Wyoming is a JSONL-over-TCP protocol: one JSON header line per event, then
# `data_length` bytes of JSON and `payload_length` bytes of audio. Implemented
# here rather than taking the `wyoming` dependency, which would have to be added
# to the trainer image for a script that never runs inside it.
# --------------------------------------------------------------------------

def _send(sock, etype, data=None, payload=None):
    header = {"type": etype}
    if data is not None:
        header["data"] = data
    if payload is not None:
        header["payload_length"] = len(payload)
    sock.sendall((json.dumps(header) + "\n").encode())
    if payload is not None:
        sock.sendall(payload)


def _read_event(sock, buf):
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return None, buf
        buf += chunk
    line, _, buf = buf.partition(b"\n")
    header = json.loads(line)
    n = header.get("data_length") or 0
    while len(buf) < n:
        buf += sock.recv(65536)
    data = json.loads(buf[:n]) if n else header.get("data", {})
    buf = buf[n:]
    p = header.get("payload_length") or 0
    while len(buf) < p:
        buf += sock.recv(65536)
    return {"type": header.get("type"), "data": data}, buf[p:]


def transcribe(pcm16, host, port, timeout=120):
    """Transcribe int16 mono PCM via a Wyoming ASR service."""
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.settimeout(timeout)
        fmt = {"rate": SR, "width": 2, "channels": 1}
        _send(sock, "transcribe", {"language": "en"})
        _send(sock, "audio-start", {**fmt, "timestamp": 0})
        raw = pcm16.tobytes()
        for i in range(0, len(raw), 2 * SR):
            _send(sock, "audio-chunk", {**fmt, "timestamp": i // 2},
                  payload=raw[i:i + 2 * SR])
        _send(sock, "audio-stop", {"timestamp": len(raw) // 2})
        buf = b""
        while True:
            event, buf = _read_event(sock, buf)
            if event is None:
                return ""
            if event["type"] == "transcript":
                return (event["data"].get("text") or "").strip()
    finally:
        sock.close()


# --------------------------------------------------------------------------
# TTS rendering, over the repo protocol
#
# The catalog answer carries the shape: (voice, speaker) pairs mean a piper-style
# speaker model, plain names a kokoro-style one. Piper phonemises with
# espeak-ng, which is per-MODEL and not per-speaker, so every speaker inside one
# voice shares a pronunciation. That is the opposite of Kokoro, where the six
# bad voices each guessed differently. In practice it means auditing a
# 904-speaker model is mostly one decision about the model plus a sweep for
# speakers whose audio is simply bad - LibriTTS is scraped audiobook read speech
# and its per-speaker quality is uneven.
#
# --piper-speakers caps how many speakers of a multi-speaker voice get sampled,
# so a first pass over en_US-libritts_r-medium does not mean 904 x len(speeds)
# renderings.
# --------------------------------------------------------------------------

def tts_targets(spec, languages, max_speakers, voice_filter):
    """([(voice, speaker_or_None), ...], the TtsClient) for the server at
    `spec`. The client comes back because its catalog fetch is what set
    server_engine - the caller's header prints the engine's own name
    (wire.py), and a client built here and never queried would leave it None."""
    c = TtsClient(spec)
    # All speakers come back flat (one (voice, speaker) pair per speaker): the
    # server's own max_speakers cap takes the FIRST N (speaker ids are
    # corpus-ordered, not representative), and this tool's whole point is a
    # representative sample, so spacing is done here, per voice.
    catalog = c.voices(languages=languages or None)
    if voice_filter:
        wanted = set(voice_filter)
        catalog = [v for v in catalog
                   if (v if isinstance(v, str) else v[0]) in wanted]
    if not c.speaker_voices:
        return [(v, None) for v in catalog], c
    groups = {}
    for v in catalog:
        if isinstance(v, (list, tuple)):
            groups.setdefault(v[0], []).append(v[1] if len(v) > 1 else None)
        else:
            groups.setdefault(v, []).append(None)
    out = []
    for voice, speakers in groups.items():
        nicks = [s for s in speakers if s is not None]
        if max_speakers and len(nicks) > max_speakers:
            # Evenly spaced rather than the first N: speaker ids are ordered by
            # the source corpus, so the head is not a representative sample.
            idx = np.linspace(0, len(nicks) - 1, max_speakers).astype(int)
            keep = set(nicks[i] for i in sorted(set(idx)))
            speakers = [s for s in speakers if s is None or s in keep]
        out.extend((voice, s) for s in speakers)
    return out, c


def tts_render(c: TtsClient, voice, speaker, text, speed):
    """Synthesize over the protocol; `speed` is the engine's own, no resampling.

    The old Wyoming path had no rate control and emulated speed by resampling
    (which also moves pitch - fine, because the point of several speeds is to
    deny the ASR's language model a comfortable rendering, not to model delivery
    rate faithfully). The protocol carries speed, so both engines get a real
    one.
    """
    voice = (voice, speaker) if speaker is not None else voice
    audio = c.render(voice, text, speed)
    return audio


# --------------------------------------------------------------------------

def final_token(text):
    """Last alphabetic token, lowercased.

    ASR invents leading filler on slow renderings ("Ahay Siri", "Ah hey Siri") but
    the word being tested is always last, so this ignores the noise and keeps the
    signal.
    """
    words = re.findall(r"[a-z']+", text.lower())
    return words[-1] if words else ""


def main():
    p = argparse.ArgumentParser(
        description="Screen TTS voices for mispronunciation of the wake word")
    p.add_argument("--wake-word", required=True)
    p.add_argument("--tts", required=True,
                   help="tcp:// URL of the TTS protocol server to audit "
                        "(the one the corpus runs against)")
    p.add_argument("--piper-speakers", type=int, default=12,
                   help="Speakers to sample per multi-speaker voice, evenly "
                        "spaced. 0 audits every one - 904 for libritts_r "
                        "(default: %(default)s)")
    p.add_argument("--languages", default="en_US,en_GB",
                   help="Language prefixes to include (default: %(default)s)")
    p.add_argument("--asr", default="localhost:10300",
                   help="Wyoming ASR service, host:port (default: %(default)s)")
    p.add_argument("--repeats", type=int, default=2,
                   help="Renderings per speed. Piper is stochastic - the same "
                        "speaker says it differently each call - so one pass "
                        "mislabels borderline voices (default: %(default)s)")
    p.add_argument("--speeds", default=",".join(str(s) for s in DEFAULT_SPEEDS),
                   help="Comma-separated render speeds. More speeds catch more "
                        "marginal voices; 1.0 alone misses them (default: %(default)s)")
    p.add_argument("--voices", default="",
                   help="Comma-separated subset of voice names to audit "
                        "(default: all matching --languages)")
    p.add_argument("--out-dir", default="voice_audit",
                   help="Where to write clips for the ear check (default: %(default)s)")
    args = p.parse_args()

    host, _, port = args.asr.partition(":")
    port = int(port or 10300)
    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]

    langs = tuple(l.strip() for l in args.languages.split(",") if l.strip())
    filters = ([v.strip() for v in args.voices.split(",") if v.strip()])
    try:
        targets, client = tts_targets(args.tts, langs, args.piper_speakers,
                                      filters)
    except Exception as e:
        print(f"cannot audit: {type(e).__name__}: {e}")
        return 1
    # tts_targets' client made the catalog request, so this is the engine's
    # own name (wire.py), not the protocol client's constant: a URL pointed
    # at the wrong server says so here, not at the end of the audit.
    engine = client.server_engine or client.name
    source = f"{args.tts} (engine={engine})"

    if not targets:
        print("No voices found.")
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reps = max(1, args.repeats)
    print(f"Auditing {len(targets)} voices x {len(speeds)} speeds x {reps} repeats "
          f"= {len(targets) * len(speeds) * reps} renderings")
    print(f"  TTS {source}   ASR {host}:{port}\n")

    results = {}
    for i, (name, speaker) in enumerate(targets, 1):
        voice = name if speaker is None else f"{name}:{speaker}"
        rows = []
        for speed in speeds:
          for rep in range(reps):
            try:
                audio = tts_render(client, name, speaker, args.wake_word, speed)
            except Exception as e:
                print(f"[{i}/{len(targets)}] {voice} @ {speed}x r{rep}: "
                      f"render failed ({type(e).__name__})")
                audio = np.zeros(0, dtype=np.int16)
            if len(audio) == 0:
                continue
            text = transcribe(audio, host, port)
            rows.append((speed, text))
            wav_path = out / f"{_wav_name(voice)}_{speed}_{rep:02d}.wav"
            scipy.io.wavfile.write(str(wav_path), SR, audio)
            print(f"[{i}/{len(targets)}] {voice} @ {speed}x r{rep}: {text!r}")
        results[voice] = rows

    if not results:
        print("Nothing was rendered.")
        return 1

    # Consensus over final tokens, weighted by speed x repeats.
    token_votes = Counter()
    for voice, rows in results.items():
        for _, text in rows:
            tok = final_token(text)
            if tok:
                token_votes[tok] += 1
    consensus, consensus_count = token_votes.most_common(1)[0]
    total = sum(token_votes.values())

    print("\nConsensus final token: %r  (%d/%d renderings)" %
          (consensus, consensus_count, total))
    print()

    bad = []
    for voice in sorted(results):
        toks = Counter(final_token(t) for _, t in results[voice])
        n = sum(toks.values())
        hits = toks.get(consensus, 0)
        pct = 100.0 * hits / n if n else 0.0
        other = ", ".join(f"{t!r}x{c}" for t, c in toks.most_common(3)
                          if t != consensus)
        flag = "OK  " if pct == 100.0 else "FLAG"
        print(f"  {flag} {voice:35s} {pct:5.1f}%   {other}")
        if pct < 100.0:
            bad.append(voice)

    if bad:
        print("\nPaste into train/corpus/piper.py (MISPRONOUNCING_VOICES):")
        print("    " + ",\n    ".join(repr(v) for v in bad))
    else:
        print("\nNo mispronouncing voices found.")

    return 0


def _wav_name(voice):
    return re.sub(r"[^A-Za-z0-9._-]", "_", voice)


if __name__ == "__main__":
    sys.exit(main() or 0)
