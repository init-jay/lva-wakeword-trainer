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
        --kokoro-url http://192.168.2.26:8880 --asr 192.168.2.14:10300

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
import requests
import scipy.io.wavfile
from scipy.signal import resample

SR = 16000
DEFAULT_SPEEDS = (0.75, 1.0, 1.3)


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
# Piper rendering, over Wyoming TTS
#
# Piper phonemises with espeak-ng, which is per-MODEL and not per-speaker, so every
# speaker inside one voice shares a pronunciation. That is the opposite of Kokoro,
# where the six bad voices each guessed differently. In practice it means auditing a
# 904-speaker model is mostly one decision about the model plus a sweep for speakers
# whose audio is simply bad - LibriTTS is scraped audiobook read speech and its
# per-speaker quality is uneven.
#
# --piper-speakers caps how many speakers of a multi-speaker voice get sampled, so a
# first pass over en_US-libritts_r-medium does not mean 904 x len(speeds) renderings.
# --------------------------------------------------------------------------

def piper_info(host, port):
    sock = socket.create_connection((host, port), timeout=30)
    try:
        sock.settimeout(30)
        _send(sock, "describe")
        event, _ = _read_event(sock, b"")
        return event["data"] if event else {}
    finally:
        sock.close()


def piper_voices(host, port, languages=("en_US", "en_GB"), max_speakers=0):
    """[(voice, speaker_or_None), ...] for the requested languages."""
    out = []
    for program in piper_info(host, port).get("tts", []):
        for voice in program.get("voices", []):
            langs = voice.get("languages") or [voice.get("language")]
            if languages and not any(str(l).startswith(tuple(languages)) for l in langs):
                continue
            speakers = [s.get("name") for s in (voice.get("speakers") or [])]
            if not speakers:
                out.append((voice["name"], None))
                continue
            if max_speakers and len(speakers) > max_speakers:
                # Evenly spaced rather than the first N: speaker ids are ordered by
                # the source corpus, so the head is not a representative sample.
                idx = np.linspace(0, len(speakers) - 1, max_speakers).astype(int)
                speakers = [speakers[i] for i in sorted(set(idx))]
            out.extend((voice["name"], s) for s in speakers)
    return out


def piper_render(host, port, voice, speaker, text, speed):
    """Synthesize via Wyoming TTS. `speed` is applied by resampling afterwards.

    Wyoming's synthesize event carries no rate control, so speed is emulated the
    same way asetrate does - which also moves pitch. That is acceptable here because
    the point of several speeds is to deny the ASR's language model a comfortable
    rendering to smooth over, not to model delivery rate faithfully.
    """
    sock = socket.create_connection((host, port), timeout=120)
    try:
        sock.settimeout(120)
        v = {"name": voice}
        if speaker is not None:
            v["speaker"] = str(speaker)
        _send(sock, "synthesize", {"text": text, "voice": v})
        buf, pcm, rate = b"", b"", 22050
        while True:
            event, buf, payload = _read_event_payload(sock, buf)
            if event is None:
                break
            if event["type"] in ("audio-start", "audio-chunk"):
                rate = event["data"].get("rate", rate)
                pcm += payload
            elif event["type"] == "audio-stop":
                break
    finally:
        sock.close()

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if len(audio) == 0:
        return np.zeros(0, dtype=np.int16)
    target = int(len(audio) * SR / rate / speed)
    audio = resample(audio, max(1, target))
    return np.clip(audio, -32768, 32767).astype(np.int16)


def _read_event_payload(sock, buf):
    """_read_event, but returning the audio payload rather than discarding it."""
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return None, buf, b""
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
    return {"type": header.get("type"), "data": data}, buf[p:], buf[:p]


# --------------------------------------------------------------------------
# Kokoro rendering
# --------------------------------------------------------------------------

def kokoro_voices(url):
    raw = requests.get(f"{url}/v1/audio/voices", timeout=30).json().get("voices", [])
    voices = [v["id"] if isinstance(v, dict) else v for v in raw]
    return sorted(v for v in voices if v.startswith(("af_", "am_", "bf_", "bm_")))


def kokoro_render(url, voice, text, speed):
    r = requests.post(f"{url}/v1/audio/speech",
                      json={"model": "kokoro", "voice": voice, "input": text,
                            "speed": speed, "response_format": "wav"}, timeout=120)
    r.raise_for_status()
    import io
    sr, data = scipy.io.wavfile.read(io.BytesIO(r.content))
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SR:
        data = resample(data.astype(np.float64), int(len(data) * SR / sr))
        data = np.clip(data, -32768, 32767)
    return data.astype(np.int16)


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
    p.add_argument("--tts", choices=("kokoro", "piper"), default="kokoro",
                   help="Which engine to audit (default: %(default)s)")
    p.add_argument("--kokoro-url", default="http://localhost:8880")
    p.add_argument("--piper", default="localhost:10200",
                   help="Wyoming TTS service for --tts piper (default: %(default)s)")
    p.add_argument("--piper-speakers", type=int, default=12,
                   help="Speakers to sample per multi-speaker Piper voice, evenly "
                        "spaced. 0 audits every one - 904 for libritts_r "
                        "(default: %(default)s)")
    p.add_argument("--languages", default="en_US,en_GB",
                   help="Piper language prefixes to include (default: %(default)s)")
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
                   help="Comma-separated subset to audit (default: all English)")
    p.add_argument("--out-dir", default="voice_audit",
                   help="Where to write clips for the ear check (default: %(default)s)")
    args = p.parse_args()

    host, _, port = args.asr.partition(":")
    port = int(port or 10300)
    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]

    if args.tts == "piper":
        p_host, _, p_port = args.piper.partition(":")
        p_port = int(p_port or 10200)
        langs = tuple(l.strip() for l in args.languages.split(",") if l.strip())
        if args.voices:
            # "voice:speaker" or bare "voice"
            targets = []
            for spec in args.voices.split(","):
                name, _, spk = spec.strip().partition(":")
                targets.append((name, spk or None))
        else:
            targets = piper_voices(p_host, p_port, langs, args.piper_speakers)
        render = lambda name, spk, speed: piper_render(
            p_host, p_port, name, spk, args.wake_word, speed)
        source = f"piper {p_host}:{p_port}"
    else:
        names = ([v.strip() for v in args.voices.split(",") if v.strip()]
                 or kokoro_voices(args.kokoro_url))
        targets = [(n, None) for n in names]
        render = lambda name, spk, speed: kokoro_render(
            args.kokoro_url, name, args.wake_word, speed)
        source = f"kokoro {args.kokoro_url}"

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
                audio = render(name, speaker, speed)
            except Exception as e:
                print(f"  {voice} @ {speed}: render failed ({e})")
                continue
            if len(audio) == 0:
                continue
            text = transcribe(audio, host, port)
            rows.append((speed, text, len(audio) / (SR / 1000)))
            safe = voice.replace(":", "-").replace("/", "-")
            suffix = f"{speed}" if reps == 1 else f"{speed}_{rep}"
            scipy.io.wavfile.write(str(out / f"{safe}_{suffix}.wav"), SR, audio)
        results[voice] = rows
        print(f"\r  {i}/{len(targets)} {voice:34s}", end="", flush=True)
    print("\r" + " " * 60 + "\r", end="")

    # Consensus is the most common final token across every rendering. It is what
    # the ASR reliably hears for a correct pronunciation - "siri" for "hey seeree" -
    # not the wake word itself, which it will never spell right.
    tokens = [final_token(t) for rows in results.values() for _, t, _ in rows]
    tokens = [t for t in tokens if t]
    if not tokens:
        print("ASR returned nothing for any voice - is the service reachable?")
        return 1
    consensus, n_consensus = Counter(tokens).most_common(1)[0]
    print(f'Consensus final token: "{consensus}" '
          f"({n_consensus}/{len(tokens)} renderings)\n")

    durations = [d for rows in results.values() for _, _, d in rows]
    median_ms = float(np.median(durations))

    scored = []
    for voice, rows in results.items():
        if not rows:
            continue
        hits = sum(1 for _, t, _ in rows if final_token(t) == consensus)
        scored.append((hits / len(rows), voice, rows))
    scored.sort(key=lambda r: (r[0], r[1]))

    width = max(16, min(34, max(len(v) for v in results)))
    print(f"{'voice':{width}s} {'agree':>6s} {'dur':>7s}  transcripts")
    print("-" * (width + 78))
    suspect = []
    for frac, voice, rows in scored:
        dur = np.median([d for _, _, d in rows])
        texts = " | ".join(t or "(silence)" for _, t, _ in rows)
        mark = "" if frac == 1.0 else ("  <-- BAD" if frac < 0.5 else "  <-- CHECK")
        if frac < 1.0:
            suspect.append(voice)
        print(f"{voice:{width}s} {frac*100:5.0f}% {dur:6.0f}ms  {texts[:58]:58s}{mark}")

    print(f"\nmedian duration {median_ms:.0f}ms. Clips written to {out}/")
    if suspect:
        print(f"\n{len(suspect)} voice(s) to check BY EAR - this is a screen, not a "
              f"verdict.\nListen to {out}/<voice>_*.wav, then paste the ones that are "
              f"genuinely wrong:\n")
        safe = args.wake_word.replace(" ", "_").lower()
        print(f'    "{safe}": [')
        print("        " + ", ".join(f'"{v}"' for v in sorted(suspect)) + ",")
        print("    ],")
    else:
        print("\nEvery voice matched consensus at every speed. Still worth spot-"
              "checking a few by ear before trusting a new engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
