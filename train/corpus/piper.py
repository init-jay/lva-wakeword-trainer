"""Piper sample generation over Wyoming TTS.

microWakeWord generates its positives with Piper, so this is the second engine the
shared corpus layer needs. It is deliberately shaped like the Kokoro path in
train.py - render a phrase at a spread of speeds across a spread of voices, write
16 kHz mono WAVs into a directory - so both trainers can consume either engine's
output, or both at once.

THE SPEED PROBLEM. Wyoming's `synthesize` event carries no rate control, so speed
has to be applied after synthesis. audit_voices.py:191 does it by resampling, which
moves pitch as well as rate - correct there, where the point is to deny the ASR a
comfortable rendering, and wrong here. `PLAIN_SPEEDS = (0.7, 1.6)` in train.py means
DELIVERY RATE: Kokoro's speed parameter re-times the phrase without turning the
speaker into a chipmunk, and the model is meant to learn that the phrase can be
said quickly, not that it can be said by someone with a shorter vocal tract. Pitch
is already covered, separately and on purpose, by add_child_range_copies.

So speed here goes through `time_stretch` (WSOLA, pitch-preserving) instead. Using
resampling would silently entangle the speed sweep with the child-range lever and
make run 13's result impossible to attribute.

PIPER IS STOCHASTIC. VITS samples noise and durations per call, so the same
(voice, speaker, text, speed) renders differently every time - measured in
audit_voices.py, where one speaker scored 0% and then 100% across passes. For a
corpus that is free diversity and needs no special handling. For the audit it is
why `--repeats` exists. It also means the corpus is not reproducible from a seed,
which is already true of the Kokoro path (train.py sets no seed).

EXERCISED AGAINST A LIVE SERVICE. piper_voices enumerated 2005 (voice, speaker)
pairs across en_US/en_GB, and piper_render returns 16 kHz int16 at exact speed
ratios (0.7000, 1.2501, 1.6001 measured on one clip). Not yet used to build a
training corpus - that is --piper-fraction, staged as run 17 in tuning.md.

MEASURE SPEED ON ONE CLIP, NOT ACROSS CALLS. Three separate renderings at 1.0/1.6/0.7
gave 0.964 s / 0.501 s / 1.194 s, which looks wrong and is not: VITS samples
durations per call, so the base clip differs every time. Stretch ratios are only
meaningful against a single rendering.
"""

import json
import socket
import sys
import uuid
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly
from tqdm import tqdm

from .augment import time_stretch

SR = 16000

# Voices that mispronounce the wake word, per wake word. THE PIPER EQUIVALENT OF
# MISPRONOUNCING_VOICES IN corpus/negatives.py, AND IT IS NOT OPTIONAL.
#
# Six of Kokoro's 42 voices say something other than "hey seeree" - ~14% of that
# corpus mislabelled as positives, undetected for eleven runs.
#
# THE EXCLUSION UNIT IS THE SPEAKER, NOT THE MODEL. The expectation going in was the
# opposite: espeak-ng phonemises per model, so every speaker inside one voice gets
# the same phoneme string, and it seemed to follow that they would all pronounce it
# the same. The audit says otherwise - en_US-l2arctic-medium ranges from :ASI at 0%
# to :PNV at 100%. Identical phonemes, different acoustic models, and intelligibility
# varies with the speaker. Keys here are therefore "voice:speaker" wherever the audit
# scored a speaker.
#
# Below is the 2026-09-02 audit of 96 voices against ASR consensus, everything under
# 83% agreement. The transcripts make the distinction clear: excluded voices produce
# a CONSISTENTLY different phrase ("his theory", four renderings running), while the
# ones kept produce "hey siri" with an occasional slip, which is VITS sampling
# durations per call rather than a pronunciation problem.
#
# en_US-l2arctic-medium is a non-native-speaker corpus and 7 of its 12 audited
# speakers land here. That is not automatically a reason to exclude - accented
# renderings of the CORRECT phrase are good training data, since real users have
# accents. It is a reason here because the benefit cannot be measured: there is no
# accented speaker in my_real_samples_holdout/, so the contamination is measurable
# and the upside is not. Revisit if an accented speaker is ever recorded.
#
# WHAT THIS METHOD CANNOT SEE: the score is agreement with the consensus ACROSS
# voices, so an error every voice shares is invisible. Every good voice here
# transcribes as "Hey Siri"; if espeak-ng renders "seeree" as /'sIri/ rather than
# /si:'ri:/, all 96 are uniformly wrong and all 96 score 100%. Check that by ear
# against a real recording, once, per wake word - not from this table.
MISPRONOUNCING_PIPER_VOICES: dict[str, list[str]] = {
    "hey_seeree": [
        # < 50% - consistently a different phrase
        "en_US-l2arctic-medium:ASI",            # 0%   "Here's your week" / "Peace, Yuri"
        "en_GB-southern_english_female-low",    # 17%  "Hey, Sirius" / "Paisiru"
        "en_US-l2arctic-medium:BWC",            # 17%  "His theory" x4
        "en_US-l2arctic-medium:YBAA",           # 17%  "He's silly" / "History"
        "en_US-l2arctic-medium:LXC",            # 33%  "his theory" x3
        "en_US-l2arctic-medium:YKWK",           # 33%  "K-series" / "Here's theory"
        # 50-67% - right more often than not, still ~1 bad clip in 3
        "en_US-arctic-medium:slp",              # 50%  "He's a re" / "Hesiery"
        "en_US-l2arctic-medium:HQTV",           # 50%  "History" / "Peace, Siri"
        "en_US-l2arctic-medium:SVBI",           # 50%  "He's sorry" / "He's silly"
        "en_US-arctic-medium:aup",              # 67%
        "en_US-danny-low",                      # 67%
        "en_US-l2arctic-medium:HKK",            # 67%  "STAE" / "A-Siri"
        "en_US-l2arctic-medium:SKA",            # 67%  "Case Theory"
        "en_US-l2arctic-medium:TXHC",           # 67%  "Hey, see you, Rhi!"
    ],
}

# Voices excluded because they were NEVER AUDITED, not because they are wrong.
#
# Kept separate from MISPRONOUNCING_PIPER_VOICES deliberately. Everything in that
# list was measured and failed; everything here is simply unknown, and merging the
# two would destroy the only record of which is which - so a later reader would have
# no way to tell that these are cheap to reclaim.
#
# HOW THEY GOT HERE. The 2026-09-02 audit ran against one Piper instance and covered
# 96 voices. Corpus generation later ran against the compose `piper` service, which
# exposes 106. The extra ten arrived unaudited and unmapped, and contributed 600 of
# 5520 synthetic positives - 10.9% of the Piper corpus, from voices whose
# pronunciation had never been checked. The Kokoro equivalent was ~14% and went
# unnoticed for eleven runs, so this is the same failure caught earlier.
#
# THE REAL LESSON IS THE MISMATCH: audit and generation must talk to the SAME Piper
# service. An audit of a different instance is only accidentally relevant.
#
# TO RECLAIM THEM: audit these ten against the instance that generates the corpus,
# then move them into MISPRONOUNCING_PIPER_VOICES or delete them from here, and add
# their F0 to PIPER_VOICE_SEX. Note cori and ljspeech appear at two qualities each,
# so this is eight distinct voices, and quality variants share a phonemisation but
# not an acoustic model - l2arctic ranged 0-100% across speakers on identical
# phonemes, so do not assume -high and -medium agree.
UNAUDITED_PIPER_VOICES: dict[str, list[str]] = {
    "hey_seeree": [
        "en_GB-cori-high",
        "en_GB-cori-medium",
        "en_US-bryce-medium",
        "en_US-john-medium",
        "en_US-kristin-medium",
        "en_US-ljspeech-high",
        "en_US-ljspeech-medium",
        "en_US-norman-medium",
        "en_US-reza_ibrahim-medium",
        "en_US-sam-medium",
    ],
}

# Voice sex, for the child-range lever (corpus/augment.py). Keys are the voice name,
# or "voice:speaker" for a multi-speaker model.
#
# MEASURED, NOT LISTENED TO. Generated by measure_voice_f0.py from the 1.0x audit
# clips: median F0 per voice, split at 185 Hz. Regenerate it for a new engine or a
# new voice set rather than extending it by ear - 96 entries is more listening than
# anyone will actually do, and skipping it silently costs the run-13 lever its reach.
#
# Validated against the ten voices whose NAME states the answer - hfc_male 147 Hz,
# hfc_female 268 Hz, northern_english_male 117 Hz, southern_english_female 248 Hz,
# joe 116 Hz, ryan 162 Hz, amy 195 Hz, alba 190 Hz, jenny 205 Hz, lessac 231 Hz.
# All ten agree with the split.
#
# THE 160-200 Hz BAND IS GENUINELY AMBIGUOUS (vctk:p239 184 Hz, p288 184, p293 182,
# kathleen 177) and it does not matter much: sex here is only a proxy for F0, and the
# two ratio ranges nearly coincide at the boundary. At 177 Hz the male range gives
# 204-230 Hz and the female range 212-239 Hz. The ranges were calibrated against
# am_adam at 132 Hz and af_bella at 227 Hz, so they are least distinguishable exactly
# where the classification is least certain.
#
# A voice absent from this map is written piper_pu_* and gets NO child-range copy.
# That is deliberate: run 12 measured male voices as "useless above R1.30
# (chipmunk)", so a wrongly-shifted clip is worse than an absent one - training on an
# artefact teaches the artefact.
PIPER_VOICE_SEX: dict[str, str] = {
    "en_GB-alan-low": "m",  # 98 Hz
    "en_GB-alan-medium": "m",  # 93 Hz
    "en_GB-alba-medium": "f",  # 190 Hz
    "en_GB-aru-medium:01": "m",  # 111 Hz
    "en_GB-aru-medium:02": "m",  # 104 Hz
    "en_GB-aru-medium:03": "f",  # 207 Hz
    "en_GB-aru-medium:04": "f",  # 226 Hz
    "en_GB-aru-medium:05": "m",  # 120 Hz
    "en_GB-aru-medium:06": "m",  # 145 Hz
    "en_GB-aru-medium:07": "f",  # 214 Hz
    "en_GB-aru-medium:08": "f",  # 215 Hz
    "en_GB-aru-medium:09": "m",  # 149 Hz
    "en_GB-aru-medium:10": "m",  # 154 Hz
    "en_GB-aru-medium:11": "f",  # 215 Hz
    "en_GB-aru-medium:12": "m",  # 129 Hz
    "en_GB-jenny_dioco-medium": "f",  # 205 Hz
    "en_GB-northern_english_male-medium": "m",  # 117 Hz
    "en_GB-semaine-medium:obadiah": "m",  # 116 Hz
    "en_GB-semaine-medium:poppy": "f",  # 229 Hz
    "en_GB-semaine-medium:prudence": "f",  # 226 Hz
    "en_GB-semaine-medium:spike": "m",  # 113 Hz
    "en_GB-southern_english_female-low": "f",  # 248 Hz
    "en_GB-vctk-medium:p239": "m",  # 184 Hz
    "en_GB-vctk-medium:p241": "m",  # 114 Hz
    "en_GB-vctk-medium:p253": "f",  # 221 Hz
    "en_GB-vctk-medium:p273": "m",  # 150 Hz
    "en_GB-vctk-medium:p286": "m",  # 141 Hz
    "en_GB-vctk-medium:p288": "m",  # 184 Hz
    "en_GB-vctk-medium:p293": "m",  # 182 Hz
    "en_GB-vctk-medium:p294": "m",  # 167 Hz
    "en_GB-vctk-medium:p299": "m",  # 159 Hz
    "en_GB-vctk-medium:p307": "f",  # 229 Hz
    "en_GB-vctk-medium:p334": "m",  # 95 Hz
    "en_GB-vctk-medium:p362": "f",  # 210 Hz
    "en_US-amy-low": "f",  # 208 Hz
    "en_US-amy-medium": "f",  # 195 Hz
    "en_US-arctic-medium:aew": "m",  # 121 Hz
    "en_US-arctic-medium:aup": "m",  # 161 Hz
    "en_US-arctic-medium:awb": "m",  # 139 Hz
    "en_US-arctic-medium:axb": "f",  # 241 Hz
    "en_US-arctic-medium:bdl": "m",  # 134 Hz
    "en_US-arctic-medium:clb": "m",  # 180 Hz
    "en_US-arctic-medium:fem": "m",  # 115 Hz
    "en_US-arctic-medium:gka": "m",  # 142 Hz
    "en_US-arctic-medium:ksp": "m",  # 134 Hz
    "en_US-arctic-medium:rms": "m",  # 99 Hz
    "en_US-arctic-medium:rxr": "m",  # 164 Hz
    "en_US-arctic-medium:slp": "f",  # 238 Hz
    "en_US-danny-low": "m",  # 133 Hz
    "en_US-hfc_female-medium": "f",  # 268 Hz
    "en_US-hfc_male-medium": "m",  # 147 Hz
    "en_US-joe-medium": "m",  # 116 Hz
    "en_US-kathleen-low": "m",  # 177 Hz
    "en_US-kusal-medium": "m",  # 98 Hz
    "en_US-l2arctic-medium:ASI": "m",  # 159 Hz
    "en_US-l2arctic-medium:BWC": "m",  # 109 Hz
    "en_US-l2arctic-medium:ERMS": "m",  # 109 Hz
    "en_US-l2arctic-medium:HKK": "m",  # 115 Hz
    "en_US-l2arctic-medium:HQTV": "f",  # 192 Hz
    "en_US-l2arctic-medium:LXC": "f",  # 224 Hz
    "en_US-l2arctic-medium:PNV": "f",  # 187 Hz
    "en_US-l2arctic-medium:SKA": "f",  # 208 Hz
    "en_US-l2arctic-medium:SVBI": "f",  # 237 Hz
    "en_US-l2arctic-medium:TXHC": "m",  # 141 Hz
    "en_US-l2arctic-medium:YBAA": "m",  # 158 Hz
    "en_US-l2arctic-medium:YKWK": "m",  # 145 Hz
    "en_US-lessac-high": "f",  # 220 Hz
    "en_US-lessac-low": "f",  # 233 Hz
    "en_US-lessac-medium": "f",  # 231 Hz
    "en_US-libritts-high:p1271": "m",  # 134 Hz
    "en_US-libritts-high:p1311": "m",  # 106 Hz
    "en_US-libritts-high:p1779": "f",  # 224 Hz
    "en_US-libritts-high:p2012": "m",  # 106 Hz
    "en_US-libritts-high:p2085": "m",  # 181 Hz
    "en_US-libritts-high:p3025": "f",  # 234 Hz
    "en_US-libritts-high:p335": "m",  # 181 Hz
    "en_US-libritts-high:p3922": "f",  # 186 Hz
    "en_US-libritts-high:p6686": "f",  # 193 Hz
    "en_US-libritts-high:p8113": "f",  # 207 Hz
    "en_US-libritts-high:p8474": "m",  # 104 Hz
    "en_US-libritts-high:p8677": "m",  # 174 Hz
    "en_US-libritts_r-medium:1241": "f",  # 195 Hz
    "en_US-libritts_r-medium:1271": "m",  # 133 Hz
    "en_US-libritts_r-medium:1311": "m",  # 104 Hz
    "en_US-libritts_r-medium:1379": "m",  # 163 Hz
    "en_US-libritts_r-medium:1779": "f",  # 258 Hz
    "en_US-libritts_r-medium:2012": "m",  # 119 Hz
    "en_US-libritts_r-medium:2085": "m",  # 181 Hz
    "en_US-libritts_r-medium:2137": "m",  # 129 Hz
    "en_US-libritts_r-medium:3025": "f",  # 245 Hz
    "en_US-libritts_r-medium:3922": "f",  # 188 Hz
    "en_US-libritts_r-medium:8113": "f",  # 219 Hz
    "en_US-libritts_r-medium:8474": "m",  # 117 Hz
    "en_US-ryan-high": "f",  # 204 Hz
    "en_US-ryan-low": "m",  # 172 Hz
    "en_US-ryan-medium": "m",  # 162 Hz
}


def voice_sex(voice: str, speaker=None) -> str:
    """'f', 'm', or 'u' (unknown) for the child-range lever.

    Checked most specific first: a multi-speaker model can hold both sexes, so a
    "voice:speaker" entry must win over the model-wide one.
    """
    if speaker is not None:
        specific = PIPER_VOICE_SEX.get(f"{voice}:{speaker}")
        if specific:
            return specific
    return PIPER_VOICE_SEX.get(voice, "u")

# The Wyoming JSONL-over-TCP framing: one JSON header line per event, then
# `data_length` bytes of JSON and `payload_length` bytes of audio.
#
# Duplicated from audit_voices.py rather than shared. That script is standalone by
# design - it runs on the host against live TTS and ASR services and takes no
# dependency on this package - and collapsing the two would drag the corpus layer
# into it. Worth revisiting if a third caller appears.


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
    """Next event, plus the remaining buffer and this event's audio payload."""
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


def piper_voices(host, port, languages=("en_US", "en_GB"), max_speakers=0):
    """[(voice, speaker_or_None), ...] for the requested languages.

    `max_speakers` caps how many speakers of a multi-speaker model are sampled -
    en_US-libritts_r-medium alone carries 904, and taking all of them would swamp
    the corpus with one model's phonemisation. The sample is evenly spaced rather
    than the first N, because speaker ids are ordered by the source corpus and the
    head of that list is not representative.
    """
    sock = socket.create_connection((host, port), timeout=30)
    try:
        sock.settimeout(30)
        _send(sock, "describe")
        event, _, _ = _read_event(sock, b"")
        info = event["data"] if event else {}
    finally:
        sock.close()

    out = []
    for program in info.get("tts", []):
        for voice in program.get("voices", []):
            langs = voice.get("languages") or [voice.get("language")]
            if languages and not any(str(l).startswith(tuple(languages)) for l in langs):
                continue
            speakers = [s.get("name") for s in (voice.get("speakers") or [])]
            if not speakers:
                out.append((voice["name"], None))
                continue
            if max_speakers and len(speakers) > max_speakers:
                idx = np.linspace(0, len(speakers) - 1, max_speakers).astype(int)
                speakers = [speakers[i] for i in sorted(set(idx))]
            out.extend((voice["name"], s) for s in speakers)
    return out


def piper_render(host, port, voice, speaker, text, speed=1.0):
    """Synthesize one phrase, returned as 16 kHz mono int16.

    `speed` > 1 is faster. Applied with time_stretch after resampling to 16 kHz, so
    it changes delivery rate without moving pitch - see the module docstring.
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
            event, buf, payload = _read_event(sock, buf)
            if event is None:
                break
            if event["type"] in ("audio-start", "audio-chunk"):
                rate = event["data"].get("rate", rate)
                pcm += payload
            elif event["type"] == "audio-stop":
                break
    finally:
        sock.close()

    audio = np.frombuffer(pcm, dtype=np.int16)
    if audio.size == 0:
        return np.zeros(0, dtype=np.int16)

    if rate != SR:
        # resample_poly rather than resample: rational up/down, no FFT-length
        # sensitivity, and it is what vocal_tract_shift already uses.
        from fractions import Fraction
        frac = Fraction(SR, int(rate)).limit_denominator(1000)
        audio = resample_poly(audio.astype(np.float64), frac.numerator, frac.denominator)

    if abs(speed - 1.0) > 1e-3:
        # speed 1.6 = 1.6x faster = 1/1.6 the duration.
        audio = time_stretch(np.asarray(audio, dtype=np.float64), 1.0 / float(speed), sr=SR)

    return np.clip(audio, -32768, 32767).astype(np.int16)


def generate_piper_samples(host, port, voices, output_dir: Path,
                           samples_per_voice: int, texts, speeds, desc="Piper"):
    """Render `samples_per_voice` clips for each voice into `output_dir`.

    Signature and sampling deliberately mirror train.py's generate_kokoro_samples,
    so the two are substitutable clip-for-clip: same per-voice budget, same
    text-offset-per-voice (without which every voice renders texts[0:n] and a list
    longer than the budget never gets past its own beginning), and the speed drawn
    from the same grid, in the job-building loop rather than in a worker, so the
    corpus does not depend on thread scheduling.

    `speeds` is passed in rather than imported: PLAIN_SPEED_GRID lives in train.py
    and this package must not depend on it.

    Filenames are `piper_{voice}_{speaker}_{uuid}.wav`. Note that this does NOT
    match the `kokoro_`/`runon_` prefixes add_child_range_copies looks for, so Piper
    clips are skipped by the child-range lever rather than mis-shifted - Piper voice
    names carry no sex marker to pick a ratio from. See corpus/augment.py.

    VOICE IS THE OUTER LOOP ON PURPOSE - DO NOT REORDER. wyoming-piper holds exactly
    one loaded voice in a module-level global and reloads it whenever a request names
    a different one (handler.py:333-346, `if voice_name != _VOICE_NAME`). Iterating
    texts or speeds outside voices would rebuild the InferenceSession on every single
    request - and under --use-cuda that means a fresh CUDA session each time, which
    is far more expensive than the synthesis itself.

    The same global is why one server serves strictly one request at a time, and why
    client concurrency measured as pure queueing (docker-compose.yml). Parallelism
    has to come from separate instances, each with its own voice - which also means
    sharding a multi-voice corpus BY VOICE across instances, never round-robin.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for v, (voice, speaker) in enumerate(voices):
        for i in range(samples_per_voice):
            text = texts[(v * samples_per_voice + i) % len(texts)]
            speed = float(np.random.choice(speeds))
            jobs.append((voice, speaker, text, speed))

    written = 0
    unknown_sex = set()
    for voice, speaker, text, speed in tqdm(jobs, desc=desc, unit="clip"):
        try:
            audio = piper_render(host, port, voice, speaker, text, speed)
        except Exception as e:
            print(f"  Error rendering {voice}/{speaker} at {speed}x: {e}")
            continue
        if audio.size < 480:
            continue

        # piper_p{sex}_{voice}[_{speaker}]_{uuid}.wav
        #
        # The `p{sex}` group is second on purpose: add_child_range_copies reads the
        # sex from parts[1][1], which is where Kokoro's af_/am_ prefix puts it. Same
        # position, same code, no special case for the engine.
        sex = voice_sex(voice, speaker)
        if sex == "u":
            unknown_sex.add(voice if speaker is None else f"{voice}:{speaker}")
        tag = f"{voice}_{speaker}" if speaker is not None else voice
        name = f"piper_p{sex}_{tag}_{uuid.uuid4().hex[:8]}.wav".replace("/", "_")
        scipy.io.wavfile.write(str(output_dir / name), SR, audio)
        written += 1

    print(f"  Wrote {written} Piper clips from {len(jobs)} jobs")
    if unknown_sex:
        print(f"  WARNING: {len(unknown_sex)} voice(s) have no sex in "
              f"PIPER_VOICE_SEX, so their clips get NO child-range copy: "
              f"{', '.join(sorted(unknown_sex)[:6])}"
              f"{' ...' if len(unknown_sex) > 6 else ''}")
    return written


def select_piper_voices(host, port, wake_word: str, languages=("en_US", "en_GB"),
                        max_speakers: int = 12) -> list:
    """Enumerate Piper voices, drop the ones that say the wrong thing, report cover.

    The exclusion step is the whole point. Six of 42 Kokoro voices mispronounce
    "hey seeree" and that was ~14% of the synthetic corpus mislabelled as positives
    for eleven runs before anyone noticed. Piper is not exempt, and with 84 voices
    available an unaudited list is a bigger exposure, not a smaller one.
    """
    safe_name = wake_word.replace(" ", "_").lower()
    try:
        found = piper_voices(host, int(port), languages=tuple(languages),
                             max_speakers=max_speakers)
    except Exception as e:
        print(f"  ERROR: could not reach Piper at {host}:{port}: {e}")
        print("         Start it with `docker compose up -d piper`.")
        sys.exit(1)

    bad = set(MISPRONOUNCING_PIPER_VOICES.get(safe_name, []))
    unaudited = set(UNAUDITED_PIPER_VOICES.get(safe_name, []))
    excluded = bad | unaudited
    if not bad:
        print(f"  WARNING: no MISPRONOUNCING_PIPER_VOICES entry for '{safe_name}'.")
        print("           Nothing has been excluded, so any voice whose espeak-ng")
        print("           g2p guesses the wake word wrong is contributing")
        print("           MISLABELLED POSITIVES. Six of 42 Kokoro voices did exactly")
        print("           that (~14% of that corpus). Run audit_voices.py --tts piper,")
        print("           listen to the shortlist, and fill the list in.")

    # Match both forms. The audit scores SPEAKERS - en_US-l2arctic-medium ran from
    # :ASI at 0% to :PNV at 100% on identical phonemes - so most entries are
    # "voice:speaker". A bare voice name still excludes the whole model.
    def is_excluded(v, s):
        return v in excluded or f"{v}:{s}" in excluded

    kept = [(v, s) for (v, s) in found if not is_excluded(v, s)]
    n_bad = sum(1 for v, s in found if v in bad or f"{v}:{s}" in bad)
    n_unaudited = sum(1 for v, s in found
                      if v in unaudited or f"{v}:{s}" in unaudited)

    print(f"  Piper voices: {len(kept)} of {len(found)} "
          f"({n_bad} mispronouncing, {n_unaudited} unaudited)")

    # A voice the service offers that appears in NEITHER list has never been checked
    # and is not being excluded - which is the exact hole the ten unaudited voices
    # fell through. Say so loudly rather than letting it show up later as a
    # child-range coverage number.
    unknown = sum(1 for v, s in kept if voice_sex(v, s) == "u")
    if unknown:
        names = sorted({v if s is None else f"{v}:{s}"
                        for v, s in kept if voice_sex(v, s) == "u"})
        print(f"  WARNING: {unknown} kept voice(s) are in no list and have no F0 -")
        print("           unaudited AND unmapped, so they contribute possibly")
        print("           mislabelled positives and get no child-range copy:")
        print(f"           {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
        print("           Audit them against THIS Piper instance, or add them to")
        print("           UNAUDITED_PIPER_VOICES in corpus/piper.py.")
    return kept
