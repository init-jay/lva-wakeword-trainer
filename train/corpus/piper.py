"""Piper sample generation for the wake-word corpus (policy layer over tts-protocol).

microWakeWord generates its positives with Piper, so this is the second engine the
shared corpus layer needs. Deliberately shaped like the Kokoro path in train.py -
phrase at a spread of speeds across a spread of voices, 16 kHz mono WAVs into a
directory - so both trainers can consume either engine's output, or both at once.

SPLIT, 2026-09-08: the transport (the Wyoming framing, and since this same date
the in-process variant - see below) moved out of this package. This module speaks
the repo's TTS PROTOCOL (tts-service/tts_protocol/) as a TCP client: it points at
a `tcp://` URL (the protocol port a Piper engine publishes) and does not care
what the server runs behind it. What stays here is wake-word TRAINING POLICY:
the exclusion tables, the sex table, voice selection, and the corpus generator.
A new engine does not get these tables; a new wake word does.

TWO MACHINES, ONE ENGINE: on a Mac it is the `uv` project
(`uv run --project tts-service/engines/piper python -m piper_engine`, piper-tts
1.7.0 loaded directly, no Wyoming at all); on the CUDA box the Docker image
(docker/Dockerfile.piper) bakes in that SAME project - one code path, one G2P
pin, both machines render from it. Both speak the identical protocol, so the
code in this file is the same on both machines - only the URL differs. The
protocol's own justifications (THE SPEED PROBLEM - why speed is WSOLA, not
resampling; PIPER IS STOCHASTIC; the one-server-one-request constraint) live in
that engine's docstring (tts-service/engines/piper), because the code they
explain lives there.
"""

import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

from tts_protocol.audio import SR  # noqa: E402  (re-exported name, now defined once)
from tts_protocol.client import TtsClient

# SR = 16000 lives in tts_protocol.audio; the name stays public from this module.

# One TtsClient per tcp:// URL, shared across calls (the client caches the
# voice catalog itself).
_CLIENTS = {}
_CLIENTS_LOCK = threading.Lock()


def _client(piper_url: str) -> TtsClient:
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(piper_url)
        if c is None:
            c = TtsClient(piper_url)
            _CLIENTS[piper_url] = c
        return c


def _piper_render(piper_url, voice, speaker, text, speed=1.0):
    """One Piper clip over the protocol. Raises on failure (the PIPER
    convention: the error envelope names the voice) - the caller reports which
    (voice, speaker) failed."""
    v = (voice, speaker) if speaker is not None else voice
    audio, _ = _client(piper_url).timed_render(v, text, speed)
    return audio


# Voices that mispronounce the wake word, per wake word. THE PIPER EQUIVALENT OF
# MISPRONOUNCING_VOICES in corpus/negatives.py, AND IT IS NOT OPTIONAL.
#
# Six of Kokoro's 42 voices say something other than "hey seeree" - ~14% of that
# corpus mislabelled as positives, undetected for eleven runs.
#
# THE EXCLUSION UNIT IS THE SPEAKER, NOT THE MODEL. The expectation was the opposite:
# espeak-ng phonemises per model, so every speaker inside one voice gets the same
# phoneme string and seemed bound to pronounce it the same. The audit says
# otherwise - en_US-l2arctic-medium ranges from :ASI at 0% to :PNV at 100%: identical
# phonemes, different acoustic models. Keys are therefore "voice:speaker" wherever
# the audit scored a speaker.
#
# Below is the 2026-09-02 audit of 96 voices against ASR consensus, everything under
# 83% agreement. The transcripts make the distinction: excluded voices produce a
# CONSISTENTLY different phrase ("his theory", four renderings running), while the
# ones kept produce "hey siri" with an occasional slip - VITS sampling durations per
# call, not a pronunciation problem.
#
# en_US-l2arctic-medium is a non-native-speaker corpus and 7 of its 12 audited
# speakers land here. Not automatically disqualifying - accented renderings of the
# CORRECT phrase are good training data, since real users have accents. Disqualified
# here because the benefit cannot be measured: there is no accented speaker in
# data/recordings/holdout/, so the contamination is measurable and the upside is not.
# Revisit if an accented speaker is ever recorded.
#
# WHAT THIS METHOD CANNOT SEE: the score is agreement with the consensus ACROSS
# voices, so an error every voice shares is invisible. Every good voice here
# transcribes as "Hey Siri"; if espeak-ng renders "seeree" as /'sIri/ rather than
# /si:'ri:/, all 96 are uniformly wrong and all score 100%. Check that by ear against
# a real recording, once, per wake word - not from this table.
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
# Kept separate from MISPRONOUNCING_PIPER_VOICES deliberately: everything in that
# list was measured and failed; everything here is simply unknown, and merging the
# two would destroy the only record of which is which - i.e. that these are cheap to
# reclaim.
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
# THE CATALOG GROWS, IN BOTH UNITS AT ONCE. Measured 2026-09-07 against the 2.4.3
# wheel (identical in the Docker image and the host venv, scripts/start-piper-host.sh):
# 163 voices in the bundled catalog, against 96 at audit time and the 106 the compose
# service exposed. With units: 96 and 163 are VOICE counts; 106 was a PAIR count. The
# default selection (en_US/en_GB, 12-speaker cap) measures 37 voices / 2005 pairs raw
# / 106 pairs capped - the same 106 the compose-era run saw, so the English selection
# set has not moved since the audit era's known exposure. The growth is voices in
# other languages, which the languages filter already excludes. Widening
# --piper-languages is a new unaudited set until tools/audit_voices.py has
# run against the instance that generates the corpus.
#
# TO RECLAIM THEM: audit these ten against the instance that generates the corpus,
# then move them into MISPRONOUNCING_PIPER_VOICES or delete them, and add their F0 to
# PIPER_VOICE_SEX. Note cori and ljspeech appear at two qualities each, so this is
# eight distinct voices, and quality variants share a phonemisation but not an
# acoustic model - l2arctic ranged 0-100% across speakers on identical phonemes, so
# do not assume -high and -medium agree.
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
# clips: median F0 per voice, split at 185 Hz. Regenerate it for a new engine or
# voice set rather than extending by ear - 96 entries is more listening than anyone
# will actually do, and skipping it silently costs the run-13 lever its reach.
#
# Validated against the ten voices whose NAME states the answer (hfc_male 147 Hz,
# hfc_female 268, northern_english_male 117, southern_english_female 248, joe 116,
# ryan 162, amy 195, alba 190, jenny 205, lessac 231 Hz). All ten agree with the split.
#
# THE 160-200 Hz BAND IS GENUINELY AMBIGUOUS (vctk:p239 184, p288 184, p293 182,
# kathleen 177) and it does not matter much: sex is only a proxy for F0, and the two
# ratio ranges nearly coincide at the boundary (at 177 Hz the male range gives
# 204-230 Hz, the female 212-239). The ranges were calibrated against am_adam at
# 132 Hz and af_bella at 227 Hz, so they are least distinguishable exactly where the
# classification is least certain.
#
# A voice absent from this map is written piper_pu_* and gets NO child-range copy.
# Deliberate: run 12 measured male voices as "useless above R1.30 (chipmunk)", so a
# wrongly-shifted clip is worse than an absent one - training on an artefact teaches
# the artefact.
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


def generate_piper_samples(piper_url: str, voices, output_dir: Path,
                           samples_per_voice: int, texts, speeds, desc="Piper"):
    """Render `samples_per_voice` clips for each voice into `output_dir`.

    `piper_url` is a `tcp://` protocol URL - the port a Piper engine publishes
    (the in-process server on a Mac, the wrapped Wyoming service in Docker).

    Signature and sampling deliberately mirror train.py's generate_kokoro_samples,
    so the two are substitutable clip-for-clip: same per-voice budget, same
    text-offset-per-voice (without which every voice renders texts[0:n] and a list
    longer than the budget never gets past its own beginning), and the speed drawn
    from the same grid, in the job-building loop rather than a worker, so the corpus
    does not depend on thread scheduling.

    `speeds` is passed in rather than imported: PLAIN_SPEED_GRID lives in train.py
    and this package must not depend on it.

    Filenames are `piper_p{sex}_{voice}[_{speaker}]_{uuid}.wav`. This does NOT match
    the `kokoro_`/`runon_` prefixes add_child_range_copies looks for, so Piper clips
    are skipped by the child-range lever rather than mis-shifted - Piper voice names
    carry no sex marker to pick a ratio from. See corpus/augment.py.

    VOICE IS THE OUTER LOOP ON PURPOSE - DO NOT REORDER. Every Piper server holds
    exactly one loaded voice at a time and reloads it when a request names a
    different one: the Wyoming server in a module-level global (handler.py:333-346,
    `if voice_name != _VOICE_NAME`), the in-process server in its single kept model.
    Iterating texts or speeds outside voices would rebuild the InferenceSession on
    every request - under --use-cuda a fresh CUDA session each time, far more
    expensive than the synthesis itself.

    The same one-voice-at-a-time property is why one server serves strictly one
    request at a time, and why client concurrency measured as pure queueing
    (docker-compose.yml). Parallelism has to come from separate instances, each
    with its own voice - which means sharding a multi-voice corpus BY VOICE across
    instances, never round-robin.
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
            audio = _piper_render(piper_url, voice, speaker, text, speed)
        except Exception as e:
            print(f"  Error rendering {voice}/{speaker} at {speed}x: {e}")
            continue
        if audio is None or audio.size < 480:
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


def select_piper_voices(piper_url: str, wake_word: str, languages=("en_US", "en_GB"),
                        max_speakers: int = 12) -> list:
    """Enumerate Piper voices from the `tcp://` server, drop the ones that say
    the wrong thing, report cover.

    The exclusion step is the whole point. Six of 42 Kokoro voices mispronounce
    "hey seeree" and that was ~14% of the synthetic corpus mislabelled as positives
    for eleven runs before anyone noticed. Piper is not exempt, and with 84 voices
    available an unaudited list is a bigger exposure, not a smaller one.
    """
    safe_name = wake_word.replace(" ", "_").lower()
    try:
        found = _client(piper_url).voices(languages=tuple(languages),
                                          max_speakers=max_speakers)
    except Exception as e:
        print(f"  ERROR: could not reach the Piper protocol server at {piper_url}: {e}")
        print("         Mac:  `uv run --project tts-service/engines/piper "
              "python -m piper_engine --port 8898`")
        print("         Docker: `docker compose up -d piper` (publishes 8898),")
        print("         then point --piper-url at tcp://127.0.0.1:8898.")
        sys.exit(1)

    bad = set(MISPRONOUNCING_PIPER_VOICES.get(safe_name, []))
    unaudited = set(UNAUDITED_PIPER_VOICES.get(safe_name, []))
    excluded = bad | unaudited
    if not bad:
        print(f"  WARNING: no MISPRONOUNCING_PIPER_VOICES entry for '{safe_name}'.")
        print("           Nothing has been excluded, so any voice whose espeak-ng")
        print("           g2p guesses the wake word wrong is contributing")
        print("           MISLABELLED POSITIVES. Six of 42 Kokoro voices did exactly")
        print("           that (~14% of that corpus). Run audit_voices.py --tts",)
        print("           tcp://<that instance>, listen to the shortlist, and fill the list in.")

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
