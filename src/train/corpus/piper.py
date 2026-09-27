"""Piper sample generation for the wake-word corpus - the training-policy layer
over the shared corpus layer (exclusion tables, voice selection, generator),
deliberately shaped like the Kokoro path so both trainers consume either
engine's output, or both at once. A new engine does not get these tables; a
new wake word does.

Transport is the repo's TTS PROTOCOL (src/tts-service/tts_protocol/) over a
`tcp://` URL - on a Mac the `uv` project at
src/tts-service/engines/piper, in Docker the image that bakes in the same
project: one code path, one G2P pin, only the URL differs. The engine-side
justifications (WSOLA speed, stochastic output, the one-server-one-request
constraint) live in that engine's docstring, because the code they explain
lives there.

The URL is also a COMMA-SEPARATED fleet list (the --piper-url shape). One
instance is one resident model and one serial lane (tts_protocol/server.py
locks every engine call), so throughput scales with PROCESSES: PiperFleet
shards the corpus BY VOICE - each model, all of its speakers, pinned to ONE
instance for the whole run. Never round-robin: round-robin reloads a model on
most requests (a 0.6 s reload against a clip costing hundreds of ms, the
engine's docstring). Fleet measurement: docs/SPEED.md - on ONE machine a
fleet does not help; the fleet is the only route to a multi-machine corpus.
The probe refuses a fleet whose instances serve different catalogs, because a
sharded voice an instance lacks fails per-clip and shrinks the corpus
silently.
"""

import json
import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from tts_protocol.audio import SR  # noqa: E402  (re-exported name, now defined once)
from tts_protocol.client import TtsClient
from .kokoro import run_jobs  # noqa: E402  (the shared thread-pool runner; kokoro imports no piper, no cycle)

# The per-word voice exclusions this module reads. Importable because both trainers
# put src/ on sys.path before importing the corpus package; the recipe package
# imports nothing from here, so there is no cycle.
import recipe  # noqa: E402

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


class PiperFleet:
    """N Piper protocol instances, addressed as one comma-separated `tcp://` list -
    the --piper-url shape. One instance holds one model resident and serves
    one request at a time, so N instances are N independent serial lanes.
    shard() runs after the voices round-trip (probe) and is memoised for the
    object's life - which is the run - so a model always hits the same
    instance from its first clip to its last, and each instance loads each of
    its models once rather than per request.

    On ONE machine a fleet does not help - measured, not assumed:
    docs/SPEED.md "Piper fleet". It is the only route to a MULTI-machine
    corpus; the sharding is the non-obvious part (a model must hit the same
    instance for the whole run or nearly every request pays the 0.6 s
    reload), and the probe catches a fleet whose instances serve different
    catalogs, which would silently shrink the corpus.
    """

    def __init__(self, spec):
        # Accepts a PiperFleet, a list, or a string - one URL or a
        # comma-separated list, the two shapes --piper-url arrives in.
        if isinstance(spec, PiperFleet):
            self._urls = list(spec.urls)
        elif isinstance(spec, str):
            self._urls = [u.strip() for u in spec.split(",") if u.strip()]
        else:
            self._urls = [str(u).strip() for u in spec if str(u).strip()]
        if not self._urls:
            raise ValueError("PiperFleet: no URLs")
        self._assignment = None

    @property
    def urls(self) -> list:
        return self._urls

    def probe(self, languages=None, max_speakers=None) -> list:
        """A `voices` round trip against EVERY instance, not a connect: a bound port
        owned by a dead or foreign listener answers a connect and still fails
        the corpus stage.

        The fleet must all serve the SAME catalog: shard pins a voice to an
        instance, and a voice the instance lacks fails per clip, silently
        shrinking the corpus instead of failing the run.
        """
        kwargs = {}
        if languages:
            kwargs["languages"] = tuple(languages)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)
        multi = len(self._urls) > 1
        ref_catalog = None
        for i, url in enumerate(self._urls):
            client = _client(url)
            try:
                catalog = client.voices(**kwargs)
            except Exception as e:
                print(f"  ERROR: could not reach the Piper protocol server at {url}: {e}")
                if multi:
                    print("         A fleet is a comma-separated --piper-url; every URL in it "
                          "must answer. The fleet script (src/scripts/start-tts-fleet.sh N) "
                          "reports which of its instances died.")
                print("         Mac:  `uv run --project src/tts-service/engines/piper "
                      "python -m piper_engine --port 8898`")
                print("         Docker: `docker compose up -d piper` (publishes 8898),")
                print("         then point --piper-url at tcp://127.0.0.1:8898.")
                sys.exit(1)
            if ref_catalog is None:
                ref_catalog = catalog
            elif catalog != ref_catalog:
                first = self._urls[0]
                print(f"  ERROR: the Piper instances disagree: {first} serves "
                      f"{len(ref_catalog)} (voice, speaker) pairs, {url} serves "
                      f"{len(catalog)}.")
                print("         A fleet is sharded BY VOICE, so a voice is sent to whichever "
                      "instance holds its shard - an instance that lacks the voice fails that "
                      "clip, and the corpus shrinks instead of the run failing. Rebuild the "
                      "fleet from one voices directory (data/external/piper/voices on a Mac) "
                      "or point --piper-url at the one instance that was audited.")
                sys.exit(1)
            if multi:
                engine = client.server_engine or "?"
                print(f"  Piper instance OK: {url} (engine={engine}, {len(catalog)} pairs)")
        return ref_catalog

    def shard(self, voices) -> dict:
        """(voice, speaker) -> URL, decided ONCE and memoised: for the life of this
        object - which is the run - a voice always hits the same instance.

        The unit of placement is the MODEL, not the speaker pair: the engine
        reloads when the model NAME changes and reuses the loaded session
        when only the speaker does, so splitting one model's speakers across
        instances makes every one of them load it. Weight is the model's pair
        count, placement is least-loaded, ties to the lower index. Input
        order (the server's catalog order) plus the deterministic tie-break
        make the mapping reproducible for a given fleet size and voice list.
        """
        if self._assignment is not None:
            return self._assignment
        # Keys are NORMALISED (voice, speaker) pairs, bare-string voices
        # included: the caller (generate_piper_samples) builds (voice, speaker)
        # tuples with speaker=None and looks the job's URL up by that key.
        norm = lambda v: v if isinstance(v, (tuple, list)) else (v, None)
        if len(self._urls) == 1:
            self._assignment = {norm(v): self._urls[0] for v in voices}
            return self._assignment

        models = {}  # model name -> [pairs, in input order]
        for v in voices:
            name = v[0] if isinstance(v, (tuple, list)) else v
            models.setdefault(name, []).append(v)

        loads = [0] * len(self._urls)
        assignment = {}
        for name, pairs in models.items():
            i = min(range(len(self._urls)), key=lambda j: (loads[j], j))
            loads[i] += len(pairs)
            for v in pairs:
                assignment[norm(v)] = self._urls[i]
        self._assignment = assignment
        return assignment


def _piper_render(piper_url, voice, speaker, text, speed=1.0):
    """One Piper clip over the protocol. Raises on failure (the PIPER
    convention: the error envelope names the voice) - the caller reports which
    (voice, speaker) failed."""
    v = (voice, speaker) if speaker is not None else voice
    audio, _ = _client(piper_url).timed_render(v, text, speed)
    return audio


# Per-word Piper exclusions - voices that render the wake word wrong, and
# voices nobody has audited - live in recipes/<word>.yaml under
# `voices.piper.{mispronouncing,unaudited}`, read through
# recipe.voice_exclusions(). PIPER_VOICE_SEX below stays here: F0 is a
# property of the voice, not of the phrase it renders.

# Voice sex, for the child-range lever (corpus/augment.py). Keys are the voice name,
# or "voice:speaker" for a multi-speaker model.
#
# MEASURED, NOT LISTENED TO. Generated by measure_voice_f0.py from the 1.0x audit
# clips: median F0 per voice, split at 185 Hz (validated against the ten voices
# whose NAME states the answer - all ten agree). Regenerate it for a new engine
# or voice set rather than extending by ear.
#
# The 160-200 Hz band is genuinely ambiguous and does not matter much: sex is a
# proxy for F0, and the two ratio ranges nearly coincide at the boundary.
#
# A voice absent from this map is written piper_pu_* and gets NO child-range copy.
# Deliberate: a wrongly-shifted clip is worse than an absent one - training on an
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
    """'f', 'm', or 'u' (unknown) for the child-range lever. Checked most specific
    first: a multi-speaker model can hold both sexes, so a "voice:speaker"
    entry must win over the model-wide one."""
    if speaker is not None:
        specific = PIPER_VOICE_SEX.get(f"{voice}:{speaker}")
        if specific:
            return specific
    return PIPER_VOICE_SEX.get(voice, "u")


def generate_piper_samples(piper_url, voices, output_dir: Path,
                           samples_per_voice: int, texts, speeds, desc="Piper"):
    """Render `samples_per_voice` clips for each voice into `output_dir`.

    `piper_url` is a `tcp://` protocol URL or a comma-separated fleet list
    (the --piper-url shape): PiperFleet.shard pins every model to ONE instance
    for the whole run, so each instance loads each of its models once and
    then only synthesises. select_piper_voices enforces the same catalog on
    every instance before any clip is rendered; an instance that dies mid-run
    surfaces as per-clip errors naming the voice, not as a shrunken corpus
    nobody explains.

    Signature and sampling deliberately mirror generate_kokoro_samples, so
    the two are substitutable clip-for-clip: same per-voice budget, same
    text-offset-per-voice (without which every voice renders texts[0:n]), and
    the speed drawn from the same grid, in the job-building loop rather than
    a worker, so the corpus does not depend on thread scheduling. `speeds` is
    passed in rather than imported: the speed grid lives in train.py and this
    package must not depend on it.

    Filenames are `piper_p{sex}_{voice}[_{speaker}]_{uuid}.wav`. This does NOT
    match the `kokoro_`/`runon_` prefixes add_child_range_copies looks for, so
    Piper clips are skipped by the child-range lever rather than mis-shifted.
    See corpus/augment.py.

    VOICE IS THE OUTER LOOP ON PURPOSE - DO NOT REORDER. Every Piper server
    holds exactly one loaded voice at a time and reloads it when a request
    names a different one; iterating texts or speeds outside voices would
    rebuild the InferenceSession on every request - under --use-cuda a fresh
    CUDA session each time. The same one-voice-at-a-time property is why the
    sharding across a fleet is BY VOICE, never round-robin. One job per
    model, one job per instance lane at a time (workers = fleet size: extra
    workers queue at the engine's lock and add no throughput).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fleet = PiperFleet(piper_url)
    assignment = fleet.shard(voices)

    # Same text-offset-per-voice as the serial loop; speed drawn HERE in the
    # build loop, never in a worker, so the corpus does not depend on thread
    # scheduling.
    clips = []  # (voice, speaker, text, speed)
    for v, pair in enumerate(voices):
        voice, speaker = (pair if isinstance(pair, (tuple, list))
                          else (pair, None))
        for i in range(samples_per_voice):
            text = texts[(v * samples_per_voice + i) % len(texts)]
            speed = float(np.random.choice(speeds))
            clips.append((voice, speaker, text, speed))

    # Group clips by MODEL, in first-seen order: one job per model keeps every
    # model's clips contiguous on its pinned instance (the catalog lists a
    # model's speakers contiguously, so with one URL this is the serial order).
    by_model = {}
    for clip in clips:
        by_model.setdefault(clip[0], []).append(clip)
    models = list(by_model)
    weights = [len(by_model[m]) for m in models]

    unknown_sex = set()
    sex_lock = threading.Lock()

    def _job(model):
        # One URL for the whole job: shard pinned the model, so every clip in
        # it names the same instance and rides one loaded session.
        url = assignment[by_model[model][0][:2]]
        written = 0
        for voice, speaker, text, speed in by_model[model]:
            try:
                audio = _piper_render(url, voice, speaker, text, speed)
            except Exception as e:
                print(f"  Error rendering {voice}/{speaker} at {speed}x: {e}")
                continue
            if audio is None or audio.size < 480:
                continue

            # piper_p{sex}_{voice}[_{speaker}]_{uuid}.wav - the `p{sex}` group is
            # second on purpose: add_child_range_copies reads the sex from parts[1][1],
            # where Kokoro's af_/am_ prefix puts it. Same position, same code.
            sex = voice_sex(voice, speaker)
            if sex == "u":
                with sex_lock:
                    unknown_sex.add(voice if speaker is None else f"{voice}:{speaker}")
            tag = f"{voice}_{speaker}" if speaker is not None else voice
            name = f"piper_p{sex}_{tag}_{uuid.uuid4().hex[:8]}.wav".replace("/", "_")
            scipy.io.wavfile.write(str(output_dir / name), SR, audio)
            written += 1
        return written

    produced = run_jobs(models, _job, desc=desc, workers=len(fleet.urls),
                        weights=weights)
    # The job manifest: which instance carried which (model, speaker) pair and
    # how many clips - the fleet's provenance, written into the tree it was
    # rendered in rather than only the run log.
    per_pair = {}
    for clip in clips:
        pair = clip[:2]
        per_pair[pair] = per_pair.get(pair, 0) + 1
    (output_dir / "piper").mkdir(parents=True, exist_ok=True)
    (output_dir / "piper" / "jobs.json").write_text(json.dumps(
        {"piper_urls": fleet.urls,
         "jobs": [{"model": m, "speaker": s,
                   "n_clips": per_pair[(m, s)],
                   "url": assignment[(m, s)]}
                  for m, s in sorted(per_pair,
                                     key=lambda p: (p[0], p[1] or ""))]},
        indent=2))
    print(f"  Wrote {produced} Piper clips from {len(clips)} jobs")
    if unknown_sex:
        print(f"  WARNING: {len(unknown_sex)} voice(s) have no sex in "
              f"PIPER_VOICE_SEX, so their clips get NO child-range copy: "
              f"{', '.join(sorted(unknown_sex)[:6])}"
              f"{' ...' if len(unknown_sex) > 6 else ''}")
    return produced


def select_piper_voices(piper_url: str, wake_word: str, languages=("en_US", "en_GB"),
                        max_speakers: int = 12) -> list:
    """Enumerate Piper voices from the `tcp://` server (or fleet - a
    comma-separated list probes every instance and requires them to agree,
    see PiperFleet.probe), drop the ones that say the wrong thing, report
    cover.

    The exclusion step is the whole point, and it is per wake word: the lists
    come from `voices.piper` in recipes/<word>.yaml. An unaudited list is a
    bigger exposure, not a smaller one.
    """
    found = PiperFleet(piper_url).probe(languages=languages,
                                        max_speakers=max_speakers)

    # A word with no recipe cannot build a corpus that means anything - the same
    # hard stop build_negative_phrases makes rather than a traceback.
    try:
        data = recipe.load(wake_word)
    except recipe.RecipeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    exclusions = recipe.voice_exclusions(data, "piper")
    recipe_name = Path(data["_path"]).name
    bad = set(exclusions["mispronouncing"])
    unaudited = set(exclusions["unaudited"])
    excluded = bad | unaudited
    if not bad:
        print(f"  WARNING: {recipe_name} has no voices.piper.mispronouncing entry.")
        print("           Nothing has been excluded, so any voice whose espeak-ng")
        print("           g2p guesses the wake word wrong is contributing")
        print("           MISLABELLED POSITIVES. Six of 42 Kokoro voices did exactly")
        print("           that on the example word (~14% of that corpus). Run")
        print("           src/scripts/audit_voices.py --tts tcp://<that instance>,")
        print("           listen to the shortlist, and paste what it prints into")
        print(f"           {recipe_name}.")

    # Match both forms. The audit scores SPEAKERS - the l2arctic model ran from
    # 0% to 100% across its speakers - so most entries are "voice:speaker".
    # A bare voice name still excludes the whole model.
    def is_excluded(v, s):
        return v in excluded or f"{v}:{s}" in excluded

    kept = [(v, s) for (v, s) in found if not is_excluded(v, s)]
    n_bad = sum(1 for v, s in found if v in bad or f"{v}:{s}" in bad)
    n_unaudited = sum(1 for v, s in found
                      if v in unaudited or f"{v}:{s}" in unaudited)

    print(f"  Piper voices: {len(kept)} of {len(found)} "
          f"({n_bad} mispronouncing, {n_unaudited} unaudited)")

    # A voice the service offers that appears in NEITHER list has never been
    # checked and is not being excluded. Say so loudly rather than letting it
    # show up later as a child-range coverage number.
    unknown = sum(1 for v, s in kept if voice_sex(v, s) == "u")
    if unknown:
        names = sorted({v if s is None else f"{v}:{s}"
                        for v, s in kept if voice_sex(v, s) == "u"})
        print(f"  WARNING: {unknown} kept voice(s) are in no list and have no F0 -")
        print("           unaudited AND unmapped, so they contribute possibly")
        print("           mislabelled positives and get no child-range copy:")
        print(f"           {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
        print("           Audit them against THIS Piper instance, or add them to")
        print(f"           voices.piper.unaudited in {recipe_name}.")
    return kept
