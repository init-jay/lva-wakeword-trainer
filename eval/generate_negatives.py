#!/usr/bin/env python3
"""Generate a targeted negative corpus for wake-word evaluation, using an
OpenAI-compatible TTS server (tested against Kokoro-FastAPI).

A hundred random sentences would mostly measure nothing: a wake-word model that is
already quiet on ordinary speech scores zero on all of them. The useful negatives are
the ones that probe the decision the model actually makes — words that *begin* like
the wake phrase and then continue into something else, and the wake word's own first
syllable attached to a different ending.

The corpus is therefore grouped into categories, and results should be read per
category rather than pooled. Measured against `hey_seeree.onnx`, the informative rows
were `extend` (13/20 false accepts) and `hey_other` (5/12); `general` was 0/36 and is
the background rate.

Categories:
    extend      the phrase, then the word keeps going ("hey serious", "hey series")
    running     wake-word-like sounds inside ordinary speech, with no "hey"
    hey_other   "hey" followed by something else ("hey Sarah", "hey Cindy")
    command     bare commands with no wake word, to check speech alone cannot trip it
    other_ww    other assistants' wake words
    general     ordinary conversation, for a baseline false-accept rate

A copy of this script lives in both the openWakeWord repo (`scripts/`) and the
training repo, since it is useful either side of the fence. Keep them identical.

Examples
--------
    # against a Kokoro-FastAPI server on the LAN
    python generate_negatives.py \\
        --url http://192.168.2.14:8880/v1/audio/speech --out negatives_tts

    # see what would be produced without calling the server
    python generate_negatives.py --dry-run

    # top up one category after editing its wordlist
    python generate_negatives.py --categories extend --out negatives_tts

Then score a model against the result with `eval_model.py --negatives ...`, reading
the output per category rather than pooled — the corpus is adversarial by
construction, so a pooled false-accept rate means little.

The wordlists below are tuned for a "hey siri"-like phrase. Retarget `EXTEND`,
`RUNNING` and `HEY_OTHER` for a different wake word — those three carry nearly all of
the signal.
"""

import argparse
import concurrent.futures as cf
import json
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 16000  # openWakeWord operates on 16 kHz mono audio

# The phrase, then the word keeps going. This is the category a model is most likely
# to fail on, and the one that trailing context is supposed to protect against.
EXTEND = [
    "hey serious", "hey seriously", "hey series", "hey Sirius", "hey syrup",
    "hey cereal", "hey searing pain", "hey sear it on both sides",
    "hey ceiling", "hey seedling", "hey sequel", "hey secret",
    "hey CD player", "hey seagull", "hey senior", "hey Sierra",
    "hey silly", "hey city hall", "hey seeker", "hey seaweed",
]
RUNNING = [
    "I watched the whole series last night and it was seriously good.",
    "He takes everything far too seriously these days.",
    "Sirius radio was playing in the background the entire time.",
    "We need to buy cereal and syrup before the shop closes.",
    "The series finale was a serious disappointment to everyone.",
    "She was searing the steak when the smoke alarm went off.",
    "The ceiling in the sitting room needs repainting this year.",
    "It is a serious situation and we should treat it as such.",
    "Sierra Nevada is beautiful in the early spring.",
    "The secretary said the seminar starts at seven.",
    "Can you see the sea from the seaside cottage?",
    "That is a seriously impressive series of results.",
]
HEY_OTHER = [
    "hey Sarah", "hey Cindy", "hey Sydney", "hey Sammy", "hey Sonny",
    "hey there, how are you doing today?", "hey you, over here!",
    "hey Sophie, can you pass me that?", "hey Sally, what time is it?",
    "hey, did you remember to lock the door?", "hey Sean, are you coming?",
    "hey Cecily, the food is ready.",
]
COMMAND = [
    "what's the time?", "turn on the lights please", "play some music",
    "set a timer for five minutes", "what's the weather like today?",
    "turn the volume down a bit", "add milk to the shopping list",
    "how long until the oven is ready?", "remind me to call my sister",
    "what's on my calendar tomorrow?", "switch off the kitchen lights",
    "play the next track please",
]
OTHER_WW = [
    "hey Google, what's the time?", "OK Google, turn on the lights",
    "Alexa, play some music", "hey Alexa, set a timer",
    "computer, open the pod bay doors", "hey Portal, call mum",
    "Cortana, what's my schedule?", "hey Bixby, take a photo",
]
GENERAL = [
    "The train leaves from platform nine in about twenty minutes.",
    "I think we should probably head home before it gets dark.",
    "Could you pass me the salt and pepper when you get a chance?",
    "The meeting has been moved to Thursday afternoon instead.",
    "It rained all weekend so we stayed inside and watched films.",
    "My brother is coming to visit us at the end of the month.",
    "There is a new bakery that opened on the corner last week.",
    "She has been learning to play the piano since she was six.",
    "We drove for hours without seeing another car on the road.",
    "The garden needs weeding but I keep putting it off.",
    "He said he would call back later this afternoon.",
    "That restaurant is usually booked up weeks in advance.",
    "I left my umbrella on the bus again this morning.",
    "The dog barked at the postman for a solid five minutes.",
    "They are renovating the house before they move in.",
    "It took three attempts to get the recipe right.",
    "The concert was cancelled due to the weather forecast.",
    "I have been meaning to read that book for ages.",
    "We should leave earlier to avoid the traffic.",
    "The battery on this laptop barely lasts an hour now.",
    "Everyone agreed it was the best decision available.",
    "She works from home three days a week these days.",
    "The package should arrive sometime on Wednesday.",
    "I could not find my keys anywhere this morning.",
    "The film was much longer than I expected it to be.",
    "We planted tomatoes and basil in the back garden.",
    "He has been training for the marathon since January.",
    "The wifi keeps dropping out in the upstairs rooms.",
    "There were far more people there than we anticipated.",
    "I will send you the details once everything is confirmed.",
    "The coffee machine broke down again this morning.",
    "They are talking about moving to the coast next year.",
    "It is much colder today than the forecast suggested.",
    "The children spent the afternoon building a fort.",
    "I need to renew my passport before we travel.",
    "The lecture covered far more material than expected.",
]

CATEGORIES = {
    "extend": EXTEND, "running": RUNNING, "hey_other": HEY_OTHER,
    "command": COMMAND, "other_ww": OTHER_WW, "general": GENERAL,
}

# A spread of accents and pitches; a single voice would measure that voice, not the model.
VOICES = ["af_bella", "af_nicole", "af_sarah", "af_sky", "af_heart", "af_nova",
          "am_adam", "am_michael", "am_eric", "am_liam", "am_onyx", "am_puck",
          "bf_emma", "bf_lily", "bf_alice", "bm_george", "bm_lewis", "bm_daniel"]


def build_corpus(categories):
    out = []
    for name in categories:
        out += [(name, text) for text in CATEGORIES[name]]
    return out


def synth(index, category, text, args):
    """Render one utterance and write it as a 16 kHz mono 16-bit WAV."""
    rng = np.random.default_rng(index)
    voice = VOICES[index % len(VOICES)]
    speed = round(float(rng.uniform(*args.speed_range)), 3)

    payload = json.dumps({"model": args.model, "input": text, "voice": voice,
                          "response_format": "wav", "speed": speed}).encode()
    request = urllib.request.Request(args.url, data=payload, headers={
        "Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"})

    for attempt in range(args.retries):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                raw = response.read()
            break
        except Exception as exc:                                     # noqa: BLE001
            if attempt == args.retries - 1:
                return None, f"FAIL {category}_{index:03d}: {type(exc).__name__}: {exc}"

    scratch = Path(args.out) / f".raw_{index:03d}.wav"
    scratch.write_bytes(raw)
    with warnings.catch_warnings():
        # Streaming servers emit a placeholder RIFF length; the data itself is fine.
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        sr, data = wavfile.read(scratch)
    scratch.unlink()

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = resample_poly(data.astype(np.float32), SR, sr)
    data = np.clip(data, -32768, 32767).astype(np.int16)

    name = f"{category}_{index:03d}_{voice}.wav"
    wavfile.write(Path(args.out) / name, SR, data)
    return name, f"{name:<34} {len(data)/SR:5.2f}s  speed={speed:<5} {text[:44]}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8880/v1/audio/speech",
                   help="OpenAI-compatible speech endpoint (default: %(default)s)")
    p.add_argument("--api-key", default="not-needed",
                   help="bearer token; Kokoro-FastAPI ignores it (default: %(default)s)")
    p.add_argument("--model", default="kokoro", help="TTS model name")
    p.add_argument("--out", default="negatives_tts", help="output directory for the WAVs")
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                   choices=list(CATEGORIES), help="which categories to generate")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel requests; keep modest, the server is doing the work")
    p.add_argument("--speed-range", type=float, nargs=2, default=(0.88, 1.18),
                   metavar=("MIN", "MAX"), help="speaking-rate jitter")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be generated without calling the server")
    args = p.parse_args()

    corpus = build_corpus(args.categories)
    if args.dry_run:
        for i, (category, text) in enumerate(corpus):
            print(f"  {category}_{i:03d}_{VOICES[i % len(VOICES)]:<12} {text}")
        print(f"\n{len(corpus)} utterances across {len(args.categories)} categories "
              f"(nothing written; drop --dry-run to generate)")
        return

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    print(f"generating {len(corpus)} negatives -> {out}")

    written, failures = 0, []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(synth, i, c, t, args) for i, (c, t) in enumerate(corpus)]
        for future in cf.as_completed(futures):
            name, line = future.result()
            if name is None:
                failures.append(line)
            else:
                written += 1

    for line in failures:
        print(" ", line)
    print(f"\n{written} written, {len(failures)} failed")
    if written:
        print(f"evaluate with:\n  eval_model.py --model MODEL --negatives {out}")


if __name__ == "__main__":
    main()
