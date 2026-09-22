#!/usr/bin/env python3
"""Fleet throughput for the Piper corpus path: clips/s vs N instances.

P2.1 in improvement.md shipped the PiperFleet sharding (train/corpus/piper.py)
and start-tts-fleet.sh but never measured the N-way throughput the plan
hypothesised (N x the single-instance rate). This is that measurement.

WHY THIS IS NOT bench_tts.py: bench_tts points at one URL (or does an
--instances fan-out that sends EVERY clip to EVERY instance). The corpus does
the opposite - it SHARDS by voice: PiperFleet.shard pins each model (all its
speakers) to ONE instance, least-loaded by pair count, and generate_piper_samples
runs one worker per instance. This tool drives that exact path: the real
select_piper_voices voice list, the real PiperFleet shard, workers == N, one
clip per request (Piper is serial, no batch, engine.py). It synthesises and
discards instead of writing WAVs, so it measures the fleet, not the disk.

SAME WORKLOAD AT EVERY N. The clip list is built once, deterministically,
from the voice list and the corpus's own text/speed grids (plain_positive_texts,
PLAIN_SPEED_GRID) and re-used unchanged across N = 1/2/4/6/8, so a change in
clips/s between sizes is the fleet, not the workload. The first clip of each
model pays its ~0.6 s load (engine docstring); with 25 models that is real
overhead the corpus also pays, and it is included, not warmed away.

    PIPER_URLS="$(./scripts/start-tts-fleet.sh 4)" \
        train-applesilicon/.venv/bin/python tools/bench_piper_fleet.py
    # or: --urls tcp://127.0.0.1:8898,tcp://127.0.0.1:8897 --clips-per-pair 24

Run with the trainer venv (it carries tts_protocol + the train.corpus client).
"""
import argparse
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train.corpus.piper import PiperFleet, _client, select_piper_voices  # noqa: E402
from train.corpus.positives import PLAIN_SPEED_GRID, plain_positive_texts  # noqa: E402

WAKE_WORD = "hey seeree"


def build_clip_list(voices, clips_per_pair):
    """(voice, speaker, text, speed) per pair, built ONCE and re-used at every N.

    Deterministic rotation over the corpus's own grids (modulo, not
    np.random): the corpus draws the speed with np.random per clip, but a
    benchmark that has to be comparable across five N values cannot depend on
    a different random mix each run, and the modulo cycle gives the same mean
    render cost with a fixed workload."""
    texts = plain_positive_texts(WAKE_WORD)
    speeds = list(PLAIN_SPEED_GRID)
    clips = []
    for v, pair in enumerate(voices):
        voice, speaker = (pair if isinstance(pair, (tuple, list)) else (pair, None))
        for i in range(clips_per_pair):
            k = v * clips_per_pair + i
            clips.append((voice, speaker,
                          texts[k % len(texts)],
                          speeds[k % len(speeds)]))
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urls", default=os.environ.get("PIPER_URLS"),
                    help="comma-separated tcp:// fleet (default: $PIPER_URLS)")
    ap.add_argument("--clips-per-pair", type=int, default=24,
                    help="clips rendered per (voice, speaker) pair; the real "
                         "mww corpus is ~84, 24 keeps a full sweep to a few "
                         "minutes while amortising each model's one 0.6 s load")
    args = ap.parse_args()
    urls = args.urls
    if not urls:
        ap.error("need --urls or $PIPER_URLS (start-tts-fleet.sh N prints it)")

    fleet = PiperFleet(urls)
    N = len(fleet.urls)
    catalog = fleet.probe(languages=("en_US", "en_GB"), max_speakers=12)
    # The exact oww voice set (select_piper_voices applies the same exclusions).
    voices = select_piper_voices(urls, WAKE_WORD,
                                 languages=("en_US", "en_GB"), max_speakers=12)
    assignment = fleet.shard(voices)

    clips = build_clip_list(voices, args.clips_per_pair)
    # One job per MODEL (all its speaker pairs), exactly generate_piper_samples:
    # keying on clip[0] (the model name) groups a multi-speaker model's pairs
    # into a single job so its 0.6 s load is paid once, not per speaker.
    by_model = {}
    for clip in clips:
        by_model.setdefault(clip[0], []).append(clip)
    model_names = list(by_model)

    import threading
    tlock = threading.Lock()
    counts = {}  # url -> clips rendered
    failures = []

    def _job(model):
        url = assignment[by_model[model][0][:2]]
        n = 0
        for voice, speaker, text, speed in by_model[model]:
            try:
                audio, _ = _client(url).timed_render((voice, speaker), text, speed)
            except Exception as e:
                # The corpus convention (piper.py _piper_render): report and skip
                # the clip, do not let one dead instance take down the sweep.
                with tlock:
                    failures.append(f"{url} {voice}/{speaker}: {type(e).__name__}: {e}")
                continue
            if audio is None or audio.size < 480:
                continue  # a failed/silence clip, as the corpus skips it
            n += 1
            with tlock:
                counts[url] = counts.get(url, 0) + 1
        return n

    total = 0
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        for n in ex.map(_job, model_names):
            total += n
    wall = time.monotonic() - t0

    rate = total / wall if wall > 0 else 0.0
    print(f"\n=== Piper fleet throughput  (P2.1) ===")
    print(f"  instances (N)        : {N}   {', '.join(fleet.urls)}")
    print(f"  models / pairs       : {len(model_names)} models / {len(voices)} (voice,speaker) pairs")
    print(f"  clips rendered       : {total}  ({args.clips_per_pair} per pair, synthesize-and-discard)")
    print(f"  wall                 : {wall:.1f} s")
    print(f"  aggregate throughput : {rate:.2f} clips/s   ({rate/N:.2f} clips/s per instance)")
    print(f"  scaling vs 1 inst    : (fill in after the N=1 run)")
    # Which instance carried the most work - the least-loaded shard should keep
    # these close; a big spread is the load-imbalance that caps scaling.
    if counts:
        spread = max(counts.values()) - min(counts.values())
        print(f"  per-instance clips   : {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
        print(f"  max-min spread       : {spread} clip(s) across {len(counts)} instance(s)")
    if failures:
        print(f"  FAILED CLIPS ({len(failures)}): {failures[:3]}{' ...' if len(failures) > 3 else ''}")


if __name__ == "__main__":
    main()
