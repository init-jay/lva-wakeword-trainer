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

--trials repeats the SAME workload back to back and reports min/max over trials
(C4, bug.md: a single-trial reading in this repo has twice turned out to be a
machine-load artefact - the MLX table in SPEED.md carries two numbers per cell
for exactly that reason), recording the machine's 1/5/15-min load average at
the start and end of each trial. --sample-cpu measures each piper instance's
CPU during the run two ways (psutil, the trainer venv carries it): the exact
cumulative-CPU-time mean over the whole trial (the answer to "how many cores
does this instance consume") plus a 10 Hz peak poll: that is the mechanism
probe - if ONE instance already measures at or near N_cores x 100%, no fleet
can beat it, and the no-scaling verdict is closed by construction instead of
by a single afternoon's numbers.

Run with the trainer venv (it carries tts_protocol + the train.corpus client).
"""
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import threading
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


class _CpuSampler:
    """Per-instance CPU usage over one trial, two complementary numbers:

    - EXACT mean: the process's cumulative CPU time (psutil.cpu_times) diffed
      across the whole trial. No sampling error - this is the answer to
      "how many cores does this instance actually consume".
    - 10 Hz peak: a one-tenth-second poll of psutil's interval cpu_percent.
      1 Hz aliases with the ~60 ms request period (a request that busy-bursts
      its intra-op threads for half its life reads as a duty-cycle average at
      1 Hz); 10 Hz resolves whether the bursts use 10 threads or 5.

    primed before the trial so the first counted sample is a real interval.
    One final sample is appended at stop() so the last tenth of the run is
    included too.
    """
    SAMPLE_HZ = 10

    def __init__(self, urls):
        import psutil
        self._psutil = psutil
        self._procs = {}
        self._t0 = None
        for url in urls:
            port = url.rsplit(":", 1)[1]
            out = subprocess.run(
                ["lsof", "-nP", "-iTCP:" + port, "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True)
            procs = []
            for pid in out.stdout.split():
                try:
                    p = psutil.Process(int(pid))
                    p.cpu_percent(None)  # prime: discard the interval to now
                    procs.append(p)
                except psutil.Error:
                    pass  # process gone before we primed it; the bench fails loudly instead
            self._procs[url] = procs
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def _row(self):
        return {url: sum(p.cpu_percent(None) for p in procs)
                for url, procs in self._procs.items()}

    def _run(self):
        while not self._stop.wait(1.0 / self.SAMPLE_HZ):
            self._samples.append(self._row())

    @staticmethod
    def _cpu_times_sum(procs):
        # (user, system, children user, children system) seconds, summed
        # field-wise across the pids on a port (pcputimes has no + or -).
        cols = [[getattr(p.cpu_times(), f) for p in procs]
                for f in ("user", "system", "children_user", "children_system")]
        return [sum(c) for c in cols]

    def start(self):
        self._t0 = {url: self._cpu_times_sum(procs)
                    for url, procs in self._procs.items()}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._samples.append(self._row())  # the last tenth of the run
        t1 = {url: self._cpu_times_sum(procs)
              for url, procs in self._procs.items()}
        # CPU seconds over the whole trial: field-wise diff. The g2p stage
        # runs espeak as a child process; children_* is its share, included.
        exact = {u: sum(a - b for a, b in zip(t1[u], self._t0[u])) for u in t1}
        return self.summary(exact)

    def summary(self, exact):
        """((mean%, max%) per URL, (exact mean%, peak% ) totals) over the trial.
        mean% = 10 Hz interval samples; exact mean% = CPU-seconds/wall-seconds.
        """
        if not self._samples:
            return None
        keys = sorted(self._samples[0])
        per = {k: (sum(r[k] for r in self._samples) / len(self._samples),
                   max(r[k] for r in self._samples)) for k in keys}
        exact_total = sum(exact.values())
        wall = len(self._samples) / self.SAMPLE_HZ  # every URL ran the same interval
        exmean = 100.0 * exact_total / wall if wall > 0 else 0.0
        peak = max(max(r.values()) for r in self._samples)
        return per, (exmean, peak), exact


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urls", default=os.environ.get("PIPER_URLS"),
                    help="comma-separated tcp:// fleet (default: $PIPER_URLS)")
    ap.add_argument("--clips-per-pair", type=int, default=24,
                    help="clips rendered per (voice, speaker) pair; the real "
                         "mww corpus is ~84, 24 keeps a full sweep to a few "
                         "minutes while amortising each model's one 0.6 s load")
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat the same workload back to back this many times "
                         "and report min/max over trials (the one-shot default "
                         "is unchanged)")
    ap.add_argument("--sample-cpu", action="store_true",
                    help="poll each instance's CPU%% at 10 Hz during each trial "
                         "and report mean/max per instance (the mechanism "
                         "probe; needs the trainer venv's psutil)")
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

    trial_rates = []
    for trial in range(1, args.trials + 1):
        load1, load5, load15 = os.getloadavg()
        counts = {}  # url -> clips rendered, fresh per trial
        failures = []
        sampler = _CpuSampler(fleet.urls) if args.sample_cpu else None
        if sampler:
            sampler.start()

        total = 0
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=N) as ex:
            for n in ex.map(_job, model_names):
                total += n
        wall = time.monotonic() - t0

        cpu = sampler.stop() if sampler else None  # stop() returns the summary
        rate = total / wall if wall > 0 else 0.0
        trial_rates.append(rate)
        l1, l5, l15 = os.getloadavg()
        head = (f"\n=== Piper fleet throughput  (P2.1)  trial {trial}/{args.trials} ==="
                if args.trials > 1 else "\n=== Piper fleet throughput  (P2.1) ===")
        print(head)
        print(f"  instances (N)        : {N}   {', '.join(fleet.urls)}")
        print(f"  models / pairs       : {len(model_names)} models / {len(voices)} (voice,speaker) pairs")
        print(f"  load avg (start)     : {load1:.2f} / {load5:.2f} / {load15:.2f}  (1/5/15 min, all N cores)")
        print(f"  load avg (end)       : {l1:.2f} / {l5:.2f} / {l15:.2f}")
        print(f"  clips rendered       : {total}  ({args.clips_per_pair} per pair, synthesize-and-discard)")
        print(f"  wall                 : {wall:.1f} s")
        print(f"  aggregate throughput : {rate:.2f} clips/s   ({rate/N:.2f} clips/s per instance)")
        if cpu:
            per, (exmean, peak), _exact = cpu
            detail = "  ".join(f"{u.rsplit(':', 1)[1]}: {m:.0f}/{x:.0f}%"
                               for u, (m, x) in sorted(per.items()))
            threads = "+".join(str(sum(p.num_threads() for p in sampler._procs[u]))
                               for u in sorted(per))
            print(f"  piper CPU: exact {exmean:.0f}% of {N * 100}% (CPU-seconds/wall), 10Hz peak {peak:.0f}%, "
                  f"threads {threads}")
            print(f"  piper CPU (10Hz mean/max): total {sum(m for m, _ in per.values()):.0f}% / {peak:.0f}%   [{detail}]")
        # Which instance carried the most work - the least-loaded shard should keep
        # these close; a big spread is the load-imbalance that caps scaling.
        if counts:
            spread = max(counts.values()) - min(counts.values())
            print(f"  per-instance clips   : {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
            print(f"  max-min spread       : {spread} clip(s) across {len(counts)} instance(s)")
        if failures:
            print(f"  FAILED CLIPS ({len(failures)}): {failures[:3]}{' ...' if len(failures) > 3 else ''}")

    if args.trials > 1:
        # Two numbers per cell, the MLX-table rule: a single reading here has
        # turned out to be a machine-load artefact before.
        print(f"\n=== min/max over {args.trials} trials (same workload each) ===")
        print(f"  throughput : {min(trial_rates):.2f} - {max(trial_rates):.2f} clips/s   "
              f"(mean {sum(trial_rates)/len(trial_rates):.2f})")
        print(f"  scaling vs 1 inst: (fill in from the N=1 run's min/max)")


if __name__ == "__main__":
    main()
