#!/usr/bin/env python3
"""P2.3 probe: CoreML EP vs CPU EP for the oww feature-stage models.

The augmentation+feature stage is the largest non-TTS host stage (the 2026-09-21
bar-test run spent ~12 min of it recomputing four arrays at 22,144 clips;
SPEED.md's range is 11-12m30s), and it runs onnxruntime on CPU:
openwakeword/openwakeword/utils.py pins providers to CUDA or CPU and there is
no CoreML branch (patches/feature-device-selection.py keys the device off
onnxruntime's real providers, which is where a CoreML branch would go).
SPEED.md's closed-Metal section is about TRAINING (torch/tensorflow), not
these two ONNX models - so this question was open, and improvement.md P2.3
prescribes exactly this probe: the real models on the real batch shape, on
this machine, and the result written down either way.

    src/train/train-applesilicon/.venv/bin/python src/scripts/coreml_probe.py

What it runs
------------
The two real feature models (openwakeword/openwakeword/resources/models/
melspectrogram.onnx, embedding_model.onnx) at the real feature-stage shapes:
augmentation_batch_size=16 clips of total_length=1.2 s (19,200 samples at
16 kHz), the shape compute_features_from_generator feeds.

CPU = the path that runs today: CPUExecutionProvider with oww's thread pool
(os.cpu_count()//2 workers) making PER-CLIP melspectrogram calls and
PER-WINDOW embedding calls (utils.py _get_melspectrograms /
_get_embeddings_batch). CoreML = the natural integration: one batched call
per stage, the way the CUDA branch works, with CoreMLExecutionProvider
ahead of the CPU fallback.

Model shapes (read off the graphs): melspectrogram (batch, samples) ->
(batch, 117, 32) at 19,200 samples; embedding takes (N, 76, 32, 1) windows
- note the trailing channel dim, 4D - and gives (N, 96).

The drift numbers matter as much as the speed: the features are baked into
every trained model, so a CoreML path whose outputs differ from the CPU
path by more than float noise would change the model it feeds. The probe
reports max/mean abs drift per stage and the verdict has to clear both the
speed and the drift bars.
"""

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "src/train/openwakeword/openwakeword/resources/models")

# The real feature-stage shapes (src/train/oww/train.py defaults):
BATCH = 16                 # augmentation_batch_size
SAMPLES = 16000 * 6 // 5   # total_length 1.2 s at 16 kHz = 19,200
NCPU = max(1, (os.cpu_count() or 2) // 2)   # train.py: os.cpu_count()//2
WINDOW = 76                # utils.py: the embedding window in frames


def bench(label, fn, reps, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    print(f"  {label:<34} {med*1000:9.1f} ms  (min {min(ts)*1000:.0f} max {max(ts)*1000:.0f} over {reps})")
    return out, med


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}  providers: {ort.get_available_providers()}")
    print(f"batch {BATCH} x {SAMPLES} samples  ncpu {NCPU} (as in the feature stage)\n")

    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        print("CoreMLExecutionProvider not available - nothing to probe. "
              "Record that in SPEED.md and stop.")
        return

    # One fixed input: deterministic, and the same array feeds every session.
    rng = np.random.default_rng(1234)
    audio = (rng.standard_normal((BATCH, SAMPLES)) * 3000).astype(np.float32)

    def make_session(name, providers):
        so = ort.SessionOptions()
        so.inter_op_num_threads = NCPU
        so.intra_op_num_threads = NCPU
        return ort.InferenceSession(f"{MODELS}/{name}", sess_options=so, providers=providers)

    # --- sessions (session-creation cost is part of the answer: a cold oww
    # run pays it four times, once per feature array) --------------------
    t0 = time.perf_counter()
    mel_cpu = make_session("melspectrogram.onnx", ["CPUExecutionProvider"])
    emb_cpu = make_session("embedding_model.onnx", ["CPUExecutionProvider"])
    print(f"CPU sessions created in {time.perf_counter()-t0:.2f} s")

    t0 = time.perf_counter()
    try:
        mel_m = make_session("melspectrogram.onnx",
                             ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        emb_m = make_session("embedding_model.onnx",
                             ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    except Exception as e:
        print(f"CoreML session creation FAILED ({type(e).__name__}: {e}) - record in SPEED.md and stop.")
        return
    print(f"CoreML sessions created in {time.perf_counter()-t0:.2f} s")
    print(f"  melspec providers in effect: {mel_m.get_providers()}")
    print(f"  embedding providers in effect: {emb_m.get_providers()}\n")

    # --- the two call patterns -------------------------------------------
    def cpu_melspec(x):
        # today's path: one session call per clip, from a thread pool
        with ThreadPoolExecutor(max_workers=NCPU) as pool:
            return np.array([r[0].squeeze() for r in
                             pool.map(lambda c: mel_cpu.run(None, {'input': c[None, :]})[0], x)])

    def cm_melspec(x):
        # (B, 1, 117, 32) -> (B, 117, 32), the per-call squeeze shape
        return mel_m.run(None, {'input': x})[0].squeeze(1)

    def windows(mels):
        # the _get_embeddings_batch loop: 76-frame windows, stride 8
        jobs = []
        for spec in mels:
            for i in range(0, spec.shape[0] - WINDOW + 1, 8):
                jobs.append(spec[i:i + WINDOW])
        return np.array(jobs)

    def cpu_embed(mels):
        arr = windows(mels).astype(np.float32)[:, :, :, None]   # 4D: (N, 76, 32, 1)
        with ThreadPoolExecutor(max_workers=NCPU) as pool:
            return np.array([r[0].squeeze() for r in
                             pool.map(lambda w: emb_cpu.run(None, {'input_1': w[None, ...]})[0], arr)])

    def cm_embed(mels):
        arr = windows(mels).astype(np.float32)[:, :, :, None]
        # (N, 1, 1, 96) -> (N, 96), the per-call squeeze shape
        return emb_m.run(None, {'input_1': arr})[0].reshape(len(arr), -1)

    print(f"melspectrogram stage (batch of {BATCH} clips):")
    out_cpu_m, t_cpu_m = bench("CPU threaded (today's path)", lambda: cpu_melspec(audio), args.reps)
    out_mel_m, t_mel_m = bench("CoreML batched", lambda: cm_melspec(audio), args.reps)
    print(f"  -> {t_cpu_m / t_mel_m:.2f}x\n")

    # embedding stage, fed the SAME mels (CPU's own, so the drift comparison
    # isolates the execution provider, not the melspec)
    print(f"embedding stage ({BATCH} clips x {out_cpu_m.shape[1]} frames):")
    out_cpu_e, t_cpu_e = bench("CPU threaded (today's path)", lambda: cpu_embed(out_cpu_m), args.reps)
    out_mel_e, t_mel_e = bench("CoreML batched", lambda: cm_embed(out_cpu_m), args.reps)
    print(f"  -> {t_cpu_e / t_mel_e:.2f}x\n")

    # --- drift -------------------------------------------------------------
    print("drift (CoreML vs CPU; the features are baked into the model):")
    dm = np.abs(out_cpu_m - out_mel_m)
    de = np.abs(out_cpu_e - out_mel_e)
    print(f"  melspec:   max {dm.max():.2e}  mean {dm.mean():.2e} (values ~{np.abs(out_cpu_m).mean():.1f})")
    print(f"  embedding: max {de.max():.2e}  mean {de.mean():.2e} (values ~{np.abs(out_cpu_e).mean():.2f})\n")

    # --- the stage-level answer -------------------------------------------
    total_cpu = t_cpu_m + t_cpu_e
    total_m = t_mel_m + t_mel_e
    print(f"per-batch total: CPU {total_cpu*1000:.0f} ms vs CoreML {total_m*1000:.0f} ms "
          f"({total_cpu/total_m:.2f}x)")
    n_batches = 22144 // BATCH
    print(f"Extrapolated to the 22,144-clip feature stage (~{n_batches} batches at batch "
          f"{BATCH}), at the measured ~12 min: CoreML would be "
          f"~{12*total_m/total_cpu:.1f} min of model time.")


if __name__ == "__main__":
    main()
