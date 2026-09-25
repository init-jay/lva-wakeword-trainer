#!/usr/bin/env python3
"""TensorFlow microbenchmark: the op mix microWakeWord's mixednet is made of.

The torch probe for openWakeWord (SPEED.md) sent GEMM to the host via
Accelerate and convolutions to the container via oneDNN; that result did not
transfer to a convolutional model, and this was the measurement it prescribed: the same probe in `docker/Dockerfile.mww.cpu` and in a host TF
install, both pinned to 2.21.0. The result is in SPEED.md, "microWakeWord on
the Mac: host vs. container" - the host won, 1.17x on the full train step.

The shapes are the real model's, transcribed from
microwakeword/mixednet.py + MODEL_FLAGS in train/mww/train.py: spectrogram
(time=200, mels=16), batch 128, first conv (5,1) stride 3 -> 32, then MDConv
blocks - depthwise (7,1)/(11,1) split across the channels, pointwise (1,1) ->
64 - matching "[5], [7,11], [9,15], [23]" and pointwise_filters 64. The GEMM
row is Phase 1b's, kept so the two probes read against each other.

    python src/scripts/tf_probe.py [--interop N]

`--interop` sets inter-op parallelism before any layer is built. Both sides
run once at the install default - that is the comparison against the measured
26m06s run, which used defaults inside Docker - and once with the full core
count, because the macOS wheel is the one with a known single-core default.
Each block is timed as a k.Functional slice of the one graph, so the number
is the op itself, not layer construction.
"""

import argparse
import platform
import statistics
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as k

BATCH = 128
# 1500 ms clip at a 10 ms step - CLIP_DURATION_MS / WINDOW_STEP_MS in
# train/mww/config.py. This probe uses 200 slices rather than 150 because it
# runs the block's depthwise kernels in series, so their valid-padding drops
# stack instead of the real MDConv's one ring buffer of max(ksize)-1; 200 keeps
# the (23,1) in the clear (200 -> 66 -> 50 -> 28 -> 6). The per-element cost is
# the same, and both sides under comparison run the same shapes.
TIME, MELS = 200, 16


def bench(label, fn, reps=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"{label:34s} {statistics.fmean(times):9.3f} ms/op   (min {min(times):.3f})")
    return statistics.fmean(times)


def mdconv_block(x, kernel_sizes, pw_filters):
    """One mixconv block, in FLOP-equivalent serial form: the real MDConv
    splits the channels across the depthwise kernels and ring-buffers the
    concat, which a probe cannot reproduce; running each kernel in sequence
    over all channels costs the same and uses the same kernels."""
    for ksize in kernel_sizes:
        x = k.layers.DepthwiseConv2D((ksize, 1), padding="valid")(x)
    return k.layers.Conv2D(pw_filters, (1, 1), use_bias=False)(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interop", type=int, default=0,
                   help="inter-op threads to set before building (0 = install "
                        "default, which is the fair comparison)")
    p.add_argument("--intra", type=int, default=0,
                   help="intra-op threads to set before building")
    args = p.parse_args()

    if args.interop:
        tf.config.threading.set_inter_op_parallelism_threads(args.interop)
    if args.intra:
        tf.config.threading.set_intra_op_parallelism_threads(args.intra)

    print(f"tensorflow {tf.__version__}  python {sys.version.split()[0]}  "
          f"platform {sys.platform}  machine {platform.machine()}")
    print(f"logical CPUs TF sees: {len(tf.config.list_logical_devices('CPU'))}"
          + (f"  (inter-op set to {args.interop})" if args.interop else ""))

    data = tf.random.normal((BATCH, TIME, 1, MELS))
    inp = k.layers.Input(shape=(TIME, 1, MELS), batch_size=BATCH)

    first = k.layers.Conv2D(32, (5, 1), strides=(3, 1), padding="valid",
                            use_bias=False)
    h1 = first(inp)  # (128, 66, 1, 32)

    block_specs = [((7, 11), 64), ((9, 15), 64), ((23,), 64)]
    hs = [h1]
    for kernels, pw in block_specs:
        hs.append(mdconv_block(hs[-1], kernels, pw))

    # One stable callable per stage, sliced out of the single graph. Each gets a
    # tensor of its own expected shape - a sliced model will not take the raw
    # input.
    stage_shapes = [(BATCH, 66, 1, 32), (BATCH, 50, 1, 64), (BATCH, 28, 1, 64)]
    stages = [("first conv (5,1) s3 -> 32f", k.Model(inp, h1), data)]
    for (kernels, _), h_prev, h, shape in zip(block_specs, hs[:-1], hs[1:],
                                             stage_shapes):
        stages.append((f"mdconv {kernels} -> 64f", k.Model(h_prev, h),
                       tf.random.normal(shape)))

    g = tf.constant(np.random.normal(size=(1024, 1024)), dtype=np.float32)

    full = k.Model(inp, k.layers.Dense(2, name="out")(
        k.layers.GlobalAveragePooling2D()(hs[-1])), name="proxy")
    optimizer = k.optimizers.Adam()
    y = tf.one_hot(tf.cast(tf.random.uniform((BATCH,), maxval=2), tf.int32), 2)

    def train_step():
        with tf.GradientTape() as tape:
            loss = k.losses.categorical_crossentropy(y, full(data, training=True))
            grads = tape.gradient(loss, full.trainable_variables)
        optimizer.apply_gradients(zip(grads, full.trainable_variables))

    for label, model, stage_data in stages:
        bench(label, lambda m=model, d=stage_data: m(d))
    bench("gemm 1024^3", lambda: tf.matmul(g, g))
    bench("full train step (fwd+grad+update)", train_step)
    print("done")


if __name__ == "__main__":
    main()
