#!/usr/bin/env python3
"""Convert an openWakeWord ONNX model to tflite.

onnx2tf changes the input shape [1,16,96] -> [1,96,16], and how it does so depends
on the onnx2tf version: older ones relabel the axes without moving the underlying
data (a tf.reshape recovers the original), newer ones genuinely move it (a
tf.transpose is then correct). Applying the wrong one produces a model with the
right shape and the wrong numbers.

So rather than assume, this tries each adaptation, scores it against the source
ONNX on random inputs, and keeps the one that matches. If none match it fails
rather than writing a model. That check is the whole point of the script: a
wrong-axis tflite loads without warning, reports a plausible input shape, and
returns plausible-looking scores in the 0-1 range while never detecting anything.
Multiple trials are needed because a wrong adaptation can agree with the ONNX by
coincidence on any single input.

Needs tensorflow, onnxruntime, and the onnx2tf CLI. onnx2tf does not declare
several of its own imports, so in practice the working set is:

    pip install tensorflow onnxruntime onnx onnx2tf \\
                onnx-graphsurgeon sng4onnx tf_keras psutil ai-edge-litert

    python onnx2tflite.py hey_seeree.onnx -o hey_seeree_v0.1.tflite
"""

import argparse
import itertools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf


def adaptations(shape, sm_shape):
    """Ways the ONNX input might map onto the saved_model's expected input.

    Yields (name, fn) pairs. Reshape covers the relabel-only case; a transpose
    covers the case where onnx2tf actually moved the data. Only permutations that
    produce the target shape are worth trying.
    """
    yield "reshape", lambda x: tf.reshape(x, sm_shape)
    identity = tuple(range(len(shape)))
    for perm in itertools.permutations(identity):
        if perm != identity and [shape[i] for i in perm] == list(sm_shape):
            yield f"transpose{perm}", lambda x, p=perm: tf.transpose(x, p)


def build_tflite(sig, key, out_key, shape, adapt):
    """Re-export the saved_model through `adapt`, returning tflite bytes."""
    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32, name="input")])
    def wrapped(x):
        return sig(**{key: adapt(x)})[out_key]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [wrapped.get_concrete_function()]
    )
    return converter.convert()


def max_diff(tflite_bytes, sess, input_name, shape, trials):
    """Worst absolute difference against the ONNX over `trials` random inputs."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    worst = 0.0
    for _ in range(trials):
        x = np.random.randn(*shape).astype(np.float32)
        ref = sess.run(None, {input_name: x})[0]
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        worst = max(worst, float(np.abs(ref - interp.get_tensor(out["index"])).max()))
    return worst, [int(d) for d in inp["shape"]]


def convert(onnx_path: Path, out_path: Path, trials: int = 5, tol: float = 1e-4) -> float:
    workdir = Path(tempfile.mkdtemp())
    try:
        print("==> onnx2tf")
        # Output is captured because onnx2tf is extremely chatty on success, but it
        # has to be re-emitted on failure - swallowing it leaves a CalledProcessError
        # with no indication of what went wrong.
        result = subprocess.run(
            ["onnx2tf", "-i", str(onnx_path), "-o", str(workdir / "sm"), "-nuo", "-osd"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout[-4000:], file=sys.stderr)
            print(result.stderr[-4000:], file=sys.stderr)
            raise SystemExit(f"FAILED: onnx2tf exited {result.returncode} (output above)")

        sess = ort.InferenceSession(str(onnx_path))
        name = sess.get_inputs()[0].name
        shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[0].shape]

        loaded = tf.saved_model.load(str(workdir / "sm"))
        sig = loaded.signatures["serving_default"]
        key = next(iter(sig.structured_input_signature[1]))
        sm_shape = [int(d) for d in sig.structured_input_signature[1][key].shape]
        out_key = next(iter(sig.structured_outputs))

        print(f"==> re-exporting: {shape} -> ? -> {sm_shape}")

        results = []
        for label, adapt in adaptations(shape, sm_shape):
            try:
                tflite_bytes = build_tflite(sig, key, out_key, shape, adapt)
                worst, input_shape = max_diff(tflite_bytes, sess, name, shape, trials)
            except Exception as exc:                                   # noqa: BLE001
                print(f"    {label:<22} could not be built: {exc}")
                continue
            print(f"    {label:<22} max diff vs onnx {worst:.2e}"
                  f"  {'MATCH' if worst < tol else 'wrong numbers'}")
            results.append((worst, label, tflite_bytes, input_shape))

        if not results:
            raise SystemExit("FAILED: no adaptation could be built")

        worst, label, tflite_bytes, input_shape = min(results, key=lambda r: r[0])
        if worst >= tol:
            raise SystemExit(
                f"FAILED: the closest adaptation ({label}) still differs from the "
                f"ONNX by {worst:.2e}, over the {tol:.0e} tolerance. Writing it would "
                "ship a model that loads cleanly and never detects."
            )

        out_path.write_bytes(tflite_bytes)
        print(f"==> {label}, input shape {input_shape}, max diff vs onnx {worst:.2e}")
        print(f"==> wrote {out_path}")
        return worst
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("onnx", type=Path, help="input .onnx model")
    p.add_argument("-o", "--output", type=Path, help="output .tflite (default: alongside input)")
    p.add_argument("--trials", type=int, default=5, help="verification trials (default: 5)")
    p.add_argument("--tol", type=float, default=1e-4, help="max allowed diff (default: 1e-4)")
    args = p.parse_args()

    if not args.onnx.is_file():
        sys.exit(f"not found: {args.onnx}")

    convert(args.onnx, args.output or args.onnx.with_suffix(".tflite"), args.trials, args.tol)


if __name__ == "__main__":
    main()
