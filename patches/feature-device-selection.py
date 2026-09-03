"""Patch openwakeword's train.py to pick the feature-computation device correctly.

Feature computation runs through onnxruntime (melspectrogram + embedding models),
but upstream chooses its device from PyTorch:

    device="gpu" if torch.cuda.is_available() else "cpu",
    ncpu=n_cpus if not torch.cuda.is_available() else 1)

Those are different runtimes. On a CUDA box with the CPU build of onnxruntime -
which is what openwakeword's own setup.py pins ('onnxruntime>=1.10.0,<2') - torch
sees the GPU, so AudioFeatures is asked for CUDAExecutionProvider, onnxruntime does
not have it, warns, and falls back to CPU. Meanwhile ncpu has been set to 1 because
torch said GPU. The result is single-threaded CPU inference: measured at 2.46 it/s,
36 minutes of an 83-minute run, on a machine with a 3090 idle and every core but
one idle too.

This patch keys the decision off onnxruntime's actual providers instead, so:
  - with onnxruntime-gpu installed and working -> GPU, ncpu=1 (as intended)
  - otherwise                                  -> CPU across n_cpus threads

Only the two feature-computation arguments are touched. torch.cuda.is_available()
is left alone everywhere else, because model training really does use torch on the
GPU and that part works.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

helper = '''

def _onnx_has_gpu():
    """Whether onnxruntime itself can use the GPU (not whether torch can)."""
    try:
        import onnxruntime
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False

'''

replacements = [
    ('device="gpu" if torch.cuda.is_available() else "cpu",',
     'device="gpu" if _onnx_has_gpu() else "cpu",'),
    ('ncpu=n_cpus if not torch.cuda.is_available() else 1)',
     'ncpu=1 if _onnx_has_gpu() else n_cpus)'),
]

if all(old not in content for old, _ in replacements):
    print("WARNING: patch target not found in", path)
    sys.exit(0)

count = 0
for old, new in replacements:
    count += content.count(old)
    content = content.replace(old, new)

# Define the helper once, after the imports.
if "_onnx_has_gpu" in content and "def _onnx_has_gpu" not in content:
    marker = "\nif __name__ == '__main__':"
    if marker not in content:
        marker = '\nif __name__ == "__main__":'
    if marker in content:
        content = content.replace(marker, helper + marker, 1)
    else:
        print("WARNING: could not place helper in", path)
        sys.exit(0)

with open(path, 'w') as f:
    f.write(content)

print(f"Patched: {path} ({count} argument(s) rewired to onnxruntime's providers)")
