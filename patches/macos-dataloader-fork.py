"""Patch openwakeword's train.py so its DataLoader workers start under `fork` on macOS.

THE ONLY PATCH HERE THAT IS ABOUT AN OPERATING SYSTEM RATHER THAN A DEVICE, and it is
needed the moment the trainer runs outside a Linux container:

    _pickle.PicklingError: Can't pickle <function <lambda> at 0x14761ae80>:
    attribute lookup <lambda> on __main__ failed

macOS defaults multiprocessing to `spawn`; Linux defaults to `fork`. Under fork the
worker inherits the parent's memory and nothing is pickled. Under spawn every argument
must pickle, and the training DataLoader is wrapped around two things that cannot:

  * `label_transforms[key] = lambda x: [1 for i in x]` (train.py:859,861), passed into
    mmap_batch_generator and captured by the generator the dataset holds
  * `class IterDataset` (train.py:874), defined INSIDE the method, so it has no
    importable qualified name to pickle by

Either alone is fatal, which is why this is not fixed by naming the lambdas.

WHY multiprocessing_context AND NOT set_start_method. Setting the start method
globally would also fork the ONNX Runtime and OpenMP threads already running by this
point, and fork-after-threads is the classic deadlock. Scoping it to this DataLoader
keeps the change to the one place that needs it.

WHY NOT num_workers=0, which would also make the error go away: it changes the thing
being measured. This environment exists to compare host training speed against the
container's 26 it/s, and the container loads data with n_cpus//2 workers and
prefetch_factor=16. Removing the workers would compare a different pipeline. Forcing
fork makes macOS do exactly what Linux already does.

Guarded by sys.platform so applying it to a Linux image is a no-op - the images do not
apply it, but a patch that silently changed their behaviour if they ever did would be
a poor trade for two saved lines.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

old = '''        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)'''

new = '''        # macOS spawns rather than forks, and neither the label-transform lambdas
        # nor the locally-defined IterDataset can be pickled. See
        # patches/macos-dataloader-fork.py.
        _dl_kwargs = {}
        if sys.platform == "darwin":
            _dl_kwargs["multiprocessing_context"] = "fork"
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=n_cpus,
                                              prefetch_factor=16, **_dl_kwargs)'''

# Check the applied state BEFORE the anchor: application replaces the DataLoader
# line itself, so `old` is gone once patched and a bare anchor check misreports
# an applied patch as "target not found" on every setup re-run.
if "multiprocessing_context" in content:
    print("Already patched:", path)
    sys.exit(0)

if old not in content:
    print("WARNING: patch target not found in", path)
    sys.exit(0)

content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)

print("Patched:", path, "(DataLoader forks on macOS)")
