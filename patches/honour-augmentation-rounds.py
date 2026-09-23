"""Patch openwakeword's train.py so `augmentation_rounds` actually multiplies data.

Upstream builds the clip list multiplied by the setting:

    positive_clips_train = [...glob("*.wav")]*config["augmentation_rounds"]

but then sizes the output array from the unmultiplied directory:

    compute_features_from_generator(gen, n_total=len(os.listdir(dir)), ...)

and compute_features_from_generator stops at n_total rows
(openwakeword/utils.py:690, `if row_counter >= n_total: break`). So with
augmentation_rounds > 1 the generator augments every clip N times, and all but the
first pass is computed and then discarded - pure cost, no extra data.

That matters because augmentation is the cheapest source of variety we have. Each
round re-augments the same clip with a different room impulse response, background
and gain, which is the variation deployment actually has, and it costs no extra TTS
calls. Without this patch the only way to add data is to synthesise more of it.

The patch multiplies n_total by the same factor, so the array is sized for what the
generator will actually produce.

Idempotency is load-bearing here because the setup script re-applies this patch on
every run. The original version matched only the unmultiplied line, so it re-matched
its OWN output: every run multiplied one more `*config["augmentation_rounds"]` onto
the same line, and a working tree that had been patched three times allocated 27N
rows in the feature array instead of 3N. This version includes the factor in the
match and collapses any number of existing factors to exactly one, so re-application
is a no-op and an over-patched tree heals itself on the next run.
"""
import re
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

# n_total=len(os.listdir(<dir>))[*factor...[*factor]] -> exactly one factor. The
# repeated group is what makes an already-applied state match and collapse instead
# of multiplying again - matching only the unmultiplied line is exactly the bug
# this patch used to have (see the docstring).
pattern = re.compile(
    r'n_total=len\(os\.listdir\((\w+)\)\)((?:\*config\["augmentation_rounds"\])*)')

matches = [(m.group(1), m.group(2).count('*')) for m in pattern.finditer(content)]

if not matches:
    print("WARNING: patch target not found in", path)
    sys.exit(0)

dirs = [d for d, _ in matches]
n_existing = sum(k for _, k in matches)

content = pattern.sub(
    r'n_total=len(os.listdir(\1))*config["augmentation_rounds"]', content)

if n_existing == len(matches):  # every site had exactly one factor: nothing to do
    print(f"Already patched: {path} ({len(matches)} call sites: {', '.join(dirs)})")
    sys.exit(0)

with open(path, 'w') as f:
    f.write(content)

if n_existing:
    print(f"Patched: {path} ({len(matches)} call sites: {', '.join(dirs)}); "
          f"normalised {n_existing} existing factor(s) to 1")
else:
    print(f"Patched: {path} ({len(matches)} call sites: {', '.join(dirs)})")
