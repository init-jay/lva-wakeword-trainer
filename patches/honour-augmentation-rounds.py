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
"""
import re
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

# n_total=len(os.listdir(<dir>)) -> n_total=len(os.listdir(<dir>))*config["augmentation_rounds"]
pattern = re.compile(r'n_total=len\(os\.listdir\((\w+)\)\)')
matches = pattern.findall(content)

if not matches:
    print("WARNING: patch target not found in", path)
    sys.exit(0)

content = pattern.sub(
    r'n_total=len(os.listdir(\1))*config["augmentation_rounds"]', content)

with open(path, 'w') as f:
    f.write(content)

print(f"Patched: {path} ({len(matches)} call sites: {', '.join(matches)})")
