"""Patch openwakeword's train.py to take the corpus directory from the config.

Upstream DERIVES it, and derives it from the same two values that name the output
model (openwakeword/train.py:655-659):

    positive_train_output_dir = os.path.join(
        config["output_dir"], config["model_name"], "positive_train")
    ...
    feature_save_dir = os.path.join(config["output_dir"], config["model_name"])

So the corpus is forced to live at <output_dir>/<model_name>/, and the only lever
that moves it - `model_name` - also names the exported `.onnx` and is used as the
model's label. Putting a subdirectory in it to relocate the corpus would rename the
model with it.

That matters here because this repo now keeps two corpora side by side:

    my_custom_model/<wake_word>/oww/   built by train.py, for openWakeWord
    my_custom_model/<wake_word>/mww/   built by mww/corpus.py, for microWakeWord

Without this patch, train.py writes the corpus into .../oww/ and upstream then looks
for it one level up, finds nothing, creates an empty positive_train, and augments
zero clips - a silent wrong answer rather than an error, which is the failure mode
this repo keeps paying for.

The patch introduces `config["corpus_dir"]`, defaulting to the upstream expression,
so an unset config behaves exactly as before.
"""
import re
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

anchor = ('    positive_train_output_dir = os.path.join('
          'config["output_dir"], config["model_name"], "positive_train")')
if anchor not in content:
    print("WARNING: patch target not found in", path)
    sys.exit(0)

replacement = (
    '    # PATCHED: corpus location is configurable, defaulting to upstream\'s.\n'
    '    corpus_dir = config.get(\n'
    '        "corpus_dir", os.path.join(config["output_dir"], config["model_name"]))\n'
    '    if not os.path.exists(corpus_dir):\n'
    '        os.makedirs(corpus_dir, exist_ok=True)\n'
    + anchor
)
content = content.replace(anchor, replacement, 1)

# Re-point the four corpus directories and the feature save dir at corpus_dir.
before = content
for name in ("positive_train", "positive_test", "negative_train", "negative_test"):
    content = content.replace(
        f'os.path.join(config["output_dir"], config["model_name"], "{name}")',
        f'os.path.join(corpus_dir, "{name}")')
content = content.replace(
    '    feature_save_dir = os.path.join(config["output_dir"], config["model_name"])',
    '    feature_save_dir = corpus_dir')

n = len(re.findall(r'os\.path\.join\(corpus_dir, "', content))
if content == before or n != 4:
    print(f"WARNING: expected 4 corpus dirs re-pointed, got {n} in {path}")
    sys.exit(1)

with open(path, 'w') as f:
    f.write(content)

print(f"Patched: {path} (corpus_dir configurable, {n} directories + feature_save_dir)")
