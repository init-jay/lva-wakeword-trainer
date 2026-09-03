"""Patch openwakeword's train.py to make piper generate_samples import conditional.

Our pipeline uses Kokoro TTS instead of Piper, so we don't have
piper-sample-generator installed. The upstream train.py unconditionally
imports from it at startup, even when only --augment_clips is used.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

old = '''\
    sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
    from generate_samples import generate_samples'''

new = '''\
    if config.get("piper_sample_generator_path"):
        sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
        from generate_samples import generate_samples'''

if old not in content:
    print("WARNING: patch target not found in", path)
    sys.exit(0)

content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)

print("Patched:", path)
