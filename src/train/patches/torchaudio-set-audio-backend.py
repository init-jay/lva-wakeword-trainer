"""Patch torch_audiomentations so it imports against torchaudio >= 2.2.

WHY A THIRD-PARTY PACKAGE AND NOT openwakeword. Every other patch here edits the
openWakeWord clone. This one edits an installed dependency, because the breakage is
between two of ITS dependencies and neither is ours to pin:

    File "torch_audiomentations/utils/io.py", line 27, in <module>
        torchaudio.set_audio_backend("soundfile")
    AttributeError: module 'torchaudio' has no attribute 'set_audio_backend'

torchaudio deprecated `set_audio_backend` in 2.0 and REMOVED it in 2.2.
torch-audiomentations 0.11.0 - the version openWakeWord's setup.py asks for, and the
one docker/requirements.txt pins - still calls it at module import time.

WHY THE CUDA IMAGE NEVER HIT THIS, WHICH IS THE PART WORTH UNDERSTANDING.
openWakeWord's setup.py pins `torchaudio>=0.13.1,<1`, so on that image
`pip install -e ./openwakeword` DOWNGRADES torchaudio to a 0.x release, where the
function still exists. That downgrade is impossible on docker/Dockerfile.oww.cpu:
it is Python 3.12, and torchaudio <1 is from 2022 with cp310 wheels at newest. So
the CPU image necessarily runs a torchaudio the pinned torch-audiomentations predates,
and no amount of pinning fixes it - the wheel matrix does not contain a combination
that satisfies both.

WHY REMOVING THE CALL IS SAFE RATHER THAN MERELY EXPEDIENT. It selects the soundfile
backend, and in torchaudio >= 2.1 soundfile IS the dispatcher's backend for the reads
this does - the line was a no-op before it became an error. The neighbouring
`USE_SOUNDFILE_LEGACY_INTERFACE = False` is left alone: setting an attribute that
nothing reads is harmless, and removing it would be a second change to justify.

Guarded with hasattr rather than deleted, so the same patched tree still works if a
torchaudio <2.2 is ever installed under it - which is exactly what happens if anyone
runs this patch against the CUDA image's environment.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

old = '''torchaudio.set_audio_backend("soundfile")'''

new = '''if hasattr(torchaudio, "set_audio_backend"):  # removed in torchaudio 2.2
    torchaudio.set_audio_backend("soundfile")'''

if old not in content:
    print("WARNING: patch target not found in", path)
    sys.exit(0)

if new in content:
    print("Already patched:", path)
    sys.exit(0)

content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)

print("Patched:", path)
