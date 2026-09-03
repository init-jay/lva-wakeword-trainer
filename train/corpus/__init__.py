"""Engine-agnostic corpus construction, shared by both trainers.

Everything in here operates on 16 kHz mono WAVs and knows nothing about how those
clips will later be turned into features. That is the seam between openWakeWord
(melspectrogram -> embedding model -> 96-dim embeddings, 2000 ms window) and
microWakeWord (40 features per 10 ms into a streaming MixConv net, 1500 ms clip):
everything up to a directory of WAVs is shared, everything after it is not.

See plan.md. The modules here were moved out of train.py without behaviour change -
sixteen runs of tuning.md are calibrated against that behaviour, so this package is
code motion, not cleanup. Anything that looks like it wants tidying probably encodes
a measured result; check tuning.md before changing it.
"""
