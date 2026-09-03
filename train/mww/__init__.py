"""microWakeWord training backend - the ESP32 target.

Separate from the openWakeWord pipeline by necessity (mWW needs numpy>=2, this repo
pins numpy<2) and by design: the two produce different artifacts for different
targets, and their numbers are not comparable. See plan.md.

What IS shared is `corpus/` - both trainers consume the same directories of 16 kHz
mono WAVs. mWW reads them through `Clips(input_directory, file_pattern)`, so nothing
generated here needs converting into its feature format.
"""
