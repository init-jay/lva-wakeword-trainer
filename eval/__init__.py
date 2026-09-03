"""Shared evaluation layer for both trainers (plan.md phase 3).

`backends` is the whole point: one streaming contract that an openWakeWord .onnx and
a microWakeWord streaming .tflite can both honour, so every measurement downstream of
it - per-category false accepts, per-speaker scoring, matched-precision comparison,
latency - is arithmetic over scores and does not care which trainer produced them.
"""
