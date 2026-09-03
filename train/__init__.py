"""Corpus generation and the two trainers (architecture.md, step 2).

`train.corpus` is engine-agnostic and shared; `train.oww` and `train.mww` are the two
trainers, which cannot share a Python environment - openWakeWord pins numpy<2 for
torch, microWakeWord requires numpy>=2. They share audio, never an interpreter.
"""
