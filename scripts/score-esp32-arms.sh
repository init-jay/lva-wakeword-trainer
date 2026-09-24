#!/bin/bash
# Score the eight 10k ESP32 models (4 flat + 4 balanced) on ONE negative set, by
# explicit run-dir path. Every earlier ESP32 number in this repo came from a /tmp
# copy whose name did not match its bytes, so nothing here goes through /tmp.
#
# Why corpus ids are hardcoded: the ESP32 table was rebuilt once already from
# c5f9430c-* dirs (the 5000-step corpus, which also has real_copies=10 and so
# looked like a 10x run). Only c574e978 (flat) and c5a47eeb (balanced) are 10k.
cd "$(dirname "$0")/.." || exit 1
OUT=/tmp/arms8.txt
: > "$OUT"
for TAG in \
  55e182a-dirty-c574e978-hd6b366e 55e182a-dirty-c574e978-hbd0e5de \
  55e182a-dirty-c574e978-h3be78bf 55e182a-dirty-c574e978-hd5948fa \
  55e182a-dirty-c5a47eeb-hd6b366e 55e182a-dirty-c5a47eeb-hbd0e5de \
  55e182a-dirty-c5a47eeb-h3be78bf 55e182a-dirty-c5a47eeb-hd5948fa ; do
  M=output/hey_seeree/mww/$TAG/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
  if [ ! -f "$M" ]; then echo "$TAG MISSING" >> "$OUT"; continue; fi
  echo "### $TAG md5=$(md5 -q "$M")" >> "$OUT"
  eval/.venv/bin/python tools/score_margins.py --model "$M" --sliding-window-size 5 \
      --adv-fa-budget 5 --top 0 >> "$OUT" 2>>"$OUT"
done
echo DONE >> "$OUT"
