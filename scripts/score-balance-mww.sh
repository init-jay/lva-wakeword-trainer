#!/bin/bash
# Score the eight balance-sweep mww models + incumbent + 42f8982 from their OWN run dirs.
# Why: the corpus dir under data/corpus/hey_seeree/mww/<cid>/ holds ONE stream_state file per
# corpus, shared by all four runs built against it - copying from there gives per-ARM bytes, not
# per-seed bytes. The run dir copy is the only per-run artifact.
cd /Users/jay/Documents/Repos/lva-wakeword-trainer
mkdir -p logs/scorecurves logs/balancescore
for tag in 55e182a-dirty-c574e978-hd6b366e 55e182a-dirty-c574e978-hbd0e5de 55e182a-dirty-c574e978-h3be78bf 55e182a-dirty-c574e978-hd5948fa 55e182a-dirty-c5a47eeb-hd6b366e 55e182a-dirty-c5a47eeb-hbd0e5de 55e182a-dirty-c5a47eeb-h3be78bf 55e182a-dirty-c5a47eeb-hd5948fa; do
  f=output/hey_seeree/mww/$tag/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
  [ -f "$f" ] || { echo "$tag MISSING"; continue; }
  echo "### $tag md5=$(md5 -q "$f" | cut -c1-8)"
  eval/.venv/bin/python tools/score_margins.py --model "$f" --sliding-window-size 5 --adv-fa-budget 5 \
      --csv logs/scorecurves/$tag.csv > logs/balancescore/$tag.txt 2>&1
  grep -aE "operating point|PER SPEAKER|^    (jay|jen|ryan) |POOLED|n=.*extend|BEST-AVAILABLE" logs/balancescore/$tag.txt | head -12
  echo
done
echo "### incumbent (deploy copy) md5=$(md5 -q deploy/esp32-mww/hey_seeree_ecbf160-dirty-da01854d.tflite | cut -c1-8)"
eval/.venv/bin/python tools/score_margins.py --model deploy/esp32-mww/hey_seeree_ecbf160-dirty-da01854d.tflite --sliding-window-size 5 --adv-fa-budget 5 > /tmp/inc.txt 2>&1
grep -aE "operating point|^    (jay|jen|ryan) |POOLED" /tmp/inc.txt | head -6
echo "### 42f8982-h5f6f353"
eval/.venv/bin/python tools/score_margins.py --model output/hey_seeree/mww/42f8982-ce1500f5-h5f6f353/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite --sliding-window-size 5 --adv-fa-budget 5 > /tmp/m42.txt 2>&1
grep -aE "operating point|^    (jay|jen|ryan) |POOLED" /tmp/m42.txt | head -6
echo DONE
