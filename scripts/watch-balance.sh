#!/bin/bash
# Progress check for the balance sweeps. The trainer logs put TF progress on one
# \r-separated line, so `tail` costs 13K tokens for nothing — normalise first
# (tr '\r' '\n'), then grep the sweep's own markers, and never print raw log lines.
cd /Users/jay/Documents/Repos/lva-wakeword-trainer || exit 1
echo "now $(date '+%H:%M:%S %Z')  running procs: $(pgrep -f 'sweep.py|train\.(oww|mww)\.|microwakeword.model' | wc -l | tr -d ' ')"
echo "chain: $(cat logs/sweep-chain.log 2>/dev/null | tr '\n' ' ')"
for t in oww mww; do
  f=logs/sweep-$t-balance.log
  [ -f "$f" ] || continue
  echo "--- $t ($f)"
  tr '\r' '\n' < "$f" \
    | grep -aiE "^# point|SWEEP FAILED|Training FAILED|done: [0-9]+ run|filed:|=== (corpus|features|training|eval)|matched-FA|balance|REFUSE|refus|die|exit=" \
    | grep -avE "Accuracy|loss" | tail -8
done
echo "--- corpus dirs:"; ls -dT data/corpus/hey_seeree/*/ 2>/dev/null | sed 's|.*hey_seeree/||'
echo "--- ledger: $(wc -l < output/hey_seeree/runs.jsonl | tr -d ' ') rows"
