#!/bin/zsh
# D: branch and prune, 16 tasks, 8 at a time x 2 attempts, on pie; stall guard; then scoring.
cd ${0:A:h}
log() { echo "- $(date '+%H:%M') $*" >> NIGHT.md; }
rm -rf runs/bp-8x2 pie_code.jsonl; touch ../.bench-running
log "Started D: pie-code with branch and prune, 16 tasks, 8 at a time x 2 attempts (16 agents), pie-bp.toml."
start=$(date +%s)
PIE_CODE_SHARE=all PIE_CODE_CONFIG=pie-bp.toml PYTHONPATH=$PWD MSWEA_COST_TRACKING=ignore_errors .venv/bin/python bp.py --attempts 2 --tasks 8 -o runs/bp-8x2 > runs/bp.log 2>&1 &
pid=$! last=$(date +%s) seen=0
while kill -0 $pid 2>/dev/null; do
  sleep 30
  n=$(wc -c < pie_code.jsonl 2>/dev/null || echo 0)
  if [ "$n" != "$seen" ]; then seen=$n; last=$(date +%s); fi
  if [ $(( $(date +%s) - last )) -gt 900 ]; then log "Stall guard stopped D (no progress for 15 min)."; kill $pid; break; fi
done
echo "{\"engine\": \"pie-code-bp\", \"agents\": 16, \"tasks_at_once\": 8, \"attempts\": 2, \"wall_s\": $(( $(date +%s) - start ))}" >> runs.jsonl
pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; docker ps -q | xargs -r docker rm -f >/dev/null 2>&1
cp pie_code.jsonl runs/bp-8x2/ 2>/dev/null
log "D done: $(tail -1 runs.jsonl); OOM errors: $(grep -c Insufficient ../pie.log). Scoring D."
.venv/bin/python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified --predictions_path runs/bp-8x2/preds.json \
  --max_workers 6 --run_id bp-8x2 --report_dir runs/bp-8x2 > runs/bp-8x2/eval.log 2>&1
log "D scored: $(grep -E 'Instances (submitted|resolved)' runs/bp-8x2/eval.log | tr '\n' ' ')"
rm -f ../.bench-running
