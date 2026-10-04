#!/bin/zsh
# 8 agents at once, equal memory: pie (B8) then llama.cpp (A8), each with a stall guard, then scoring;
# then the long-horizon pair at 8 agents.
cd ${0:A:h}
log() { echo "- $(date '+%H:%M') $*" >> NIGHT.md; }
guarded() {  # run "$@" in the background; stop it if no agent request completes for 15 minutes
  "$@" &
  local pid=$! last=$(date +%s) seen=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    local n=$(cat ../serve.jsonl ../llama.log 2>/dev/null | wc -c)  # bytes of completed-request logs
    if [ "$n" != "$seen" ]; then seen=$n; last=$(date +%s); fi
    if [ $(( $(date +%s) - last )) -gt 900 ]; then
      log "Stall guard: no progress for 15 min in '$*'; stopped it."
      pkill -P $pid; kill $pid; pkill -f "mini-extra swebench"; pkill -f "python -u serve.py"; pkill -9 -f "pie-mpp/target/release/pie -c"; pkill -f "llama-b11312/llama-server"
      docker ps -q | xargs -r docker rm -f >/dev/null 2>&1; return 1
    fi
  done
}
score() { .venv/bin/python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path runs/$1/preds.json --max_workers 6 --run_id $1 --report_dir runs/$1 > runs/$1/eval.log 2>&1; }
touch ../.bench-running
log "Night plan reset: 16 agents plus 16 containers OOMs the GPU mid-run (B hung 3 h after an OOM at 01:03). Now 8 agents at once on both engines, equal memory (~24 GB each), stall guard on."
guarded ./run.sh pie 8;   log "B8 done: $(tail -1 runs.jsonl)"
guarded ./run.sh llama 8; log "A8 done: $(tail -1 runs.jsonl)"
score pie-8; score llama-8; log "Scored B8 and A8."
guarded ./run_long.sh pie 8;   log "E8 pie done: $(tail -1 runs.jsonl)"
guarded ./run_long.sh llama 8; log "E8 llama done: $(tail -1 runs.jsonl)"
score pie-8-long; score llama-8-long; log "Scored E8."
rm -f ../.bench-running
