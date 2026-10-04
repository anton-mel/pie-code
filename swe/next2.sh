#!/bin/zsh
# After next.sh: the long-horizon pair (E), then the co-designed branch-and-prune agent (D), scoring each.
cd ${0:A:h}
while pgrep -f "next.sh" > /dev/null; do sleep 60; done
score() { .venv/bin/python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path runs/$1/preds.json --max_workers 8 --run_id $1 --report_dir runs/$1 > runs/$1/eval.log 2>&1; }
echo "- $(date '+%H:%M') Started E: pie, 16 agents, 120 steps." >> NIGHT.md
./run_long.sh pie 16
echo "- $(date '+%H:%M') E pie done: $(tail -1 runs.jsonl). Started E: llama.cpp, 16 agents on 8 slots, 120 steps." >> NIGHT.md
./run_long.sh llama 16
echo "- $(date '+%H:%M') E llama done: $(tail -1 runs.jsonl). Scoring E." >> NIGHT.md
score pie-16-long; score llama-16-long
echo "- $(date '+%H:%M') Started D: branch and prune, 16 tasks x 2 attempts, on pie." >> NIGHT.md
pkill -f "python -u serve.py"; pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv
start=$(date +%s)
PIE_CODE_SHARE=all PIE_CODE_CONFIG=pie-swe.toml PYTHONPATH=$PWD MSWEA_COST_TRACKING=ignore_errors .venv/bin/python bp.py --attempts 2 --tasks 16 -o runs/bp-16x2 > runs/bp.log 2>&1
echo "{\"engine\": \"pie-code-bp\", \"agents\": 32, \"wall_s\": $(( $(date +%s) - start ))}" >> runs.jsonl
pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv
echo "- $(date '+%H:%M') D done: $(tail -1 runs.jsonl). Scoring D." >> NIGHT.md
score bp-16x2
echo "- $(date '+%H:%M') All runs scored." >> NIGHT.md
