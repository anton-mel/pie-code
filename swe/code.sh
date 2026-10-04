#!/bin/zsh
# pie-code (native harness): N agents at once.
cd ${0:A:h}
n=$1; out=runs/pie-code-$n; rm -rf $out; mkdir -p $out; rm -f pie_code.jsonl
filter="^($(python3 -c "import json;print('|'.join(json.load(open('tasks16.json'))))"))$"
start=$(date +%s)
PYTHONPATH=$PWD MSWEA_COST_TRACKING=ignore_errors .venv/bin/mini-extra swebench --subset verified --split test --filter "$filter" \
  --workers $n -c swebench_backticks.yaml -c local.yaml --model-class pie_code.PieCodeModel -o $out > $out/run.log 2>&1
echo "{\"engine\": \"pie-code\", \"agents\": $n, \"wall_s\": $(( $(date +%s) - start ))}" | tee -a runs.jsonl
pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; cp pie_code.jsonl $out/ 2>/dev/null
