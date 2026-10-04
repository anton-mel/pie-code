#!/bin/zsh
# After B (pie, 16): A (llama.cpp, 16), then official scoring of both.
cd ${0:A:h}
while kill -0 20968 2>/dev/null; do sleep 30; done
echo "- $(date '+%H:%M') B finished: $(grep '"engine": "pie", "agents": 16' runs.jsonl | tail -1). Started A (llama.cpp, 16)." >> NIGHT.md
./run.sh llama 16
echo "- $(date '+%H:%M') A finished: $(grep '"engine": "llama", "agents": 16' runs.jsonl | tail -1). Scoring A and B." >> NIGHT.md
for name in pie-16 llama-16; do
  .venv/bin/python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path runs/$name/preds.json --max_workers 8 --run_id $name --report_dir runs/$name > runs/$name/eval.log 2>&1
done
echo "- $(date '+%H:%M') Scored A and B." >> NIGHT.md
