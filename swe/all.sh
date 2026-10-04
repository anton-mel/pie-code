#!/bin/zsh
# SWE-bench Verified, 16 easy tasks: the 16-agent runs first, then the smaller ones, then official scoring.
cd ${0:A:h}
touch ../.bench-running
./run.sh pie 16
./run.sh llama 16
./code.sh 16
./run.sh pie 1
./run.sh pie 4
./run.sh llama 4
./run.sh ollama 4
rm -f ../.bench-running
for r in runs/*/; do
  name=${r:t}
  [ -f $r/preds.json ] || continue
  .venv/bin/python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path $r/preds.json --max_workers 8 --run_id $name --report_dir $r > $r/eval.log 2>&1
  echo "scored $name"
done
