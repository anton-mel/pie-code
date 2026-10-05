#!/bin/zsh
# Overnight: the agent army against one agent and majority vote on pie, and the same team on
# llama.cpp, AIME 2025 (first 15 problems), Qwen3.8-27B. One engine on the GPU at a time, so every
# time is clean. Then LatentMAS (pie runs and the authors' reference), then the results table.
cd ${0:A:h}
py=latentmas-ref/.venv/bin/python
n=${1:-15}

bench() {
  local engine=$1 mode=$2
  [ -f army-runs/$engine-$mode.done ] && return
  $py army.py bench --engine $engine --mode $mode --n $n > army-runs/$engine-$mode.out 2>&1 && touch army-runs/$engine-$mode.done
}

mkdir -p army-runs
# The headline comparison first: the same team on both engines.
bench pie team
bench llama team
bench pie single
bench llama single
bench pie vote

./run_latent.sh &
(cd latentmas-ref && ./runs.sh > runs/runs.log 2>&1) &
wait
$py results.py > RESULTS.md
