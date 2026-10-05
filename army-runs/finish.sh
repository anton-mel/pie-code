#!/bin/zsh
# After the 07:00 cap: let the llama.cpp single run finish, clear every llama.cpp server, rerun the
# pie baselines on a clean GPU, then LatentMAS and the results page.
cd ${0:A:h:h}
py=latentmas-ref/.venv/bin/python
for run in pie:single pie:vote llama:single; do
  engine=${run%:*} mode=${run#*:}
  [ -f army-runs/$engine-$mode.done ] || { $py army.py bench --engine $engine --mode $mode --n 15 > army-runs/$engine-$mode.out 2>&1 && touch army-runs/$engine-$mode.done; }
  lsof -ti tcp:8098 -sTCP:LISTEN | xargs kill 2>/dev/null; sleep 5
done
./run_latent.sh &
(cd latentmas-ref && ./runs.sh > runs/runs.log 2>&1) &
wait
$py results.py > RESULTS.md
