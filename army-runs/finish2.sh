#!/bin/zsh
# After llama.cpp single: the pie vote again with room for eight voters, then LatentMAS and results.
cd ${0:A:h:h}
py=latentmas-ref/.venv/bin/python
while pgrep -f "army.py bench --engine llama" > /dev/null; do sleep 30; done
grep -q '"solved"' army-runs/llama-single.out && touch army-runs/llama-single.done
lsof -ti tcp:8098 -sTCP:LISTEN | xargs kill 2>/dev/null; pkill -f llama-server; sleep 8
$py army.py bench --engine pie --mode vote --n 15 --config pie-army-vote.toml > army-runs/pie-vote.out 2>&1 && touch army-runs/pie-vote.done
./run_latent.sh &
(cd latentmas-ref && ./runs.sh > runs/runs.log 2>&1) &
wait
$py results.py > RESULTS.md
