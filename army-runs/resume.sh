#!/bin/zsh
# Waits for the team run already in flight, marks it done, then resumes the queue in its new order.
cd /Users/antonmelnychuk/pie-agents
while kill -0 22440 2>/dev/null; do sleep 30; done
grep -q '"solved"' army-runs/pie-team.out && touch army-runs/pie-team.done
./run_army.sh 15 >> army-runs/night.log 2>&1
