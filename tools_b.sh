#!/bin/zsh
# Part B: context size, tool time, pie with everything in memory, llama.cpp with one slot, repeats.
cd ${0:A:h}
run() { uv run -q --project pie/python/client python -u tools.py "$@" 2>&1 | tail -1 | cut -c1-200; pkill -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; sleep 5; }
touch .bench-running
for c in 2000 12000; do
  run pie --agents 8 --context $c --variant parked --timeout 600
  run llama --agents 8 --slots 8 --context $c --variant slot-per-agent
done
for t in 0 10; do
  run pie --agents 8 --tool $t --variant parked --timeout 600
  run llama --agents 8 --slots 8 --tool $t --variant slot-per-agent
done
for n in 8 16; do run pie --agents $n --config pie-wide.toml --variant in-memory --timeout 600; done
for n in 8 16; do run pie --agents $n --variant parked --timeout 600; done
run llama --agents 8 --slots 1 --variant one-slot
run ollama --agents 8 --context 2000
run ollama --agents 8 --tool 10
run ollama --agents 8 --context 12000
rm -f .bench-running
