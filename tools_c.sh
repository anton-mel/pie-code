#!/bin/zsh
# Part C: llama.cpp with a prompt cache big enough for every agent, and pie at 16 agents again.
cd ${0:A:h}
run() { uv run -q --project pie/python/client python -u tools.py "$@" 2>&1 | tail -1 | cut -c1-200; pkill -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; sleep 5; }
touch .bench-running
for n in 8 16; do run llama --agents $n --slots $n --variant slot-per-agent-big-cache --llama-args "--cache-ram 24576"; done
rm -f .bench-running
