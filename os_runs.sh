#!/bin/zsh
# The agent-OS comparison, small: team, search and undo at 4 and 8 agents.
cd ${0:A:h}
clean() { pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; sleep 5; }
team() { uv run -q --project pie/python/client python -u team.py "$@" 2>&1 | tail -1 | cut -c1-200; clean; }
os() { uv run -q --project pie/python/client python -u os_bench.py "$@" 2>&1 | tail -1 | cut -c1-200; clean; }
touch .bench-running
for n in 4 8; do team team --specialists $n --timeout 600; done
team separate --specialists 4 --timeout 600
for n in 4 8; do team llama --specialists $n; done
for n in 4 8; do os search team --branches $n --timeout 600; done
os search separate --branches 4 --timeout 600
for n in 4 8; do os search llama --branches $n; done
os search ollama --branches 4
os undo team --agents 4 --timeout 600
os undo separate --agents 4 --timeout 600
os undo llama --agents 4
rm -f .bench-running
