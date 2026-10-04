#!/bin/zsh
# Part A of the tool-loop benchmark: 1 to 16 agents on every engine. Rows go to tools.jsonl.
cd ${0:A:h}
run() { uv run -q --project pie/python/client python -u tools.py "$@" 2>&1 | tail -1 | cut -c1-200; pkill -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; sleep 5; }
touch .bench-running
for n in 1 2 4 8 16; do run pie --agents $n --variant parked --timeout 600; done
for n in 1 2 4 8 16; do run llama --agents $n --slots $n --variant slot-per-agent; done
for n in 1 2 4 8; do run ollama --agents $n; done
rm -f .bench-running
