#!/bin/zsh
# The team workflow on every engine. Rows go to team.jsonl.
cd ${0:A:h}
run() { uv run -q --project pie/python/client python -u team.py "$@" 2>&1 | tail -1 | cut -c1-220; pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; sleep 5; }
touch .bench-running
for n in 4 8 16; do run team --specialists $n --timeout 600; done
for n in 4 8 16; do run separate --specialists $n --timeout 600; done
for n in 4 8 16; do run llama --specialists $n; done
for n in 4 8; do run ollama --specialists $n; done
rm -f .bench-running
