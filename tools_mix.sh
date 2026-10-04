#!/bin/zsh
# Tool-call share sweep at 8 agents: thinking steps (256 tokens, no pause) mixed with tool steps.
cd ${0:A:h}
run() { uv run -q --project pie/python/client python -u tools.py "$@" --agents 8 --steps 8 2>&1 | tail -1 | cut -c1-200; pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; sleep 5; }
touch .bench-running
for s in 0 0.25 0.5 0.75 1; do run pie --config pie-full.toml --variant mix --tool-share $s --timeout 600; done
for s in 0 0.25 0.5 0.75 1; do run llama --slots 8 --variant mix --tool-share $s; done
for s in 0 0.5 1; do run ollama --variant mix --tool-share $s; done
rm -f .bench-running
