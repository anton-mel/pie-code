#!/bin/zsh
# Runs every arm of the comparison, one model at a time, into results.jsonl.
cd ${0:A:h}
arm() {
  until [ ! -e .mem-running ] && ! pgrep -qf "llama-server|ollama runner|Runner.Worker"; do sleep 5; done
  touch .bench-running
  "$@" | tee /dev/stderr | grep '^{' >> results.jsonl
  rm -f .bench-running
}
pie() { uv run -q --project pie/python/client python -u bench.py "$@" }
: > results.jsonl
arm pie --agents 6 --context 6000
arm pie --agents 6 --context 6000 --pie ~/pie/target/release/pie --timeout 90
arm uv run -q --project pie/python/client python -u llama.py --agents 6 --context 6000
arm uv run -q --project pie/python/client python -u llama.py --agents 6 --context 6000 --cache-ram 0
