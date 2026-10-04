#!/bin/zsh
# How memory and switch time scale with the number of live agents, per engine.
cd ${0:A:h}
arm() {
  until [ ! -e .bench-running ] && ! pgrep -qf "llama-server|ollama runner|Runner.Worker|release/pie -c"; do sleep 5; done
  touch .bench-running
  sleep 20  # let the last engine's memory return to the system
  uv run -q --project pie/python/client python -u "$@" | tee /dev/stderr | grep '^{' >> capacity.jsonl
  rm -f .bench-running
}
[ ${2:-all} = ollama ] || arm bench.py --agents ${1:-20} --rounds 2
[ ${2:-all} = pie ] || arm ollama.py --agents ${1:-20} --rounds 2
