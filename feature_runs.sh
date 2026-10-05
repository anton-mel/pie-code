#!/bin/zsh
# Prefetch and shared prompts, small: before/after on pie, and llama.cpp for the army.
cd ${0:A:h}
clean() { pkill -9 -f "pie-mpp/target/release/pie -c"; pkill -f "python -u serve.py"; rm -rf ~/.pie/cache/kv; sleep 5; }
touch .bench-running
# Prefetch: 8 agents on the small pool, so they park on SSD between tools.
for i in 1 2; do
  for lead in 0 1.0; do
    uv run -q --project pie/python/client python -u tools.py pie --config pie.toml --agents 8 --variant prefetch-$lead --prefetch $lead --timeout 600 2>&1 | tail -1 | cut -c1-200; clean
  done
done
# Shared prompts: an OpenCode-style army through the OpenAI-style proxy, and llama.cpp.
for n in 4 8; do
  ./oc_serve.sh && uv run -q python -u os_army.py pie --agents $n --variant shared 2>&1 | tail -1; clean
  ./oc_serve.sh --no-share && uv run -q python -u os_army.py pie --agents $n --variant no-share 2>&1 | tail -1; clean
  uv run -q python -u os_army.py llama --agents $n 2>&1 | tail -1; clean
done
rm -f .bench-running
