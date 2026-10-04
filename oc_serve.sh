#!/bin/zsh
# Restarts serve.py (and its pie) from scratch, then waits until it listens.
cd ${0:A:h}
pkill -f "python -u serve.py"; pkill -9 -f "pie-mpp/target/release/pie -c"; sleep 2; rm -rf ~/.pie/cache/kv
(uv run -q --project pie/python/client --with aiohttp --with transformers --with jinja2 python -u serve.py "$@" > serve.out 2>&1 &)
for i in $(seq 1 100); do sleep 3; grep -q "serving" serve.out && exit 0; done
echo "serve.py did not start"; exit 1
