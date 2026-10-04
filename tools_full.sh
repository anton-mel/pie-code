#!/bin/zsh
# pie with the KV pool sized to free GPU memory, for every setting in TOOL-LOOP.md.
cd ${0:A:h}
run() { uv run -q --project pie/python/client python -u tools.py pie --config pie-full.toml --variant full-pool --timeout 600 "$@" 2>&1 | tail -1 | cut -c1-200; grep -m1 -o "free=[0-9]*/[0-9]*" pie.log; pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; sleep 5; }
touch .bench-running
for n in 1 2 4 8 16; do run --agents $n; done
for c in 2000 12000; do run --agents 8 --context $c; done
for t in 0 10; do run --agents 8 --tool $t; done
rm -f .bench-running
