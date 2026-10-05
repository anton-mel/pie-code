#!/bin/zsh
# Real OpenCode, N agents at once fixing a bug each: llama.cpp, then pie with and without shared prompts.
cd ${0:A:h}
touch .bench-running
./oc_llama.sh 8 && ./oc_bench.sh llama 8
pkill -f "llama-b11312/llama-server"; sleep 3
for n in 1 4 8; do ./oc_serve.sh && ./oc_bench.sh pie $n; done
for n in 4 8; do ./oc_serve.sh --no-share && ./oc_bench.sh pie $n | sed 's/"engine": "pie"/"engine": "pie-no-share"/'; done
pkill -f "python -u serve.py"; pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv
rm -f .bench-running
