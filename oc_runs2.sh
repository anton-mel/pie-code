#!/bin/zsh
# Real OpenCode armies through one shared OpenCode server: llama.cpp at 4, then pie with and without shared prompts.
cd ${0:A:h}
export OPENCODE_PASSWORD=$(grep -o "server password [^ ]*" oc-server.log | awk '{print $3}') OC_SERVER=http://127.0.0.1:4096
touch .bench-running
./oc_llama.sh 4 && ./oc_bench.sh llama 4
pkill -f "llama-b11312/llama-server"; sleep 3
for n in 1 4 8; do ./oc_serve.sh && ./oc_bench.sh pie $n; done
for n in 4 8; do ./oc_serve.sh --no-share && ./oc_bench.sh pie-no-share $n; done
pkill -f "python -u serve.py"; pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv
rm -f .bench-running
