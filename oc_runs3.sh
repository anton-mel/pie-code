#!/bin/zsh
# Real OpenCode through one shared OpenCode server: pie at 4 and 8, with and without shared prompts.
cd ${0:A:h}
export OPENCODE_PASSWORD=$(grep -o "server password [^ ]*" oc-server.log | awk '{print $3}') OC_SERVER=http://127.0.0.1:4096
touch .bench-running
for n in 4 8; do ./oc_serve.sh && ./oc_bench.sh pie $n; done
for n in 4 8; do ./oc_serve.sh --no-share && ./oc_bench.sh pie-no-share $n; done
pkill -f "python -u serve.py"; pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv
rm -f .bench-running
