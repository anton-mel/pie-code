#!/bin/zsh
# N OpenCode agents fix the same bug at once, each in its own copy of oc-demo.
#   ./oc_bench.sh pie 4      (serve.py must be listening on :8100)
#   ./oc_bench.sh ollama 4   (a private Ollama on :11500)
# Prints one JSON line: seconds per agent, total, and how many fixed the bug.
cd ${0:A:h}
engine=$1; n=$2
case $engine in
  pie|pie-no-share) url=http://127.0.0.1:8100/v1; model=pie-qwen3.8-27b; opts='{}' ;;
  ollama) url=http://127.0.0.1:11500/v1; model=qwen3.8:27b; opts='{"reasoningEffort": "none"}' ;;
  llama) url=http://127.0.0.1:8098/v1; model=qwen3.8-27b; opts='{}' ;;
esac
if [ -n "$OC_SERVER" ]; then ocflags=(--server $OC_SERVER); else ocflags=(--standalone); pkill -x opencode 2>/dev/null; sleep 1; fi  # stale OpenCode processes from earlier runs block new ones
rm -rf oc-runs/$engine-$n; mkdir -p oc-runs/$engine-$n
start=$(date +%s.%N)
for k in $(seq 1 $n); do
  d=oc-runs/$engine-$n/$k; mkdir -p $d; cp oc-demo/calc.py.orig $d/calc.py; cp oc-demo/test_calc.py $d/
  cat > $d/opencode.json <<JSON
{"\$schema": "https://opencode.ai/config.json",
 "provider": {"local": {"npm": "@ai-sdk/openai-compatible", "name": "local", "options": {"baseURL": "$url"},
   "models": {"$model": {"name": "$model", "tool_call": true, "options": $opts}}}},
 "model": "local/$model"}
JSON
  cp -R oc-runs/home-template $d/.home  # a home that has already fetched the provider package
  sleep 1                                # 8 OpenCode starts in the same instant hang before any request
  (cd $d && git init -q && git add -A && GIT_AUTHOR_NAME=bench GIT_AUTHOR_EMAIL=bench@local GIT_COMMITTER_NAME=bench GIT_COMMITTER_EMAIL=bench@local git commit -qm "agent $engine-$n-$k" && s=$(date +%s.%N) && HOME=$PWD/.home XDG_DATA_HOME=$PWD/.home/data XDG_STATE_HOME=$PWD/.home/state XDG_CONFIG_HOME=$PWD/.home/config XDG_CACHE_HOME=$PWD/.home/cache opencode run $ocflags --auto "There is a bug in calc.py. Run python3 test_calc.py, find the bug, fix it, and run the test again to confirm." > run.out 2>&1; echo $(( $(date +%s.%N) - s )) > seconds) &
done
wait
total=$(( $(date +%s.%N) - start ))
python3 - $engine $n $total <<'PY'
import json, subprocess, sys, pathlib
engine, n, total = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
runs = sorted(pathlib.Path(f"oc-runs/{engine}-{n}").iterdir(), key=lambda p: int(p.name))
secs = [round(float((r / "seconds").read_text()), 1) for r in runs]
fixed = sum(subprocess.run(["python3", "test_calc.py"], cwd=r, capture_output=True).returncode == 0 for r in runs)
row = {"engine": engine, "agents": n, "total_s": round(total, 1), "agent_s": secs, "fixed": f"{fixed}/{n}"}
print(json.dumps(row))
open("oc_bench.jsonl", "a").write(json.dumps(row) + "\n")
PY
