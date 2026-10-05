#!/bin/zsh
# One question, the same model (Qwen3.8-27B) two ways: OpenCode, one agent on llama.cpp, the usual
# local stack; then our agent army on pie, live. Ends with both answers and times side by side.
#
#   ./demo.sh                       # AIME 2025 #4 (answer 117)
#   ./demo.sh "your question" code  # mode: code (default), tree, team, single, vote
cd ${0:A:h}
q=${1:-"Find the number of ordered pairs (x,y), where both x and y are integers between -100 and 100, inclusive, such that 12x^2-xy-6y^2=0."}
mode=${2:-code}
py=latentmas-ref/.venv/bin/python
ask="$q

Reason step by step, and put your final answer within \\boxed{}."

print -P "%B━━ 1. OpenCode, one agent, on llama.cpp ━━%b"
# A background OpenCode service left by an earlier run captures new runs and hangs them.
pkill -f "opencode serve --service"; pkill -f "llama-b11312/llama-server"; sleep 2
./oc_llama.sh 1 || exit 1
d=$(mktemp -d)
cp -R oc-runs/home-template $d/.home
cat > $d/opencode.json <<JSON
{"\$schema": "https://opencode.ai/config.json",
 "provider": {"local": {"npm": "@ai-sdk/openai-compatible", "name": "local", "options": {"baseURL": "http://127.0.0.1:8098/v1"},
   "models": {"local": {"name": "local", "tool_call": true}}}},
 "model": "local/local"}
JSON
start=$(date +%s)
(cd $d && HOME=$d/.home XDG_DATA_HOME=$d/.home/data XDG_STATE_HOME=$d/.home/state XDG_CONFIG_HOME=$d/.home/config \
  XDG_CACHE_HOME=$d/.home/cache opencode run --standalone --auto "$ask" | tee $d/answer.txt)
oc_secs=$(( $(date +%s) - start ))
oc_answer=$(grep -o '\\boxed{[^}]*}' $d/answer.txt | tail -1)
pkill -f "llama-b11312/llama-server"; pkill -f "opencode serve --service"; sleep 3

print -P "\n%B━━ 2. Our agent army, on pie ━━%b"
start=$(date +%s)
$py army.py ask "$q" --mode $mode | tee $d/army.txt
army_secs=$(( $(date +%s) - start ))
army_answer=$(sed 's/\x1b\[[0-9;]*[A-Za-z]//g' $d/army.txt | grep -o 'Answer: [^ ]*' | tail -1)

print -P "\n%B━━ Result ━━%b"
printf "  OpenCode on llama.cpp   %-22s %4d s\n" "${oc_answer:-no boxed answer}" $oc_secs
printf "  agent army on pie       %-22s %4d s\n" "${army_answer#Answer: }" $army_secs
print "  (OpenCode's time starts once llama.cpp has loaded the model; the army's includes starting pie)"
