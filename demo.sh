#!/bin/zsh
# One question, the same model (Qwen3.8-27B) two ways: OpenCode, one agent on llama.cpp, the usual
# local stack; then our agent on pie. Each runs the question `runs` times and both clocks measure
# solving only, with the model already loaded. Ends with the median and every run.
#
#   ./demo.sh                                  # AIME 2025 #4 (answer 117), 3 runs each
#   ./demo.sh "your question" [runs] [mode]    # mode: code (default), tree, team, single, vote
cd ${0:A:h}
q=${1:-"Find the number of ordered pairs (x,y), where both x and y are integers between -100 and 100, inclusive, such that 12x^2-xy-6y^2=0."}
runs=${2:-3}
mode=${3:-code}
py=latentmas-ref/.venv/bin/python
ask="$q

Reason step by step, and put your final answer within \\boxed{}."
median() { print -l "$@" | sort -n | awk '{a[NR]=$1} END {print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}' }

print -P "%B━━ 1. OpenCode, one agent, on llama.cpp ━━%b"
# A background OpenCode service left by an earlier run captures new runs and hangs them.
pkill -f "opencode serve --service"; pkill -f "llama-b11312/llama-server"; sleep 2
./oc_llama.sh 1 || exit 1
oc_times=(); oc_answers=()
for k in $(seq 1 $runs); do
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
    XDG_CACHE_HOME=$d/.home/cache opencode run --standalone --auto "$ask" > $d/answer.txt 2>&1)
  oc_times+=($(( $(date +%s) - start )))
  oc_answers+=("$(grep -o '\\boxed{[^}]*}' $d/answer.txt | tail -1 | sed 's/\\boxed{\(.*\)}/\1/')")
  print "  run $k: ${oc_answers[-1]:-no answer}  ${oc_times[-1]} s"
  pkill -f "opencode serve --service"
done
pkill -f "llama-b11312/llama-server"
# The GPU memory llama.cpp held must be free before pie loads the model.
while pgrep -f "llama-b11312/llama-server" > /dev/null; do sleep 1; done; sleep 8

print -P "\n%B━━ 2. Our agent on pie ($mode mode) ━━%b"
army_times=(); army_answers=()
for k in $(seq 1 $runs); do
  # The live view on the first run only. The time is the run's own, which starts once pie has
  # loaded the model, like OpenCode's above.
  if [ $k -eq 1 ]; then
    $py army.py ask "$q" --mode $mode --seed $k | tee /tmp/army-demo.txt
  else
    $py army.py ask "$q" --mode $mode --seed $k > /tmp/army-demo.txt 2>&1
  fi
  line=$(sed 's/\x1b\[[0-9;]*[A-Za-z]//g' /tmp/army-demo.txt | grep 'Answer:' | tail -1)
  army_answers+=("$(print -r -- $line | awk '{print $2}')")
  army_times+=($(print -r -- $line | grep -oE '[0-9]+ s on pie' | grep -oE '^[0-9]+'))
  print "  run $k: ${army_answers[-1]:-no answer}  ${army_times[-1]} s"
done

print -P "\n%B━━ Result: time to solve, model already loaded, median of $runs runs ━━%b"
printf "  OpenCode on llama.cpp   %5s s   runs: %-16s answers: %s\n" $(median $oc_times) "${oc_times[*]}" "${oc_answers[*]}"
printf "  our agent on pie        %5s s   runs: %-16s answers: %s\n" $(median $army_times) "${army_times[*]}" "${army_answers[*]}"
