#!/bin/zsh
# SWE-bench Verified, 16 easy tasks, mini-swe-agent (text-based), N agents at once on one engine.
#   swe/run.sh pie 16 | swe/run.sh llama 16 | swe/run.sh ollama 4
cd ${0:A:h}
engine=$1; n=$2
root=${0:A:h:h}
out=runs/$engine-$n; rm -rf $out; mkdir -p $out
export MSWEA_COST_TRACKING=ignore_errors
case $engine in
  pie)
    : > $root/serve.jsonl
    PIE_CONTENTION_TRACE_MS=2000 $root/oc_serve.sh --config pie-swe.toml --sessions $(( n + 2 )) || exit 1
    base=http://127.0.0.1:8100/v1; model=openai/local ;;
  llama)
    pkill -f "llama-b11312/llama-server"; sleep 2
    gguf=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/*.gguf | head -1)
    (~/.cache/pie-evals/baselines/llama-b11312/llama-server -m $gguf --port 8098 -ngl all -c $(( 32768 * n )) -np $n \
      -ctk q8_0 -ctv q8_0 --jinja --chat-template-kwargs '{"enable_thinking": false}' > $root/llama.log 2>&1 &)
    for i in $(seq 1 120); do sleep 2; curl -s http://127.0.0.1:8098/health | grep -q ok && break; done
    base=http://127.0.0.1:8098/v1; model=openai/local ;;
  ollama)
    (OLLAMA_HOST=127.0.0.1:11500 OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=$n ollama serve > $root/ollama.log 2>&1 &)
    for i in $(seq 1 60); do sleep 2; curl -s http://127.0.0.1:11500/ | grep -q running && break; done
    base=http://127.0.0.1:11500/v1; model=openai/qwen3.8:27b ;;
esac
filter="^($(python3 -c "import json;print('|'.join(json.load(open('tasks16.json'))))"))$"
start=$(date +%s)
.venv/bin/mini-extra swebench --subset verified --split test --filter "$filter" --workers $n \
  -c swebench_backticks.yaml -c local.yaml -c model.model_name=$model -c model.model_kwargs.api_base=$base \
  -o $out > $out/run.log 2>&1
end=$(date +%s)
echo "{\"engine\": \"$engine\", \"agents\": $n, \"wall_s\": $(( end - start ))}" | tee -a runs.jsonl
case $engine in
  pie) pkill -f "python -u serve.py"; pkill -9 -f "pie-mpp/target/release/pie -c"; rm -rf ~/.pie/cache/kv; cp $root/serve.jsonl $out/serve.jsonl ;;
  llama) pkill -f "llama-b11312/llama-server" ;;
  ollama) lsof -ti tcp:11500 -sTCP:LISTEN | xargs kill 2>/dev/null ;;  # only the private server; the Ollama app on 11434 stays
esac
