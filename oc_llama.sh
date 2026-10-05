#!/bin/zsh
# llama-server for the OpenCode benchmark: a slot per agent, OpenAI API, thinking off.
cd ${0:A:h}
n=$1
pkill -f "llama-b11312/llama-server"; sleep 2
model=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/*.gguf | head -1)
(~/.cache/pie-evals/baselines/llama-b11312/llama-server -m $model --port 8098 -ngl all -c $(( 16384 * n )) -np $n --jinja \
  --chat-template-kwargs '{"enable_thinking": false}' > llama.log 2>&1 &)
for i in $(seq 1 100); do sleep 2; curl -s http://127.0.0.1:8098/health | grep -q ok && exit 0; done
echo "llama-server did not start"; exit 1
