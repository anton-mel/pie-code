#!/bin/zsh
# LatentMAS on pie: the paper's settings on Qwen3.5-4B and on its own model, Qwen3-4B (bf16), 100
# questions each. A run whose results file exists is skipped, so a restart resumes.
cd ${0:A:h}
py=latentmas-ref/.venv/bin/python
q35=(--model Qwen/Qwen3.5-4B --config pie-latent.toml --scale 0.6558 --teams 32 --n 100)
q3=(--model Qwen/Qwen3-4B --config pie-latent-q3.toml --scale 1.0974 --teams 12 --n 100)

run() {
  local name=$1; shift
  [ -f latent-runs/$name.json ] && return
  $py latentmas.py "$@" > latent-runs/$name.out 2>&1
}

for task in gsm8k arc_challenge; do
  run Qwen3.5-4B-$task-single --method single --task $task $q35
  run Qwen3.5-4B-$task-latent10 --method latent --task $task --steps 10 $q35
  run Qwen3.5-4B-$task-text --method text --task $task $q35
done
for task in arc_challenge gsm8k; do
  run Qwen3-4B-$task-single --method single --task $task $q3
  run Qwen3-4B-$task-latent0 --method latent --task $task --steps 0 $q3
  run Qwen3-4B-$task-latent10 --method latent --task $task --steps 10 $q3
  run Qwen3-4B-$task-text --method text --task $task $q3
done
