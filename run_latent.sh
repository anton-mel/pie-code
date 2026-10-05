#!/bin/zsh
# LatentMAS on pie: the paper's settings on Qwen3.5-4B and on its own model, Qwen3-4B (bf16), 100 questions each.
cd ${0:A:h}
py=latentmas-ref/.venv/bin/python
while pgrep -f "latent_runs.sh|queue.sh" > /dev/null; do sleep 30; done
q35=(--model Qwen/Qwen3.5-4B --config pie-latent.toml --scale 0.6558 --teams 32 --n 100)
$py latentmas.py --method single --task gsm8k $q35 > latent-runs/q35-gsm8k-single.out 2>&1
$py latentmas.py --method text --task gsm8k $q35 > latent-runs/q35-gsm8k-text.out 2>&1
$py latentmas.py --method latent --task arc_challenge --steps 10 $q35 > latent-runs/q35-arc_challenge-latent10.out 2>&1
$py latentmas.py --method text --task arc_challenge $q35 > latent-runs/q35-arc_challenge-text.out 2>&1
q3=(--model Qwen/Qwen3-4B --config pie-latent-q3.toml --scale 1.0974 --teams 12 --n 100)
for task in arc_challenge gsm8k; do
  $py latentmas.py --method single --task $task $q3 > latent-runs/q3-$task-single.out 2>&1
  $py latentmas.py --method latent --task $task --steps 0 $q3 > latent-runs/q3-$task-latent0.out 2>&1
  $py latentmas.py --method latent --task $task --steps 10 $q3 > latent-runs/q3-$task-latent10.out 2>&1
  $py latentmas.py --method text --task $task $q3 > latent-runs/q3-$task-text.out 2>&1
done
