"""MLX's decode speed on the same notebook context pie reads: tokens per second after a 6,000-token prompt.

    ~/.cache/pie-evals/baselines/mlx-lm-0.31.3/bin/python mlx_decode.py <hf repo>
"""
import sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_probe_notebook import notebook

model, tokenizer = load(sys.argv[1])
for agent, size in ((0, 300), (1, 6000), (2, 6000), (3, 0)):
    ask = (notebook(agent, size) if size else "") + "\nWrite a long story about a fox and a river.\n"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": ask}], tokenize=False, add_generation_prompt=True)
    stamps = []
    for piece in stream_generate(model, tokenizer, prompt, max_tokens=96):
        stamps.append(time.perf_counter())
    if len(stamps) < 2:
        continue
    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    if agent:
        print(f"{sys.argv[1]} context {size:5d}: decode {rate:6.1f} tok/s over {len(stamps)} tokens (mlx reports {piece.generation_tps:.1f})")
