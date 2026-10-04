"""How fast MLX reads the same notebooks pie reads, and where its time goes.

    ~/.cache/pie-evals/baselines/mlx-lm-0.31.3/bin/python mlx_probe.py
"""

import time
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

WORDS = "amber birch cedar delta ember fjord grove heron iris jade kelp lotus".split()


def notebook(agent: int, tokens: int) -> str:  # bench.notebook, copied: this runs in MLX's own environment
    lines = [f"Notebook of agent {agent}. Each item has a code word."]
    for item in range(tokens // 14):
        lines.append(f"Item {item}: the code word is {WORDS[(item * 7 + agent * 5) % len(WORDS)]}{agent}.")
    return "\n".join(lines)


def read(model, ids, step: int) -> float:
    """Seconds to push `ids` through the model in passes of `step` tokens."""
    cache = make_prompt_cache(model)
    start = time.perf_counter()
    for at in range(0, len(ids), step):
        logits = model(mx.array(ids[at : at + step])[None], cache=cache)
        mx.eval(logits, [c.state for c in cache])
    return time.perf_counter() - start


model, tokenizer = load("mlx-community/Qwen3.8-27B-4bit")
read(model, tokenizer.encode(notebook(0, 300)), 2048)  # warm-up
for agent, step in ((1, 2048), (2, 2048), (3, 8192), (4, 512)):
    ids = tokenizer.encode(notebook(agent, 6000))
    seconds = read(model, ids, step)
    print(f"{len(ids)} new tokens in passes of {step}: {seconds:.2f} s = {len(ids) / seconds:.0f} tok/s")

# Where each 2,048-token pass goes. Inputs are finished before a module starts, so its time is its own.
spent = defaultdict(float)


def timed(owner, name, label):
    inner = getattr(owner, name)

    def call(*args, **kwargs):
        mx.eval([a for a in args if isinstance(a, mx.array)])
        start = time.perf_counter()
        out = inner(*args, **kwargs)
        mx.eval(out)
        spent[label] += time.perf_counter() - start
        return out

    setattr(owner, name, call)


def timed_module(kind, label):
    inner = kind.__call__

    def call(self, x, *args, **kwargs):
        mx.eval(x)
        start = time.perf_counter()
        out = inner(self, x, *args, **kwargs)
        mx.eval(out)
        spent[label] += time.perf_counter() - start
        return out

    kind.__call__ = call


timed_module(nn.QuantizedLinear, "quantized matmul")
timed(mx.fast, "scaled_dot_product_attention", "attention")
ids = tokenizer.encode(notebook(5, 6000))
cache = make_prompt_cache(model)
for at in range(0, len(ids), 2048):
    spent.clear()
    start = time.perf_counter()
    logits = model(mx.array(ids[at : at + 2048])[None], cache=cache)
    mx.eval(logits, [c.state for c in cache])
    total = time.perf_counter() - start
    rest = total - sum(spent.values())
    print(f"pass at {at}: {total:.2f} s = matmul {spent['quantized matmul']:.2f} + attention {spent['attention']:.2f} + rest {rest:.2f}")
