"""Answer quality of one pie build: pie-evals' questions and scoring, greedy decoding.

    uv run --project pie/python/client --with transformers --with jinja2 --with pyarrow --with huggingface_hub \\
        python quality.py new pie-mpp/target/release/pie --tasks gsm8k arc --n 100

Appends one row per task to quality.jsonl and keeps every answer in quality-<label>-<task>.json.
"""

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from bench import HERE, PORT
from pie_client import Event, PieClient

EVALS = Path.home() / "actions-runner/_work/pie-evals/pie-evals/src/pie_evals/quality"
SNAPSHOTS = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
WASM = Path.home() / "pie/examples/target/wasm32-wasip2/release/text_completion_bench.wasm"
SYSTEM = "You are a helpful assistant."

sys.path.insert(0, str(EVALS))
from tasks import TASKS, wilson  # noqa: E402  (pie-evals' loaders and scorers)


def render(prompts: list[str]) -> list[list[int]]:
    """The chat-templated prompt as token ids, thinking off, as pie-evals renders it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(next(SNAPSHOTS.iterdir())))
    out = []
    for prompt in prompts:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out.append(tokenizer.encode(text, add_special_tokens=False))
    return out


async def generate(binary: str, config: str, items: list, tokens: list, max_tokens: int, concurrency: int) -> list[str]:
    server = subprocess.Popen([binary, "-c", HERE / config, "serve"], stdout=open(HERE / "pie.log", "w"), stderr=subprocess.STDOUT)
    texts = [""] * len(items)
    try:
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            if server.poll() is not None:
                raise SystemExit("pie did not start; see pie.log")
            time.sleep(0.5)
        client = PieClient(f"ws://127.0.0.1:{PORT}")
        await client.connect()
        await client.authenticate("bench")
        program = await client.install_program(WASM, force_overwrite=True)
        gate = asyncio.Semaphore(concurrency)

        async def one(at: int) -> None:
            async with gate:
                ask = {"system": SYSTEM, "prompt": items[at].prompt, "prompt_tokens": tokens[at], "max_tokens": max_tokens,
                       "temperature": 0.0, "top_p": 1.0, "ignore_eos": False, "return_text": True, "report_timing": False}
                process = await client.launch_process(program, ask)
                while True:
                    event, value = await asyncio.wait_for(process.recv(), 900)
                    if event == Event.Return:
                        texts[at] = json.loads(value).get("text", "")
                        return
                    if event != Event.Message:
                        return

        await asyncio.gather(*(one(at) for at in range(len(items))))
        await client.close()
    finally:
        server.terminate()
        server.wait()
    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("binary")
    parser.add_argument("--config", default="pie.toml")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "arc"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    for task in args.tasks:
        load, score, max_tokens = TASKS[task]
        items = load(args.n)
        start = time.monotonic()
        texts = asyncio.run(generate(args.binary, args.config, items, render([item.prompt for item in items]), max_tokens, args.concurrency))
        scores = [score(item, text) for item, text in zip(items, texts)]
        low, high = wilson(sum(scores), len(scores))
        row = {"build": args.label, "task": task, "n": len(scores), "correct": sum(scores), "score": round(sum(scores) / len(scores), 3),
               "ci95": [round(low, 3), round(high, 3)], "empty": sum(not text for text in texts), "minutes": round((time.monotonic() - start) / 60, 1)}
        samples = [{"id": item.id, "score": s, "text": text} for item, s, text in zip(items, scores, texts)]
        (HERE / f"quality-{args.label}-{task}.json").write_text(json.dumps(samples, indent=1))
        with open(HERE / "quality.jsonl", "a") as log:
            log.write(json.dumps(row) + "\n")
        print(json.dumps(row))
