"""One document, N questions at once. Reports seconds until all N are answered.

    uv run --project pie/python/client python fanout.py pie
    uv run --project pie/python/client python fanout.py ollama

pie: one agent reads the document, then forks itself per question.
Ollama: the N prompts share the document as a prefix and are sent together.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from bench import HERE, PIE, PORT, WASM, notebook, question
from pie_client import Event, PieClient

SIZES = (1, 2, 4, 8, 16)


def report(engine: str, read: float, rows: list) -> None:
    for n, seconds, right in rows:
        print(f"{engine}: {n:2d} questions answered in {seconds:6.2f} s, {right}/{n} correct")
    result = {"engine": engine, "read_s": round(read, 2), "questions": [n for n, _, _ in rows]}
    print(json.dumps({**result, "seconds": [round(s, 2) for _, s, _ in rows], "correct": [r for _, _, r in rows]}))


async def pie(args):
    log = open(HERE / "pie.log", "w")
    server = subprocess.Popen([PIE, "-c", HERE / args.config, "serve"], stdout=log, stderr=log)
    try:
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            time.sleep(0.5)
        client = PieClient(f"ws://127.0.0.1:{PORT}")
        await client.connect()
        await client.authenticate("bench")
        program = await client.install_program(WASM, force_overwrite=True)
        start = time.monotonic()
        agent = await client.launch_process(program, {"context": notebook(0, args.context), "max_tokens": args.max_tokens})

        async def reply() -> dict:
            while True:
                event, value = await asyncio.wait_for(agent.recv(), 600)
                if event == Event.Message:
                    return json.loads(value)
                if event in (Event.Error, Event.Return):
                    raise RuntimeError(f"agent ended: {value}")

        await reply()
        read, rows = time.monotonic() - start, []
        for n in args.sizes:
            asks = [question(0, turn) for turn in range(n)]
            start = time.monotonic()
            await agent.signal(json.dumps([ask for ask, _ in asks]))
            texts = (await reply())["texts"]
            rows.append((n, time.monotonic() - start, sum(answer in text for (_, answer), text in zip(asks, texts))))
        await agent.signal("quit")
        await client.close()
    finally:
        server.terminate()
        server.wait()
    report("pie", read, rows)


def ollama(args):
    url = "http://127.0.0.1:11500"  # a private server; the Ollama app on 11434 is left alone
    env = {**os.environ, "OLLAMA_HOST": url[7:], "OLLAMA_NUM_PARALLEL": str(max(SIZES)), "OLLAMA_MAX_LOADED_MODELS": "1"}
    log = open(HERE / "ollama.log", "w")
    server = subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, env=env)

    def complete(prompt: str) -> str:
        options = {"num_ctx": args.context + 1024, "num_predict": args.max_tokens, "temperature": 0}
        body = {"model": args.model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": -1, "options": options}
        request = urllib.request.Request(f"{url}/api/generate", json.dumps(body).encode())
        return json.load(urllib.request.urlopen(request))["response"]

    try:
        while True:
            try:
                urllib.request.urlopen(url)
                break
            except OSError:
                time.sleep(0.5)
        start = time.monotonic()
        document = notebook(0, args.context)
        document += complete(document)
        read, rows = time.monotonic() - start, []
        for n in SIZES:
            asks = [question(0, turn) for turn in range(n)]
            start = time.monotonic()
            with ThreadPoolExecutor(n) as pool:
                texts = list(pool.map(complete, [document + ask for ask, _ in asks]))
            rows.append((n, time.monotonic() - start, sum(answer in text for (_, answer), text in zip(asks, texts))))
    finally:
        server.terminate()
        server.wait()
    report("ollama", read, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=("pie", "ollama"))
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--config", default="pie-wide.toml", help="pie only: a pool with room for every fork")
    parser.add_argument("--context", type=int, default=6000, help="tokens in the document, roughly")
    parser.add_argument("--max-tokens", type=int, default=48, help="tokens per answer")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), help="how many questions per round")
    args = parser.parse_args()
    asyncio.run(pie(args)) if args.engine == "pie" else ollama(args)
