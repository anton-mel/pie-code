"""A rule on every token: never write a token containing the letter "e".

    uv run --project pie/python/client python rule.py pie
    uv run --project pie/python/client python rule.py ollama

pie: the rule runs inside the engine (the agent's `ban` input).
Ollama: the client asks for one token and its candidates, applies the rule, sends the text back.
Reports milliseconds per generated token, with and without the rule.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import time
import urllib.request

from bench import HERE, PIE, PORT, WASM, notebook
from pie_client import Event, PieClient

TASK = "\nWrite a long story about a fox and a river.\n"
BAN = "e"


def prompt(context: int) -> str:
    return (notebook(0, context) if context else "") + TASK


async def pie(args):
    log = open(HERE / "pie.log", "w")
    server = subprocess.Popen([PIE, "-c", HERE / "pie.toml", "serve"], stdout=log, stderr=log)
    rows = []
    try:
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            time.sleep(0.5)
        client = PieClient(f"ws://127.0.0.1:{PORT}")
        await client.connect()
        await client.authenticate("bench")
        program = await client.install_program(WASM, force_overwrite=True)
        for context in args.contexts:
            for ban in ("", BAN):
                agent = await client.launch_process(program, {"context": prompt(context), "max_tokens": args.tokens, "ban": ban})
                while True:
                    event, value = await asyncio.wait_for(agent.recv(), 600)
                    if event == Event.Message:
                        break
                    if event in (Event.Error, Event.Return):
                        raise RuntimeError(f"agent ended: {value}")
                stats = json.loads(value)
                await agent.signal("quit")
                per_token = (stats["total_ms"] - stats["ttft_ms"]) / (args.tokens - 1)
                rows.append({"context": context, "rule": bool(ban), "ms_per_token": round(per_token, 1), "text": stats["text"][:70]})
                print(rows[-1])
        await client.close()
    finally:
        server.terminate()
        server.wait()
    print(json.dumps({"engine": "pie", "rows": rows}))


def ollama(args):
    url = "http://127.0.0.1:11500"  # a private server; the Ollama app on 11434 is left alone
    env = {**os.environ, "OLLAMA_HOST": url[7:], "OLLAMA_MAX_LOADED_MODELS": "1"}
    log = open(HERE / "ollama.log", "w")
    server = subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, env=env)

    def generate(text: str, count: int, **extra) -> dict:
        options = {"num_ctx": max(args.contexts) + 1024, "num_predict": count, "temperature": 0}
        body = {"model": args.model, "prompt": text, "raw": True, "stream": False, "keep_alive": -1, "options": options, **extra}
        return json.load(urllib.request.urlopen(urllib.request.Request(f"{url}/api/generate", json.dumps(body).encode())))

    rows = []
    try:
        while True:
            try:
                urllib.request.urlopen(url)
                break
            except OSError:
                time.sleep(0.5)
        for context in args.contexts:
            text = prompt(context)
            plain = generate(text, args.tokens)
            rows.append({"context": context, "rule": False, "ms_per_token": round(plain["eval_duration"] / plain["eval_count"] / 1e6, 1), "text": plain["response"][:70]})
            print(rows[-1])
            generate(text, 1)  # the context is read once, outside the timed loop
            start, story = time.monotonic(), ""
            for _ in range(args.tokens):
                step = generate(text + story, 1, logprobs=True, top_logprobs=20)
                candidates = [candidate["token"] for candidate in step["logprobs"][0]["top_logprobs"]]
                story += next((token for token in candidates if BAN not in token.lower()), candidates[0])
            per_token = (time.monotonic() - start) * 1e3 / args.tokens
            rows.append({"context": context, "rule": True, "ms_per_token": round(per_token, 1), "text": story[:70]})
            print(rows[-1])
    finally:
        server.terminate()
        server.wait()
    print(json.dumps({"engine": "ollama", "rows": rows}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=("pie", "ollama"))
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--contexts", type=int, nargs="+", default=[0, 6000], help="tokens before the task, roughly")
    parser.add_argument("--tokens", type=int, default=96, help="tokens to generate")
    args = parser.parse_args()
    asyncio.run(pie(args)) if args.engine == "pie" else ollama(args)
