"""The same agents and turns as bench.py, against Ollama with one slot per agent, with peak memory.

    uv run --project pie/python/client python ollama.py --agents 4 --context 6000
"""

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from bench import HERE, notebook, question
from mem import Peak

URL = "http://127.0.0.1:11500"  # a private server; the Ollama app on 11434 is left alone


def complete(args, prompt: str) -> tuple[float, str]:
    """Returns seconds to the first token and the text."""
    body = {
        "model": args.model,
        "prompt": prompt,
        "raw": True,
        "stream": True,
        "keep_alive": -1,
        "options": {"num_ctx": args.context + 1024, "num_predict": args.max_tokens, "temperature": 0},
    }
    request = urllib.request.Request(f"{URL}/api/generate", json.dumps(body).encode())
    start, first, text = time.monotonic(), None, ""
    with urllib.request.urlopen(request) as stream:
        for line in stream:
            piece = json.loads(line).get("response", "")
            if piece:
                first = first or time.monotonic() - start
                text += piece
    return first or time.monotonic() - start, text


def main(args):
    log = open(HERE / "ollama.log", "w")
    env = {
        **os.environ,
        "OLLAMA_HOST": URL[7:],
        "OLLAMA_NUM_PARALLEL": str(args.agents),  # one slot per agent, so each keeps its KV
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    server = subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, env=env)
    try:
        while True:
            try:
                urllib.request.urlopen(URL)
                break
            except OSError:
                if server.poll() is not None:
                    raise SystemExit("ollama did not start; see ollama.log")
                time.sleep(0.5)
        prompts, cold, warm, right, growth, together = [], [], [], 0, [], []
        with Peak("Ollama.app/Contents/Resources/llama-server") as peak:
            for k in range(args.agents):
                prompts.append(notebook(k, args.context))
                ttft, text = complete(args, prompts[k])
                prompts[k] += text
                cold.append(ttft)
                growth.append(round(peak.system / 1024, 1))
                print(f"launch agent {k}: first token after {ttft:.2f} s")
            sizes = [size for size in (1, 2, 4, 8, 16) if size <= args.agents] if args.together else []
            for turn, size in enumerate(sizes):  # `size` agents are asked at the same moment
                asks = [question(k, turn) for k in range(size)]
                for k, (ask, _) in enumerate(asks):
                    prompts[k] += ask
                start = time.monotonic()
                with ThreadPoolExecutor(size) as pool:
                    replies = list(pool.map(lambda prompt: complete(args, prompt), prompts[:size]))
                seconds = time.monotonic() - start
                for k, (_, text) in enumerate(replies):
                    prompts[k] += text
                right = sum(answer in text for (_, answer), (_, text) in zip(asks, replies))
                together.append([size, round(seconds, 2), right])
                print(f"{size:2d} agents asked together: all answered in {seconds:.2f} s, {right}/{size} correct")
            for turn in range(0 if args.together else args.rounds):
                for k in range(args.agents):
                    ask, answer = question(k, turn)
                    prompts[k] += ask
                    ttft, text = complete(args, prompts[k])
                    prompts[k] += text
                    warm.append(ttft)
                    right += answer in text
                    print(f"round {turn} agent {k}: first token after {ttft * 1e3:.0f} ms, said {text.strip()[:24]!r} (expect {answer})")
            loaded = json.load(urllib.request.urlopen(f"{URL}/api/ps"))["models"][0]
    finally:
        server.terminate()
        server.wait()
    if args.together:
        print(json.dumps({"engine": "ollama", "together": together, "system_growth_gb": round(peak.system / 1024, 2)}))
        return
    result = {
        "engine": "ollama",
        "agents": args.agents,
        "context_tokens": args.context,
        "launch_ttft_s": round(statistics.median(cold), 2),
        "switch_ttft_s": round(statistics.median(warm), 3),
        "switch_ttft_max_s": round(max(warm), 3),
        "slow_switches": sum(seconds > 1 for seconds in warm),
        "correct": f"{right}/{len(warm)}",
        "growth_gb_by_agents": growth,
        "peak_footprint_gb": round(peak.mb / 1024, 2),
        "system_growth_gb": round(peak.system / 1024, 2),
        "ollama_reported_gb": round(loaded["size"] / 2**30, 2),
        "on_gpu_gb": round(loaded["size_vram"] / 2**30, 2),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--context", type=int, default=6000, help="tokens per agent, roughly")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--together", action="store_true", help="ask 1, 2, 4, ... agents at the same moment")
    main(parser.parse_args())
