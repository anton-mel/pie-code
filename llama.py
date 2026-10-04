"""The same agents and turns as bench.py, against llama-server.

    uv run --project pie/python/client python llama.py --agents 6 --context 6000 --cache-ram 0
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

from bench import HERE, notebook, question

HOME = Path.home()
SERVER = HOME / ".cache/pie-evals/baselines/llama-b11312/llama-server"
MODEL = next((HOME / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots").glob("*/*.gguf"))
URL = "http://127.0.0.1:8098"


def complete(prompt: str, max_tokens: int) -> tuple[float, str]:
    """Returns seconds to the first token and the text."""
    body = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0, "cache_prompt": True, "stream": True}
    request = urllib.request.Request(f"{URL}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    start, first, text = time.monotonic(), None, ""
    with urllib.request.urlopen(request) as stream:
        for line in stream:
            if not line.startswith(b"data: "):
                continue
            first = first or time.monotonic() - start
            text += json.loads(line[6:]).get("content", "")
    return first, text


def main(args, extra):
    log = open(HERE / "llama.log", "w")
    command = [SERVER, "-m", MODEL, "--port", "8098", "-ngl", "all", "-c", str(args.context + 1024), "-np", "1", *extra]
    server = subprocess.Popen(command, stdout=log, stderr=log)
    try:
        while True:
            try:
                urllib.request.urlopen(f"{URL}/health")
                break
            except OSError:
                if server.poll() is not None:
                    raise SystemExit("llama-server did not start; see llama.log")
                time.sleep(0.5)
        prompts, cold, warm, right = [], [], [], 0
        for k in range(args.agents):
            prompts.append(notebook(k, args.context))
            ttft, text = complete(prompts[k], args.max_tokens)
            prompts[k] += text
            cold.append(ttft)
            print(f"launch agent {k}: first token after {ttft:.2f} s")
        for turn in range(args.rounds):
            for k in range(args.agents):
                ask, answer = question(k, turn)
                prompts[k] += ask
                ttft, text = complete(prompts[k], args.max_tokens)
                prompts[k] += text
                warm.append(ttft)
                right += answer in text
                print(f"round {turn} agent {k}: first token after {ttft * 1e3:.0f} ms, said {text.strip()[:24]!r} (expect {answer})")
    finally:
        server.terminate()
        server.wait()
    result = {
        "engine": "llama.cpp " + " ".join(extra),
        "agents": args.agents,
        "context_tokens": args.context,
        "launch_ttft_s": round(statistics.median(cold), 2),
        "switch_ttft_s": round(statistics.median(warm), 3),
        "switch_ttft_max_s": round(max(warm), 3),
        "correct": f"{right}/{len(warm)}",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--context", type=int, default=6000, help="tokens per agent, roughly")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    known, extra = parser.parse_known_args()
    main(known, [word for word in extra if word != "--"])
