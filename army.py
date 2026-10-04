"""An OpenCode-style army without OpenCode: N agents on one OpenAI-compatible endpoint.

    uv run python army.py pie --agents 8        # serve.py on :8100 (start it first)
    uv run python army.py llama --agents 8      # starts llama-server with a slot per agent

Each agent's system prompt is the same 8,000-token block (an agent app's instructions and tool
docs) plus one line of its own near the end (its working directory), as OpenCode's are. Then
it runs its task: T turns of reply, tool output, reply. All agents start at once.
Appends one row to army.jsonl.
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
LLAMA_SERVER = Path.home() / ".cache/pie-evals/baselines/llama-b11312/llama-server"
LLAMA_MODELS = Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots"
URLS = {"pie": "http://127.0.0.1:8100/v1", "llama": "http://127.0.0.1:8098/v1"}


def system_prompt(agent: int, tokens: int) -> str:
    """Common instructions and tool docs, then this agent's environment, then a closing rule."""
    tools = []
    for i in range(tokens // 40):
        tools.append(f"Tool t{i}: takes a path and a flag; returns status code {i % 7}, a list of entries and a log line.")
    return ("You are a coding agent. Use the tools below to finish the user's task.\n\n" + "\n".join(tools)
            + f"\n\n<env>\nWorking directory: /work/agent-{agent}\nPlatform: darwin\n</env>\n"
            + "Answer briefly.")


def agent_run(args, agent: int) -> dict:
    url = URLS[args.engine] + "/chat/completions"
    messages = [{"role": "system", "content": system_prompt(agent, args.prompt)},
                {"role": "user", "content": f"Task {agent}: list the files, then run the tests."}]
    times = []
    for turn in range(args.turns):
        body = {"model": "local", "messages": messages, "max_tokens": args.max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        start = time.monotonic()
        request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(request, timeout=args.timeout))["choices"][0]["message"]
        times.append(time.monotonic() - start)
        messages.append({"role": "assistant", "content": reply.get("content") or ""})
        messages.append({"role": "user", "content": f"Tool output for turn {turn}:\n" + "\n".join(
            f"entry {row}: file_{agent}_{turn}_{row}.py ok" for row in range(10))})
    return {"first_s": times[0], "later_s": times[1:]}


def main(args):
    server = None
    if args.engine == "llama":
        model = next(LLAMA_MODELS.glob("*/*.gguf"))
        per_slot = args.prompt + 4096
        server = subprocess.Popen([LLAMA_SERVER, "-m", model, "--port", "8098", "-ngl", "all", "-c", str(per_slot * args.agents),
                                   "-np", str(args.agents), "--jinja"], stdout=open(HERE / "llama.log", "w"), stderr=subprocess.STDOUT)
        while True:
            try:
                urllib.request.urlopen("http://127.0.0.1:8098/health")
                break
            except OSError:
                time.sleep(0.5)
    try:
        start = time.monotonic()
        with ThreadPoolExecutor(args.agents) as pool:
            runs = list(pool.map(lambda a: agent_run(args, a), range(args.agents)))
        total = time.monotonic() - start
    finally:
        if server:
            server.terminate()
            server.wait()
    first = [r["first_s"] for r in runs]
    later = [t for r in runs for t in r["later_s"]]
    row = {"engine": args.engine, "variant": args.variant, "agents": args.agents, "prompt_tokens": args.prompt, "turns": args.turns,
           "total_s": round(total, 1), "first_reply_s": round(statistics.median(first), 1), "first_reply_max_s": round(max(first), 1),
           "turn_s": round(statistics.median(later), 2) if later else None}
    print(json.dumps(row))
    with open(HERE / "army.jsonl", "a") as log:
        log.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=("pie", "llama"))
    parser.add_argument("--variant", default="default")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--prompt", type=int, default=8000, help="tokens in the shared system prompt, roughly")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800)
    main(parser.parse_args())
