"""N agents, each in a tool loop: write a call, wait for the tool, read its output, write the next call.

    uv run --project pie/python/client python tools.py pie --agents 8
    uv run --project pie/python/client python tools.py ollama --agents 8
    uv run --project pie/python/client python tools.py llama --agents 8 --slots 8

Every agent holds its own context and all of them loop at once. Reports seconds per step
(tool output sent -> next call written), the seconds until every agent has finished, and the
memory the Mac gave up. Appends one row to tools.jsonl. See TOOL-LOOP.md.
"""

import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench import HERE, PIE, PORT, WASM, notebook, question, swaps
from mem import Peak
from pie_client import Event, PieClient

OLLAMA = "http://127.0.0.1:11500"  # a private server; the Ollama app on 11434 is left alone
LLAMA = "http://127.0.0.1:8098"
LLAMA_SERVER = Path.home() / ".cache/pie-evals/baselines/llama-b11312/llama-server"
LLAMA_MODELS = Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots"
KV_ON_DISK = Path.home() / ".pie/cache/kv"


def tool_output(agent: int, step: int) -> str:
    """What the tool sends back: about 120 tokens only this agent sees."""
    rows = "\n".join(f"row {row}: agent {agent} step {step} status ok value {(agent * 31 + step * 17 + row * 7) % 997}" for row in range(8))
    return f"\nTool output:\n{rows}"


def is_tool(args, agent: int, step: int) -> bool:
    """Whether this step is a tool call: `--tool-share` of each agent's steps, spread evenly, offset per agent."""
    at = (step + agent) % args.steps
    return int((at + 1) * args.tool_share + 1e-9) > int(at * args.tool_share + 1e-9)


def ask(args, agent: int, step: int) -> tuple[str, str, int]:
    """The message of one step, the answer it must contain, and how many tokens it may generate.
    A tool step reads the previous tool's output and writes a short call; a thinking step writes at length."""
    text, answer = question(agent, step)
    before = tool_output(agent, step) if step and is_tool(args, agent, step - 1) else ""
    if is_tool(args, agent, step):
        return before + text, answer, args.max_tokens
    item = 3 + step
    return before + f"\nList items {item} to {item + 40} with their code words, one per line.\nItem {item}: the code word is", answer, args.think_tokens


def report(args, read: float, loop: float, steps: list, peak: Peak, extra: dict) -> None:
    """`steps` holds (step index, seconds, right, text, agent, tool, tokens) for every step of every agent."""
    first = [seconds for step, seconds, *_ in steps if step == 0]
    later = [seconds for step, seconds, *_ in steps if step > 0]
    tool = [seconds for _, seconds, _, _, _, kind, _ in steps if kind]
    think = [seconds for _, seconds, _, _, _, kind, _ in steps if not kind]
    made = sum(tokens for *_, tokens in steps)
    every = sorted(seconds for _, seconds, *_ in steps)
    row = {
        "engine": args.engine,
        "variant": args.variant,
        "agents": args.agents,
        "context_tokens": args.context,
        "steps": args.steps,
        "tool_s": args.tool,
        "tool_share": args.tool_share,
        "tool_step_s": round(statistics.median(tool), 2) if tool else None,
        "think_step_s": round(statistics.median(think), 2) if think else None,
        "tokens_made": made,
        "tok_per_s": round(made / loop, 1),
        "read_s": round(read, 1),
        "loop_s": round(loop, 1),
        "step_s": round(statistics.median(every), 2),
        "first_step_s": round(statistics.median(first), 2),
        "later_step_s": round(statistics.median(later), 2),
        "step_p90_s": round(every[int(len(every) * 0.9)], 2),
        "step_max_s": round(every[-1], 2),
        "correct": f"{sum(right for _, _, right, *_ in steps)}/{len(steps)}",
        "wrong": [f"agent {k} step {step}: {text.strip()[:60]}" for step, _, right, text, k, *_ in steps if not right][:8],
        "system_growth_gb": round(peak.system / 1024, 1),
        "footprint_gb": round(peak.mb / 1024, 1),
        **extra,
    }
    print(json.dumps(row))
    with open(HERE / "tools.jsonl", "a") as log:
        log.write(json.dumps(row) + "\n")


def disk_gb(path: Path) -> float:
    out = subprocess.run(["du", "-sk", path], capture_output=True, text=True).stdout.split()
    return round(int(out[0]) / 2**20, 1) if out else 0.0


async def pie(args, peak: Peak):
    log = open(HERE / "pie.log", "w")
    env = {**os.environ, "PIE_CONTENTION_TRACE_MS": "50"}
    server = subprocess.Popen([args.pie, "-c", HERE / args.config, "serve"], stdout=log, stderr=log, env=env)
    steps = []
    try:
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            if server.poll() is not None:
                raise SystemExit("pie did not start; see pie.log")
            time.sleep(0.5)
        client = PieClient(f"ws://127.0.0.1:{PORT}")
        await client.connect()
        await client.authenticate("bench")
        program = await client.install_program(WASM, force_overwrite=True)

        async def reply(agent) -> dict:
            while True:
                event, value = await asyncio.wait_for(agent.recv(), args.timeout)
                if event == Event.Message:
                    return json.loads(value)
                if event in (Event.Error, Event.Return):
                    raise RuntimeError(f"agent ended: {value}")

        start, agents = time.monotonic(), []
        for k in range(args.agents):
            agents.append(await client.launch_process(program, {"context": notebook(args.first_agent + k, args.context), "max_tokens": args.max_tokens}))
            await reply(agents[-1])
        read = time.monotonic() - start

        async def loop(k: int) -> None:
            for step in range(args.steps):
                agent = args.first_agent + k
                text, answer, limit = ask(args, agent, step)
                sent = time.monotonic()
                await agents[k].signal(json.dumps({"text": text, "max_tokens": limit}))
                said = (await reply(agents[k]))["text"]
                tool = is_tool(args, agent, step)
                steps.append((step, time.monotonic() - sent, answer in said, said, k, tool, limit))
                if tool:
                    await asyncio.sleep(args.tool)  # the tool runs; the agent holds its context and waits

        start = time.monotonic()
        await asyncio.gather(*(loop(k) for k in range(args.agents)))
        spent = time.monotonic() - start
        disk = disk_gb(KV_ON_DISK)
        for agent in agents:
            await agent.signal("quit")
        await client.close()
    finally:
        server.terminate()
        server.wait()
    report(args, read, spent, steps, peak, {"kv_on_disk_gb": disk, **swaps()})


async def http(args, peak: Peak, server, ready: str, complete):
    """Engines behind a completion endpoint: the client keeps each agent's text and resends it."""
    steps = []
    try:
        while True:
            try:
                urllib.request.urlopen(ready)
                break
            except OSError:
                if server.poll() is not None:
                    raise SystemExit(f"{args.engine} did not start; see its log")
                time.sleep(0.5)
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(args.agents))
        start, prompts = time.monotonic(), []
        for k in range(args.agents):
            prompts.append(notebook(k, args.context))
            prompts[k] += complete(k, prompts[k], args.max_tokens)[0]
        read = time.monotonic() - start

        async def loop(k: int) -> None:
            for step in range(args.steps):
                text, answer, limit = ask(args, k, step)
                prompts[k] += text
                sent = time.monotonic()
                said, made = await asyncio.to_thread(complete, k, prompts[k], limit)
                tool = is_tool(args, k, step)
                steps.append((step, time.monotonic() - sent, answer in said, said, k, tool, made))
                prompts[k] += said
                if tool:
                    await asyncio.sleep(args.tool)

        start = time.monotonic()
        await asyncio.gather(*(loop(k) for k in range(args.agents)))
        spent = time.monotonic() - start
    finally:
        server.terminate()
        server.wait()
    report(args, read, spent, steps, peak, {})


def ollama(args, peak: Peak):
    env = {**os.environ, "OLLAMA_HOST": OLLAMA[7:], "OLLAMA_NUM_PARALLEL": str(args.agents), "OLLAMA_MAX_LOADED_MODELS": "1"}
    log = open(HERE / "ollama.log", "w")
    server = subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, env=env)
    num_ctx = args.context + 1024 + args.steps * (args.think_tokens + 256)

    def complete(k: int, prompt: str, limit: int) -> tuple[str, int]:
        options = {"num_ctx": num_ctx, "num_predict": limit, "temperature": 0}
        body = {"model": args.model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": -1, "options": options}
        request = urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps(body).encode())
        out = json.load(urllib.request.urlopen(request, timeout=args.timeout))
        return out["response"], out.get("eval_count", 0)

    return http(args, peak, server, OLLAMA, complete)


def llama(args, peak: Peak):
    """llama-server with `--slots` slots; with one slot per agent every agent keeps its KV in memory."""
    per_slot = args.context + 1024 + args.steps * (args.think_tokens + 256)
    log = open(HERE / "llama.log", "w")
    model = next(LLAMA_MODELS.glob("*/*.gguf"))
    command = [LLAMA_SERVER, "-m", model, "--port", LLAMA[-4:], "-ngl", "all", "-c", str(per_slot * args.slots), "-np", str(args.slots)]
    server = subprocess.Popen([*command, *args.llama_args.split()], stdout=log, stderr=log)

    def complete(k: int, prompt: str, limit: int) -> tuple[str, int]:
        body = {"prompt": prompt, "n_predict": limit, "temperature": 0, "cache_prompt": True}
        if args.slots > 1:
            body["id_slot"] = k % args.slots
        request = urllib.request.Request(f"{LLAMA}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(request, timeout=args.timeout))
        return out["content"], out.get("tokens_predicted", 0)

    return http(args, peak, server, f"{LLAMA}/health", complete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=("pie", "ollama", "llama"))
    parser.add_argument("--variant", default="default", help="a name for this setup, kept in the row")
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--pie", default=PIE)
    parser.add_argument("--config", default="pie.toml")
    parser.add_argument("--slots", type=int, default=1, help="llama only: slots; one per agent keeps every KV in memory")
    parser.add_argument("--llama-args", default="", help="llama only: extra llama-server flags")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--first-agent", type=int, default=0, help="number agents from here, to rerun one agent alone")
    parser.add_argument("--context", type=int, default=6000, help="tokens each agent holds, roughly")
    parser.add_argument("--steps", type=int, default=6, help="tool calls per agent")
    parser.add_argument("--tool", type=float, default=2.0, help="seconds a tool takes")
    parser.add_argument("--max-tokens", type=int, default=16, help="tokens per tool call")
    parser.add_argument("--tool-share", type=float, default=1.0, help="share of steps that call a tool; the rest are thinking steps")
    parser.add_argument("--think-tokens", type=int, default=256, help="tokens per thinking step")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    patterns = {"pie": "target/release/pie -c", "ollama": "Ollama.app/Contents/Resources/llama-server", "llama": "llama-b11312/llama-server"}
    with Peak(patterns[args.engine]) as peak:  # started before the engine, so growth includes the weights
        asyncio.run({"pie": pie, "ollama": ollama, "llama": llama}[args.engine](args, peak))
