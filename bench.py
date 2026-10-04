"""N agents share one model and take turns. Reports time to first token per turn.

    uv run --project pie/python/client python bench.py --agents 6 --context 6000
"""

import argparse
import asyncio
import json
import os
import re
import socket
import statistics
import subprocess
import time
from pathlib import Path

from mem import Peak
from pie_client import Event, PieClient

HERE = Path(__file__).parent
PIE = HERE / "pie-mpp/target/release/pie"  # current main plus the branch under test
WASM = HERE / "agent/target/wasm32-wasip2/release/agent.wasm"
PORT = 8097
WORDS = "amber birch cedar delta ember fjord grove heron iris jade kelp lotus".split()


def notebook(agent: int, tokens: int) -> str:
    """A context only this agent holds: numbered items, each with a code word."""
    lines = [f"Notebook of agent {agent}. Each item has a code word."]
    for item in range(tokens // 14):
        word = WORDS[(item * 7 + agent * 5) % len(WORDS)]
        lines.append(f"Item {item}: the code word is {word}{agent}.")
    return "\n".join(lines)


def question(agent: int, turn: int) -> tuple[str, str]:
    item = 3 + turn
    word = WORDS[(item * 7 + agent * 5) % len(WORDS)]
    return f"\nQuestion: what is the code word of item {item}?\nAnswer: the code word is", f"{word}{agent}"


async def reply(process, timeout: float) -> dict:
    """One turn's stats, with the first-token time as the client saw it."""
    start = time.monotonic()
    while True:
        event, value = await asyncio.wait_for(process.recv(), timeout)
        if event == Event.Message:
            stats = json.loads(value)
            wall = (time.monotonic() - start) * 1e3
            stats["ttft_ms"] = wall - (stats["total_ms"] - stats["ttft_ms"])
            return stats
        if event in (Event.Error, Event.Return):
            raise RuntimeError(f"agent ended: {value}")


def swaps() -> dict:
    """What the planner moved, from its last trace line in pie.log."""
    lines = re.findall(r"\[planner-trace\].*", (HERE / "pie.log").read_text())
    keys = ("evictions", "restores", "disk_w_pages", "disk_r_pages", "disk_w_ms", "disk_r_ms")
    return {key: int(re.search(rf"\b{key}=(\d+)", lines[-1]).group(1)) for key in keys} if lines else {}


async def main(args):
    log = open(HERE / "pie.log", "w")
    env = {**os.environ, "PIE_CONTENTION_TRACE_MS": "50"}
    peak = Peak("target/release/pie -c")  # started before the engine, so growth includes the weights
    peak.thread.start()
    server = subprocess.Popen([args.pie, "-c", HERE / args.config, "serve"], stdout=log, stderr=log, env=env)
    try:
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            if server.poll() is not None:
                raise SystemExit("pie did not start; see pie.log")
            time.sleep(0.5)
        client = PieClient(f"ws://127.0.0.1:{PORT}")
        await client.connect()
        await client.authenticate("bench")
        program = await client.install_program(WASM, force_overwrite=True)

        agents, cold, warm, right, growth, together = [], [], [], 0, [], []
        for k in range(args.agents):
            process = await client.launch_process(
                program, {"context": notebook(k, args.context), "max_tokens": args.max_tokens}
            )
            stats = await reply(process, args.timeout)
            agents.append(process)
            cold.append(stats["ttft_ms"])
            growth.append(round(peak.system / 1024, 1))
            print(f"launch agent {k}: read {stats['fed']} tokens, first token after {stats['ttft_ms'] / 1e3:.2f} s")
        sizes = [size for size in (1, 2, 4, 8, 16) if size <= args.agents] if args.together else []
        for turn, size in enumerate(sizes):  # `size` agents are asked at the same moment
            asks = [question(k, turn) for k in range(size)]
            start = time.monotonic()
            for process, (ask, _) in zip(agents, asks):
                await process.signal(ask)
            replies = await asyncio.gather(*(reply(process, args.timeout) for process in agents[:size]))
            seconds = time.monotonic() - start
            right = sum(answer in stats["text"] for (_, answer), stats in zip(asks, replies))
            together.append([size, round(seconds, 2), right])
            print(f"{size:2d} agents asked together: all answered in {seconds:.2f} s, {right}/{size} correct")
        for turn in range(0 if args.together else args.rounds):
            for k, process in enumerate(agents):
                ask, answer = question(k, turn)
                await process.signal(ask)
                stats = await reply(process, args.timeout)
                warm.append(stats["ttft_ms"])
                right += answer in stats["text"]
                print(f"round {turn} agent {k}: first token after {stats['ttft_ms']:.0f} ms, said {stats['text'].strip()[:24]!r} (expect {answer})")
        for process in agents:
            await process.signal("quit")
        await client.close()
    except TimeoutError:
        stalled = {"engine": "pie", "stalled": f"no reply in {args.timeout:.0f} s", "agents_launched": len(agents)}
        print(json.dumps({**stalled, "growth_gb_by_agents": growth}))
        return
    finally:
        peak.stop.set()
        server.terminate()
        server.wait()
    if args.together:
        print(json.dumps({"engine": "pie", "together": together, "system_growth_gb": round(peak.system / 1024, 2)}))
        return
    result = {
        "engine": "pie",
        "agents": args.agents,
        "context_tokens": args.context,
        "launch_ttft_s": round(statistics.median(cold) / 1e3, 2),
        "switch_ttft_s": round(statistics.median(warm) / 1e3, 3),
        "switch_ttft_max_s": round(max(warm) / 1e3, 3),
        "slow_switches": sum(ms > 1000 for ms in warm),
        "correct": f"{right}/{len(warm)}",
        "system_growth_gb": round(peak.system / 1024, 2),
        "growth_gb_by_agents": growth,
        **swaps(),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pie.toml")
    parser.add_argument("--pie", default=PIE)
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--context", type=int, default=6000, help="tokens per agent, roughly")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--together", action="store_true", help="ask 1, 2, 4, ... agents at the same moment")
    parser.add_argument("--timeout", type=float, default=120, help="seconds a turn may take")
    asyncio.run(main(parser.parse_args()))
