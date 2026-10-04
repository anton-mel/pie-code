"""Agent workloads that need an operating system's process tricks, on every engine of team.py.

    uv run --project pie/python/client python os_bench.py search team --branches 8
    uv run --project pie/python/client python os_bench.py undo llama --agents 4

search (fork + kill: branch and prune): an agent holds a 6,000-token notebook and must find which
  of N candidate items has a given code word. It checks all N at once, one forked branch per
  candidate; the first branch to say YES wins and the others are killed mid-answer.
undo (checkpoint + rollback): K agents each try a tool action that fails, undo it, and retry, at
  every step.
Appends one row to os.jsonl.
"""

import argparse
import asyncio
import json
import re
import statistics
import time

from bench import HERE, WORDS, notebook
from mem import Peak
from team import Llama, Ollama, Separate, Team, question, user
from tools import tool_output

ENGINES = {"team": Team, "separate": Separate, "ollama": Ollama, "llama": Llama}
PATTERNS = {"team": "target/release/pie -c", "separate": "target/release/pie -c",
            "ollama": "Ollama.app/Contents/Resources/llama-server", "llama": "llama-b11312/llama-server"}
SYSTEM = "<|im_start|>system\nYou answer questions about a notebook.<|im_end|>\n"


async def read_notebook(engine, args):
    await engine.read(0, SYSTEM + user("The notebook:\n" + notebook(0, args.context) + "\n\nReply OK once you have read it.", first=True), 16)


def word_of(item: int) -> str:
    return WORDS[(item * 7) % len(WORDS)] + "0"


async def search(args, engine) -> dict:
    """Rounds of: N branches forked from the reader, each checking one candidate item against the
    code word sought; exactly one candidate is right, and the first YES wins."""
    items = args.context // 14
    start = time.monotonic()
    await read_notebook(engine, args)
    read = time.monotonic() - start
    found, right, next_k = [], 0, 1
    for round_ in range(args.rounds):
        target = 20 + round_ * 53 % (items - 40)
        word = word_of(target)
        candidates = [c for c in range(5, items) if word_of(c) != word][round_ * 31 % 40:][: args.branches - 1]
        candidates.insert((round_ * 3) % args.branches, target)
        began = time.monotonic()
        tasks = {}
        for c in candidates:
            k = next_k
            next_k += 1
            ask = f"Is the code word of item {c} exactly {word}? Reply YES or NO only."
            tasks[asyncio.create_task(engine.fork(0, k, user(ask) + "Verdict:", args.answer_tokens))] = (k, c == target)
        winner = None
        pending = set(tasks)
        while pending and winner is None:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                k, holds = tasks[task]
                said = "" if task.exception() else task.result()
                if args.verbose:
                    print(f"round {round_} target {target}: branch {k} (right: {holds}) said {said.strip()[:40]!r}")
                if winner is None and said.strip().upper().startswith("YES"):
                    winner = (k, holds)
        found.append(time.monotonic() - began)
        right += bool(winner and winner[1])
        for task in pending:  # prune: the rest stop now
            task.cancel()
        await asyncio.gather(*(engine.kill(tasks[t][0]) for t in tasks), return_exceptions=True)
    total = time.monotonic() - start
    return {"branches": args.branches, "read_s": round(read, 1), "total_s": round(total, 1),
            "found_s": round(statistics.median(found), 2), "found_each_s": [round(f, 1) for f in found],
            "correct": f"{right}/{args.rounds}"}


async def undo(args, engine) -> dict:
    """K agents; at every step each saves itself, tries an action that fails, rolls back, retries."""
    start = time.monotonic()
    await read_notebook(engine, args)
    await asyncio.gather(*(engine.fork(0, k, user(f"You are agent {k}. Reply OK."), 16) for k in range(1, args.agents + 1)))
    ready = time.monotonic() - start
    steps, right = [], 0

    async def agent(k):
        nonlocal right
        for step in range(args.steps):
            ask, answer = question(k, step)
            began = time.monotonic()
            name = f"{k}.{step}"
            await engine.checkpoint(k, name)
            await engine.send(k, user("Tool output:\nERROR: the action failed and changed nothing.\n\n" + ask), 16)
            await engine.rollback(k, name)  # undo the failed attempt
            said = await engine.send(k, user(tool_output(k, step).strip() + "\n\n" + ask), 16)
            steps.append(time.monotonic() - began)
            right += answer in said
            await asyncio.sleep(args.tool)

    await asyncio.gather(*(agent(k) for k in range(1, args.agents + 1)))
    total = time.monotonic() - start
    return {"agents": args.agents, "ready_s": round(ready, 1), "total_s": round(total, 1),
            "step_s": round(statistics.median(steps), 2), "correct": f"{right}/{len(steps)}"}


async def main(args, peak):
    engine = ENGINES[args.engine]()
    args.specialists = max(args.branches, args.agents) * (args.rounds if args.scenario == "search" else 1) + 1
    await engine.start(args)
    try:
        row = await {"search": search, "undo": undo}[args.scenario](args, engine)
    finally:
        await engine.stop()
    row = {"scenario": args.scenario, "engine": args.engine, "context_tokens": args.context, **row,
           "system_growth_gb": round(peak.system / 1024, 1)}
    print(json.dumps(row))
    with open(HERE / "os.jsonl", "a") as log:
        log.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("search", "undo"))
    parser.add_argument("engine", choices=tuple(ENGINES))
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--pie", default=str(HERE / "pie-mpp/target/release/pie"))
    parser.add_argument("--config", default="pie-full.toml")
    parser.add_argument("--context", type=int, default=6000)
    parser.add_argument("--branches", type=int, default=8, help="search: branches per round")
    parser.add_argument("--rounds", type=int, default=3, help="search: targets to find")
    parser.add_argument("--answer-tokens", type=int, default=8, help="search: tokens per branch's answer")
    parser.add_argument("--agents", type=int, default=4, help="undo: agents")
    parser.add_argument("--steps", type=int, default=4, help="undo: steps per agent")
    parser.add_argument("--tool", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    with Peak(PATTERNS[args.engine]) as peak:
        asyncio.run(main(args, peak))
