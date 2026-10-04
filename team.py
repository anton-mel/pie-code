"""A team of agents: one coordinator, K specialists, tool calls, and agents talking to each other.

    uv run --project pie/python/client python team.py team --specialists 8      # pie, agents as one OS (team/)
    uv run --project pie/python/client python team.py separate --specialists 8  # pie, one process per agent
    uv run --project pie/python/client python team.py ollama --specialists 8
    uv run --project pie/python/client python team.py llama --specialists 8

The workflow, the same on every engine:
  1. The coordinator reads a shared context (a 6,000-token notebook).
  2. K specialists start from the coordinator: each knows the context, plus its own role.
  3. Each specialist makes S tool calls: it answers a question, the tool runs, it reads the output.
  4. The coordinator reads every specialist's last answer and writes a summary.
  5. Every specialist reads the summary. Steps 3-5 repeat for R rounds.
On `team` step 2 is a fork (the context is shared, never re-read) and steps 4-5 are pipes (one
agent's tokens go into another's memory inside the engine). Elsewhere step 2 re-reads the context
and steps 4-5 pass text through the harness. Appends one row to team.jsonl.
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

from bench import HERE, PIE, PORT, WORDS, notebook
from mem import Peak
from pie_client import Event, PieClient
from tools import LLAMA, LLAMA_MODELS, LLAMA_SERVER, OLLAMA, tool_output

TEAM = HERE / "team/target/wasm32-wasip2/release/team.wasm"
AGENT = HERE / "agent/target/wasm32-wasip2/release/agent.wasm"


CUE = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def user(text: str, first: bool = False) -> str:
    """`text` as the next user turn, ending on the assistant's cue. After the first, it also
    closes the agent's own last reply."""
    return ("" if first else "<|im_end|>\n") + f"<|im_start|>user\n{text}<|im_end|>\n" + CUE


def question(k: int, step: int) -> tuple[str, str]:
    """A question about the shared notebook only specialist k is asked, and its answer."""
    item = 3 + k * 16 + step
    word = WORDS[(item * 7) % len(WORDS)]
    return f"Question: what is the code word of item {item}? Reply with the code word only.", f"{word}0"


class Pie:
    """Common to both pie engines: the server and a client."""

    async def start(self, args):
        self.args = args
        self.server = subprocess.Popen([args.pie, "-c", HERE / args.config, "serve"],
                                       stdout=open(HERE / "pie.log", "w"), stderr=subprocess.STDOUT)
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            if self.server.poll() is not None:
                raise SystemExit("pie did not start; see pie.log")
            await asyncio.sleep(0.5)
        self.client = PieClient(f"ws://127.0.0.1:{PORT}")
        await self.client.connect()
        await self.client.authenticate("team")

    async def stop(self):
        await self.client.close()
        self.server.terminate()
        self.server.wait()


class Team(Pie):
    """pie, co-designed: every agent lives in one process; fork and pipe are system calls."""

    async def start(self, args):
        await super().start(args)
        program = await self.client.install_program(TEAM, force_overwrite=True)
        self.process = await self.client.launch_process(program, {})
        self.waiting: dict[int, asyncio.Future] = {}
        assert (await self.process.recv())[0] == Event.Message  # {"ready": true}
        self.pump = asyncio.create_task(self.route())

    async def route(self):
        while True:
            event, value = await self.process.recv()
            if event != Event.Message:
                for future in self.waiting.values():
                    future.set_exception(RuntimeError(f"team ended: {value}"))
                return
            out = json.loads(value)
            future = self.waiting.pop(out["k"])
            future.set_exception(RuntimeError(out["error"])) if "error" in out else future.set_result(out)

    async def op(self, op: str, k: int, text: str, n: int, sources=(), prefix: str = "") -> str:
        self.waiting[k] = asyncio.get_running_loop().create_future()
        await self.process.signal(json.dumps({"op": op, "k": k, "from": list(sources), "prefix": prefix, "text": text, "max_tokens": n}))
        return (await asyncio.wait_for(self.waiting[k], self.args.timeout))["text"]

    async def read(self, k, text, n):
        return await self.op("read", k, text, n)

    async def fork(self, parent, k, text, n):
        return await self.op("fork", k, text, n, [parent])

    async def send(self, k, text, n):
        return await self.op("send", k, text, n)

    async def pipe(self, sources, k, prefix, text, n):
        return await self.op("pipe", k, text, n, sources, prefix)

    async def stop(self):
        await self.process.signal("quit")
        self.pump.cancel()
        await super().stop()


class Separate(Pie):
    """pie, decoupled: one process per agent; a new agent reads everything itself, and agents
    talk through the harness as text."""

    async def start(self, args):
        await super().start(args)
        self.program = await self.client.install_program(AGENT, force_overwrite=True)
        self.agents, self.history, self.last = {}, {}, {}

    async def reply(self, k) -> str:
        while True:
            event, value = await asyncio.wait_for(self.agents[k].recv(), self.args.timeout)
            if event == Event.Message:
                self.last[k] = json.loads(value)["text"]
                self.history[k] += self.last[k]
                return self.last[k]
            if event in (Event.Error, Event.Return):
                raise RuntimeError(f"agent {k} ended: {value}")

    async def read(self, k, text, n):
        self.history[k] = text
        self.agents[k] = await self.client.launch_process(self.program, {"context": text, "max_tokens": n})
        return await self.reply(k)

    async def fork(self, parent, k, text, n):
        return await self.read(k, self.history[parent] + text, n)

    async def send(self, k, text, n):
        self.history[k] += text
        await self.agents[k].signal(json.dumps({"text": text, "max_tokens": n}))
        return await self.reply(k)

    async def pipe(self, sources, k, prefix, text, n):
        return await self.send(k, prefix + "".join(self.last[s] for s in sources) + text, n)

    async def stop(self):
        for agent in self.agents.values():
            await agent.signal("quit")
        await super().stop()


class Http:
    """Ollama or llama.cpp: the harness keeps each agent's text and resends all of it."""

    async def start(self, args):
        self.args, self.history, self.last = args, {}, {}
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(args.specialists + 1))
        self.server = self.launch()
        while True:
            try:
                urllib.request.urlopen(self.ready)
                break
            except OSError:
                if self.server.poll() is not None:
                    raise SystemExit("the server did not start; see its log")
                await asyncio.sleep(0.5)

    async def turn(self, k, text, n):
        self.history[k] = self.history.get(k, "") + text
        self.last[k] = await asyncio.to_thread(self.complete, k, self.history[k], n)
        self.history[k] += self.last[k]
        return self.last[k]

    async def read(self, k, text, n):
        return await self.turn(k, text, n)

    async def fork(self, parent, k, text, n):
        self.history[k] = self.history[parent]
        return await self.turn(k, text, n)

    async def send(self, k, text, n):
        return await self.turn(k, text, n)

    async def pipe(self, sources, k, prefix, text, n):
        return await self.turn(k, prefix + "".join(self.last[s] for s in sources) + text, n)

    async def stop(self):
        self.server.terminate()
        self.server.wait()


class Ollama(Http):
    def launch(self):
        self.ready = OLLAMA
        env = {**os.environ, "OLLAMA_HOST": OLLAMA[7:], "OLLAMA_NUM_PARALLEL": str(self.args.specialists + 1), "OLLAMA_MAX_LOADED_MODELS": "1"}
        return subprocess.Popen(["ollama", "serve"], stdout=open(HERE / "ollama.log", "w"), stderr=subprocess.STDOUT, env=env)

    def complete(self, k, prompt, n):
        options = {"num_ctx": self.args.context + 4096, "num_predict": n, "temperature": 0}
        body = {"model": self.args.model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": -1, "options": options}
        request = urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps(body).encode())
        return json.load(urllib.request.urlopen(request, timeout=self.args.timeout))["response"]


class Llama(Http):
    def launch(self):
        self.ready = f"{LLAMA}/health"
        slots = self.args.specialists + 1
        model = next(LLAMA_MODELS.glob("*/*.gguf"))
        command = [LLAMA_SERVER, "-m", model, "--port", LLAMA[-4:], "-ngl", "all", "-c", str((self.args.context + 4096) * slots), "-np", str(slots)]
        return subprocess.Popen(command, stdout=open(HERE / "llama.log", "w"), stderr=subprocess.STDOUT)

    def complete(self, k, prompt, n):
        body = {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": True, "id_slot": k}
        request = urllib.request.Request(f"{LLAMA}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(request, timeout=self.args.timeout))["content"]


async def run(args, engine, peak: Peak):
    K, steps, marks = args.specialists, [], {}
    right, wrong = 0, []
    await engine.start(args)
    try:
        start = time.monotonic()
        system = f"<|im_start|>system\nYou coordinate {K} specialists who answer questions about a notebook.<|im_end|>\n"
        await engine.read(0, system + user("The notebook:\n" + notebook(0, args.context) + "\n\nReply OK once you have read it.", first=True), 16)
        marks["context_read_s"] = time.monotonic() - start
        await asyncio.gather(*(engine.fork(0, k, user(f"You are now specialist {k}. You will get questions about the notebook and tool outputs. Reply OK."), 16)
                               for k in range(1, K + 1)))
        marks["team_ready_s"] = time.monotonic() - start

        async def specialist(k, round_):
            nonlocal right
            for step in range(args.steps):
                index = round_ * args.steps + step
                ask, answer = question(k, index)
                text = user((tool_output(k, index - 1).strip() + "\n\n" if step else "") + ask)
                sent = time.monotonic()
                said = await engine.send(k, text, 16)
                steps.append(time.monotonic() - sent)
                right += answer in said
                if answer not in said:
                    wrong.append(f"{k}/{index}: {said.strip()[:50]}")
                await asyncio.sleep(args.tool)

        for round_ in range(args.rounds):
            await asyncio.gather(*(specialist(k, round_) for k in range(1, K + 1)))
            await engine.pipe(list(range(1, K + 1)), 0, "<|im_end|>\n<|im_start|>user\nThe specialists' last answers:\n",
                              "\nSummarize them in one line.<|im_end|>\n" + CUE, 64)
            await asyncio.gather(*(engine.pipe([0], k, "<|im_end|>\n<|im_start|>user\nThe coordinator's summary:\n",
                                               "\nContinue with your next question. Reply OK.<|im_end|>\n" + CUE, 16) for k in range(1, K + 1)))
        total = time.monotonic() - start
    finally:
        await engine.stop()
    row = {"engine": args.engine, "specialists": K, "context_tokens": args.context, "steps": args.steps, "rounds": args.rounds,
           "tool_s": args.tool, "total_s": round(total, 1), "context_read_s": round(marks["context_read_s"], 1),
           "team_ready_s": round(marks["team_ready_s"], 1), "spawn_s": round(marks["team_ready_s"] - marks["context_read_s"], 1),
           "step_s": round(statistics.median(steps), 2), "step_p90_s": round(sorted(steps)[int(len(steps) * 0.9)], 2),
           "correct": f"{right}/{len(steps)}", "wrong": wrong[:6], "system_growth_gb": round(peak.system / 1024, 1)}
    print(json.dumps(row))
    with open(HERE / "team.jsonl", "a") as log:
        log.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=("team", "separate", "ollama", "llama"))
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--pie", default=PIE)
    parser.add_argument("--config", default="pie-full.toml")
    parser.add_argument("--specialists", type=int, default=8)
    parser.add_argument("--context", type=int, default=6000, help="tokens in the shared notebook, roughly")
    parser.add_argument("--steps", type=int, default=4, help="tool calls per specialist per round")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tool", type=float, default=2.0, help="seconds a tool takes")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    engines = {"team": Team, "separate": Separate, "ollama": Ollama, "llama": Llama}
    patterns = {"team": "target/release/pie -c", "separate": "target/release/pie -c",
                "ollama": "Ollama.app/Contents/Resources/llama-server", "llama": "llama-b11312/llama-server"}
    with Peak(patterns[args.engine]) as peak:
        asyncio.run(run(args, engines[args.engine](), peak))
