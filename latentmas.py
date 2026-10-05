"""LatentMAS, TextMAS and a single agent on pie, scored exactly as the paper's code scores them.

    latentmas-ref/.venv/bin/python latentmas.py --method latent --task gsm8k --n 100 --steps 10

Prompts, datasets and answer checks are the authors' own (latentmas-ref: prompts.py, data.py,
utils.py), so only the engine differs. Each question's team is one pie process (latent/) with one
context: in LatentMAS each agent reads its prompt into it and thinks `steps` latent steps there,
and the judger answers from it, so every earlier agent's thoughts are already in its KV. All
questions' teams run at once and pie batches them.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).parent
REF = HERE / "latentmas-ref"
sys.path.insert(0, str(REF))

from data import load_arc_challenge, load_arc_easy, load_gsm8k  # noqa: E402
from methods import default_agents  # noqa: E402
from prompts import (  # noqa: E402
    build_agent_message_sequential_latent_mas,
    build_agent_messages_sequential_text_mas,
    build_agent_messages_single_agent,
)
from transformers import AutoTokenizer  # noqa: E402
from utils import extract_gsm8k_answer, normalize_answer  # noqa: E402

from pie_client import Event, PieClient  # noqa: E402


def port_of(config) -> int:
    """The server port a pie config names."""
    import tomllib
    return tomllib.loads((HERE / config).read_text())["server"]["port"]

WASM = HERE / "latent/target/wasm32-wasip2/release/latent.wasm"
PORT = 8097
LOADERS = {"gsm8k": load_gsm8k, "arc_challenge": load_arc_challenge, "arc_easy": load_arc_easy}


class Team:
    """One question's agents on pie: one process, one context."""

    def __init__(self, client, program, tok, args):
        self.client, self.program, self.tok, self.args = client, program, tok, args
        self.process = None
        self.stops = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
        self.prompt_tokens = self.made_tokens = 0

    def render(self, messages) -> list[int]:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.args.think:
            text += "<think>"
        return self.tok.encode(text, add_special_tokens=False)

    async def ask(self, request: dict) -> dict:
        if self.process is None:
            self.process = await self.client.launch_process(self.program, request)
        else:
            await self.process.signal(json.dumps(request))
        event, value = await self.process.recv()
        if event != Event.Message:
            raise RuntimeError(f"team process ended: {value}")
        reply = json.loads(value)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    async def think(self, messages):
        tokens = self.render(messages)
        self.prompt_tokens += len(tokens)
        await self.ask({"op": "think", "tokens": tokens, "steps": self.args.steps, "scale": self.args.scale})

    async def say(self, messages, fresh: bool, seed: int) -> str:
        tokens = self.render(messages)
        self.prompt_tokens += len(tokens)
        a = self.args
        reply = await self.ask({"op": "say", "tokens": tokens, "max_tokens": a.max_tokens, "temperature": a.temperature,
                                "top_p": a.top_p, "seed": seed, "stops": self.stops, "fresh": fresh})
        made = [t for t in reply["tokens"] if t not in self.stops]
        self.made_tokens += len(reply["tokens"])
        return self.tok.decode(made, skip_special_tokens=True)

    async def close(self):
        if self.process is not None:
            await self.process.signal("quit")


async def solve(team: Team, item: dict, k: int) -> str:
    a = team.args
    ns = SimpleNamespace(task=a.task, think=a.think, prompt="sequential", model_name=a.model, text_mas_context_length=-1)
    q = item["question"]
    if a.method == "single":
        ns.method = "baseline"
        return await team.say(build_agent_messages_single_agent(question=q, args=ns), True, k)
    if a.method == "text":
        context, final = "", ""
        for i, agent in enumerate(default_agents()):
            messages = build_agent_messages_sequential_text_mas(role=agent.role, question=q, context=context, method="text_mas", args=ns)
            out = (await team.say(messages, True, k * 8 + i)).strip()
            if agent.role == "judger":
                final = out
            else:
                context += f"[{agent.name}]:\n{out}\n\n"
        return final
    # latent: every agent but the judger reads its prompt into the shared context and thinks there
    for agent in default_agents():
        messages = build_agent_message_sequential_latent_mas(role=agent.role, question=q, context="", method="latent_mas", args=ns)
        if agent.role != "judger":
            await team.think(messages)
        else:
            # as the paper's code: with no latent steps the judger answers on a fresh context
            return await team.say(messages, a.steps == 0, k)


async def main(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    items = [x for _, x in zip(range(a.n), LOADERS[a.task](split="test"))]
    log = open(HERE / f"pie-latent-{port_of(a.config)}.log", "w")
    server = subprocess.Popen([a.pie, "-c", HERE / a.config, "serve"], stdout=log, stderr=log)
    try:
        while socket.socket().connect_ex(("127.0.0.1", port_of(a.config))):
            if server.poll() is not None:
                raise SystemExit("pie did not start; see pie-latent.log")
            time.sleep(0.5)
        client = PieClient(f"ws://127.0.0.1:{port_of(a.config)}")
        await client.connect()
        await client.authenticate("latentmas")
        program = await client.install_program(WASM, force_overwrite=True)
        gate = asyncio.Semaphore(a.teams)
        teams = []

        async def one(k, item):
            async with gate:
                team = Team(client, program, tok, a)
                teams.append(team)
                try:
                    text = await solve(team, item, k)
                finally:
                    await team.close()
                pred = normalize_answer(extract_gsm8k_answer(text))
                ok = bool(pred and item["gold"] and pred == item["gold"])
                print(f"#{k} pred={pred} gold={item['gold']} ok={ok}", flush=True)
                return {"k": k, "pred": pred, "gold": item["gold"], "correct": ok, "text": text}

        began = time.monotonic()
        results = await asyncio.gather(*(one(k, item) for k, item in enumerate(items)))
        secs = time.monotonic() - began
        await client.close()
    finally:
        server.terminate()
        server.wait()
    acc = sum(r["correct"] for r in results) / len(results)
    summary = {"engine": "pie", "model": a.model, "method": a.method, "task": a.task, "n": len(results), "steps": a.steps,
               "accuracy": round(acc, 4), "seconds": round(secs, 1), "teams": a.teams,
               "prompt_tokens": sum(t.prompt_tokens for t in teams), "made_tokens": sum(t.made_tokens for t in teams)}
    print(json.dumps(summary), flush=True)
    out = HERE / "latent-runs"
    out.mkdir(exist_ok=True)
    name = f"{a.model.split('/')[-1]}-{a.task}-{a.method}" + (f"{a.steps}" if a.method == "latent" else "")
    (out / f"{name}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1))
    with open(out / "summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["single", "text", "latent"], required=True)
    p.add_argument("--task", choices=list(LOADERS), default="gsm8k")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--steps", type=int, default=10, help="latent steps per agent")
    p.add_argument("--scale", type=float, default=0.6558, help="mean input-embedding norm (Qwen3.5-4B)")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--teams", type=int, default=32, help="questions in flight at once")
    p.add_argument("--think", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--config", default="pie-latent.toml")
    p.add_argument("--pie", default=str(HERE / "pie-mpp/target/release/pie"))
    asyncio.run(main(p.parse_args()))
