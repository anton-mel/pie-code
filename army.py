"""An agent army on one hard question: pie runs it in one process, llama.cpp runs the same algorithm.

    python army.py ask "question"                      # live view of every agent, then the answer
    python army.py ask "question" --engine llama       # the same team on llama.cpp
    python army.py bench --mode team --n 15            # AIME 2025, scored; also --mode single|vote
    python army.py bench --engine llama --mode team

On pie the team lives in one process (army/): an agent is a copy-on-write fork, a finding is
appended once to a blackboard new agents fork from and handed to running agents as a note, a
killed agent's memory is freed at once, and each round surer agents get more tokens. On llama.cpp
the same prompts and the same plan run over its HTTP API with parallel slots and prompt caching: an
agent re-reads its text where pie forks, and is re-issued with a note where pie writes into a
running context. Results go to army-runs/.
"""

import argparse
import asyncio
import json
import re
import socket
import subprocess
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
WASM = HERE / "army/target/wasm32-wasip2/release/army.wasm"
TOKENIZER = next((Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots").iterdir())
GGUF = next((Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots").glob("*/*.gguf"), None)
LLAMA = Path.home() / ".cache/pie-evals/baselines/llama-b11312/llama-server"
LLAMA_PORT = 8098
OPEN, CLOSE, SEAL = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_end|>\n"
ASK = "{q}\n\nReason step by step, and put your final answer within \\boxed{{}}."


def rust_constants() -> dict:
    """The prompts and limits army/src/lib.rs runs with, so the llama.cpp team cannot drift from it."""
    src = (HERE / "army/src/lib.rs").read_text()
    out = {}
    for name, body in re.findall(r'const (\w+): &str = "((?:[^"\\]|\\.)*)";', src, re.S):
        out[name] = re.sub(r"\\\n\s*", "", body).replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
    for name, value in re.findall(r"const (\w+): usize = ([\d_]+);", src):
        out[name] = int(value.replace("_", ""))
    return out


def boxed(text: str):
    at = text.rfind("\\boxed{")
    if at < 0:
        return None
    depth, out = 0, ""
    for c in text[at + 7:]:
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return out.strip()
            depth -= 1
        out += c
    return None


def same(pred, gold: str) -> bool:
    if not pred:
        return False
    pred = re.sub(r"\\text\{[^}]*\}|\\,|\s|\$|\^\{?\\circ\}?", "", pred)
    try:
        return int(float(pred)) == int(gold)
    except ValueError:
        return pred == gold


def aime(n: int):
    from datasets import load_dataset
    rows = []
    for config in ("AIME2025-I", "AIME2025-II"):
        rows += [{"question": r["question"], "answer": str(r["answer"]).strip()} for r in load_dataset("opencompass/AIME2025", config, split="test")]
    return rows[:n]


class View:
    """The live team, redrawn in place in the terminal."""

    COLORS = {"live": "\033[34m", "done": "\033[32m", "killed": "\033[90m"}

    def __init__(self, live: bool):
        self.live, self.lines, self.said = live, 0, []

    def say(self, text: str):
        self.said.append(text)

    def board(self, b: dict):
        if not self.live:
            return
        rows = [f"\033[1mwave {b['wave']} · {len(b['agents'])} agents · {sum(a['state'] == 'live' for a in b['agents'])} running · "
                f"{b['findings']} findings shared · {b['written']:,} tokens · {b['seconds']:.0f} s\033[0m"]
        for a in b["agents"]:
            mark = "★" if a["winner"] else {"live": "●", "done": "✓", "killed": "✗"}[a["state"]]
            what = a["answer"] or a["finding"] or a["task"]
            tail = f"doubt {a['unsure']:.2f}" if a["state"] == "live" else a["fate"]
            rows.append(f" {self.COLORS[a['state']]}{mark} {a['id']:>2} {a['role']:<8}\033[0m {a['tokens']:>5} tok  {tail:<10} {what[:90]}")
        rows += [f" \033[35m» {s[:150]}\033[0m" for s in self.said[-6:]]
        sys.stdout.write(f"\033[{self.lines}F\033[J" if self.lines else "")
        sys.stdout.write("\n".join(rows) + "\n")
        sys.stdout.flush()
        self.lines = len(rows)


# ── pie ─────────────────────────────────────────────────────────────────────────────────────────

class Pie:
    def __init__(self, config: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
        self.config = HERE / config
        self.port = tomllib.loads(self.config.read_text())["server"]["port"]

    async def __aenter__(self):
        from pie_client import PieClient
        self.server = subprocess.Popen([str(HERE / "pie-mpp/target/release/pie"), "-c", str(self.config), "serve"],
                                       stdout=open(HERE / "pie-army.log", "w"), stderr=subprocess.STDOUT)
        while socket.socket().connect_ex(("127.0.0.1", self.port)):
            if self.server.poll() is not None:
                raise SystemExit("pie did not start; see pie-army.log")
            time.sleep(0.5)
        self.client = PieClient(f"ws://127.0.0.1:{self.port}")
        await self.client.connect()
        await self.client.authenticate("army")
        self.program = await self.client.install_program(WASM, force_overwrite=True)
        return self

    async def __aexit__(self, *_):
        await self.client.close()
        self.server.terminate()
        self.server.wait()

    def enc(self, text: str):
        return self.tok.encode(text, add_special_tokens=False)

    async def run(self, question: str, mode: str, width: int, budget: int, seed: int, view: View) -> dict:
        from pie_client import Event
        full = self.tok.apply_chat_template([{"role": "user", "content": ASK.format(q=question)}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        head = full[: full.rindex("<|im_end|>")]
        job = {"head": self.enc(head), "turn_open": self.enc(SEAL + OPEN), "turn_close": self.enc(CLOSE), "seal": self.enc(SEAL),
               "stops": [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id],
               "mode": mode, "width": width, "budget": budget, "seed": seed}
        process = await self.client.launch_process(self.program, job)
        traces: dict = {}
        while True:
            event, value = await process.recv()
            if event != Event.Message:
                raise RuntimeError(f"army process ended: {value}")
            m = json.loads(value)
            if m["event"] == "say":
                view.say(m["text"])
            elif m["event"] == "board":
                view.board(m)
            elif m["event"] == "trace":
                traces[m["id"]] = traces.get(m["id"], "") + m["text"]
            elif m["event"] == "done":
                m["traces"] = traces
                return m


# ── llama.cpp: the same team over HTTP ──────────────────────────────────────────────────────────

class Llama:
    """The army's algorithm on llama.cpp: one slot per agent where slots allow, prompt caching on."""

    def __init__(self, slots: int):
        self.slots = slots
        self.k = rust_constants()

    async def __aenter__(self):
        import aiohttp
        if not (GGUF and LLAMA.exists()):
            raise SystemExit("llama.cpp server or the Qwen3.8-27B GGUF is missing")
        self.server = subprocess.Popen([str(LLAMA), "-m", str(GGUF), "--port", str(LLAMA_PORT), "-ngl", "all",
                                        "-c", str(32768 * self.slots), "-np", str(self.slots), "-ctk", "q8_0", "-ctv", "q8_0"],
                                       stdout=open(HERE / "llama-army.log", "w"), stderr=subprocess.STDOUT)
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200))
        for _ in range(600):
            try:
                async with self.http.get(f"http://127.0.0.1:{LLAMA_PORT}/health") as r:
                    if r.status == 200:
                        return self
            except Exception:
                pass
            await asyncio.sleep(1)
        raise SystemExit("llama.cpp did not start; see llama-army.log")

    async def __aexit__(self, *_):
        await self.http.close()
        self.server.terminate()
        self.server.wait()

    async def complete(self, prompt: str, n: int) -> tuple[str, int, bool]:
        body = {"prompt": prompt, "n_predict": n, "temperature": 0.7, "top_p": 0.95, "cache_prompt": True, "stop": ["<|im_end|>"]}
        async with self.http.post(f"http://127.0.0.1:{LLAMA_PORT}/completion", json=body) as r:
            out = await r.json()
        return out["content"], out.get("tokens_predicted", 0), bool(out.get("stopped_eos") or out.get("stopped_word"))

    async def run(self, question: str, mode: str, width: int, budget: int, seed: int, view: View) -> dict:
        k, began = self.k, time.monotonic()
        head = "<|im_start|>user\n" + ASK.format(q=question)
        turn = lambda text: SEAL + OPEN + text + CLOSE
        written = 0
        agents: list[dict] = []

        def snapshot(winner=None):
            view.board({"wave": wave, "written": written, "seconds": time.monotonic() - began, "findings": len(board_notes),
                        "agents": [{"id": i + 1, "role": a["role"], "task": a["task"], "state": a["state"], "fate": a["fate"],
                                    "tokens": a["tokens"], "unsure": 0.0, "answer": a.get("answer"), "finding": a.get("finding"),
                                    "winner": winner == i} for i, a in enumerate(agents)]})

        async def advance(a: dict, n: int):
            nonlocal written
            text, made, stopped = await self.complete(a["prompt"] + a["text"], n)
            written += made
            a["text"] += text
            a["tokens"] += made
            return stopped or a["tokens"] >= a["max"]

        def spawn(base: str, role: str, task: str, ask: str, cap: int):
            agents.append({"prompt": base + turn(ask), "text": "", "role": role, "task": task, "state": "live", "fate": "",
                           "tokens": 0, "max": cap})
            return agents[-1]

        board_notes: list[str] = []
        wave = 0

        def decided() -> bool:
            votes = Counter(a["answer"] for a in agents if a["state"] == "done" and a.get("answer") and a["role"] in ("solver", "referee"))
            top = votes.most_common(2)
            if mode == "team" and top and top[0][1] >= k["QUORUM"] and (len(top) < 2 or top[0][1] > top[1][1]):
                for a in agents:
                    if a["state"] == "live":
                        a["state"], a["fate"] = "killed", "stopped"
                return True
            return False

        async def until_done(busy: list) -> bool:
            """Rounds until `busy` holds no live agent, as the pie team does; true once decided."""
            while any(a["state"] == "live" for a in busy) and written < budget:
                live = [a for a in agents if a["state"] == "live"]
                finished = await asyncio.gather(*(advance(a, k["ROUND"]) for a in live))
                for a, fin in zip(live, finished):
                    reported = a["role"].startswith("T") and "RESULT:" in a["text"] and "\n" in a["text"][a["text"].rfind("RESULT:"):]
                    if not (fin or reported):
                        continue
                    a["state"] = "done"
                    if a["role"] == "main":
                        a["fate"] = "planned"
                    elif a["role"].startswith("T"):
                        found = a["text"][a["text"].rfind("RESULT:") + 7:].strip().split("\n")[0] if "RESULT:" in a["text"] else a["text"][-300:]
                        a["finding"], a["fate"] = found, "shared"
                        board_notes.append(SEAL + OPEN + f"Team finding from {a['role']} ({a['task']}): {found}")
                        for other in agents:
                            if other["state"] == "live" and other["role"].startswith("T"):
                                # llama.cpp cannot write into a running generation: the note is
                                # appended and the agent re-issued, re-reading what is not cached.
                                other["text"] += turn(f"Note from teammate {a['role']}: {found}\nUse it if it helps your task, then continue.")
                    else:
                        a["answer"] = boxed(a["text"])
                        a["fate"] = "answered" if a["answer"] else "no answer"
                if decided():
                    return True
                snapshot()
            return False

        if mode == "team":
            for _ in range(k["SCOUTS"]):
                spawn(head, "solver", "scout: a direct attempt, alone", k["SOLVE_ALONE"], k["SOLVER_MAX"])
            done_tasks: list[str] = []
            settled = False
            for wave in range(1, k["WAVES"] + 1):
                if written >= budget // 2:
                    break
                least = min(k["FIRST_WAVE_MIN"], width)
                base = head + "".join(board_notes)
                ask = (k["LEAD_FIRST"].replace("{min}", str(least)).replace("{width}", str(width)) if wave == 1 else
                       k["LEAD_NEXT"].replace("{width}", str(width)) + " Already done: " + "; ".join(done_tasks) + ".")
                main = spawn(base, "main", f"plan wave {wave}", ask, 500)
                if settled := await until_done([main]):
                    break
                plan = [m.group(1).strip() for m in re.finditer(r"^\s*T\d+:\s*(.+)$", main["text"], re.M)][:width]
                plan = [t for t in plan if t not in done_tasks]
                if not plan:
                    break
                workers = [spawn(base, f"T{len(done_tasks) + i + 1}", t, k["WORK"].replace("{task}", t), k["WORKER_MAX"]) for i, t in enumerate(plan)]
                done_tasks += plan
                if settled := await until_done(workers):
                    break
            if not settled:
                base = head + "".join(board_notes)
                for _ in range(k["SOLVERS"]):
                    spawn(base, "solver", "the whole solution, from the findings", k["SOLVE"], k["SOLVER_MAX"])
                await until_done(agents)
        else:
            for _ in range(width if mode == "vote" else 1):
                spawn(head, "solver", "the whole solution, alone", k["SOLVE_ALONE"], k["SOLVER_MAX"])
            snapshot()
            await until_done(agents)
        votes = Counter(a["answer"] for a in agents if a.get("answer"))
        answer, n = votes.most_common(1)[0] if votes else ("", 0)
        winner = next((i for i, a in enumerate(agents) if a.get("answer") == answer), None)
        snapshot(winner)
        return {"answer": answer, "votes": n, "agents": len(agents), "workers": sum(a["role"].startswith("T") for a in agents),
                "findings": len(board_notes), "written": written, "seconds": round(time.monotonic() - began, 1),
                "solution": agents[winner]["text"] if winner is not None else None}


# ── commands ────────────────────────────────────────────────────────────────────────────────────

def engine(a):
    return Pie(a.config) if a.engine == "pie" else Llama(a.slots)


async def ask(a):
    async with engine(a) as e:
        view = View(live=True)
        done = await e.run(a.question, a.mode, a.width, a.budget, a.seed, view)
    print(f"\n\033[1mAnswer: {done['answer']}\033[0m  ({done['votes']} votes, {done['agents']} agents, {done['workers']} parallel workers, "
          f"{done['findings']} findings shared, {done['written']:,} tokens, {done['seconds']:.0f} s on {a.engine})")
    if a.show and done.get("solution"):
        print("\n" + done["solution"])
    if a.engine == "pie" and done.get("traces"):
        out = HERE / "army-runs/last-ask.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(done, indent=1))
        print(f"every agent's own transcript: {out}")


async def bench(a):
    rows = aime(a.n)
    out = HERE / "army-runs"
    out.mkdir(exist_ok=True)
    name = f"{a.engine}-{a.mode}"
    results = []
    async with engine(a) as e:
        for k, row in enumerate(rows):
            done = await e.run(row["question"], a.mode, a.width, a.budget, k, View(live=False))
            ok = same(done["answer"], row["answer"])
            done.pop("traces", None)
            results.append({"k": k, "gold": row["answer"], "correct": ok, **{x: done[x] for x in ("answer", "votes", "agents", "workers", "findings", "written", "seconds")}})
            print(f"#{k + 1:>2} {name}: {done['answer']!s:>8} vs {row['answer']:>4} {'✓' if ok else '✗'}  {done['agents']:>2} agents  "
                  f"{done['written']:>6,} tok  {done['seconds']:>5.0f} s", flush=True)
            (out / f"{name}.json").write_text(json.dumps(results, indent=1))
    summary = {"engine": a.engine, "mode": a.mode, "n": len(results), "solved": sum(r["correct"] for r in results),
               "seconds": round(sum(r["seconds"] for r in results), 1), "tokens": sum(r["written"] for r in results)}
    print(json.dumps(summary))
    with open(out / "summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("ask", "bench"):
        s = sub.add_parser(name)
        if name == "ask":
            s.add_argument("question")
            s.add_argument("--show", action="store_true", help="print the agreed solution")
        else:
            s.add_argument("--n", type=int, default=15, help="AIME 2025 problems")
        s.add_argument("--engine", choices=["pie", "llama"], default="pie")
        s.add_argument("--mode", choices=["team", "single", "vote"], default="team")
        s.add_argument("--width", type=int, default=8)
        s.add_argument("--budget", type=int, default=65536)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--slots", type=int, default=8, help="llama.cpp parallel slots")
        s.add_argument("--config", default="pie-army.toml")
    a = p.parse_args()
    asyncio.run(ask(a) if a.command == "ask" else bench(a))
