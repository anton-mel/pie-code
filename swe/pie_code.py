"""pie-code: mini-swe-agent's coding loop, run natively on pie.

    PYTHONPATH=swe MSWEA_COST_TRACKING=ignore_errors swe/.venv/bin/mini-extra swebench ... \\
        --model-class pie_code.PieCodeModel

Same prompts, sandbox and parsing as mini-swe-agent, so solve rates compare. What changes is the
engine interface: each agent is a pie session process (../session) that keeps the KV of its whole
conversation and parks on SSD while its command runs; the harness sends it only the tokens that
are new, with no HTTP and no re-rendering; and a new agent maps the system prompt it shares with
earlier ones, named by its content, instead of reading it.
"""

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import litellm
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PIE = ROOT / "pie-mpp/target/release/pie"
CONFIG = ROOT / os.environ.get("PIE_CODE_CONFIG", "pie-swe.toml")
SESSION = ROOT / "session/target/wasm32-wasip2/release/session.wasm"
SNAPSHOTS = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
PORT, PAGE, SHARE_EVERY = 8097, 32, 1024
# Checkpoints only up to the first user turn (agents on different tasks share the system prompt),
# or through the whole first prompt (attempts at the same task share all of it but their hint).
SHARE_ALL = os.environ.get("PIE_CODE_SHARE", "") == "all"
LOG = ROOT / "swe/pie_code.jsonl"


class Engine:
    """One pie server and one client for every agent of this run, on a loop of their own."""

    _one = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "Engine":
        with cls._lock:
            if cls._one is None:
                cls._one = Engine()
            return cls._one

    def __init__(self):
        from pie_client import PieClient

        self.tokenizer = AutoTokenizer.from_pretrained(str(next(SNAPSHOTS.iterdir())))
        self.im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.user = self.tokenizer.encode("user", add_special_tokens=False)[0]
        self.server = subprocess.Popen([PIE, "-c", CONFIG, "serve"], stdout=open(ROOT / "pie.log", "w"), stderr=subprocess.STDOUT)
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            if self.server.poll() is not None:
                raise RuntimeError("pie did not start; see pie.log")
            time.sleep(0.5)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.shared: dict[str, asyncio.Event] = {}

        async def connect():
            client = PieClient(f"ws://127.0.0.1:{PORT}")
            await client.connect()
            await client.authenticate("pie-code")
            return client, await client.install_program(SESSION, force_overwrite=True)

        self.client, self.program = self.run(connect())

    def run(self, coroutine, timeout: float = 3600):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    def render(self, messages: list[dict]) -> list[int]:
        text = self.tokenizer.apply_chat_template([{"role": m["role"], "content": m["content"]} for m in messages],
                                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def checkpoints(self, tokens: list[int]) -> list[dict]:
        """Content-named checkpoints every SHARE_EVERY tokens up to the first user turn."""
        if SHARE_ALL:
            start = len(tokens) - 1
        else:
            start = next((i for i in range(len(tokens) - 1) if tokens[i] == self.im_start and tokens[i + 1] == self.user), 0)
        end = start // PAGE * PAGE
        points = list(range(SHARE_EVERY, end, SHARE_EVERY)) + ([end] if end >= 256 else [])
        digest, out, at = hashlib.sha256(), [], 0
        for point in points:
            digest.update(json.dumps(tokens[at:point]).encode())
            at = point
            out.append({"key": "share." + digest.copy().hexdigest()[:32], "at": point})
        return out

    async def reply(self, process, published: dict) -> dict:
        from pie_client import Event

        while True:
            event, value = await process.recv()
            if event != Event.Message:
                raise RuntimeError(f"session ended: {value}")
            message = json.loads(value)
            if "published" in message:
                if message["published"] in published:
                    published[message["published"]].set()
                continue
            return message

    async def start(self, tokens: list[int], limit: int):
        """A new agent: wait for the longest checkpoint another agent is still reading, then launch."""
        share = self.checkpoints(tokens)
        known = [p for p in share if p["key"] in self.shared]
        if known:
            await self.shared[known[-1]["key"]].wait()
        mine = {p["key"]: asyncio.Event() for p in share if p["key"] not in self.shared}
        self.shared.update(mine)
        process = await self.client.launch_process(self.program, {"tokens": tokens, "max_tokens": limit, "share": share})
        try:
            return process, await self.reply(process, mine)
        finally:
            for key, event in mine.items():
                if not event.is_set():
                    del self.shared[key]
                    event.set()

    async def turn(self, process, tokens: list[int], limit: int) -> dict:
        await process.signal(json.dumps({"tokens": tokens, "max_tokens": limit}))
        return await self.reply(process, {})


class PieCodeModel(LitellmTextbasedModel):
    """A mini-swe-agent model whose every agent is a live pie session."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engine = Engine.get()
        self.process, self.read = None, []
        self.pruned: threading.Event | None = None  # set by a runner to stop this agent: its pie process ends
        self.abort_exceptions = [*getattr(self, "abort_exceptions", []), Pruned]  # a pruned attempt stops; never retried

    def close(self):
        """Ends this agent's pie process, freeing its KV and state at once."""
        if self.process is not None:
            try:
                self.engine.run(self.process.signal("quit"), timeout=10)
            except Exception:
                pass
            self.process = None

    def _query(self, messages: list[dict], **kwargs):
        if self.pruned is not None and self.pruned.is_set():
            self.close()
            raise Pruned("another attempt at this task already submitted")
        engine = self.engine
        limit = int(self.config.model_kwargs.get("max_tokens", 2048))
        tokens = engine.render(messages)
        began = time.monotonic()
        if self.process is not None and tokens[: len(self.read)] == self.read:
            new = tokens[len(self.read):]
            out = engine.run(engine.turn(self.process, new, limit))
        else:  # the first query, or a history that no longer extends what the agent read
            if self.process is not None:
                engine.run(self.process.signal("quit"))
            self.process, out = engine.run(engine.start(tokens, limit))
        self.read = tokens
        with open(LOG, "a") as log:
            log.write(json.dumps({"prompt": len(tokens), "read": out["fed"], "mapped": out.get("mapped", 0), "made": out["tokens"],
                                  "ttft_s": round(out["ttft_ms"] / 1e3, 3), "wall_s": round(time.monotonic() - began, 3)}) + "\n")
        return litellm.ModelResponse(
            model="pie-code", choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": out["text"]}}],
            usage={"prompt_tokens": len(tokens), "completion_tokens": out["tokens"], "total_tokens": len(tokens) + out["tokens"]})

    def __del__(self):
        if getattr(self, "process", None) is not None:
            self.close()


class Pruned(Exception):
    """This attempt was killed: another attempt at the same task submitted first."""
