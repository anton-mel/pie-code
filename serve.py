"""An OpenAI-compatible server in front of pie that keeps every conversation alive.

    uv run --project pie/python/client --with aiohttp --with transformers --with jinja2 python serve.py

Agent apps (OpenCode, ...) resend the whole conversation on every request. Here each
conversation is one live pie process (session/), which keeps the KV of everything it has
read and parks on SSD while the app runs its tools. A request whose tokens extend a live
session's sends only the new tokens; anything else starts a session. A new session maps
the system prompt it shares with earlier ones (session/: shared memory) instead of
reading it. Listens on :8100.
"""

import argparse
import asyncio
import hashlib
import json
import re
import socket
import subprocess
import time
import uuid
from pathlib import Path

from aiohttp import web
from pie_client import Event, PieClient
from transformers import AutoTokenizer

HERE = Path(__file__).parent
PIE = HERE / "pie-mpp/target/release/pie"
WASM = HERE / "session/target/wasm32-wasip2/release/session.wasm"
SNAPSHOTS = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
PORT = 8097
MODEL = "pie-qwen3.8-27b"
CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


class Session:
    def __init__(self, process):
        self.process, self.tokens, self.lock, self.used = process, [], asyncio.Lock(), time.monotonic()

    async def turn(self, tokens: list[int], max_tokens: int, timeout: float) -> dict:
        await self.process.signal(json.dumps({"tokens": tokens, "max_tokens": max_tokens}))
        return await reply(self.process, timeout)


async def reply(process, timeout: float, published=None) -> dict:
    while True:
        event, value = await asyncio.wait_for(process.recv(), timeout)
        if event == Event.Message:
            message = json.loads(value)
            if message.get("published"):
                if published is not None:
                    published.set()
                continue
            return message
        if event in (Event.Error, Event.Return):
            raise RuntimeError(f"session ended: {value}")


def text_of(content) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def messages_for_template(messages: list[dict]) -> list[dict]:
    """OpenAI messages as the model's chat template reads them."""
    out = []
    for message in messages:
        message = {**message, "content": text_of(message.get("content"))}
        if message["role"] == "developer":
            message["role"] = "system"
        calls = []
        for call in message.get("tool_calls") or []:
            arguments = call.get("function", {}).get("arguments", "")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) and arguments else arguments
            except json.JSONDecodeError:
                arguments = {"input": arguments}
            calls.append({"function": {"name": call["function"]["name"], "arguments": arguments or {}}})
        if calls:
            message["tool_calls"] = calls
        out.append(message)
    return out


def parse(text: str) -> tuple[str, list[dict]]:
    """The reply's prose and its tool calls in OpenAI form."""
    calls = []
    for name, body in CALL.findall(text):
        arguments = {}
        for key, value in PARAM.findall(body):
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key] = value
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(arguments)}})
    prose = text.split("<tool_call>")[0].strip() if calls else text.strip()
    return prose, calls


class Server:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(str(next(SNAPSHOTS.iterdir())))
        self.sessions: list[Session] = []
        self.client = self.program = None
        self.log = open(HERE / "serve.jsonl", "a")
        self.im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.user = self.tokenizer.encode("user", add_special_tokens=False)[0]
        self.shared: dict[str, asyncio.Event] = {}  # set once a session has published that key

    def share(self, tokens: list[int]) -> dict | None:
        """The page-aligned run of tokens before the first user turn: the app's system prompt
        and tools, common to its sessions. Named by its hash."""
        if self.args.no_share:
            return None
        start = next((i for i in range(len(tokens) - 1) if tokens[i] == self.im_start and tokens[i + 1] == self.user), 0)
        at = start // self.args.page * self.args.page
        if at < 256:
            return None
        key = "share." + hashlib.sha256(json.dumps(tokens[:at]).encode()).hexdigest()[:32]
        return {"key": key, "at": at}

    async def start(self):
        self.engine = subprocess.Popen([self.args.pie, "-c", HERE / self.args.config, "serve"],
                                       stdout=open(HERE / "pie.log", "w"), stderr=subprocess.STDOUT)
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            if self.engine.poll() is not None:
                raise SystemExit("pie did not start; see pie.log")
            await asyncio.sleep(0.5)
        self.client = PieClient(f"ws://127.0.0.1:{PORT}")
        await self.client.connect()
        await self.client.authenticate("serve")
        self.program = await self.client.install_program(WASM, force_overwrite=True)

    def render(self, body: dict) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages_for_template(body["messages"]), tools=body.get("tools") or None, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        if self.args.dump:
            (HERE / f"prompt-{len(list(HERE.glob('prompt-*.txt')))}.txt").write_text(text)
        return self.tokenizer.encode(text, add_special_tokens=False)

    async def complete(self, body: dict) -> tuple[str, list[dict], dict]:
        tokens = self.render(body)
        limit = int(body.get("max_tokens") or body.get("max_completion_tokens") or self.args.max_tokens)
        limit = min(limit, self.args.max_tokens)
        start = time.monotonic()
        # The live session this request extends: the longest whose tokens are a prefix of it.
        best = None
        for session in self.sessions:
            n = len(session.tokens)
            if 0 < n < len(tokens) and not session.lock.locked() and tokens[:n] == session.tokens:
                if best is None or n > len(best.tokens):
                    best = session
        if best is not None:
            session, reused = best, len(best.tokens)
            async with session.lock:
                out = await session.turn(tokens[reused:], limit, self.args.timeout)
        else:
            reused = 0
            await self.retire()
            share = self.share(tokens)
            published = None
            if share is not None:
                if share["key"] in self.shared:
                    await self.shared[share["key"]].wait()  # mapped once its first reader publishes it
                else:
                    published = self.shared[share["key"]] = asyncio.Event()
            process = await self.client.launch_process(self.program, {"tokens": tokens, "max_tokens": limit, "share": share})
            session = Session(process)
            self.sessions.append(session)
            async with session.lock:
                try:
                    out = await reply(process, self.args.timeout, published)
                finally:
                    if published is not None and not published.is_set():
                        del self.shared[share["key"]]  # it never published; the next one reads and tries
                        published.set()
        session.tokens, session.used = tokens, time.monotonic()
        prose, calls = parse(out["text"])
        stats = {"prompt": len(tokens), "reused": reused, "mapped": out.get("mapped", 0), "missed": out.get("missed", ""), "read": out["fed"], "made": out["tokens"],
                 "ttft_s": round(out["ttft_ms"] / 1e3, 3), "engine_s": round(out["total_ms"] / 1e3, 3),
                 "wall_s": round(time.monotonic() - start, 3), "calls": len(calls), "sessions": len(self.sessions)}
        self.log.write(json.dumps(stats) + "\n")
        self.log.flush()
        print(json.dumps(stats), flush=True)
        return prose, calls, {"prompt_tokens": len(tokens), "completion_tokens": out["tokens"],
                              "total_tokens": len(tokens) + out["tokens"],
                              "prompt_tokens_details": {"cached_tokens": reused}}

    async def retire(self):
        """Keeps at most `--sessions` alive: the least recently used idle one goes."""
        idle = [s for s in self.sessions if not s.lock.locked()]
        while len(self.sessions) >= self.args.sessions and idle:
            oldest = min(idle, key=lambda s: s.used)
            idle.remove(oldest)
            self.sessions.remove(oldest)
            await oldest.process.signal("quit")

    async def chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prose, calls, usage = await self.complete(body)
        ident, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        finish = "tool_calls" if calls else "stop"
        message = {"role": "assistant", "content": prose or None}
        if calls:
            message["tool_calls"] = calls
        if not body.get("stream"):
            return web.json_response({"id": ident, "object": "chat.completion", "created": created, "model": MODEL,
                                      "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                      "usage": usage})
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)

        async def chunk(delta: dict, finish_reason=None, **extra):
            data = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra}
            await response.write(f"data: {json.dumps(data)}\n\n".encode())

        await chunk({"role": "assistant", "content": ""})
        if prose:
            await chunk({"content": prose})
        if calls:
            await chunk({"tool_calls": [{"index": i, **call} for i, call in enumerate(calls)]})
        await chunk({}, finish, usage=usage)
        await response.write(b"data: [DONE]\n\n")
        return response

    async def models(self, _):
        return web.json_response({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "pie"}]})


async def main(args):
    server = Server(args)
    await server.start()
    app = web.Application(client_max_size=64 * 2**20)
    app.router.add_post("/v1/chat/completions", server.chat)
    app.router.add_get("/v1/models", server.models)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.port).start()
    print(f"serving {MODEL} on http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        server.engine.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pie", default=PIE)
    parser.add_argument("--config", default="pie-full.toml")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--sessions", type=int, default=48, help="conversations kept alive at once")
    parser.add_argument("--max-tokens", type=int, default=4096, help="cap on tokens per reply")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--page", type=int, default=32, help="the engine's kv_page_size")
    parser.add_argument("--no-share", action="store_true", help="read every system prompt, mapping none")
    parser.add_argument("--dump", action="store_true", help="write every rendered prompt to prompt-N.txt")
    asyncio.run(main(parser.parse_args()))
