"""Reproduces new sessions waiting behind finished ones (the SWE-bench stall).

    uv run --project pie/python/client --with transformers python admit_repro.py --pie pie-mpp/target/release/pie

Wave 1: 8 sessions each grow to about 15k tokens over 10 turns, then sit idle, as finished
agents do behind a proxy that never closes them. Wave 2: 8 new sessions start. Reports how long
each new session waited for its first reply; a stall shows as a timeout.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from pie_client import Event, PieClient

HERE = Path(__file__).parent
WASM = HERE / "session/target/wasm32-wasip2/release/session.wasm"
PORT = 8097


def words(n: int, seed: int) -> list[int]:
    """n token ids of plain text (ids in a range the tokenizer treats as ordinary words)."""
    return [1000 + (seed * 7919 + i * 104729) % 30000 for i in range(n)]


async def reply(process, timeout: float) -> dict:
    while True:
        event, value = await asyncio.wait_for(process.recv(), timeout)
        if event == Event.Message:
            message = json.loads(value)
            if "published" in message:
                continue
            return message
        raise RuntimeError(f"session ended: {value}")


async def main(args):
    env = {**os.environ, "PIE_CONTENTION_TRACE_MS": "2000"}
    log = open(HERE / "pie.log", "w")
    server = subprocess.Popen([args.pie, "-c", HERE / args.config, "serve"], stdout=log, stderr=log, env=env)
    try:
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            if server.poll() is not None:
                raise SystemExit("pie did not start; see pie.log")
            time.sleep(0.5)
        client = PieClient(f"ws://127.0.0.1:{PORT}")
        await client.connect()
        await client.authenticate("repro")
        program = await client.install_program(WASM, force_overwrite=True)

        async def grow(k: int):
            process = await client.launch_process(program, {"tokens": words(1500, k), "max_tokens": 8})
            await reply(process, args.timeout)
            for turn in range(9):
                await asyncio.sleep(1.0)
                await process.signal(json.dumps({"tokens": words(1500, k * 100 + turn), "max_tokens": 8}))
                await reply(process, args.timeout)
            return process  # left alive and idle, like a finished agent

        start = time.monotonic()
        idle = await asyncio.gather(*(grow(k) for k in range(8)))
        print(f"wave 1: 8 sessions at ~15k tokens in {time.monotonic() - start:.0f} s, now idle")

        async def newcomer(k: int):
            began = time.monotonic()
            process = await client.launch_process(program, {"tokens": words(2000, 50 + k), "max_tokens": 8})
            try:
                await reply(process, args.timeout)
                return round(time.monotonic() - began, 1)
            except TimeoutError:
                return None

        waits = await asyncio.gather(*(newcomer(k) for k in range(8)))
        stalled = sum(w is None for w in waits)
        print(json.dumps({"binary": args.label, "first_reply_s": waits, "stalled": stalled}))
        for process in idle:
            await process.signal("quit")
        await client.close()
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pie", default=str(HERE / "pie-mpp/target/release/pie"))
    parser.add_argument("--config", default="pie-swe.toml")
    parser.add_argument("--label", default="")
    parser.add_argument("--timeout", type=float, default=300)
    asyncio.run(main(parser.parse_args()))
