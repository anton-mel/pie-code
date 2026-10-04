"""Seconds to the first token for a new 6,000-token prompt, on a warm server.

    uv run --project pie/python/client python e2e.py <label> <pie binary> <config>     # appends to e2e.jsonl
"""
import asyncio, json, socket, subprocess, sys, time
from bench import HERE, PORT, notebook
from pie_client import PieClient

WASM = "/Users/antonmelnychuk/pie/examples/target/wasm32-wasip2/release/text_completion_bench.wasm"
label, binary, config = sys.argv[1:4]


async def main():
    server = subprocess.Popen([binary, "-c", HERE / config, "serve"], stdout=open(HERE / "pie.log", "w"), stderr=subprocess.STDOUT)
    seconds = []
    try:
        while socket.socket().connect_ex(("127.0.0.1", PORT)):
            if server.poll() is not None:
                raise SystemExit("pie did not start; see pie.log")
            time.sleep(0.5)
        client = PieClient(f"ws://127.0.0.1:{PORT}")
        await client.connect()
        await client.authenticate("bench")
        program = await client.install_program(WASM, force_overwrite=True)
        for agent, size in ((0, 300), (1, 6000), (2, 6000), (3, 6000)):  # the first is a warm-up
            ask = {"prompt": notebook(agent, size), "max_tokens": 8, "temperature": 0, "ignore_eos": True, "report_timing": True, "return_text": False}
            result = await (await client.launch_process(program, ask)).result()
            if agent:
                seconds.append(result["ttft_us"] / 1e6)
        await client.close()
    finally:
        server.terminate()
        server.wait()
    row = {"label": label, "ttft_s": [round(s, 3) for s in seconds]}
    with open(HERE / "e2e.jsonl", "a") as log:
        log.write(json.dumps(row) + "\n")
    print(json.dumps(row))


asyncio.run(main())
