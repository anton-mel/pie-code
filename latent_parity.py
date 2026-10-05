"""Checks pie's latent steps against Hugging Face: same prompt, same steps, compare each latent input.

    latentmas-ref/.venv/bin/python latent_parity.py --steps 8

pie runs the 4-bit build and HF the bf16 checkpoint, so vectors match closely, not exactly. A
cosine near 1 after the prompt says the hidden readout is right; staying high across the steps
says each latent row is fed in and written to the KV and recurrent state correctly.
"""

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "latentmas-ref"))
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402

from pie_client import Event, PieClient  # noqa: E402


def port_of(config) -> int:
    """The server port a pie config names."""
    import tomllib
    return tomllib.loads((HERE / config).read_text())["server"]["port"]

WASM = HERE / "latent/target/wasm32-wasip2/release/latent.wasm"
QUESTION = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"


def hf_inputs(model, ids, steps, scale):
    out = model(input_ids=ids, use_cache=True, output_hidden_states=True)
    past, h = out.past_key_values, out.hidden_states[-1][:, -1]
    inputs = []
    for _ in range(steps + 1):
        e = (h.float() * (scale / h.float().norm(dim=-1, keepdim=True))).to(h.dtype)
        inputs.append(e[0].float())
        if len(inputs) == steps + 1:
            break
        out = model(inputs_embeds=e[:, None], past_key_values=past, use_cache=True, output_hidden_states=True)
        past, h = out.past_key_values, out.hidden_states[-1][:, -1]
    return inputs


async def pie_inputs(a, tokens):
    log = open(HERE / f"pie-latent-{port_of(a.config)}.log", "w")
    server = subprocess.Popen([a.pie, "-c", HERE / a.config, "serve"], stdout=log, stderr=log)
    try:
        while socket.socket().connect_ex(("127.0.0.1", port_of(a.config))):
            if server.poll() is not None:
                raise SystemExit("pie did not start; see pie-latent.log")
            time.sleep(0.5)
        client = PieClient(f"ws://127.0.0.1:{port_of(a.config)}")
        await client.connect()
        await client.authenticate("parity")
        program = await client.install_program(WASM, force_overwrite=True)
        process = await client.launch_process(program, {"op": "probe", "tokens": tokens, "steps": a.steps, "scale": a.scale})
        event, value = await process.recv()
        if event != Event.Message:
            raise RuntimeError(value)
        await process.signal("quit")
        await client.close()
        return [torch.tensor(v).flatten() for v in json.loads(value)["inputs"]]
    finally:
        server.terminate()
        server.wait()


def main(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    ns = SimpleNamespace(task="gsm8k", think=False, prompt="sequential", model_name=a.model)
    messages = build_agent_message_sequential_latent_mas(role="planner", question=QUESTION, context="", method="latent_mas", args=ns)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tok.encode(text, add_special_tokens=False)
    pie = asyncio.run(pie_inputs(a, tokens))
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("mps").eval()
    with torch.no_grad():
        hf = hf_inputs(model, torch.tensor([tokens], device="mps"), a.steps, a.scale)
    rows = []
    for i, (p, h) in enumerate(zip(pie, hf)):
        cos = torch.nn.functional.cosine_similarity(p, h.cpu(), dim=0).item()
        rows.append({"step": i, "cosine": round(cos, 4), "pie_norm": round(p.norm().item(), 4), "hf_norm": round(h.norm().item(), 4)})
        print(json.dumps(rows[-1]))
    (HERE / "latent-runs").mkdir(exist_ok=True)
    (HERE / "latent-runs/parity.json").write_text(json.dumps({"prompt_tokens": len(tokens), "rows": rows}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--scale", type=float, default=0.6558)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--config", default="pie-latent.toml")
    p.add_argument("--pie", default=str(HERE / "pie-mpp/target/release/pie"))
    main(p.parse_args())
