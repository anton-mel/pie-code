"""Whether a build writes the same tokens as main: 64 greedy tokens per prompt.

    python3 identity.py <label> <pie binary> <config>     # appends to identity.jsonl
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
label, binary, config = sys.argv[1:4]
prompts = json.loads((HERE / "identity-prompts.json").read_text())
for name, prompt in prompts.items():
    command = [binary, "-c", str(HERE / config), "run", "text-completion-bench", "--", "--prompt", prompt,
               "--max_tokens", "64", "--temperature", "0", "--ignore_eos", "--report_timing"]
    out = subprocess.run(command, capture_output=True, text=True).stdout
    found = re.search(r'"token_ids":\[([0-9,]+)\]', out)
    tokens = [int(t) for t in found.group(1).split(",")] if found else None
    with open(HERE / "identity.jsonl", "a") as log:
        log.write(json.dumps({"label": label, "prompt": name, "tokens": tokens}) + "\n")
    print(label, name, "no output" if tokens is None else f"{len(tokens)} tokens")
