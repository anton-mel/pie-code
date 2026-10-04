"""pie-code with branch and prune: k parallel attempts per SWE-bench task, first submission wins.

    PIE_CODE_SHARE=all PYTHONPATH=swe swe/.venv/bin/python swe/bp.py --attempts 2 --tasks 16 -o swe/runs/bp-16x2

Each attempt is mini-swe-agent's agent, sandbox and prompts with a different opening hint, so the
attempts diverge. On pie every attempt is a session that maps the task's shared prompt instead
of reading it (pie_code.py); when one attempt submits a non-empty patch the others are killed,
their pie processes end, and their GPU time goes to the other tasks.
"""

import argparse
import os
import concurrent.futures
import json
import threading
import time
from pathlib import Path

import yaml
from datasets import load_dataset
from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.utils.serialize import recursive_merge

import pie_code

HERE = Path(__file__).parent
CONFIGS = Path(pie_code.__file__).parent / ".venv/lib/python3.14/site-packages/minisweagent/config/benchmarks"
HINTS = [
    "Start by writing a small script that reproduces the problem, then fix it.",
    "Start by locating the code involved with grep, read it, then fix it.",
    "Start by finding the tests that cover this code, then fix it.",
    "Start by reading the most relevant module top to bottom, then fix it.",
]


def attempt(config, instance, index, pruned, log):
    model = get_model(config=config["model"])
    model.pruned = pruned
    task = instance["problem_statement"] + f"\n\n(Approach: {HINTS[index % len(HINTS)]})"
    began, status, patch, steps = time.monotonic(), "", "", 0
    env = None
    try:
        env = get_sb_environment(config, instance)
        agent = DefaultAgent(model, env, **config.get("agent", {}))
        info = agent.run(task)
        status, patch = info.get("exit_status", ""), info.get("submission", "") or ""
        steps = sum(m.get("role") == "assistant" for m in agent.messages)
    except pie_code.Pruned:
        status = "Pruned"
    except Exception as error:
        status = type(error).__name__
    finally:
        model.close()
        if env is not None and hasattr(env, "cleanup"):
            env.cleanup()
    row = {"instance_id": instance["instance_id"], "attempt": index, "status": status, "steps": steps,
           "seconds": round(time.monotonic() - began, 1), "patch_chars": len(patch)}
    with log["lock"]:
        log["file"].write(json.dumps(row) + "\n")
        log["file"].flush()
    return status, patch


def task(config, instance, attempts, out, log, lock):
    pruned = threading.Event()
    winner = {"patch": "", "attempt": None}
    with concurrent.futures.ThreadPoolExecutor(attempts) as pool:
        futures = {pool.submit(attempt, config, instance, i, pruned, log): i for i in range(attempts)}
        for future in concurrent.futures.as_completed(futures):
            status, patch = future.result()
            if status == "Submitted" and patch.strip() and winner["attempt"] is None:
                winner.update(patch=patch, attempt=futures[future])
                pruned.set()  # the other attempts stop at their next step
    with lock:
        preds = json.loads((out / "preds.json").read_text()) if (out / "preds.json").exists() else {}
        preds[instance["instance_id"]] = {"model_name_or_path": "pie-code-bp", "instance_id": instance["instance_id"],
                                          "model_patch": winner["patch"]}
        (out / "preds.json").write_text(json.dumps(preds, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--tasks", type=int, default=16, help="tasks run at once")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    configs = [yaml.safe_load((CONFIGS / "swebench_backticks.yaml").read_text()), yaml.safe_load((HERE / "local.yaml").read_text()),
               {"model": {"model_class": "pie_code.PieCodeModel"}}]
    config = recursive_merge(*configs)
    ids = set(json.loads((HERE / os.environ.get("BP_TASKS", "tasks16.json")).read_text()))
    instances = [i for i in load_dataset("princeton-nlp/SWE-bench_Verified", split="test") if i["instance_id"] in ids]
    log = {"file": open(out / "attempts.jsonl", "a"), "lock": threading.Lock()}
    lock = threading.Lock()
    began = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(args.tasks) as pool:
        list(pool.map(lambda inst: task(config, inst, args.attempts, out, log, lock), instances))
    print(json.dumps({"engine": f"pie-code-bp-x{args.attempts}", "agents": args.tasks * args.attempts,
                      "wall_s": round(time.monotonic() - began)}))


if __name__ == "__main__":
    main()
