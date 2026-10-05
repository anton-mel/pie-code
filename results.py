"""Prints RESULTS.md from what the overnight runs left: the agent army on pie and llama.cpp, and
LatentMAS on pie against the authors' own code."""

import json
from pathlib import Path

HERE = Path(__file__).parent


def army() -> list[str]:
    rows = {}
    for name in ("pie-single", "pie-vote", "pie-team", "llama-single", "llama-team"):
        f = HERE / "army-runs" / f"{name}.json"
        if f.exists():
            rows[name] = json.loads(f.read_text())
    if not rows:
        return ["No agent-army results yet."]
    out = ["## Agent army: AIME 2025, Qwen3.8-27B, one MacBook",
           "",
           "| Engine · method | Solved | Total time | Mean per problem | Tokens written | Agents per problem |",
           "|---|---|---|---|---|---|"]
    for name, rs in rows.items():
        solved, secs, toks = sum(r["correct"] for r in rs), sum(r["seconds"] for r in rs), sum(r["written"] for r in rs)
        agents = sum(r["agents"] for r in rs) / len(rs)
        out.append(f"| {name.replace('-', ' · ')} | **{solved} / {len(rs)}** | {secs / 60:.0f} min | {secs / len(rs):.0f} s | {toks:,} | {agents:.1f} |")
    both = [k for k in ("pie-team", "llama-team") if k in rows]
    if len(both) == 2:
        pt, lt = (sum(r["seconds"] for r in rows[k]) for k in both)
        n = min(len(rows["pie-team"]), len(rows["llama-team"]))
        out += ["", f"Same team, same prompts, same model: pie finishes it **{lt / pt:.1f}x faster** than llama.cpp "
                    f"(over {n} problems)."]
    return out


def latent() -> list[str]:
    out = ["## LatentMAS on pie (agents that think in hidden states and share one KV working memory)", ""]
    parity = HERE / "latent-runs/parity-qwen3-4b-bf16.json"
    if parity.exists():
        cos = [r["cosine"] for r in json.loads(parity.read_text())["rows"]]
        out += [f"Engine check, Qwen3-4B bf16: pie's latent vectors vs Hugging Face, cosine by step: "
                + ", ".join(f"{c:.4f}" for c in cos[:4]) + " … (rounding drift after, as HF shows against itself).", ""]
    runs = {}
    for f in sorted((HERE / "latent-runs").glob("Qwen*.json")):
        s = json.loads(f.read_text())["summary"]
        runs[f.stem] = s
    ref = {}
    summary = HERE / "latentmas-ref/runs/summary.txt"
    if summary.exists():
        for line in summary.read_text().splitlines():
            if '"accuracy"' in line:
                name, rest = line.split(":", 1)
                ref[name.strip()] = float(rest.split('"accuracy":')[1].split()[0])
    out += ["| Model | Task | Method | pie accuracy | pie time | authors' code (HF) accuracy |", "|---|---|---|---|---|---|"]
    for name, s in runs.items():
        method = s["method"] + (str(s["steps"]) if s["method"] == "latent" else "")
        key = {"single": "single", "latent10": "latent10", "latent0": "latent0", "text": "text"}.get(method)
        r = ref.get(f"{s['task']}-{key}") if s["model"].endswith("Qwen3-4B") else None
        out.append(f"| {s['model'].split('/')[-1]} | {s['task']} | {method} | {s['accuracy'] * 100:.0f}% | {s['seconds']:.0f} s | "
                   f"{'' if r is None else f'{r * 100:.0f}%'} |")
    return out


if __name__ == "__main__":
    print("\n".join(["# Results", ""] + army() + [""] + latent()))
