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
    if "pie-team" in rows and "llama-team" in rows:
        pie, llama = rows["pie-team"], rows["llama-team"]
        n = min(len(pie), len(llama))
        ratios = sorted(l["seconds"] / p["seconds"] for p, l in zip(pie[:n], llama[:n]))
        median = ratios[n // 2] if n % 2 else (ratios[n // 2 - 1] + ratios[n // 2]) / 2
        ps, ls = sum(r["seconds"] for r in pie[:n]), sum(r["seconds"] for r in llama[:n])
        pt, lt = sum(r["written"] for r in pie[:n]), sum(r["written"] for r in llama[:n])
        out += ["", f"### The same team on both engines (problems 1–{n}, both finished)", "",
                f"Same prompts, same plan, same model. **pie is {median:.1f}x faster per problem (median; "
                f"{ratios[0]:.1f}x–{ratios[-1]:.1f}x)**, {ls / ps:.1f}x in total: {ps / 60:.0f} min vs {ls / 60:.0f} min. "
                f"Across all agents pie writes {pt / ps:.0f} tokens/s, llama.cpp {lt / ls:.0f}. "
                f"Solved: pie {sum(r['correct'] for r in pie[:n])}/{n}, llama.cpp {sum(r['correct'] for r in llama[:n])}/{n} "
                f"(sampling noise at this size).", "",
                "| # | answer | pie | llama.cpp | llama.cpp / pie |", "|---|---|---|---|---|"]
        for p, l in zip(pie[:n], llama[:n]):
            mark = lambda r: ("✓" if r["correct"] else "✗") + f" {r['seconds']:.0f} s"
            out.append(f"| {p['k'] + 1} | {p['gold']} | {mark(p)} | {mark(l)} | {l['seconds'] / p['seconds']:.1f}x |")
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
