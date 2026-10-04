"""Draws results.png from results.jsonl and capacity.jsonl.

    uv run --with matplotlib python plot.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ENGINES = {"pie": ("pie, idle agents parked on SSD", "#2a78d6"), "ollama": ("Ollama", "#eb6834")}

rows = []
for name in ("results.jsonl", "capacity.jsonl"):
    path = HERE / name
    if path.exists():
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
done = [row for row in rows if "switch_ttft_s" in row]
have = {(row["engine"], row["agents"]) for row in done}
missing = [f"{engine} at {agents} agents" for engine, agents in (("pie", 20), ("ollama", 8)) if (engine, agents) not in have]
partial = bool(missing)

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "text.color": INK})
figure, (left, right) = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
for axes in (left, right):
    axes.set_facecolor(SURFACE)
    axes.grid(axis="y", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.set_xlim(0, 22)
    axes.set_xticks(range(0, 21, 4))
    axes.set_xlabel("live agents, each holding about 6,000 tokens", color=MUTED)

for engine, (label, color) in ENGINES.items():
    points = sorted({row["agents"]: row["switch_ttft_s"] for row in done if row["engine"] == engine}.items())
    if points:
        xs, ys = zip(*points)
        left.plot(xs, ys, color=color, linewidth=0 if partial else 2, marker="o", markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        for index in {0, len(xs) - 1}:
            left.annotate(f"{ys[index]:.1f} s" if ys[index] > 1 else f"{ys[index]:.2f} s", (xs[index], ys[index]),
                          textcoords="offset points", xytext=(0, 10), ha="center", color=INK)
    curves = [row for row in rows if row["engine"] == engine and row.get("growth_gb_by_agents")]
    if curves:
        growth = max(curves, key=lambda row: len(row["growth_gb_by_agents"]))["growth_gb_by_agents"]
        right.plot(range(1, len(growth) + 1), growth, color=color, linewidth=2, label=label)
        right.plot(len(growth), growth[-1], "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        right.annotate(f"{growth[0]:.1f} GB", (1, growth[0]), textcoords="offset points", xytext=(4, -16), color=INK)
        right.annotate(f"{growth[-1]:.1f} GB", (len(growth), growth[-1]), textcoords="offset points", xytext=(8, -4), color=INK)

left.set_title("Compute spent before an agent's next reply starts", loc="left", color=INK, fontsize=13, pad=14)
left.set_ylabel("seconds of GPU work per turn", color=MUTED)
left.set_ylim(0, 17)
right.set_title("Memory the Mac gives up, by number of live agents", loc="left", color=INK, fontsize=13, pad=14)
right.set_ylabel("GB above idle", color=MUTED)
right.set_ylim(0, 34)
for axes in (left, right):
    axes.legend(frameon=False, loc="upper left", labelcolor=INK)

stalled = next((row for row in rows if row.get("stalled") and row.get("agents_launched") == 1), None)
notes = ["One 27B model (Qwen3.8-27B, 4-bit) on a 48 GB M5 Max. Agents take turns; every turn returns to an agent that has been idle.",
         "Past 4 agents Ollama re-reads the agent's 6,000 tokens on most turns (its RAM cache is full); pie reads only the new message."]
if stalled:
    notes.append("Stock pie, not shown: stalls at the second agent when the device pool holds one. The pie line is the patched engine.")
if partial:
    notes.append("PARTIAL: not yet run: " + ", ".join(missing) + ". Points are not joined until those land.")
figure.suptitle("Many agents, one model: what each turn costs, and the memory behind it" + ("  (partial)" if partial else ""), x=0.045, ha="left", fontsize=16, color=INK)
figure.text(0.045, 0.02, "\n".join(notes), color=MUTED, fontsize=10, va="bottom")
figure.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.24, wspace=0.18)
figure.savefig(HERE / "results.png", dpi=160, facecolor=SURFACE)
print(HERE / "results.png")

# together.png: agents with their own contexts, all asked at the same moment.
together = HERE / "together8.jsonl"
if together.exists():
    runs = {row["engine"]: row for row in map(json.loads, together.read_text().splitlines())}
    figure, axes = plt.subplots(figsize=(8.5, 6), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.grid(axis="y", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    for engine, (label, color) in ENGINES.items():
        sizes, seconds, _ = zip(*runs[engine]["together"])
        axes.plot(sizes, seconds, color=color, linewidth=2, marker="o", markersize=9, markeredgecolor=SURFACE, markeredgewidth=2,
                  label=label.replace(", idle agents parked on SSD", ", all agents decoded in one batch"))
        axes.annotate(f"{seconds[-1]:.1f} s", (sizes[-1], seconds[-1]), textcoords="offset points", xytext=(12, -4), color=INK)
    axes.set_xticks(sizes)
    axes.set_xlim(0.5, 9.2)
    axes.set_ylim(0, 85)
    axes.set_xlabel("agents asked at the same moment (8 alive, about 6,000 tokens each)", color=MUTED)
    axes.set_ylabel("seconds until every agent has answered", color=MUTED)
    axes.legend(frameon=False, loc="upper left", labelcolor=INK)
    figure.suptitle("Eight agents at work at once: time until all have answered", x=0.07, ha="left", fontsize=15, color=INK)
    notes = ["One 27B model (Qwen3.8-27B, 4-bit) on a 48 GB M5 Max; 48-token answers; one run per engine.",
             "Ollama answers one agent at a time on this model and re-reads a context once its RAM cache is full.",
             f"Memory above idle: pie {runs['pie']['system_growth_gb']:.1f} GB, Ollama {runs['ollama']['system_growth_gb']:.1f} GB."]
    figure.text(0.07, 0.02, "\n".join(notes), color=MUTED, fontsize=10, va="bottom")
    figure.subplots_adjust(left=0.1, right=0.93, top=0.9, bottom=0.26)
    figure.savefig(HERE / "together.png", dpi=160, facecolor=SURFACE)
    print(HERE / "together.png")

# How fast pie reads new text, branch by branch: one chart per result file.
CHARTS = (
    ("prefill.jsonl", "prefill.png", "pie reads prompts 2x faster on M5 Macs (light: before, dark: after)",
     (("before", "#86b6ef"), ("after", "#2a78d6"))),
    ("prefill.jsonl", "prefill-attention.png", "pie reads prompts 2.7x faster on M5 Macs (main, matmul fix, attention fix)",
     (("before", "#b7d3f6"), ("after", "#6da7ec"), ("attention", "#1c5cab"))),
    ("prefill-moe.jsonl", "prefill-moe.png", "35B MoE on M5: pie reads prompts up to 2x faster (main, earlier fixes, MoE fix)",
     (("main", "#b7d3f6"), ("dense", "#6da7ec"), ("moe", "#1c5cab"))),
    ("prefill-m2.jsonl", "prefill-m2.png", "M2: pie reads long prompts up to 1.5x faster (light: before, dark: after)",
     (("before", "#86b6ef"), ("after", "#2a78d6"))),
)
for source, name, title, series in CHARTS:
    if not (HERE / source).exists():
        continue
    builds = {row["build"]: row["read_tok_s"] for row in map(json.loads, (HERE / source).read_text().splitlines())}
    if any(build not in builds for build, _ in series):
        continue
    sizes = list(builds[series[0][0]])
    figure, axes = plt.subplots(figsize=(8.5, 6), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.grid(axis="y", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    width = 0.8 / len(series)
    for at, (build, color) in enumerate(series):
        values = [builds[build][size] for size in sizes]
        shift = (at - (len(series) - 1) / 2) * width
        bars = axes.bar([x + shift for x in range(len(sizes))], values, width=width - 0.03, color=color)
        for bar, value in zip(bars, values):
            axes.annotate(f"{value:,}", (bar.get_x() + bar.get_width() / 2, value), textcoords="offset points", xytext=(0, 5), ha="center", color=INK)
    axes.set_xticks(range(len(sizes)))
    axes.set_xticklabels([f"{int(size):,} new tokens" if size.isdigit() else size for size in sizes])
    axes.set_ylim(0, max(max(builds[build].values()) for build, _ in series) * 1.12)
    axes.set_ylabel("tokens generated per second, all agents together" if source == "decode.jsonl" else "tokens read per second", color=MUTED)
    figure.suptitle(title, x=0.07, ha="left", fontsize=14, color=INK)
    figure.subplots_adjust(left=0.11, right=0.96, top=0.9, bottom=0.1)
    figure.savefig(HERE / name, dpi=160, facecolor=SURFACE)
    print(HERE / name)

# scan.png: the staged recurrent scan on M5, as kernel time and as time to the first token.
if (HERE / "scan.jsonl").exists() and (HERE / "e2e.jsonl").exists():
    kernel = {row["build"]: row["read_tok_s"] for row in map(json.loads, (HERE / "scan.jsonl").read_text().splitlines())}
    runs = {}
    for row in map(json.loads, (HERE / "e2e.jsonl").read_text().splitlines()):
        runs.setdefault(row["label"], []).extend(row["ttft_s"])
    mean = {label: sum(values) / len(values) for label, values in runs.items()}
    models = (("Dense 27B", "dense"), ("Expert 35B", "moe"))
    panels = (
        ("Scan kernel, ms per 2,000-token pass", [kernel["before"][name] for name, _ in models], [kernel["after"][name] for name, _ in models], "{:.0f}"),
        ("Time to first token, 6,000-token prompt (s)", [mean[f"{key}-before"] for _, key in models], [mean[f"{key}-after"] for _, key in models], "{:.2f}"),
    )
    figure, both = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
    for axes, (label, before, after, form) in zip(both, panels):
        axes.set_facecolor(SURFACE)
        axes.grid(axis="y", color=GRID, linewidth=1)
        axes.set_axisbelow(True)
        axes.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.spines["bottom"].set_color(GRID)
        for values, color, shift in ((before, "#86b6ef", -0.2), (after, "#2a78d6", 0.2)):
            bars = axes.bar([x + shift for x in range(len(models))], values, width=0.37, color=color)
            for bar, value in zip(bars, values):
                axes.annotate(form.format(value), (bar.get_x() + bar.get_width() / 2, value), textcoords="offset points", xytext=(0, 5), ha="center", color=INK)
        axes.set_xticks(range(len(models)))
        axes.set_xticklabels([name for name, _ in models])
        axes.set_ylim(0, max(before + after) * 1.12)
        axes.set_ylabel(label, color=MUTED)
    figure.suptitle("M5: staged recurrent scan (light: before, dark: after)", x=0.045, ha="left", fontsize=15, color=INK)
    figure.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.1, wspace=0.2)
    figure.savefig(HERE / "scan.png", dpi=160, facecolor=SURFACE)
    print(HERE / "scan.png")

# tools.png and tools-sweeps.png: agents in a tool loop on pie, llama.cpp and Ollama (TOOL-LOOP.md).
TOOL_ENGINES = (
    ("pie", ("full-pool",), "pie, default GPU pool", "#2a78d6"),
    ("pie", ("parked", "parked-fixed"), "pie, small GPU pool (low memory)", "#9cc3f2"),
    ("llama", ("slot-per-agent",), "llama.cpp, a slot per agent", "#6b6b66"),
    ("ollama", ("default",), "Ollama", "#eb6834"),
)


def tool_rows():
    rows = []
    for name in ("tools-first.jsonl", "tools.jsonl"):
        if (HERE / name).exists():
            for row in map(json.loads, (HERE / name).read_text().splitlines()):
                rows.append({"variant": "default", **row})
    return rows


def tool_value(rows, engine, variants, key, **where):
    """Mean of `key` over the runs of one engine at one setting; None when there are none."""
    hits = [row[key] for row in rows if row["engine"] == engine and row["variant"] in variants
            and all(row[k] == v for k, v in where.items())]
    return sum(hits) / len(hits) if hits else None


def tool_panel(axes, rows, xs, labels, key, scale, fmt, heading, **fixed):
    axes.set_facecolor(SURFACE)
    axes.grid(axis="y", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    width, top = 0.8 / len(TOOL_ENGINES), 0
    for at, (engine, variants, label, color) in enumerate(TOOL_ENGINES):
        spots, values = [], []
        for x, (field, value) in enumerate(xs):
            found = tool_value(rows, engine, variants, key, **{field: value}, **fixed)
            if found is not None:
                spots.append(x)
                values.append(found / scale)
        shift = (at - (len(TOOL_ENGINES) - 1) / 2) * width
        bars = axes.bar([x + shift for x in spots], values, width=width - 0.03, color=color, label=label)
        for bar, value in zip(bars, values):
            axes.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value), textcoords="offset points",
                          xytext=(0, 4), ha="center", color=INK, fontsize=9)
        top = max([top, *values])
    axes.set_ylim(0, top * 1.12)
    axes.set_xticks(range(len(xs)))
    axes.set_xticklabels(labels)
    axes.set_title(heading, loc="left", fontsize=12, color=MUTED)


if (HERE / "tools.jsonl").exists():
    rows = tool_rows()
    base = {"context_tokens": 6000, "tool_s": 2.0}
    agents = [1, 2, 4, 8, 16]
    xs = [("agents", n) for n in agents]
    labels = [f"{n} agent" + ("s" if n > 1 else "") for n in agents]
    figure, both = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
    tool_panel(both[0], rows, xs, labels, "step_s", 1, "{:.1f}", "Seconds per tool step (median)", **base)
    tool_panel(both[1], rows, xs, labels, "loop_s", 60, "{:.1f}", "Minutes until every agent has finished", **base)
    both[0].legend(frameon=False, loc="upper left")
    figure.suptitle("Agents calling tools on an M5 Max, 6,000 tokens each", x=0.05, ha="left", fontsize=14, color=INK)
    figure.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.08, wspace=0.12)
    figure.savefig(HERE / "tools.png", dpi=160, facecolor=SURFACE)
    print(HERE / "tools.png")

    figure, both = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
    contexts = [2000, 6000, 12000]
    tool_panel(both[0], rows, [("context_tokens", c) for c in contexts], [f"{c:,} tokens" for c in contexts],
               "loop_s", 60, "{:.1f}", "By context per agent: minutes until 8 agents finish", agents=8, tool_s=2.0)
    tools = [0.0, 2.0, 10.0]
    tool_panel(both[1], rows, [("tool_s", t) for t in tools], [f"{t:.0f} s tools" for t in tools],
               "step_s", 1, "{:.1f}", "By tool time: seconds per step, 8 agents", agents=8, context_tokens=6000)
    both[0].legend(frameon=False, loc="upper left")
    figure.suptitle("8 agents on an M5 Max: pie's lead grows with longer contexts", x=0.05, ha="left", fontsize=14, color=INK)
    figure.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.08, wspace=0.12)
    figure.savefig(HERE / "tools-sweeps.png", dpi=160, facecolor=SURFACE)
    print(HERE / "tools-sweeps.png")
